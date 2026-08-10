"""
BMKG Strong Motion Analyzer (BSMA)
main.py - Batch Processing Entry Point

Author: Ahmad Didane

Fungsi:
    Menjalankan pemrosesan batch waveform menggunakan pipeline BSMA
    yang sama dengan aplikasi Streamlit.

Prinsip:
    - Satuan internal acceleration : m/s²
    - Satuan internal velocity     : m/s
    - Satuan internal displacement : m
    - PGA output                   : Gal
    - PGV output                   : cm/s
    - PGD output                   : cm
    - SIG BMKG                     : berdasarkan PGA dalam Gal
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import obspy

from core.pipeline import PipelineBuilder
from core.preprocessing.baseline import BaselineCorrectionPlugin
from core.preprocessing.taper import TaperConfig, TaperPlugin
from core.preprocessing.filter import (
    ButterworthFilterPlugin,
    FilterConfig,
)
from core.preprocessing.integration import (
    IntegrationConfig,
    KinematicIntegrationPlugin,
)
from core.processing.parameters import (
    ParameterConfig,
    ParameterExtractionPlugin,
)
from core.processing.response_spectrum import (
    ResponseSpectrumConfig,
    ResponseSpectrumPlugin,
)
from core.types.context import ProcessingContext, WaveformData
from core.types.processing_state import ProcessingState

from core.io.exporter import ResultExporter


# ============================================================================
# KONFIGURASI
# ============================================================================

DATA_DIR = Path("Data/mseed")
STATION_XML_DIR = Path("Data/stationXML")

OUTPUT_DIR = Path("outputs")
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "reports"

SUMMARY_PATH = OUTPUT_DIR / "summary_parameters.csv"

GRAVITY = 9.80665


# ============================================================================
# LOGGING
# ============================================================================


def configure_logging() -> logging.Logger:
    """
    Membuat logger untuk batch processing BSMA.

    Returns
    -------
    logging.Logger
        Logger BSMA batch processor.
    """

    logger = logging.getLogger("bsma_batch")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False

    return logger


# ============================================================================
# SIG BMKG
# ============================================================================


def estimate_sig_bmkg(pga_gal: float) -> str:
    """
    Mengklasifikasikan PGA berdasarkan kategori SIG BMKG.

    Parameters
    ----------
    pga_gal
        Peak Ground Acceleration dalam Gal.

    Returns
    -------
    str
        Klasifikasi SIG BMKG.

    Notes
    -----
    Batas klasifikasi mengikuti tabel yang digunakan pada
    implementasi BSMA sebelumnya:

        < 2.9 Gal
        2.9 - <89 Gal
        89 - <167 Gal
        167 - <564 Gal
        >=564 Gal
    """

    if pga_gal < 2.9:
        return "I - TIDAK DIRASAKAN (< 2.9 gal)"

    if pga_gal < 89.0:
        return "II - DIRASAKAN (2.9 - 88 gal)"

    if pga_gal < 167.0:
        return "III - KERUSAKAN RINGAN (89 - 167 gal)"

    if pga_gal < 564.0:
        return "IV - KERUSAKAN SEDANG (168 - 563 gal)"

    return "V - KERUSAKAN BERAT (>= 564 gal)"


# ============================================================================
# PIPELINE
# ============================================================================


def build_pipeline(
    logger: logging.Logger,
    freq_min: float = 0.1,
    freq_max: float = 25.0,
    filter_type: str = "bandpass",
    damping: float = 0.05,
):
    """
    Membuat pipeline pemrosesan BSMA standar.

    Pipeline harus identik dengan pipeline yang digunakan oleh
    aplikasi Streamlit agar hasil batch dan GUI konsisten.
    """

    return (
        PipelineBuilder(
            logger=logger,
            halt_on_error=False,
        )
        .add(
            BaselineCorrectionPlugin(
                method="linear",
            )
        )
        .add(
            TaperPlugin(
                config=TaperConfig(
                    alpha=0.05,
                )
            )
        )
        .add(
            ButterworthFilterPlugin(
                config=FilterConfig(
                    type=filter_type,
                    freq_min=freq_min,
                    freq_max=freq_max,
                    corners=4,
                    zerophase=True,
                )
            )
        )
        .add(
            KinematicIntegrationPlugin(
                config=IntegrationConfig(
                    remove_mean=True,
                    remove_linear_trend=True,
                )
            )
        )
        .add(
            ParameterExtractionPlugin(
                config=ParameterConfig(
                    gravity=GRAVITY,
                )
            )
        )
        .add(
            ResponseSpectrumPlugin(
                config=ResponseSpectrumConfig(
                    damping=damping,
                    solver="nigam_jennings",
                )
            )
        )
        .build()
    )


# ============================================================================
# STATIONXML
# ============================================================================


def find_station_inventory(
    station: str,
    xml_dir: Path = STATION_XML_DIR,
) -> obspy.Inventory | None:
    """
    Mencari StationXML berdasarkan kode stasiun.

    Prioritas:
        Data/stationXML/{station}.xml

    Returns
    -------
    obspy.Inventory | None
        Inventory jika tersedia dan berhasil dibaca.
    """

    xml_path = xml_dir / f"{station}.xml"

    if not xml_path.exists():
        return None

    try:
        return obspy.read_inventory(str(xml_path))

    except Exception:
        return None


# ============================================================================
# WAVEFORM PROCESSING
# ============================================================================


def process_trace(
    trace: obspy.Trace,
    pipeline: Any,
    inventory: obspy.Inventory | None,
    logger: logging.Logger,
):
    """
    Mengubah satu ObsPy Trace menjadi ProcessingContext BSMA.

    Parameters
    ----------
    trace
        ObsPy Trace.
    pipeline
        Pipeline BSMA yang telah dibangun.
    inventory
        StationXML inventory jika tersedia.
    logger
        Logger BSMA.

    Returns
    -------
    ProcessingContext
        Context hasil pemrosesan.
    """

    # ------------------------------------------------------------------
    # Instrument response
    # ------------------------------------------------------------------

    if inventory is not None:
        try:
            trace = trace.copy()

            trace.remove_response(
                inventory=inventory,
                output="ACC",
                water_level=60,
                pre_filt=[
                    0.05,
                    0.1,
                    30.0,
                    35.0,
                ],
            )

            logger.info(
                "Instrument response berhasil dihapus: %s",
                trace.id,
            )

        except Exception as exc:
            logger.warning(
                "Instrument response gagal untuk %s: %s",
                trace.id,
                exc,
            )

    # ------------------------------------------------------------------
    # Standardisasi waveform
    # ------------------------------------------------------------------

    data = trace.data.astype("float64", copy=True)

    waveform = WaveformData(
        data=data,
        sampling_rate=float(trace.stats.sampling_rate),
        unit="m/s^2",
    )

    # ------------------------------------------------------------------
    # Processing Context
    # ------------------------------------------------------------------

    context = ProcessingContext(
        trace_id=trace.id,
        metadata=dict(trace.stats),
        raw_waveform=waveform,
        acceleration=waveform,
        processing_state=ProcessingState(),
        history=(),
    )

    return pipeline.run(context)


# ============================================================================
# RESULT EXTRACTION
# ============================================================================


def extract_result(
    context: ProcessingContext,
) -> dict[str, Any]:
    """
    Mengekstrak parameter hasil analisis menjadi dictionary CSV.

    Semua parameter diubah ke satuan output standar BSMA.
    """

    metrics = context.metrics
    metadata = context.metadata

    pga_ms2 = float(metrics.get("PGA", 0.0))
    pgv_ms = float(metrics.get("PGV", 0.0))
    pgd_m = float(metrics.get("PGD", 0.0))

    pga_gal = pga_ms2 * 100.0
    pga_percent_g = (
        pga_ms2 / GRAVITY
    ) * 100.0

    pgv_cm_s = pgv_ms * 100.0
    pgd_cm = pgd_m * 100.0

    return {
        "Network": metadata.get("network", ""),
        "Station": metadata.get("station", ""),
        "Location": metadata.get("location", ""),
        "Channel": metadata.get("channel", ""),
        "Sampling_Rate_Hz": float(
            metadata.get("sampling_rate", 0.0)
        ),
        "PGA_Gal": round(pga_gal, 4),
        "PGA_percent_g": round(pga_percent_g, 4),
        "PGV_cm_s": round(pgv_cm_s, 4),
        "PGD_cm": round(pgd_cm, 4),
        "Arias_Intensity": round(
            float(
                metrics.get(
                    "Arias_Intensity",
                    0.0,
                )
            ),
            6,
        ),
        "Significant_Duration_D5_95_s": round(
            float(
                metrics.get(
                    "Significant_Duration_D5_95",
                    0.0,
                )
            ),
            3,
        ),
        "SIG_BMKG": estimate_sig_bmkg(
            pga_gal
        ),
    }


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """
    Entry point batch processing BSMA.
    """

    logger = configure_logging()

    logger.info(
        "============================================================"
    )
    logger.info(
        "BMKG Strong Motion Analyzer - Batch Processing"
    )
    logger.info(
        "============================================================"
    )

    # ------------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------------

    if not DATA_DIR.exists():
        logger.error(
            "Direktori waveform tidak ditemukan: %s",
            DATA_DIR,
        )
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    waveform_files = sorted(
        [
            *DATA_DIR.glob("*.mseed"),
            *DATA_DIR.glob("*.miniseed"),
            *DATA_DIR.glob("*.sac"),
        ]
    )

    if not waveform_files:
        logger.warning(
            "Tidak ditemukan waveform di %s",
            DATA_DIR,
        )
        return

    logger.info(
        "Jumlah file waveform: %d",
        len(waveform_files),
    )

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    pipeline = build_pipeline(
        logger=logger,
        freq_min=0.1,
        freq_max=25.0,
        filter_type="bandpass",
        damping=0.05,
    )

    # ------------------------------------------------------------------
    # Containers
    # ------------------------------------------------------------------

    summary_results: list[dict[str, Any]] = []

    all_contexts: list[ProcessingContext] = []

    station_grouped_contexts: defaultdict[
        str,
        list[ProcessingContext],
    ] = defaultdict(list)

    # ------------------------------------------------------------------
    # Process files
    # ------------------------------------------------------------------

    for index, file_path in enumerate(
        waveform_files,
        start=1,
    ):
        logger.info(
            "[%d/%d] Memproses %s",
            index,
            len(waveform_files),
            file_path.name,
        )

        try:
            stream = obspy.read(
                str(file_path)
            )

        except Exception as exc:
            logger.error(
                "Gagal membaca %s: %s",
                file_path.name,
                exc,
            )
            continue

        if len(stream) == 0:
            logger.warning(
                "Stream kosong: %s",
                file_path.name,
            )
            continue

        # --------------------------------------------------------------
        # Process setiap trace
        # --------------------------------------------------------------

        for trace in stream:

            try:
                station = trace.stats.station

                inventory = (
                    find_station_inventory(
                        station
                    )
                )

                if inventory is None:
                    logger.warning(
                        "StationXML tidak ditemukan "
                        "untuk stasiun %s. "
                        "Waveform akan diproses "
                        "tanpa response removal.",
                        station,
                    )

                context = process_trace(
                    trace=trace,
                    pipeline=pipeline,
                    inventory=inventory,
                    logger=logger,
                )

                all_contexts.append(
                    context
                )

                station_grouped_contexts[
                    station
                ].append(context)

                summary_results.append(
                    extract_result(context)
                )

                logger.info(
                    "Selesai: %s | PGA = %.4f Gal",
                    trace.id,
                    summary_results[-1][
                        "PGA_Gal"
                    ],
                )

            except Exception as exc:
                logger.exception(
                    "Gagal memproses trace %s: %s",
                    trace.id,
                    exc,
                )

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    if summary_results:

        try:
            csv_path = (
                ResultExporter.export_to_csv(
                    summary_results,
                    output_path=SUMMARY_PATH,
                )
            )

            logger.info(
                "CSV berhasil dibuat: %s",
                csv_path,
            )

        except Exception as exc:
            logger.exception(
                "Gagal melakukan ekspor CSV: %s",
                exc,
            )

    else:
        logger.warning(
            "Tidak terdapat hasil analisis "
            "yang dapat diekspor."
        )

    # ------------------------------------------------------------------
    # Optional visualization
    # ------------------------------------------------------------------
    #
    # Visualisasi sengaja tidak diimplementasikan langsung di main.py.
    #
    # Alasannya:
    #   main.py = orchestration
    #   visualization.plots = visualization
    #
    # Jika modul BSMAVisualizer telah mengikuti ProcessingContext baru,
    # visualisasi dapat dipanggil di sini tanpa memasukkan business logic
    # ke dalam main.py.
    #
    # ------------------------------------------------------------------

    try:
        from visualization.plots import BSMAVisualizer

    except ImportError:
        logger.warning(
            "BSMAVisualizer tidak tersedia. "
            "Tahap visualisasi dilewati."
        )

    else:

        # --------------------------------------------------------------
        # Individual plots
        # --------------------------------------------------------------

        for context in all_contexts:

            try:
                metadata = context.metadata

                station = metadata.get(
                    "station",
                    "UNKNOWN",
                )

                channel = metadata.get(
                    "channel",
                    "UNKNOWN",
                )

                file_tag = (
                    f"{station}_{channel}"
                )

                BSMAVisualizer.plot_time_series(
                    context,
                    save_path=str(
                        PLOT_DIR
                        / f"{file_tag}_timeseries.png"
                    ),
                )

                BSMAVisualizer.plot_response_spectra(
                    context,
                    save_path=str(
                        PLOT_DIR
                        / f"{file_tag}_spectra.png"
                    ),
                )

            except Exception as exc:
                logger.exception(
                    "Gagal membuat plot %s: %s",
                    context.trace_id,
                    exc,
                )

        # --------------------------------------------------------------
        # Combined spectra
        # --------------------------------------------------------------

        if all_contexts:

            try:
                BSMAVisualizer.plot_combined_spectra(
                    all_contexts,
                    save_path=str(
                        PLOT_DIR
                        / "COMBINED_response_spectra.png"
                    ),
                )

            except Exception as exc:
                logger.exception(
                    "Gagal membuat combined spectrum: %s",
                    exc,
                )

        # --------------------------------------------------------------
        # Station component comparison
        # --------------------------------------------------------------

        for (
            station,
            contexts,
        ) in station_grouped_contexts.items():

            try:
                BSMAVisualizer.plot_multi_component_comparison(
                    contexts,
                    station_name=station,
                    save_path=str(
                        PLOT_DIR
                        / f"{station}_component_comparison.png"
                    ),
                )

            except Exception as exc:
                logger.exception(
                    "Gagal membuat component comparison "
                    "stasiun %s: %s",
                    station,
                    exc,
                )

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    logger.info(
        "============================================================"
    )
    logger.info(
        "Batch processing selesai."
    )
    logger.info(
        "Trace berhasil diproses: %d",
        len(all_contexts),
    )
    logger.info(
        "Hasil parameter: %d",
        len(summary_results),
    )
    logger.info(
        "Output directory: %s",
        OUTPUT_DIR,
    )
    logger.info(
        "============================================================"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()