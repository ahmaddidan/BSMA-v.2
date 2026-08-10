"""
BMKG Strong Motion Analyzer (BSMA)
Module: run_pipeline.py

Description
-----------
GUI-ready and batch-processing API for the BSMA seismic processing system.

Responsibilities
----------------
1. Read seismic waveform files using ObsPy.
2. Optionally apply StationXML instrument response correction.
3. Convert waveform data into physical acceleration (m/s²).
4. Build and execute the standardized BSMA processing pipeline.
5. Export JSON/CSV metrics.
6. Generate a combined multi-component PDF report.
7. Provide robust logging and batch-processing support.

Scientific Notes
----------------
- Instrument correction is mandatory when waveform data are still in counts.
- A failed instrument correction MUST NOT be silently treated as m/s².
- StationXML is matched against the trace SEED identifiers:
  network.station.location.channel.
- No dynamic modification of trace.stats.location is performed.
- Pipeline processing is performed only on physically valid acceleration data.

Python
------
>= 3.12
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Final

import numpy as np
from obspy import Stream, Trace, read, read_inventory
from obspy.core.inventory import Inventory

from core.pipeline import PipelineBuilder
from core.preprocessing.baseline import BaselineCorrectionPlugin
from core.preprocessing.filter import (
    ButterworthFilterPlugin,
    FilterConfig,
)
from core.preprocessing.integration import (
    IntegrationConfig,
    KinematicIntegrationPlugin,
)
from core.preprocessing.taper import (
    TaperConfig,
    TaperPlugin,
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
from utils.exporter import BSMAExporter
from utils.logger import setup_logger
from utils.pdf_exporter import export_station_report


__all__ = [
    "build_bsma_pipeline",
    "process_earthquake_data",
    "main",
]


# =============================================================================
# Constants
# =============================================================================

GRAVITY: Final[float] = 9.80665

DEFAULT_DATA_DIRECTORY: Final[Path] = Path("Data/mseed")
DEFAULT_STATIONXML_DIRECTORY: Final[Path] = Path("Data/stationXML")
DEFAULT_OUTPUT_DIRECTORY: Final[Path] = Path("outputs")

SUPPORTED_WAVEFORM_EXTENSIONS: Final[tuple[str, ...]] = (
    "*.mseed",
    "*.MSEED",
    "*.miniseed",
    "*.MINISEED",
    "*.sac",
    "*.SAC",
    "*.seed",
    "*.SEED",
    "*.gse",
    "*.GSE",
    "*.gse2",
    "*.GSE2",
)


# =============================================================================
# Pipeline Construction
# =============================================================================


def build_bsma_pipeline(logger: logging.Logger):
    """
    Build the standardized BSMA preprocessing and analysis pipeline.

    Pipeline
    --------
    1. Baseline correction
    2. Taper
    3. Butterworth band-pass filter
    4. Kinematic integration
    5. Strong-motion parameter extraction
    6. 5% damped response spectrum

    Parameters
    ----------
    logger:
        BSMA logger instance.

    Returns
    -------
    PipelineOrchestrator
        Fully configured BSMA processing pipeline.
    """

    return (
        PipelineBuilder(
            logger=logger,
            halt_on_error=True,
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
                    type="bandpass",
                    freq_min=0.1,
                    freq_max=25.0,
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
                    damping=0.05,
                    solver="nigam_jennings",
                )
            )
        )
        .build()
    )


# =============================================================================
# StationXML Utilities
# =============================================================================


def _select_inventory_for_trace(
    trace: Trace,
    inventory: Inventory,
) -> Inventory | None:
    """
    Select StationXML metadata corresponding to a waveform trace.

    Matching is performed using the complete SEED identification where
    possible:

        NET.STA.LOC.CHA

    Parameters
    ----------
    trace:
        ObsPy waveform trace.

    inventory:
        Complete StationXML inventory.

    Returns
    -------
    Inventory | None
        Matching inventory subset or None if no match exists.
    """

    network = str(getattr(trace.stats, "network", "") or "")
    station = str(getattr(trace.stats, "station", "") or "")
    location = str(getattr(trace.stats, "location", "") or "")
    channel = str(getattr(trace.stats, "channel", "") or "")

    if not station or not channel:
        return None

    # -------------------------------------------------------------------------
    # First attempt: exact match
    # -------------------------------------------------------------------------

    try:
        selected = inventory.select(
            network=network or "*",
            station=station,
            location=location or "*",
            channel=channel,
        )

        if len(selected.networks) > 0:
            return selected
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Second attempt: location-independent match
    #
    # Useful when MiniSEED location code is blank but StationXML contains
    # the actual location code.
    # -------------------------------------------------------------------------

    try:
        selected = inventory.select(
            network=network or "*",
            station=station,
            channel=channel,
        )

        if len(selected.networks) > 0:
            return selected
    except Exception:
        pass

    return None


def _apply_instrument_correction(
    trace: Trace,
    inventory: Inventory,
    logger: logging.Logger,
) -> Trace:
    """
    Remove instrument response and convert waveform to acceleration.

    The function operates on a copy of the original trace.

    Parameters
    ----------
    trace:
        Raw ObsPy trace.

    inventory:
        StationXML inventory.

    logger:
        BSMA logger.

    Returns
    -------
    Trace
        Trace expressed as physical acceleration in m/s².

    Raises
    ------
    RuntimeError
        If no matching StationXML metadata exists or response removal fails.
    """

    corrected = trace.copy()

    trace_id = corrected.id

    logger.debug(
        "Searching StationXML response for %s",
        trace_id,
    )

    matched_inventory = _select_inventory_for_trace(
        corrected,
        inventory,
    )

    if matched_inventory is None:
        raise RuntimeError(
            f"StationXML response tidak ditemukan untuk trace {trace_id}."
        )

    logger.info(
        "Applying instrument response correction: %s",
        trace_id,
    )

    try:
        corrected.remove_response(
            inventory=matched_inventory,
            output="ACC",
            water_level=60,
            pre_filt=(
                0.05,
                0.10,
                30.0,
                35.0,
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Gagal melakukan instrument correction pada {trace_id}: {exc}"
        ) from exc

    # -------------------------------------------------------------------------
    # Numerical validation
    # -------------------------------------------------------------------------

    data = np.asarray(
        corrected.data,
        dtype=np.float64,
    )

    if data.size == 0:
        raise RuntimeError(
            f"Hasil instrument correction kosong untuk {trace_id}."
        )

    if not np.all(np.isfinite(data)):
        raise RuntimeError(
            f"Hasil instrument correction mengandung NaN/Inf pada {trace_id}."
        )

    corrected.data = data

    logger.debug(
        "Instrument correction successful | %s | "
        "min=%.6e m/s² | max=%.6e m/s² | mean=%.6e m/s²",
        trace_id,
        float(np.min(data)),
        float(np.max(data)),
        float(np.mean(data)),
    )

    return corrected


# =============================================================================
# Waveform Validation
# =============================================================================


def _validate_trace(
    trace: Trace,
    logger: logging.Logger,
) -> None:
    """
    Validate waveform before entering the mathematical pipeline.

    Raises
    ------
    ValueError
        If waveform is invalid.
    """

    if trace.stats.npts <= 0:
        raise ValueError(
            f"Trace {trace.id} tidak memiliki sample."
        )

    sampling_rate = float(
        trace.stats.sampling_rate
    )

    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError(
            f"Sampling rate tidak valid pada {trace.id}: "
            f"{sampling_rate}"
        )

    data = np.asarray(
        trace.data,
        dtype=np.float64,
    )

    if data.size == 0:
        raise ValueError(
            f"Trace {trace.id} kosong."
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(
            f"Trace {trace.id} mengandung NaN/Inf."
        )

    logger.debug(
        "Trace validation successful | %s | "
        "npts=%d | sampling_rate=%.6f Hz",
        trace.id,
        trace.stats.npts,
        sampling_rate,
    )


# =============================================================================
# Context Construction
# =============================================================================


def _build_processing_context(
    trace: Trace,
) -> ProcessingContext:
    """
    Construct the initial immutable ProcessingContext.

    The input trace MUST already represent acceleration in m/s².
    """

    data = np.asarray(
        trace.data,
        dtype=np.float64,
    ).copy()

    waveform = WaveformData(
        data=data,
        sampling_rate=float(
            trace.stats.sampling_rate
        ),
        unit="m/s^2",
    )

    return ProcessingContext(
        trace_id=trace.id,
        metadata=dict(trace.stats),
        raw_waveform=waveform,
        acceleration=waveform,
        processing_state=ProcessingState(),
        history=(),
    )


# =============================================================================
# Single File Processing
# =============================================================================


def process_earthquake_data(
    waveform_path: Path,
    xml_path: Path | None,
    exporter: BSMAExporter,
    logger: logging.Logger,
) -> dict[str, ProcessingContext]:
    """
    Process one seismic waveform file.

    Processing flow
    ---------------
    1. Read waveform.
    2. Validate stream.
    3. Load StationXML when supplied.
    4. Validate each trace.
    5. Apply instrument correction.
    6. Build ProcessingContext.
    7. Execute BSMA pipeline.
    8. Export JSON.
    9. Export CSV.
    10. Generate combined station PDF.

    Parameters
    ----------
    waveform_path:
        Input waveform path.

    xml_path:
        Optional StationXML path.

    exporter:
        BSMA exporter instance.

    logger:
        BSMA logger instance.

    Returns
    -------
    dict[str, ProcessingContext]
        Successfully processed contexts keyed by channel.

    Raises
    ------
    RuntimeError
        If file-level processing fails.
    """

    waveform_path = Path(waveform_path)

    logger.info(
        "\n%s\nMEMPROSES FILE: %s\n%s",
        "=" * 70,
        waveform_path.name,
        "=" * 70,
    )

    if not waveform_path.is_file():
        raise FileNotFoundError(
            f"Waveform file tidak ditemukan: {waveform_path}"
        )

    # -------------------------------------------------------------------------
    # Read waveform
    # -------------------------------------------------------------------------

    try:
        stream: Stream = read(str(waveform_path))
    except Exception as exc:
        raise RuntimeError(
            f"Gagal membaca waveform {waveform_path}: {exc}"
        ) from exc

    if len(stream) == 0:
        raise RuntimeError(
            f"Waveform {waveform_path.name} menghasilkan Stream kosong."
        )

    station_code = str(
        getattr(
            stream[0].stats,
            "station",
            "UNKNOWN",
        )
        or "UNKNOWN"
    )

    logger.info(
        "Stream loaded | station=%s | traces=%d",
        station_code,
        len(stream),
    )

    # -------------------------------------------------------------------------
    # Load StationXML
    # -------------------------------------------------------------------------

    inventory: Inventory | None = None

    if xml_path is not None:

        xml_path = Path(xml_path)

        if not xml_path.is_file():
            raise FileNotFoundError(
                f"StationXML tidak ditemukan: {xml_path}"
            )

        try:
            inventory = read_inventory(
                str(xml_path)
            )
        except Exception as exc:
            raise RuntimeError(
                f"Gagal membaca StationXML {xml_path}: {exc}"
            ) from exc

        logger.info(
            "StationXML loaded: %s",
            xml_path.name,
        )

    else:
        logger.warning(
            "StationXML tidak tersedia untuk %s. "
            "Waveform dianggap sudah dalam satuan fisik.",
            waveform_path.name,
        )

    # -------------------------------------------------------------------------
    # Build pipeline once per file
    # -------------------------------------------------------------------------

    pipeline = build_bsma_pipeline(
        logger
    )

    processed_contexts: dict[
        str,
        ProcessingContext,
    ] = {}

    # -------------------------------------------------------------------------
    # Process every trace
    # -------------------------------------------------------------------------

    for trace in stream:

        channel = str(
            getattr(
                trace.stats,
                "channel",
                "UNKNOWN",
            )
        )

        logger.info(
            "Processing component | %s | SR=%.6f Hz",
            trace.id,
            float(trace.stats.sampling_rate),
        )

        try:
            _validate_trace(
                trace,
                logger,
            )

            # -------------------------------------------------------------
            # Instrument correction
            # -------------------------------------------------------------

            if inventory is not None:

                corrected_trace = _apply_instrument_correction(
                    trace=trace,
                    inventory=inventory,
                    logger=logger,
                )

            else:

                # ---------------------------------------------------------
                # IMPORTANT:
                #
                # Without StationXML we cannot know whether the waveform
                # is counts, m/s², cm/s², etc.
                #
                # The existing architecture assumes physical acceleration.
                # Therefore we only accept explicit physical-unit metadata.
                # ---------------------------------------------------------

                corrected_trace = trace.copy()

                unit = str(
                    getattr(
                        corrected_trace.stats,
                        "units",
                        "",
                    )
                    or ""
                ).lower()

                if unit not in {
                    "m/s^2",
                    "m/s2",
                    "ms-2",
                    "m s-2",
                }:

                    raise RuntimeError(
                        f"StationXML tidak tersedia dan unit waveform "
                        f"{trace.id} tidak dapat dipastikan sebagai m/s². "
                        f"Unit terdeteksi: {unit!r}. "
                        "Instrument correction diperlukan."
                    )

                corrected_trace.data = np.asarray(
                    corrected_trace.data,
                    dtype=np.float64,
                )

            # -------------------------------------------------------------
            # Validate corrected data
            # -------------------------------------------------------------

            _validate_trace(
                corrected_trace,
                logger,
            )

            # -------------------------------------------------------------
            # Build ProcessingContext
            # -------------------------------------------------------------

            initial_context = _build_processing_context(
                corrected_trace
            )

            # -------------------------------------------------------------
            # Execute mathematical pipeline
            # -------------------------------------------------------------

            final_context = pipeline.run(
                initial_context
            )

            if not isinstance(
                final_context,
                ProcessingContext,
            ):
                raise TypeError(
                    "Pipeline tidak mengembalikan ProcessingContext."
                )

            processed_contexts[channel] = final_context

            # -------------------------------------------------------------
            # Export machine-readable results
            # -------------------------------------------------------------

            json_path = exporter.export_to_json(
                final_context
            )

            csv_path = exporter.export_to_csv(
                final_context
            )

            # -------------------------------------------------------------
            # Strong-motion summary
            # -------------------------------------------------------------

            pga_ms2 = float(
                final_context.metrics.get(
                    "PGA",
                    0.0,
                )
            )

            pga_gal = pga_ms2 * 100.0

            logger.info(
                "Component completed | %s | "
                "PGA=%.6f m/s² | PGA=%.3f Gal",
                channel,
                pga_ms2,
                pga_gal,
            )

            logger.debug(
                "Export completed | JSON=%s | CSV=%s",
                json_path,
                csv_path,
            )

        except Exception as exc:

            logger.error(
                "Component processing failed | %s | %s",
                trace.id,
                exc,
                exc_info=True,
                extra={
                    "bsma_context": {
                        "waveform": str(
                            waveform_path
                        ),
                        "trace_id": trace.id,
                        "station": station_code,
                        "channel": channel,
                    }
                },
            )

            # -------------------------------------------------------------
            # Fail-fast per trace.
            #
            # A station report should never mix successful and silently
            # invalid physical quantities.
            # -------------------------------------------------------------

            raise RuntimeError(
                f"Gagal memproses component {trace.id} "
                f"pada file {waveform_path.name}."
            ) from exc

    # -------------------------------------------------------------------------
    # Require at least one successfully processed component
    # -------------------------------------------------------------------------

    if not processed_contexts:
        raise RuntimeError(
            f"Tidak ada component berhasil diproses dari "
            f"{waveform_path.name}."
        )

    # -------------------------------------------------------------------------
    # Generate combined station PDF
    # -------------------------------------------------------------------------

    logger.info(
        "Generating combined PDF report | station=%s",
        station_code,
    )

    try:

        pdf_path = export_station_report(
            station_code=station_code,
            contexts=processed_contexts,
            output_dir=(
                DEFAULT_OUTPUT_DIRECTORY / "reports"
            ),
        )

    except Exception as exc:

        logger.error(
            "PDF report generation failed | station=%s | %s",
            station_code,
            exc,
            exc_info=True,
        )

        raise RuntimeError(
            f"Gagal membuat PDF report untuk stasiun "
            f"{station_code}."
        ) from exc

    logger.info(
        "PDF report completed: %s",
        pdf_path,
    )

    return processed_contexts


# =============================================================================
# StationXML Discovery
# =============================================================================


def _find_stationxml(
    waveform_path: Path,
    xml_directory: Path,
) -> Path | None:
    """
    Find a StationXML file corresponding to a waveform.

    Matching strategy
    -----------------
    1. Exact waveform stem.
    2. Station code extracted from waveform filename.
    3. Case-insensitive filename search.
    """

    if not xml_directory.is_dir():
        return None

    candidates = [
        waveform_path.with_suffix(".xml"),
        xml_directory / f"{waveform_path.stem}.xml",
    ]

    for candidate in candidates:

        if candidate.is_file():
            return candidate

    # -------------------------------------------------------------------------
    # Try station code from filename
    # -------------------------------------------------------------------------

    parts = waveform_path.stem.split("_")

    possible_station_codes = set()

    if len(parts) > 1:
        possible_station_codes.add(
            parts[1]
        )

    possible_station_codes.add(
        waveform_path.stem
    )

    xml_files = list(
        xml_directory.glob("*.xml")
    )

    for xml_file in xml_files:

        stem_lower = xml_file.stem.lower()

        for station_code in possible_station_codes:

            if station_code.lower() in stem_lower:
                return xml_file

    return None


# =============================================================================
# Batch Processing
# =============================================================================


def _discover_waveform_files(
    data_directory: Path,
) -> list[Path]:
    """
    Discover supported waveform files.

    Returns
    -------
    list[Path]
        Sorted unique waveform paths.
    """

    files: set[Path] = set()

    for pattern in SUPPORTED_WAVEFORM_EXTENSIONS:

        files.update(
            data_directory.glob(pattern)
        )

    return sorted(files)


def main() -> None:
    """
    Execute BSMA batch processing from the default data directories.
    """

    logger = setup_logger(
        "bsma_backend"
    )

    logger.info(
        "Starting BSMA Batch Processor."
    )

    data_directory = DEFAULT_DATA_DIRECTORY
    stationxml_directory = (
        DEFAULT_STATIONXML_DIRECTORY
    )

    # -------------------------------------------------------------------------
    # Validate input directory
    # -------------------------------------------------------------------------

    if not data_directory.is_dir():

        logger.error(
            "Waveform directory tidak ditemukan: %s",
            data_directory,
        )

        return

    # -------------------------------------------------------------------------
    # Discover waveforms
    # -------------------------------------------------------------------------

    waveform_files = _discover_waveform_files(
        data_directory
    )

    if not waveform_files:

        logger.warning(
            "Tidak ditemukan waveform file di %s.",
            data_directory,
        )

        return

    logger.info(
        "Discovered %d waveform file(s).",
        len(waveform_files),
    )

    # -------------------------------------------------------------------------
    # Shared exporter
    # -------------------------------------------------------------------------

    exporter = BSMAExporter(
        output_dir=str(
            DEFAULT_OUTPUT_DIRECTORY
        )
    )

    successful = 0
    failed = 0

    # -------------------------------------------------------------------------
    # Batch loop
    # -------------------------------------------------------------------------

    for index, waveform_file in enumerate(
        waveform_files,
        start=1,
    ):

        logger.info(
            "Batch item %d/%d | %s",
            index,
            len(waveform_files),
            waveform_file.name,
        )

        xml_path = _find_stationxml(
            waveform_file,
            stationxml_directory,
        )

        if xml_path is not None:

            logger.info(
                "StationXML matched: %s",
                xml_path,
            )

        else:

            logger.warning(
                "StationXML tidak ditemukan untuk %s.",
                waveform_file.name,
            )

        try:

            process_earthquake_data(
                waveform_path=waveform_file,
                xml_path=xml_path,
                exporter=exporter,
                logger=logger,
            )

            successful += 1

        except Exception as exc:

            failed += 1

            logger.error(
                "Batch item FAILED | %s | %s",
                waveform_file.name,
                exc,
                exc_info=True,
            )

            # Continue to the next waveform.
            continue

    # -------------------------------------------------------------------------
    # Batch summary
    # -------------------------------------------------------------------------

    logger.info(
        "\n%s\n"
        "BATCH PROCESSING SELESAI\n"
        "Total   : %d\n"
        "Success : %d\n"
        "Failed  : %d\n"
        "%s",
        "=" * 70,
        len(waveform_files),
        successful,
        failed,
        "=" * 70,
    )


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    main()