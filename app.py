"""BMKG Strong Motion Analyzer (BSMA) Streamlit dashboard."""

from __future__ import annotations

import io
import json
import hashlib
import logging
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import obspy
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.preprocessing.filter import FilterType
from services import (
    AnalysisConfiguration,
    AnalysisService,
    BatchService,
    ExportService,
    extract_summary_data,
)

PROJECT_ROOT = Path(__file__).resolve().parent
WAVEFORM_DIRECTORY = PROJECT_ROOT / "Data" / "mseed"
INVENTORY_DIRECTORY = PROJECT_ROOT / "Data" / "stationXML"
REPORT_DIRECTORY = PROJECT_ROOT / "outputs" / "reports"
LOGO_PATH = PROJECT_ROOT / "Logo_Judul.png"

st.set_page_config(
    page_title="BMKG Strong Motion Analyzer",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _initialise_state() -> None:
    st.session_state.setdefault("contexts_by_station", {})
    st.session_state.setdefault("batch_failures", {})
    st.session_state.setdefault("last_station", None)
    st.session_state.setdefault("benchmark_reference", None)
    st.session_state.setdefault("benchmark_tolerance_percent", 10.0)


def _ensure_directories() -> None:
    for directory in (WAVEFORM_DIRECTORY, INVENTORY_DIRECTORY, REPORT_DIRECTORY):
        directory.mkdir(parents=True, exist_ok=True)


def _waveform_files() -> list[Path]:
    files: list[Path] = []
    for suffix in ("*.mseed", "*.miniseed", "*.sac", "*.msd"):
        files.extend(WAVEFORM_DIRECTORY.glob(suffix))
    # A failed FDSN download can be saved with a MiniSEED extension.  Keep
    # that source file intact, but do not repeatedly treat its HTTP error
    # body as waveform input on every dashboard rerun.
    return sorted(path for path in set(files) if not _is_fdsn_error_response(path))


def _is_fdsn_error_response(path: Path) -> bool:
    try:
        preview = path.read_bytes()[:1024].decode("utf-8", errors="ignore").lower()
    except OSError:
        return False
    return "error 404" in preview and "fdsnws" in preview


def _load_master_stream(files: list[Path]) -> obspy.Stream:
    stream = obspy.Stream()
    failures: list[str] = []
    for path in files:
        try:
            loaded = obspy.read(str(path))
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            for trace in loaded:
                trace.stats["bsma_source_file"] = path.name
                trace.stats["bsma_source_sha256"] = source_hash
            stream += loaded
        except Exception as exc:
            failures.append(f"{path.name}: {_read_error_detail(path, exc)}")
    if failures:
        st.warning("Sebagian waveform tidak dapat dibaca: " + "; ".join(failures))
    return stream


def _read_error_detail(path: Path, error: Exception) -> str:
    """Return an actionable ingestion error without exposing a traceback."""
    try:
        preview = path.read_bytes()[:1024].decode("utf-8", errors="ignore").lower()
    except OSError:
        preview = ""

    if "error 404" in preview or "no metadata found" in preview:
        return (
            "berisi respons 404 dari layanan data, bukan rekaman MiniSEED. "
            "Unduh ulang interval/channel yang tersedia lalu unggah file baru."
        )
    return str(error)


def _find_inventory(station: str) -> obspy.Inventory | None:
    source = _find_inventory_path(station)
    if source is None:
        return None
    try:
        return obspy.read_inventory(str(source))
    except Exception as exc:
        logging.getLogger("bsma.dashboard").warning(
            "Ignoring unreadable StationXML %s for station %s: %s",
            source.name,
            station,
            exc,
        )
        return None


def _find_inventory_path(station: str) -> Path | None:
    """Find only a StationXML whose filename explicitly identifies station."""
    direct = INVENTORY_DIRECTORY / f"{station}.xml"
    candidates = [direct] if direct.is_file() else list(INVENTORY_DIRECTORY.glob("*.xml"))
    for candidate in candidates:
        if candidate.stem.upper() == station.upper() or station.upper() in candidate.stem.upper():
            return candidate
    return None


def _configuration_from_sidebar() -> tuple[AnalysisConfiguration, dict[str, Any]]:
    with st.sidebar:
        with st.expander("Processing configuration", expanded=False):
            st.caption("Default: zero-phase 4th-order Butterworth band-pass filter (0.25–25 Hz).")
            st.caption("The default is a conservative starting point; review corner frequencies against each record's signal-to-noise ratio.")
            with st.form("processing_configuration", border=False):
                filter_type = st.selectbox("Filter type", options=[member.value for member in FilterType], index=0)
                frequency_min = st.number_input("Low cutoff (Hz)", min_value=0.001, value=0.25)
                frequency_max = st.number_input("High cutoff (Hz)", min_value=0.01, value=25.0)
                adaptive_filter = st.checkbox(
                    "Apply SNR/Nyquist screening recommendation",
                    value=True,
                    help="Caps the high corner below 80% Nyquist and raises the low corner conservatively for weak SNR. The exact decision is retained in the audit log.",
                )
                damping = st.number_input("Response-spectrum damping ratio", min_value=0.0, max_value=0.99, value=0.05, step=0.01)
                unit_labels = {"Meter per second squared (m/s²)": "m/s^2", "Gal / centimeter per second squared (Gal)": "gal", "Centimeter per second squared (cm/s²)": "cm/s^2"}
                input_unit_label = st.selectbox("Unit when StationXML is unavailable", options=list(unit_labels), help="Confirm this only when MiniSEED samples are already physical acceleration, not ADC counts.")
                provenance = st.selectbox(
                    "Input data provenance",
                    ["Already processed physical acceleration", "Raw instrument counts with StationXML", "Unknown - require scientific review"],
                    help="Use the first option only when the provider confirms the samples are acceleration and any prior filtering is documented. Files labelled BP4 are treated as already filtered, not raw counts.",
                )
                st.session_state["apply_instrument_response"] = provenance == "Raw instrument counts with StationXML"
                st.session_state["input_provenance"] = provenance
                applied = st.form_submit_button("Apply configuration", icon=":material/tune:")
        if applied:
            st.session_state.pop("contexts_by_station", None)
            st.session_state["contexts_by_station"] = {}
            st.session_state["last_station"] = None

        with st.expander("Event information (optional)", expanded=False):
            event_mode = st.radio("Report event details", ["Do not include (recommended)", "Add manually for the PDF"], help="MiniSEED normally provides record start time; StationXML provides instrument/station metadata. Neither reliably contains earthquake origin, magnitude, or depth.")
            event_info = {}
            if event_mode == "Add manually for the PDF":
                event_info = {"time": st.text_input("Origin time (UTC)", placeholder="YYYY-MM-DD HH:MM:SS"), "latitude": st.text_input("Latitude"), "longitude": st.text_input("Longitude"), "magnitude": st.text_input("Magnitude"), "depth_km": st.text_input("Depth (km)"), "epicentral_distance_km": st.text_input("Epicentral distance (km)")}

        with st.expander("Reference benchmark (optional)", expanded=False):
            st.caption("Upload a CSV with channel plus any of PGA, PGV, PGD, Arias_Intensity, Significant_Duration_D5_95, or PSA. Optional record_id scopes rows to one recording window.")
            benchmark_upload = st.file_uploader("Reference metrics CSV", type=["csv"], key="benchmark_upload")
            st.session_state["benchmark_tolerance_percent"] = st.number_input(
                "Relative tolerance (%)",
                min_value=0.1,
                max_value=100.0,
                value=float(st.session_state["benchmark_tolerance_percent"]),
                step=0.5,
                help="A comparison passes when the relative difference is within this tolerance. The reference source and matching preprocessing must be documented.",
            )
            if benchmark_upload is not None:
                try:
                    reference = pd.read_csv(benchmark_upload)
                    if "channel" not in {str(column).lower() for column in reference.columns}:
                        raise ValueError("CSV must contain a 'channel' column.")
                    st.session_state["benchmark_reference"] = reference
                    st.success(f"Loaded {len(reference)} reference row(s).")
                except Exception as exc:
                    st.error(f"Reference benchmark could not be read: {exc}")

    return (
        AnalysisConfiguration(
            filter_type=filter_type,
            freq_min_hz=float(frequency_min),
            freq_max_hz=float(frequency_max),
            damping_ratio=float(damping),
            input_unit=unit_labels[input_unit_label],
            input_mode="raw_counts" if provenance == "Raw instrument counts with StationXML" else "physical_acceleration",
            adaptive_filter=bool(adaptive_filter),
        ),
        {key: value for key, value in event_info.items() if value},
    )


def _upload_data() -> None:
    with st.sidebar:
        with st.expander("Input data", expanded=False):
            with st.form("input_upload", clear_on_submit=True):
                waveforms = st.file_uploader(
                    "Waveforms", type=["mseed", "miniseed", "msd", "sac"], accept_multiple_files=True
                )
                inventories = st.file_uploader(
                    "StationXML", type=["xml", "stationxml"], accept_multiple_files=True
                )
                submitted = st.form_submit_button("Store input data", icon=":material/upload:")
            if submitted:
                for upload, target in ((upload, WAVEFORM_DIRECTORY) for upload in waveforms or []):
                    (target / Path(upload.name).name).write_bytes(upload.getvalue())
                for upload, target in ((upload, INVENTORY_DIRECTORY) for upload in inventories or []):
                    (target / Path(upload.name).name).write_bytes(upload.getvalue())
                st.success("Input data stored. Reloading stations.")
                st.rerun()


def _service(configuration: AnalysisConfiguration) -> AnalysisService:
    return AnalysisService(configuration, logger=logging.getLogger("bsma.dashboard"))


def _record_windows(stream: obspy.Stream) -> dict[str, obspy.Stream]:
    """Group one station's components by their common recording start time."""
    windows: dict[str, obspy.Stream] = {}
    for trace in sorted(stream, key=lambda item: item.stats.starttime):
        label = f"{trace.stats.station} | {trace.stats.starttime.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        windows.setdefault(label, obspy.Stream()).append(trace.copy())
    return windows


def _process_one_station(
    record_id: str,
    station_stream: obspy.Stream,
    configuration: AnalysisConfiguration,
) -> dict[str, Any]:
    contexts = _service(configuration).process_station_stream(
        station_stream,
        _find_inventory(str(station_stream[0].stats.station)) if st.session_state.get("apply_instrument_response", False) else None,
    )
    st.session_state["contexts_by_station"][record_id] = contexts
    st.session_state["last_station"] = record_id
    return contexts


def _display_metrics(contexts: dict[str, Any]) -> None:
    strongest_channel, strongest = max(
        contexts.items(), key=lambda item: float(item[1].metrics.get("PGA", 0.0))
    )
    metrics = strongest.metrics
    metadata = strongest.metadata
    st.caption(f"Network: {metadata.get('network', '-')} | Channels: {' '.join(contexts)} | Sampling: {strongest.sampling_rate:.1f} Hz | Record: {metadata.get('starttime', '-')}")
    with st.container(horizontal=True):
        st.metric("Strongest component", strongest_channel, help="Component with the largest processed peak ground acceleration.", border=True)
        st.metric("PGA", f"{float(metrics.get('PGA', 0.0)) * 100:.3f} Gal", help="Peak Ground Acceleration: maximum absolute ground acceleration.", border=True)
        st.metric("PGV", f"{float(metrics.get('PGV', 0.0)) * 100:.3f} cm/s", help="Peak Ground Velocity: maximum absolute integrated ground velocity.", border=True)
        st.metric("PGD", f"{float(metrics.get('PGD', 0.0)) * 100:.4f} cm", help="Peak Ground Displacement: maximum absolute integrated ground displacement.", border=True)
        st.metric("Arias intensity", f"{float(metrics.get('Arias_Intensity', 0.0)):.5f} m/s", help="Energy-related intensity measure computed from the processed acceleration history.", border=True)
        st.metric("D5-95", f"{float(metrics.get('Significant_Duration_D5_95', 0.0)):.2f} s", help="Time interval over which cumulative Arias energy grows from 5% to 95%.", border=True)
        sig = ExportService._sig_label(float(metrics.get("PGA", 0.0)) * 100.0)
        st.metric("SIG-BMKG", sig, help="Klasifikasi intensitas berdasarkan PGA dalam Gal.", border=True)
    sig_messages = {
        "SIG I": (st.info, "Skala I - Putih: tidak dirasakan (MMI I-II)."),
        "SIG II": (st.success, "Skala II - Hijau: dirasakan (MMI III-V)."),
        "SIG III": (st.warning, "Skala III - Kuning: kerusakan ringan (MMI VI)."),
        "SIG IV": (st.warning, "Skala IV - Jingga: kerusakan sedang (MMI VII-VIII)."),
        "SIG V": (st.error, "Skala V - Merah: kerusakan berat (MMI IX-XII)."),
    }
    render_message, message = sig_messages[sig]
    render_message(message, icon=":material/vibration:")


def _display_benchmark(record_id: str, contexts: dict[str, Any]) -> None:
    """Compare processed metrics with an operator-supplied generic reference CSV."""
    reference = st.session_state.get("benchmark_reference")
    with st.expander("Numerical benchmark", expanded=False):
        if reference is None:
            st.info("No reference CSV loaded. This is optional; use it to compare this result with an independently processed record.")
            return
        normalized = reference.rename(columns={str(column): str(column).strip().lower() for column in reference.columns})
        if "record_id" in normalized.columns:
            normalized = normalized[normalized["record_id"].astype(str).isin({record_id, "*", ""})]
        available = {
            "pga": "PGA",
            "pgv": "PGV",
            "pgd": "PGD",
            "arias_intensity": "Arias_Intensity",
            "significant_duration_d5_95": "Significant_Duration_D5_95",
            "psa": "PSA",
        }
        tolerance = float(st.session_state["benchmark_tolerance_percent"])
        rows: list[dict[str, Any]] = []
        for _, reference_row in normalized.iterrows():
            channel = str(reference_row.get("channel", "")).strip()
            context = contexts.get(channel)
            if context is None:
                rows.append({"channel": channel or "-", "metric": "-", "status": "NOT FOUND", "detail": "Reference channel is not in this record."})
                continue
            for csv_name, metric_name in available.items():
                if csv_name not in normalized.columns or pd.isna(reference_row[csv_name]):
                    continue
                reference_value = float(reference_row[csv_name])
                if metric_name == "PSA":
                    spectrum = np.asarray(context.spectral_data.get("PSA", []), dtype=float)
                    computed_value = float(np.nanmax(spectrum)) if spectrum.size else np.nan
                else:
                    computed_value = float(context.metrics.get(metric_name, np.nan))
                relative_error = abs(computed_value - reference_value) / max(abs(reference_value), 1e-12) * 100.0
                rows.append(
                    {
                        "channel": channel,
                        "metric": metric_name,
                        "computed": computed_value,
                        "reference": reference_value,
                        "difference (%)": relative_error,
                        "status": "PASS" if relative_error <= tolerance else "REVIEW",
                    }
                )
        if not rows:
            st.warning("The CSV has no comparable metric rows for this recording window.")
            return
        st.caption(f"Tolerance: {tolerance:.1f}%. A REVIEW result is a scientific review prompt, not automatic proof that either dataset is wrong.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _display_analysis(station: str, contexts: dict[str, Any], event_info: dict[str, Any]) -> None:
    _display_metrics(contexts)
    rows = extract_summary_data(station, contexts)
    summary, waveform, spectrum, husid_tab, fas, audit, report = st.tabs(
        ["Summary", "Waveforms", "Response spectrum", "Husid plot", "FAS", "QC audit", "Report"]
    )
    with summary:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        strongest_channel, strongest = max(contexts.items(), key=lambda item: float(item[1].metrics.get("PGA", 0.0)))
        pga = float(strongest.metrics.get("PGA", 0.0))
        duration = float(strongest.metrics.get("Significant_Duration_D5_95", 0.0))
        pga_gal = pga * 100.0
        percent_g = pga / 9.80665 * 100.0
        sig = ExportService._sig_label(pga_gal)
        descriptions = {"SIG I": "TIDAK DIRASAKAN", "SIG II": "DIRASAKAN", "SIG III": "KERUSAKAN RINGAN", "SIG IV": "KERUSAKAN SEDANG", "SIG V": "KERUSAKAN BERAT"}
        qc = strongest.qc
        calibration = "terkalibrasi dengan StationXML" if strongest.processing_state.response_correction.value == "SUCCESS" else "tanpa kalibrasi respons instrumen"
        clipping = "tidak terdeteksi clipping" if qc is None or not qc.has_clipping else "terdeteksi indikasi clipping"
        st.markdown(
            f"### Interpretasi hasil\n"
            f"**Percepatan tanah maksimum (PGA):** {pga_gal:.3f} Gal ({percent_g:.3f} %g).  \n"
            f"**Guncangan dominan:** komponen **{strongest_channel}**.  \n"
            f"**Klasifikasi intensitas:** **{sig} — {descriptions[sig]}**.  \n"
            f"**Durasi signifikan (D5–D95):** {duration:.2f} detik.  \n"
            f"**Status data:** {calibration}; {clipping}; pemrosesan selesai."
        )
        _display_benchmark(station, contexts)
    with waveform:
        channels = st.multiselect(
            "Components to display",
            list(contexts),
            default=list(contexts),
            help="Each selected component is drawn with its own acceleration, velocity, and displacement histories. PGA, D5, and D95 markers are derived from that component.",
        )
        for channel in channels:
            context = contexts[channel]
            acceleration = context.acceleration
            if acceleration is None:
                st.warning(f"{channel}: acceleration history is unavailable.")
                continue
            st.markdown(f"#### {channel}")
            time = np.arange(acceleration.npts) / acceleration.sampling_rate
            figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04)
            for row, data, name in (
                (1, acceleration.data, "Acceleration (m/s2)"),
                (2, context.velocity.data, "Velocity (m/s)"),
                (3, context.displacement.data, "Displacement (m)"),
            ):
                figure.add_trace(go.Scatter(x=time, y=data, mode="lines", name=name), row=row, col=1)
                figure.update_yaxes(title_text=name, row=row, col=1)
            pga_index = int(np.argmax(np.abs(acceleration.data)))
            figure.add_vline(x=float(time[pga_index]), line_color="red", line_dash="dot", annotation_text="PGA")
            husid = context.cache.husid_curve
            if husid is not None and len(husid) == len(time):
                for level, label, color in ((0.05, "D5", "orange"), (0.95, "D95", "green")):
                    index = int(np.searchsorted(np.asarray(husid), level))
                    figure.add_vline(x=float(time[min(index, len(time) - 1)]), line_color=color, line_dash="dash", annotation_text=label)
            figure.update_xaxes(title_text="Time (s)", row=3, col=1)
            figure.update_layout(height=700, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(figure, width="stretch")
    with spectrum:
        figure = go.Figure()
        for channel, context in contexts.items():
            periods = context.spectral_data.get("periods")
            psa = context.spectral_data.get("PSA")
            if periods is not None and psa is not None:
                figure.add_trace(
                    go.Scatter(x=periods, y=np.asarray(psa) / 9.80665, mode="lines", name=channel)
                )
        figure.update_xaxes(type="log", title="Period (s)")
        figure.update_yaxes(title="PSA (g)")
        figure.update_layout(height=450, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(figure, width="stretch")
    with husid_tab:
        st.caption("Husid plot menunjukkan persentase kumulatif energi Arias; D5 dan D95 membatasi durasi energi utama.")
        figure = go.Figure()
        for channel, context in contexts.items():
            curve = context.cache.husid_curve
            if curve is not None:
                time = np.arange(curve.size) / context.sampling_rate
                figure.add_trace(go.Scatter(x=time, y=np.asarray(curve) * 100, mode="lines", name=channel))
        figure.update_layout(height=450, xaxis_title="Time (s)", yaxis_title="Cumulative Arias energy (%)")
        st.plotly_chart(figure, width="stretch")
    with fas:
        st.caption("Fourier Amplitude Spectrum (FAS) memperlihatkan kandungan amplitudo terhadap frekuensi setelah pemrosesan.")
        figure = go.Figure()
        for channel, context in contexts.items():
            data = context.acceleration.data
            frequency = np.fft.rfftfreq(data.size, d=context.dt)
            amplitude = np.abs(np.fft.rfft(data)) / data.size
            figure.add_trace(go.Scatter(x=frequency[1:], y=amplitude[1:], mode="lines", name=channel))
        figure.update_layout(height=450, xaxis_type="log", yaxis_type="log", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude (m/s2)")
        st.plotly_chart(figure, width="stretch")
    with audit:
        st.caption("Audit trail berikut menyimpan keputusan ilmiah dan setiap tahap pemrosesan yang benar-benar dijalankan untuk tiap komponen.")
        narratives = {
            "ScientificProvenance": "Mencatat sumber, checksum, satuan, versi engine, dan waktu proses.",
            "RawQC": "Memeriksa integritas sampel, clipping, flatline, drift, dan indikator SNR.",
            "InstrumentResponse": "Menentukan apakah respons instrumen dikoreksi atau sengaja dilewati.",
            "FilterRecommendation": "Memilih sudut filter berdasarkan sampling rate, Nyquist, dan screening SNR.",
            "BaselineCorrection": "Menghapus offset atau tren baseline sebelum filtering.",
            "Taper": "Menerapkan taper untuk mengurangi artefak pada tepi rekaman.",
            "ButterworthFilter": "Menerapkan filter Butterworth zero-phase pada band yang diaudit.",
            "KinematicIntegration": "Mengintegrasikan percepatan untuk memperoleh kecepatan dan perpindahan.",
            "ParameterExtraction": "Mengekstrak parameter strong-motion dan durasi energi.",
            "Response_Spectrum": "Menghitung spektrum respons dengan redaman yang ditetapkan.",
        }
        for channel, context in contexts.items():
            st.subheader(channel)
            qc = context.qc
            if qc is not None:
                quality_message = f"QC score {qc.quality_score}/100"
                if qc.is_valid:
                    st.success(quality_message + ": processing permitted; review any warnings below.", icon=":material/check_circle:")
                else:
                    st.error(quality_message + ": processing blocked by the quality gate.", icon=":material/error:")
            for entry in context.history:
                stage = entry.get("step", entry.get("stage", entry.get("plugin", "Processing step")))
                status = entry.get("status", "SUCCESS")
                display_stage = str(stage).split("(", maxsplit=1)[0]
                details = {key: value for key, value in entry.items() if key not in {"step", "stage", "plugin", "status", "timestamp"}}
                with st.expander(f"{stage} [{status}]", expanded=False):
                    st.caption(narratives.get(display_stage, "Tahap pemrosesan tercatat dalam provenance."))
                    if details:
                        st.json(details)
                    else:
                        st.caption("Tahap selesai tanpa parameter tambahan yang dicatat.")
    with report:
        if st.button("Generate station PDF", icon=":material/picture_as_pdf:"):
            safe_record_id = re.sub(r'[<>:"/\\|?*]+', "_", station)
            output = REPORT_DIRECTORY / f"BSMA_Report_{safe_record_id}.pdf"
            with st.spinner("Generating PDF report..."):
                pdf_path = ExportService().export_station_pdf(
                    station,
                    contexts,
                    output,
                    event_info=event_info or None,
                )
            st.session_state["last_pdf"] = pdf_path.read_bytes()
            st.session_state["last_pdf_name"] = pdf_path.name
        if st.session_state.get("last_pdf"):
            st.download_button(
                "Download PDF report",
                data=st.session_state["last_pdf"],
                file_name=st.session_state["last_pdf_name"],
                mime="application/pdf",
                icon=":material/download:",
            )


def _batch_analysis(
    records: dict[str, obspy.Stream],
    configuration: AnalysisConfiguration,
) -> None:
    unknown_provenance = st.session_state.get("input_provenance") == "Unknown - require scientific review"
    missing_inventory = [
        record
        for record, stream in records.items()
        if configuration.input_mode == "raw_counts" and _find_inventory_path(str(stream[0].stats.station)) is None
    ]
    if unknown_provenance:
        st.warning("Processing is blocked until the input is declared as raw counts with StationXML or already processed physical acceleration.")
    if missing_inventory:
        st.error("Raw-count mode requires StationXML for every record. Missing: " + ", ".join(missing_inventory))
    if st.button(
        "Process all stations",
        type="primary",
        icon=":material/play_arrow:",
        disabled=unknown_provenance or bool(missing_inventory),
    ):
        progress = st.progress(0, text="Preparing batch analysis")

        def on_progress(index: int, total: int, station: str) -> None:
            progress.progress(index / total, text=f"Processing {station} ({index}/{total})")

        streams = records
        inventories = {record: _find_inventory(str(stream[0].stats.station)) if st.session_state.get("apply_instrument_response", False) else None for record, stream in streams.items()}
        result = BatchService(_service(configuration)).process_stations(
            streams, inventories, progress_callback=on_progress
        )
        st.session_state["contexts_by_station"].update(result.contexts_by_station)
        st.session_state["batch_failures"] = result.failures
        st.session_state["batch_rows"] = result.summary_rows()
        progress.progress(1.0, text="Batch analysis complete")

    rows = st.session_state.get("batch_rows", [])
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    if st.session_state.get("batch_failures"):
        st.error("Failed stations")
        st.json(st.session_state["batch_failures"])


def _export_batch(event_info: dict[str, Any]) -> None:
    contexts = st.session_state["contexts_by_station"]
    if not contexts:
        st.info("Process at least one station before export.")
        return
    selected = st.multiselect("Stations to export", list(contexts), default=list(contexts))
    if st.button("Build export package", type="primary", icon=":material/archive:"):
        if not selected:
            st.error("Select at least one station.")
            return
        exporter = ExportService()
        with io.BytesIO() as buffer:
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                selected_contexts = {station: contexts[station] for station in selected}
                csv_path = exporter.export_batch_csv(
                    selected_contexts, REPORT_DIRECTORY / "BSMA_Summary.csv"
                )
                archive.write(csv_path, arcname="BSMA_Summary.csv")
                provenance = {
                    station: {
                        channel: {
                            "trace_id": context.trace_id,
                            "raw_waveform_sha256": hashlib.sha256(context.raw_waveform.data.tobytes()).hexdigest(),
                            "raw_unit": context.raw_waveform.unit,
                            "acceleration_unit": context.acceleration.unit if context.acceleration is not None else None,
                            "metadata": dict(context.metadata),
                            "metrics": dict(context.metrics),
                            "processing_state": context.processing_state.to_dict(),
                            "configuration": dict(context.config),
                            "history": list(context.history),
                            "qc": context.qc.to_dict() if context.qc is not None else None,
                        }
                        for channel, context in station_contexts.items()
                    }
                    for station, station_contexts in selected_contexts.items()
                }
                archive.writestr("provenance.json", json.dumps(provenance, indent=2, default=str))
                for station, station_contexts in selected_contexts.items():
                    pdf_path = exporter.export_station_pdf(
                        station,
                        station_contexts,
                        REPORT_DIRECTORY / f"BSMA_Report_{station}.pdf",
                        event_info=event_info or None,
                    )
                    archive.write(pdf_path, arcname=f"reports/{pdf_path.name}")
                    for png_path in exporter.export_station_waveform_pngs(
                        station,
                        station_contexts,
                        REPORT_DIRECTORY / "waveforms",
                    ):
                        archive.write(png_path, arcname=f"waveforms/{png_path.name}")
            st.session_state["export_archive"] = buffer.getvalue()
        st.session_state["export_archive_name"] = "BSMA_Operational_Exports.zip"
    if st.session_state.get("export_archive"):
        st.download_button(
            "Download export package",
            data=st.session_state["export_archive"],
            file_name=st.session_state["export_archive_name"],
            mime="application/zip",
            icon=":material/download:",
        )


_initialise_state()
_ensure_directories()
_upload_data()
configuration, event_info = _configuration_from_sidebar()

if LOGO_PATH.is_file():
    st.image(str(LOGO_PATH), width=180)
st.title("BMKG Strong Motion Analyzer")
st.caption("Scientific processing and engineering review of strong-motion records with traceable quality control and exports.")

files = _waveform_files()
if not files:
    st.info("Upload MiniSEED/SAC waveforms and optional StationXML from the sidebar to begin.")
    st.stop()

master_stream = _load_master_stream(files)
records = _record_windows(master_stream)
if not records:
    st.error("No usable station traces were found in the input files.")
    st.stop()

mode = st.segmented_control(
    "Workflow",
    options=["Single-station review", "Multi-station processing", "Export results"],
    default="Single-station review",
    key="app_mode",
)

if mode == "Single-station review":
    record_id = st.selectbox("Recording window", list(records))
    station_stream = records[record_id]
    station = str(station_stream[0].stats.station)
    inventory_path = _find_inventory_path(station)
    inventory = _find_inventory(station) if st.session_state.get("apply_instrument_response", False) else None
    st.caption(f"StationXML correction enabled: {inventory_path.name}." if inventory is not None and inventory_path else "Using declared physical acceleration; StationXML response correction is disabled.")
    unknown_provenance = st.session_state.get("input_provenance") == "Unknown - require scientific review"
    missing_inventory = configuration.input_mode == "raw_counts" and inventory is None
    if unknown_provenance:
        st.warning("Processing is blocked until the input data mode is declared.")
    if missing_inventory:
        st.error("Raw-count mode requires a readable StationXML matching this station; processing has been blocked to prevent an invalid unit conversion.")
    if st.button(
        "Process selected record",
        type="primary",
        icon=":material/play_arrow:",
        disabled=unknown_provenance or missing_inventory,
    ):
        try:
            with st.spinner(f"Processing {station}..."):
                _process_one_station(record_id, station_stream, configuration)
        except Exception as exc:
            st.exception(exc)
    contexts = st.session_state["contexts_by_station"].get(record_id)
    if contexts:
        _display_analysis(record_id, contexts, event_info)
elif mode == "Multi-station processing":
    _batch_analysis(records, configuration)
else:
    _export_batch(event_info)
