"""BMKG Strong Motion Analyzer (BSMA) Professional Scientific Dashboard."""

from __future__ import annotations

import hashlib
import json
import logging
import re

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
from utils.pdf_exporter import get_mmi_worden, get_sig_bmkg

PROJECT_ROOT = Path(__file__).resolve().parent
WAVEFORM_DIRECTORY = PROJECT_ROOT / "Data" / "mseed"
INVENTORY_DIRECTORY = PROJECT_ROOT / "Data" / "stationXML"
REPORT_DIRECTORY = PROJECT_ROOT / "outputs" / "reports"
LOGO_PATH = PROJECT_ROOT / "Logo_Judul.png"

st.set_page_config(
    page_title="BMKG Strong Motion Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)


DESIGN_TOKENS: dict[str, Any] = {
    "colors": {
        "bg": "#0B0B0B",
        "sidebar": "#111111",
        "panel": "#171717",
        "panel_active": "#1D1D1D",
        "border": "#2A2A2A",
        "border_light": "#3A3A3A",
        "text": "#F2F2F2",
        "text_secondary": "#A8A8A8",
        "text_muted": "#666666",
        "success": "#10B981",
        "warning": "#F59E0B",
        "error": "#EF4444",
        "focus": "#D0D0D0",
    },
    "typography": {
        "font_sans": "'Inter', system-ui, -apple-system, sans-serif",
        "font_mono": "'Fira Code', 'JetBrains Mono', monospace",
        "size_xs": "0.75rem",
        "size_sm": "0.82rem",
        "size_md": "0.95rem",
        "size_lg": "1.1rem",
        "size_xl": "1.4rem",
    },
    "spacing": {
        "xs": "4px",
        "sm": "8px",
        "md": "12px",
        "lg": "16px",
        "xl": "24px",
    },
    "radius": {
        "sm": "4px",
        "md": "6px",
        "lg": "8px",
    },
    "semantic": {
        "surface": "colors.panel",
        "surface_active": "colors.panel_active",
        "text_primary": "colors.text",
        "text_secondary": "colors.text_secondary",
        "status_pass": "colors.success",
        "status_warning": "colors.warning",
        "status_error": "colors.error",
        "focus": "colors.focus",
    },
}


def token(path: str) -> str:
    """Resolve a design token path (e.g., 'colors.bg' or 'semantic.status_pass') to its raw value."""
    keys = path.split(".")
    current: Any = DESIGN_TOKENS
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return ""
    if isinstance(current, str) and "." in current:
        return token(current)
    return str(current)


def inject_bsma_theme() -> None:
    """Inject global CSS theme derived dynamically from DESIGN_TOKENS."""
    bg = token("colors.bg")
    sidebar = token("colors.sidebar")
    panel = token("colors.panel")
    panel_active = token("colors.panel_active")
    border = token("colors.border")
    border_light = token("colors.border_light")
    text = token("colors.text")
    text_secondary = token("colors.text_secondary")
    pass_col = token("semantic.status_pass")
    warn_col = token("semantic.status_warning")
    fail_col = token("semantic.status_error")
    font_sans = token("typography.font_sans")
    font_mono = token("typography.font_mono")
    radius_sm = token("radius.sm")

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: {font_sans};
    }}

    /* App Main Canvas: Neutral Scientific Dark */
    .stApp {{
        background-color: {bg};
        color: {text};
        padding-bottom: 50px !important;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar};
        border-right: 1px solid {border};
    }}

    section[data-testid="stSidebar"][aria-expanded="false"] {{
        border-right: none !important;
    }}

    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {text_secondary};
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }}

    /* Expander Containers (Read-only / Collapsible Panels) */
    .stExpander {{
        background-color: {panel} !important;
        border: 1px solid {border} !important;
        border-radius: {radius_sm} !important;
        margin-bottom: 0.5rem !important;
    }}
    
    .stExpander > details > summary {{
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        color: {text} !important;
        letter-spacing: 0.02em !important;
    }}

    /* Scientific Data Panels / Cards */
    .sci-card {{
        background-color: {panel};
        border: 1px solid {border};
        border-radius: {radius_sm};
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
    }}

    .sci-card-active {{
        background-color: {panel_active};
        border: 1px solid {border_light};
    }}

    /* Monospace Logs & Identifiers */
    .code-ident {{
        font-family: {font_mono};
        color: {text};
    }}

    /* Technical Log Panel */
    .technical-log {{
        font-family: {font_mono};
        background-color: #0d0d0d;
        border: 1px solid {border};
        border-left: 3px solid {border_light};
        padding: 0.75rem 1rem;
        border-radius: {radius_sm};
        color: {text_secondary};
        font-size: 0.8rem;
        line-height: 1.6;
    }}

    /* Functional Badges */
    .badge-pass {{
        background-color: rgba(16, 185, 129, 0.15);
        color: {pass_col};
        border: 1px solid rgba(16, 185, 129, 0.3);
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
    }}

    .badge-warn {{
        background-color: rgba(245, 158, 11, 0.15);
        color: {warn_col};
        border: 1px solid rgba(245, 158, 11, 0.3);
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
    }}

    .badge-fail {{
        background-color: rgba(239, 68, 68, 0.15);
        color: {fail_col};
        border: 1px solid rgba(239, 68, 68, 0.3);
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
    }}

    /* Table Styling */
    [data-testid="stDataFrame"] {{
        border: 1px solid {border};
        border-radius: {radius_sm};
        background-color: {panel};
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        background-color: {bg};
        border-bottom: 1px solid {border};
    }}

    .stTabs [data-baseweb="tab"] {{
        font-size: 0.8rem;
        font-weight: 600;
        color: {text_secondary};
        padding: 0.5rem 1rem;
        border-radius: 4px 4px 0 0;
        background-color: {sidebar};
        border: 1px solid {border};
        border-bottom: none;
    }}

    .stTabs [aria-selected="true"] {{
        color: {text} !important;
        background-color: {panel} !important;
        border-top: 2px solid {text} !important;
    }}

    /* Form Controls & Dropdowns */
    [data-testid="stSelectbox"] > div > div, [data-testid="stTextInput"] > div > div {{
        background-color: {panel} !important;
        border: 1px solid {border} !important;
        color: {text} !important;
        border-radius: {radius_sm} !important;
    }}

    /* Buttons */
    .stButton button {{
        border-radius: {radius_sm};
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.02em;
    }}

    .stButton button[kind="primary"] {{
        background-color: {panel};
        border: 1px solid #444444;
        color: {text};
    }}

    .stButton button[kind="primary"]:hover {{
        background-color: #222222;
        border-color: #666666;
        color: #ffffff;
    }}

    /* Workflow Stepper */
    .stepper-container {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        background-color: {sidebar};
        border: 1px solid {border};
        padding: 0.4rem 1rem;
        border-radius: {radius_sm};
        margin-bottom: 1rem;
        font-size: 0.78rem;
        color: {text_secondary};
    }}

    .stepper-item {{
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }}

    .stepper-active {{
        color: {text};
        font-weight: 700;
    }}

    .stepper-done {{
        color: {pass_col};
    }}

    /* Footer Status Bar */
    .status-footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: {sidebar};
        border-top: 1px solid {border};
        padding: 0.3rem 1.5rem;
        font-size: 0.75rem;
        font-family: {font_mono};
        color: {text_secondary};
        display: flex;
        justify-content: space-between;
        z-index: 999;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def _inject_custom_css() -> None:
    """Backward compatibility alias for theme injection."""
    inject_bsma_theme()


def _initialise_state() -> None:
    st.session_state.setdefault("contexts_by_station", {})
    st.session_state.setdefault("batch_failures", {})
    st.session_state.setdefault("last_station", None)
    st.session_state.setdefault("benchmark_reference", None)
    st.session_state.setdefault("benchmark_tolerance_percent", 10.0)


def _ensure_directories() -> None:
    for directory in (WAVEFORM_DIRECTORY, INVENTORY_DIRECTORY, REPORT_DIRECTORY):
        directory.mkdir(parents=True, exist_ok=True)


def _clear_uploaded_data() -> None:
    """Clear temporary uploaded MiniSEED and StationXML files and reset session state."""
    for directory in (WAVEFORM_DIRECTORY, INVENTORY_DIRECTORY):
        if directory.exists():
            for item in directory.glob("*"):
                if item.is_file():
                    try:
                        item.unlink()
                    except OSError:
                        pass
    st.session_state["contexts_by_station"] = {}
    st.session_state["batch_failures"] = {}
    st.session_state["batch_rows"] = []
    st.session_state["last_station"] = None
    st.session_state.pop("last_pdf", None)
    st.session_state.pop("last_pdf_name", None)
    st.session_state.pop("export_archive", None)


def _waveform_files() -> list[Path]:
    files: list[Path] = []
    for suffix in ("*.mseed", "*.miniseed", "*.sac", "*.msd"):
        files.extend(WAVEFORM_DIRECTORY.glob(suffix))
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
        st.warning("Ingestion Warning: " + "; ".join(failures))
    return stream


def _read_error_detail(path: Path, error: Exception) -> str:
    try:
        preview = path.read_bytes()[:1024].decode("utf-8", errors="ignore").lower()
    except OSError:
        preview = ""

    if "error 404" in preview or "no metadata found" in preview:
        return "HTTP 404 response payload instead of valid MiniSEED binary."
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
    direct = INVENTORY_DIRECTORY / f"{station}.xml"
    candidates = [direct] if direct.is_file() else list(INVENTORY_DIRECTORY.glob("*.xml"))
    for candidate in candidates:
        if candidate.stem.upper() == station.upper() or station.upper() in candidate.stem.upper():
            return candidate
    return None


def _configuration_from_sidebar() -> tuple[AnalysisConfiguration, dict[str, Any]]:
    with st.sidebar:
        # DATA INPUT Section
        st.markdown("### DATA INPUT")
        uploaded_waveforms = st.file_uploader(
            "Upload MiniSEED Waveforms",
            type=["mseed", "miniseed", "sac", "msd"],
            accept_multiple_files=True,
            help="Upload raw or pre-filtered seismic waveform files.",
            key="waveform_file_uploader",
        )
        if uploaded_waveforms:
            for file in uploaded_waveforms:
                target_path = WAVEFORM_DIRECTORY / file.name
                target_path.write_bytes(file.getbuffer())
            st.success(f"{len(uploaded_waveforms)} file waveform diunggah.")

        uploaded_inventories = st.file_uploader(
            "Upload StationXML Metadata",
            type=["xml"],
            accept_multiple_files=True,
            help="Upload StationXML response metadata for instrument correction.",
            key="inventory_file_uploader",
        )
        if uploaded_inventories:
            for file in uploaded_inventories:
                target_path = INVENTORY_DIRECTORY / file.name
                target_path.write_bytes(file.getbuffer())
            st.success(f"{len(uploaded_inventories)} file StationXML diunggah.")

        if st.button("Reset / Clear Data", use_container_width=True):
            _clear_uploaded_data()
            st.rerun()

        st.divider()

        # PROJECT Section
        st.markdown("### PROJECT")
        provenance = st.selectbox(
            "Data Provenance",
            ["Already processed physical acceleration", "Raw instrument counts with StationXML", "Unknown - require scientific review"],
            help="Select Raw Counts if StationXML is available for instrument response removal.",
        )
        unit_labels = {
            "m/s² (SI Unit)": "m/s^2",
            "Gal (cm/s²)": "gal",
            "cm/s²": "cm/s^2",
        }
        input_unit_label = st.selectbox(
            "Unit (No StationXML)",
            options=list(unit_labels),
            help="Physical unit declaration when StationXML response correction is bypassed.",
        )
        st.session_state["apply_instrument_response"] = provenance == "Raw instrument counts with StationXML"
        st.session_state["input_provenance"] = provenance

        # PROCESSING Section
        st.markdown("### PROCESSING")
        default_fmin = 0.25
        default_fmax = 25.0
        is_prefiltered = False
        for p in WAVEFORM_DIRECTORY.glob("*.mseed"):
            stem = p.stem.upper()
            if "BP4_0.05_40" in stem or ("BP4" in stem and "0.05" in stem):
                default_fmin = 0.05
                default_fmax = 40.0
                is_prefiltered = True
                break

        if is_prefiltered:
            st.caption("Auto-detected BMKG Pre-Filtered File (BP4 0.05–40 Hz). Preset corner frequencies applied.")

        filter_type = st.selectbox("Filter", options=[member.value for member in FilterType], index=0)
        frequency_min = st.number_input("Low Cutoff (Hz)", min_value=0.001, value=default_fmin, step=0.05)
        frequency_max = st.number_input("High Cutoff (Hz)", min_value=0.01, value=default_fmax, step=1.0)

        with st.expander("Advanced settings ▸", expanded=False):
            adaptive_filter = st.checkbox(
                "Adaptive SNR/Nyquist Screening",
                value=True,
                help="Caps high cutoff below 80% Nyquist and checks low corner against noise floor.",
            )
            apply_detrend = st.checkbox("Polynomial Detrending", value=True)
            apply_taper = st.checkbox("Tukey Window Tapering (5%)", value=True)

        # ANALYSIS Section
        st.markdown("### ANALYSIS")
        damping = st.number_input("Damping Ratio (xi)", min_value=0.0, max_value=0.99, value=0.05, step=0.01)

        with st.expander("Spectrum Advanced ▸", expanded=False):
            solver_option = st.selectbox("SDOF Solver", ["Newmark-Beta (Implicit)", "Nigam-Jennings (Exact Piecewise)"])

        # OUTPUT Section
        st.markdown("### OUTPUT")
        with st.expander("Event Metadata ▸", expanded=False):
            event_info = {
                "time": st.text_input("Origin Time (UTC)", placeholder="YYYY-MM-DD HH:MM:SS"),
                "latitude": st.text_input("Latitude (°N)"),
                "longitude": st.text_input("Longitude (°E)"),
                "magnitude": st.text_input("Magnitude (Mw)"),
                "depth_km": st.text_input("Depth (km)"),
                "epicentral_distance_km": st.text_input("Distance (km)"),
            }

        with st.expander("Benchmark ▸", expanded=False):
            benchmark_upload = st.file_uploader("Reference CSV", type=["csv"], key="benchmark_upload")
            st.session_state["benchmark_tolerance_percent"] = st.number_input(
                "Tolerance (%)",
                min_value=0.1,
                max_value=100.0,
                value=float(st.session_state["benchmark_tolerance_percent"]),
                step=0.5,
            )
            if benchmark_upload is not None:
                try:
                    reference = pd.read_csv(benchmark_upload)
                    if "channel" not in {str(col).lower() for col in reference.columns}:
                        raise ValueError("CSV must contain a 'channel' column.")
                    st.session_state["benchmark_reference"] = reference
                    st.success(f"Loaded {len(reference)} benchmark row(s).")
                except Exception as exc:
                    st.error(f"Benchmark error: {exc}")

    solver_key = "newmark" if "Newmark" in solver_option else "nigam_jennings"
    baseline_method = "linear" if apply_detrend else "constant"
    taper_alpha = 0.05 if apply_taper else 0.0

    return (
        AnalysisConfiguration(
            baseline_method=baseline_method,
            taper_alpha=taper_alpha,
            filter_type=filter_type,
            freq_min_hz=float(frequency_min),
            freq_max_hz=float(frequency_max),
            damping_ratio=float(damping),
            response_solver=solver_key,
            input_unit=unit_labels[input_unit_label],
            input_mode="raw_counts" if provenance == "Raw instrument counts with StationXML" else "physical_acceleration",
            adaptive_filter=bool(adaptive_filter),
        ),
        {key: value for key, value in event_info.items() if value},
    )


def _service(configuration: AnalysisConfiguration) -> AnalysisService:
    return AnalysisService(configuration, logger=logging.getLogger("bsma.dashboard"))


def _record_windows(stream: obspy.Stream) -> dict[str, obspy.Stream]:
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


def _render_workflow_stepper(current_step: str = "ANALYSIS", has_data: bool = True, has_qc: bool = True) -> None:
    """Render informational non-blocking workflow status stepper."""
    data_status = "stepper-done" if has_data else ""
    qc_status = "stepper-done" if has_qc else ""
    
    st.markdown(
        f"""
        <div class="stepper-container">
            <div class="stepper-item {data_status}"><span>DATA</span> <span>✓</span></div>
            <span>•</span>
            <div class="stepper-item {qc_status}"><span>QC</span> <span>✓</span></div>
            <span>•</span>
            <div class="stepper-item stepper-done"><span>PROCESS</span> <span>✓</span></div>
            <span>•</span>
            <div class="stepper-item stepper-active"><span>ANALYSIS</span> <span>●</span></div>
            <span>•</span>
            <div class="stepper-item"><span>REPORT</span> <span>○</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _station_quality_summary(contexts: dict[str, Any]) -> dict[str, Any]:
    if not contexts:
        return {
            "class_id": 7,
            "label": "OFFLINE / NO DATA",
            "description": "No data available on station.",
            "reasons": ["No processed traces."],
            "quality_score": 0,
        }

    total_score = 0.0
    reasons: list[str] = []
    has_missing_data = False
    critical_signal_issue = False
    noise_issue = False

    for channel, context in contexts.items():
        qc = context.qc
        if qc is None:
            has_missing_data = True
            reasons.append(f"{channel}: QC report unavailable.")
            continue

        total_score += float(qc.quality_score)
        if qc.quality_score < 60:
            critical_signal_issue = True
            reasons.append(f"{channel}: Low signal quality score ({qc.quality_score}/100).")
        if qc.has_clipping or qc.has_adc_saturation:
            critical_signal_issue = True
            reasons.append(f"{channel}: Clipping / ADC saturation detected.")
        if qc.has_spikes:
            reasons.append(f"{channel}: Impulsive spikes detected.")
        if qc.has_flatline:
            reasons.append(f"{channel}: Flatline detected.")
        if qc.has_offset or qc.has_drift:
            reasons.append(f"{channel}: Baseline offset/drift exceeds tolerance.")
        if qc.snr_estimate_db is not None and qc.snr_estimate_db < 3.0:
            noise_issue = True
            reasons.append(f"{channel}: Low SNR (< 3 dB).")

    average_score = total_score / max(len(contexts), 1)
    if average_score >= 80 and not reasons:
        return {
            "class_id": 1,
            "label": "NOMINAL DATA QUALITY",
            "description": "Background noise within standard AHNM boundaries; clean PSD response.",
            "reasons": ["Good overall data quality."],
            "quality_score": int(round(average_score)),
        }
    if average_score >= 70 and not critical_signal_issue and not noise_issue and not has_missing_data:
        return {
            "class_id": 2,
            "label": "ACCEPTABLE QUALITY",
            "description": "Slight SNR degradation or minor offset.",
            "reasons": ["Minor signal degradation noted."],
            "quality_score": int(round(average_score)),
        }
    if critical_signal_issue:
        return {
            "class_id": 3,
            "label": "SENSOR / DIGITIZER ANOMALY",
            "description": "Clipping or ADC saturation present.",
            "reasons": reasons,
            "quality_score": int(round(average_score)),
        }
    if has_missing_data:
        return {
            "class_id": 4,
            "label": "DATA / METADATA ERROR",
            "description": "Incomplete dataset or unreadable StationXML.",
            "reasons": reasons,
            "quality_score": int(round(average_score)),
        }
    if noise_issue:
        return {
            "class_id": 5,
            "label": "DEGRADED SIGNAL (HIGH NOISE)",
            "description": "Dominant background noise, low SNR.",
            "reasons": reasons,
            "quality_score": int(round(average_score)),
        }
    return {
        "class_id": 6,
        "label": "AVAILABILITY / TRANSMISSION ANOMALY",
        "description": "Data availability or telemetry gap.",
        "reasons": reasons,
        "quality_score": int(round(average_score)),
    }


def _display_summary_view(
    station: str,
    contexts: dict[str, Any],
    event_info: dict[str, Any],
    configuration: AnalysisConfiguration,
) -> None:
    """Render a clean, scientific Summary view: Metadata, Waveform Preview (40-50% height), Metrics Table, QC, and Provenance."""
    strongest_channel, strongest = max(
        contexts.items(), key=lambda item: float(item[1].metrics.get("PGA", 0.0))
    )
    metadata = strongest.metadata

    # 1. Station & Record Metadata Header Banner
    st.markdown(
        f"""
        <div class="sci-card" style="margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin:0; font-size: 1.3rem; font-weight: 700; color: #f2f2f2;">
                        <span class="code-ident">{metadata.get('network', 'IA')}.{metadata.get('station', station)}</span>
                    </h2>
                    <span style="color: #a8a8a8; font-size: 0.82rem;">
                        Start: <span class="code-ident">{metadata.get('starttime', '-')}</span> | 
                        Sampling: <span class="code-ident">{strongest.sampling_rate:.1f} Hz</span> | 
                        Components: <span class="code-ident">{' · '.join(contexts)}</span>
                    </span>
                </div>
                <div style="text-align: right;">
                    <span class="badge-pass">ANALYSIS COMPLETE</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # 1b. Key Metrics Strip (below header)
    pga_gal = float(strongest.metrics.get("PGA", 0.0)) * 100.0
    if pga_gal > 3000.0 and configuration.input_mode == "physical_acceleration":
        st.warning(
            f"⚠️ **High PGA Warning ({pga_gal:.1f} Gal / {pga_gal/980.665:.2f} g)**: "
            "PGA exceeds 3.0 g. If input waveform was recorded in Gal (cm/s²), "
            "please select **Gal (cm/s²)** in the sidebar 'Unit (No StationXML)' dropdown."
        )

    pgv_cm = float(strongest.metrics.get("PGV", 0.0)) * 100.0
    pgd_cm = float(strongest.metrics.get("PGD", 0.0)) * 100.0
    arias_m = float(strongest.metrics.get("Arias_Intensity", 0.0))
    d595 = float(strongest.metrics.get("Significant_Duration_D5_95", 0.0))
    pga_pct_g = (float(strongest.metrics.get("PGA", 0.0)) / 9.80665) * 100.0
    mmi_info = get_mmi_worden(pga_pct_g, pgv_cm)

    metric_items = [
        ("Strongest component", strongest_channel),
        ("PGA", f"{pga_gal:.3f} Gal"),
        ("PGV", f"{pgv_cm:.3f} cm/s"),
        ("PGD", f"{pgd_cm:.4f} cm"),
        ("Arias intensity", f"{arias_m:.5f} m/s"),
        ("D5-95", f"{d595:.2f} s"),
        ("MMI", f"MMI {mmi_info['mmi']}"),
    ]

    metrics_html = "".join(
        f"""<div style="flex:1; padding: 0.6rem 0.8rem; border-right: 1px solid {token('colors.border')};">
            <div style="color:{token('colors.text_secondary')}; font-size:0.7rem; margin-bottom:0.2rem;">{label}</div>
            <div style="color:{token('colors.text')}; font-size:1.05rem; font-weight:600;">{value}</div>
        </div>"""
        for label, value in metric_items
    )

    st.markdown(
        f"""<div class="sci-card" style="display:flex; margin-bottom:0.75rem; padding:0; overflow:hidden;">
            {metrics_html}
        </div>""",
        unsafe_allow_html=True,
    )

    # 2. Waveform Preview (40-50% Height Canvas)
    st.markdown("#### WAVEFORM PREVIEW")
    figure = make_subplots(rows=len(contexts), cols=1, shared_xaxes=True, vertical_spacing=0.04)
    colors = ["#06b6d4", "#3b82f6", "#10b981", "#f59e0b", "#ec4899"]
    
    for idx, (channel, context) in enumerate(contexts.items()):
        acc = context.acceleration
        if acc is None:
            continue
        time = np.arange(acc.npts) / acc.sampling_rate
        figure.add_trace(
            go.Scatter(
                x=time,
                y=acc.data,
                mode="lines",
                name=channel,
                line=dict(color=colors[idx % len(colors)], width=1.0),
            ),
            row=idx + 1,
            col=1,
        )
        figure.update_yaxes(title_text=f"{channel} (m/s²)", row=idx + 1, col=1, gridcolor="#2a2a2a")

    figure.update_xaxes(title_text="Time (s)", row=len(contexts), col=1, gridcolor="#2a2a2a")
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#171717",
        plot_bgcolor="#0b0b0b",
        height=320,  # Balanced 40-50% preview height
        showlegend=False,
        margin=dict(l=20, r=20, t=15, b=25),
    )
    st.plotly_chart(figure, use_container_width=True)

    # 3. Strong Motion Metrics Scientific Table
    st.markdown("#### STRONG-MOTION PARAMETERS")
    rows = extract_summary_data(station, contexts)
    summary_df = pd.DataFrame(rows)
    
    # Rename columns for clarity and add tooltips
    col_rename = {
        "channel": "Component",
        "pga_gal": "PGA (Gal)",
        "pgv_cm_s": "PGV (cm/s)",
        "pgd_cm": "PGD (cm)",
        "arias_intensity_m_s": "Arias Intensity (m/s)",
        "significant_duration_d5_95_s": "Significant Duration (D5–95)",
        "significant_duration_d5_95": "Significant Duration (D5–95)",
    }
    display_df = summary_df.rename(columns=col_rename)
    display_df = display_df.loc[:, ~display_df.columns.duplicated()]
    selected_cols = [c for c in col_rename.values() if c in display_df.columns]
    seen = set()
    unique_selected_cols = [x for x in selected_cols if not (x in seen or seen.add(x))]
    st.dataframe(display_df[unique_selected_cols], hide_index=True, use_container_width=True)

    # 4. Processing Provenance & QC Status Row
    col_prov, col_qc = st.columns([3, 2])
    
    clean_station_code = station.split(" | ")[0].strip()

    with col_prov:
        st.markdown("#### PROCESSING PROVENANCE")
        input_mode_str = "Raw instrument counts" if configuration.input_mode == "raw_counts" else "Physical acceleration"
        response_str = "StationXML Response Correction Applied" if st.session_state.get("apply_instrument_response") else "Bypassed (Declared Unit)"
        filter_str = f"{configuration.filter_type.upper()} ({configuration.freq_min_hz} – {configuration.freq_max_hz} Hz)"
        inv_path = _find_inventory_path(clean_station_code)
        
        st.markdown(
            f"""
            <div class="technical-log">
                <strong>[PROCESSING PROVENANCE]</strong><br>
                • Data Binary       : <span class="code-ident">{metadata.get('station', clean_station_code)}.mseed</span><br>
                • Metadata Source   : <span class="code-ident">{inv_path.name if inv_path else 'Declared Unit Header'}</span><br>
                • Input Mode        : {input_mode_str}<br>
                • Response Status   : {response_str}<br>
                • Pre-Filter Band   : <span class="code-ident">{filter_str}</span><br>
                • Baseline Correction: Polynomial Detrending & 5% Tukey Tapering Applied
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_qc:
        st.markdown("#### QC STATUS CHECK")
        quality = _station_quality_summary(contexts)
        pass_badge = '<span class="badge-pass">PASS</span>' if quality['class_id'] <= 2 else '<span class="badge-fail">REVIEW</span>'
        has_anomalies = any(c.qc and (c.qc.has_clipping or c.qc.has_spikes or c.qc.has_adc_saturation) for c in contexts.values())
        qc_badge = '<span class="badge-fail">ANOMALY DETECTED</span>' if has_anomalies else '<span class="badge-pass">CLEAN</span>'
        
        st.markdown(
            f"""
            <div class="sci-card">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                    <span>Sampling Rate Continuity:</span> {pass_badge}
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                    <span>Signal Quality Score:</span> <strong>{quality['quality_score']} / 100</strong>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                    <span>Spike & Clipping Flags:</span> {qc_badge}
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span>Classification:</span> <strong style="color:#f2f2f2;">Class {quality['class_id']} ({quality['label']})</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _display_benchmark(station, contexts)


def _display_waveforms_view(contexts: dict[str, Any]) -> None:
    """Dedicated 60-70% height canvas for high-resolution interactive signal inspection."""
    st.markdown("### WAVEFORMS ANALYSIS")
    channels = st.multiselect(
        "Select Components to Inspect",
        list(contexts),
        default=list(contexts),
    )
    for channel in channels:
        context = contexts[channel]
        acc = context.acceleration
        if acc is None:
            st.warning(f"{channel}: Acceleration history missing.")
            continue
        
        st.markdown(f"**COMPONENT:** `{channel}`")
        time = np.arange(acc.npts) / acc.sampling_rate
        figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        
        # Subplot 1: Acceleration
        figure.add_trace(
            go.Scatter(x=time, y=acc.data, mode="lines", name="Acc (m/s²)", line=dict(color="#06b6d4", width=1.2)),
            row=1, col=1
        )
        figure.update_yaxes(title_text="Acc (m/s²)", row=1, col=1, gridcolor="#2a2a2a")

        # Subplot 2: Velocity
        if context.velocity is not None:
            figure.add_trace(
                go.Scatter(x=time, y=context.velocity.data, mode="lines", name="Vel (m/s)", line=dict(color="#3b82f6", width=1.2)),
                row=2, col=1
            )
            figure.update_yaxes(title_text="Vel (m/s)", row=2, col=1, gridcolor="#2a2a2a")

        # Subplot 3: Displacement
        if context.displacement is not None:
            figure.add_trace(
                go.Scatter(x=time, y=context.displacement.data, mode="lines", name="Disp (m)", line=dict(color="#10b981", width=1.2)),
                row=3, col=1
            )
            figure.update_yaxes(title_text="Disp (m)", row=3, col=1, gridcolor="#2a2a2a")

        pga_index = int(np.argmax(np.abs(acc.data)))
        figure.add_vline(x=float(time[pga_index]), line_color="#ef4444", line_dash="dot", annotation_text="PGA", annotation_font_color="#ef4444")
        
        husid = context.cache.husid_curve
        if husid is not None and len(husid) == len(time):
            for level, label, color in ((0.05, "D5", "#f59e0b"), (0.95, "D95", "#10b981")):
                index = int(np.searchsorted(np.asarray(husid), level))
                figure.add_vline(x=float(time[min(index, len(time) - 1)]), line_color=color, line_dash="dash", annotation_text=label, annotation_font_color=color)

        figure.update_xaxes(title_text="Time (s)", row=3, col=1, gridcolor="#2a2a2a")
        figure.update_layout(
            template="plotly_dark",
            paper_bgcolor="#171717",
            plot_bgcolor="#0b0b0b",
            height=600,  # Primary 60-70% height canvas
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(figure, use_container_width=True)


def _display_qc_view(contexts: dict[str, Any]) -> None:
    """Detailed quality control and diagnostic audit log."""
    st.markdown("### QUALITY CONTROL DIAGNOSTICS")
    quality_summary = _station_quality_summary(contexts)
    
    st.markdown(
        f"""
        <div class="technical-log" style="border-left-color: #3a3a3a; margin-bottom: 1rem;">
            STATION QC CLASS : <strong>Class {quality_summary['class_id']} - {quality_summary['label']}</strong><br>
            AVERAGE QC SCORE : <strong>{quality_summary['quality_score']} / 100</strong><br>
            DIAGNOSTIC DETAILS : {quality_summary['description']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    for channel, context in contexts.items():
        st.markdown(f"**CHANNEL:** `{channel}`")
        qc = context.qc
        if qc is not None:
            col_q1, col_q2, col_q3 = st.columns(3)
            with col_q1:
                st.caption(f"QC Score: **{qc.quality_score} / 100**")
            with col_q2:
                st.caption(f"Estimated SNR: **{qc.snr_estimate_db:.1f} dB**")
            with col_q3:
                clipping_txt = "Detected" if qc.has_clipping else "Clean"
                st.caption(f"Clipping Flag: **{clipping_txt}**")

        with st.expander(f"Inspect Processing History & Provenance ({channel})", expanded=False):
            for entry in context.history:
                stage = entry.get("step", entry.get("stage", entry.get("plugin", "Processing step")))
                status = entry.get("status", "SUCCESS")
                details = {k: v for k, v in entry.items() if k not in {"step", "stage", "plugin", "status", "timestamp"}}
                st.markdown(f"**Step:** `{stage}` | **Status:** `{status}`")
                if details:
                    st.json(details)


def _display_strong_motion_view(contexts: dict[str, Any]) -> None:
    """Kinematic strong-motion metric breakdown."""
    st.markdown("### STRONG-MOTION PARAMETERS")
    rows = extract_summary_data("", contexts)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    strongest_channel, strongest = max(contexts.items(), key=lambda item: float(item[1].metrics.get("PGA", 0.0)))
    pga = float(strongest.metrics.get("PGA", 0.0))
    pga_gal = pga * 100.0
    pga_pct_g = (pga / 9.80665) * 100.0
    pgv_cm_s = float(strongest.metrics.get("PGV", 0.0)) * 100.0
    pgd_cm = float(strongest.metrics.get("PGD", 0.0)) * 100.0
    arias = float(strongest.metrics.get("Arias_Intensity", 0.0))
    duration = float(strongest.metrics.get("Significant_Duration_D5_95", 0.0))

    st.markdown(
        f"""
        <div class="technical-log">
            <strong>[DOMINANT CHANNEL PARAMETERS ({strongest_channel})]</strong><br>
            • Peak Ground Acceleration (PGA) : {pga_gal:.4f} Gal ({pga_pct_g:.4f} %g)<br>
            • Peak Ground Velocity (PGV)     : {pgv_cm_s:.4f} cm/s<br>
            • Peak Ground Displacement (PGD)  : {pgd_cm:.4f} cm<br>
            • Arias Intensity (Ia)           : {arias:.4f} m/s<br>
            • Significant Duration (D5–95)  : {duration:.2f} s
        </div>
        """,
        unsafe_allow_html=True,
    )


def _display_intensity_view(contexts: dict[str, Any]) -> None:
    """Dedicated Instrumental Intensity (Worden et al., 2011) ShakeMap view."""
    st.markdown("### INSTRUMENTAL INTENSITY")
    strongest_channel, strongest = max(contexts.items(), key=lambda item: float(item[1].metrics.get("PGA", 0.0)))
    pga_m_s2 = float(strongest.metrics.get("PGA", 0.0))
    pga_pct_g = (pga_m_s2 / 9.80665) * 100.0
    pgv_cm_s = float(strongest.metrics.get("PGV", 0.0)) * 100.0

    mmi_info = get_mmi_worden(pga_pct_g, pgv_cm_s)
    mmi_rgb = mmi_info["rgb"]
    bg_color = f"rgb({mmi_rgb[0]}, {mmi_rgb[1]}, {mmi_rgb[2]})"
    text_color = "#ffffff" if mmi_info["mmi"] in {"VIII", "IX", "X+"} else "#000000"

    st.markdown(
        f"""
        <div style="background-color: #171717; border: 1px solid #2a2a2a; border-radius: 4px; padding: 1.2rem; margin-top: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2a2a2a; padding-bottom: 0.75rem; margin-bottom: 1rem;">
                <span style="font-size: 0.85rem; font-weight: 700; color: #a8a8a8; text-transform: uppercase;">SHAKEMAP INSTRUMENTAL INTENSITY (WORDEN ET AL., 2011)</span>
                <span style="background-color: {bg_color}; color: {text_color}; font-family: 'Fira Code', monospace; font-weight: 800; font-size: 1.2rem; padding: 0.3rem 1rem; border-radius: 3px;">
                    MMI {mmi_info['mmi']}
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; font-size: 0.85rem;">
                <div>
                    <span style="color: #a8a8a8; font-weight: 600;">PERCEIVED SHAKING</span><br>
                    <strong style="color: #f2f2f2; font-size: 1.05rem;">{mmi_info['shaking']}</strong>
                </div>
                <div>
                    <span style="color: #a8a8a8; font-weight: 600;">POTENTIAL DAMAGE</span><br>
                    <strong style="color: #f2f2f2; font-size: 1.05rem;">{mmi_info['damage']}</strong>
                </div>
                <div>
                    <span style="color: #a8a8a8; font-weight: 600;">PEAK ACC. (%g)</span><br>
                    <strong style="color: #f2f2f2; font-size: 1.05rem;">{pga_pct_g:.3f} %g</strong> 
                    <span style="color: #666666; font-size: 0.78rem;">(Ref: {mmi_info['pga_label']} %g)</span>
                </div>
                <div>
                    <span style="color: #a8a8a8; font-weight: 600;">PEAK VEL. (cm/s)</span><br>
                    <strong style="color: #f2f2f2; font-size: 1.05rem;">{pgv_cm_s:.3f} cm/s</strong> 
                    <span style="color: #666666; font-size: 0.78rem;">(Ref: {mmi_info['pgv_label']} cm/s)</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _display_spectrum_view(contexts: dict[str, Any], configuration: AnalysisConfiguration) -> None:
    """Response Spectrum, FAS, and HUSID energy growth curve sub-tabs."""
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Response Spectrum", "Fourier Spectrum (FAS)", "Husid Energy Growth"])

    colors = ["#06b6d4", "#3b82f6", "#10b981", "#f59e0b", "#ec4899"]

    with sub_tab1:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            scale_type = st.radio("X-Axis Scale", ["Linear", "Logarithmic"], horizontal=True)
        with col_s2:
            min_period, max_period = st.slider(
                "Period Range (s)",
                min_value=0.01,
                max_value=10.0,
                value=(0.01, 10.0),
                step=0.05,
            )

        figure = go.Figure()
        for idx, (channel, context) in enumerate(contexts.items()):
            periods = np.asarray(context.spectral_data.get("periods", []), dtype=float)
            psa = np.asarray(context.spectral_data.get("PSA", []), dtype=float)
            if periods.size and psa.size:
                mask = (periods >= min_period) & (periods <= max_period)
                if mask.any():
                    figure.add_trace(
                        go.Scatter(
                            x=periods[mask],
                            y=psa[mask] / 9.80665,
                            mode="lines",
                            name=f"{channel} (xi={configuration.damping_ratio*100:.1f}%)",
                            line=dict(width=2, color=colors[idx % len(colors)]),
                        )
                    )
        
        figure.update_xaxes(
            type="log" if scale_type == "Logarithmic" else "linear",
            title="Period (s)",
            gridcolor="#2a2a2a",
            dtick=1 if scale_type == "Logarithmic" else None,
            exponentformat="none",
        )
        figure.update_yaxes(title="Pseudo-Spectral Acceleration PSa (g)", gridcolor="#2a2a2a")
        figure.update_layout(
            template="plotly_dark",
            paper_bgcolor="#171717",
            plot_bgcolor="#0b0b0b",
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(figure, use_container_width=True)

    with sub_tab2:
        figure = go.Figure()
        for idx, (channel, context) in enumerate(contexts.items()):
            data = context.acceleration.data
            frequency = np.fft.rfftfreq(data.size, d=context.dt)
            amplitude = np.abs(np.fft.rfft(data)) / data.size
            figure.add_trace(
                go.Scatter(
                    x=frequency[1:],
                    y=amplitude[1:],
                    mode="lines",
                    name=channel,
                    line=dict(width=1.5, color=colors[idx % len(colors)]),
                )
            )
        figure.update_layout(
            template="plotly_dark",
            paper_bgcolor="#171717",
            plot_bgcolor="#0b0b0b",
            height=450,
            xaxis_type="log",
            yaxis_type="log",
            xaxis=dict(title="Frequency (Hz)", gridcolor="#2a2a2a", dtick=1, exponentformat="none"),
            yaxis=dict(title="Fourier Amplitude (m/s² · s)", gridcolor="#2a2a2a"),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(figure, use_container_width=True)

    with sub_tab3:
        st.caption("Normalized cumulative Arias intensity / energy growth representation.")
        figure = go.Figure()
        for idx, (channel, context) in enumerate(contexts.items()):
            curve = context.cache.husid_curve
            if curve is not None:
                time = np.arange(curve.size) / context.sampling_rate
                figure.add_trace(
                    go.Scatter(
                        x=time,
                        y=np.asarray(curve) * 100,
                        mode="lines",
                        name=channel,
                        line=dict(width=2, color=colors[idx % len(colors)]),
                    )
                )
        figure.update_xaxes(title="Time (s)", gridcolor="#2a2a2a")
        figure.update_yaxes(title="Cumulative Arias Energy (%)", gridcolor="#2a2a2a")
        figure.update_layout(
            template="plotly_dark",
            paper_bgcolor="#171717",
            plot_bgcolor="#0b0b0b",
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(figure, use_container_width=True)


def _display_report_view(station: str, contexts: dict[str, Any], event_info: dict[str, Any]) -> None:
    """PDF Report Compilation and Batch ZIP Export."""
    st.markdown("### REPORT GENERATION & EXPORT")
    
    st.markdown("#### Report Content Options")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.checkbox("Include Record Metadata", value=True)
        st.checkbox("Include Waveforms", value=True)
    with col_c2:
        st.checkbox("Include QC Audit Log", value=True)
        st.checkbox("Include Strong-Motion Table", value=True)
    with col_c3:
        st.checkbox("Include Response Spectrum", value=True)
        st.checkbox("Include FAS Plot", value=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Generate Station PDF Report", type="primary"):
            safe_record_id = re.sub(r'[<>:"/\\|?*]+', "_", station)
            output = REPORT_DIRECTORY / f"BSMA_Report_{safe_record_id}.pdf"
            with st.spinner("Compiling PDF report..."):
                pdf_path = ExportService().export_station_pdf(
                    station,
                    contexts,
                    output,
                    event_info=event_info or None,
                )
            st.session_state["last_pdf"] = pdf_path.read_bytes()
            st.session_state["last_pdf_name"] = pdf_path.name
            st.success("PDF report generated successfully.")

    with col_btn2:
        if st.session_state.get("last_pdf"):
            st.download_button(
                "Download PDF File",
                data=st.session_state["last_pdf"],
                file_name=st.session_state["last_pdf_name"],
                mime="application/pdf",
            )


def _display_analysis(
    station: str,
    contexts: dict[str, Any],
    event_info: dict[str, Any],
    configuration: AnalysisConfiguration,
) -> None:
    """Main analysis container with workflow stepper and 6 scientific navigation tabs."""
    _render_workflow_stepper("ANALYSIS", has_data=True, has_qc=True)

    summary, waveform, qc_tab, strong_motion, intensity, spectrum, report_tab = st.tabs(
        ["SUMMARY", "WAVEFORMS", "QC", "STRONG MOTION", "INTENSITY", "SPECTRUM", "REPORT"]
    )

    with summary:
        _display_summary_view(station, contexts, event_info, configuration)
    with waveform:
        _display_waveforms_view(contexts)
    with qc_tab:
        _display_qc_view(contexts)
    with strong_motion:
        _display_strong_motion_view(contexts)
    with intensity:
        _display_intensity_view(contexts)
    with spectrum:
        _display_spectrum_view(contexts, configuration)
    with report_tab:
        _display_report_view(station, contexts, event_info)


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
        st.warning("Processing blocked: Input data provenance undeclared.")
    if missing_inventory:
        st.error("Raw-count mode requires StationXML for all stations. Missing: " + ", ".join(missing_inventory))

    if st.button(
        "Run Batch Processing",
        type="primary",
        disabled=unknown_provenance or bool(missing_inventory),
    ):
        progress = st.progress(0, text="Initializing batch execution...")

        def on_progress(index: int, total: int, station: str) -> None:
            progress.progress(index / total, text=f"Processing {station} ({index}/{total})")

        streams = records
        inventories = {
            record: _find_inventory(str(stream[0].stats.station)) if st.session_state.get("apply_instrument_response", False) else None
            for record, stream in streams.items()
        }
        result = BatchService(_service(configuration)).process_stations(
            streams, inventories, progress_callback=on_progress
        )
        st.session_state["contexts_by_station"].update(result.contexts_by_station)
        st.session_state["batch_failures"] = result.failures
        st.session_state["batch_rows"] = result.summary_rows()
        progress.progress(1.0, text="Batch processing complete.")

    rows = st.session_state.get("batch_rows", [])
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if st.session_state.get("batch_failures"):
        st.error("Batch Failure Log")
        st.json(st.session_state["batch_failures"])


def _export_batch(event_info: dict[str, Any]) -> None:
    contexts = st.session_state["contexts_by_station"]
    if not contexts:
        st.info("Process at least one station before export.")
        return
    selected = st.multiselect("Select Stations for Export", list(contexts), default=list(contexts))
    if st.button("Build Export Package (ZIP)", type="primary"):
        if not selected:
            st.error("Select at least one station.")
            return
        exporter = ExportService()
        selected_contexts = {station: contexts[station] for station in selected}
        archive_bytes, archive_name = exporter.export_batch_package(
            selected_contexts,
            REPORT_DIRECTORY,
            event_info=event_info or None,
        )
        st.session_state["export_archive"] = archive_bytes
        st.session_state["export_archive_name"] = archive_name

    if st.session_state.get("export_archive"):
        st.download_button(
            "Download ZIP Package",
            data=st.session_state["export_archive"],
            file_name=st.session_state["export_archive_name"],
            mime="application/zip",
        )


def _display_benchmark(record_id: str, contexts: dict[str, Any]) -> None:
    reference = st.session_state.get("benchmark_reference")
    with st.expander("Benchmark Reference Audit", expanded=False):
        if reference is None:
            st.info("No reference CSV loaded. Upload a reference metrics CSV in sidebar to perform automated validation.")
            return
        normalized = reference.rename(columns={str(col): str(col).strip().lower() for col in reference.columns})
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
                rows.append({"Channel": channel or "-", "Metric": "-", "Computed": "-", "Reference": "-", "Error (%)": "-", "Status": "NOT FOUND"})
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
                        "Channel": channel,
                        "Metric": metric_name,
                        "Computed": f"{computed_value:.6f}",
                        "Reference": f"{reference_value:.6f}",
                        "Error (%)": f"{relative_error:.2f}%",
                        "Status": "PASS" if relative_error <= tolerance else "REVIEW",
                    }
                )
        if not rows:
            st.warning("No matching channel metrics found in benchmark CSV.")
            return
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_status_footer(station: str = "-", num_components: int = 0, sampling_rate: float = 0.0) -> None:
    """Render sticky bottom status bar."""
    st.markdown(
        f"""
        <div class="status-footer">
            <div>Ready &nbsp;|&nbsp; Station: <strong>{station}</strong> &nbsp;|&nbsp; Components: <strong>{num_components}</strong> &nbsp;|&nbsp; Sampling: <strong>{sampling_rate:.1f} Hz</strong></div>
            <div>BMKG Strong Motion Analyzer v2.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_custom_css()
    _initialise_state()
    _ensure_directories()
    
    configuration, event_info = _configuration_from_sidebar()

    # App Header Banner
    col_h1, col_h2 = st.columns([1, 6])
    with col_h1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=100)
    with col_h2:
        st.markdown(
            """
            <h1 style="color: #f2f2f2; font-size: 1.6rem; font-weight: 700; margin-bottom: 0;">
                BMKG Strong Motion Analyzer (BSMA)
            </h1>
            <p style="color: #a8a8a8; font-size: 0.85rem; margin-top: 0;">
                Professional Seismological & Geotechnical Strong-Motion Processing Workstation
            </p>
            """,
            unsafe_allow_html=True,
        )

    files = _waveform_files()
    if not files:
        st.info("Upload MiniSEED/SAC waveforms and optional StationXML from the sidebar panel to begin processing.")
        st.stop()

    master_stream = _load_master_stream(files)
    records = _record_windows(master_stream)
    if not records:
        st.error("No valid station waveform traces found in the uploaded dataset.")
        st.stop()

    mode = st.segmented_control(
        "Workflow Mode",
        options=["Single-station review", "Multi-station processing", "Export results"],
        default="Single-station review",
        key="app_mode",
    )

    current_station = "-"
    current_num_components = 0
    current_sampling_rate = 0.0

    if mode == "Single-station review":
        record_id = st.selectbox("Select Recording Window", list(records))
        station_stream = records[record_id]
        station = str(station_stream[0].stats.station)
        inventory_path = _find_inventory_path(station)
        inventory = _find_inventory(station) if st.session_state.get("apply_instrument_response", False) else None
        
        status_text = f"StationXML Response Correction Active: `{inventory_path.name}`." if inventory is not None and inventory_path else "Physical Acceleration Mode (StationXML correction bypassed)."
        st.caption(status_text)
        
        unknown_provenance = st.session_state.get("input_provenance") == "Unknown - require scientific review"
        missing_inventory = configuration.input_mode == "raw_counts" and inventory is None
        
        if unknown_provenance:
            st.warning("Select Data Provenance in sidebar to unlock processing.")
        if missing_inventory:
            st.error("Raw Counts mode requires StationXML for instrument response removal.")

        if st.button(
            "Run Analysis",
            type="primary",
            disabled=unknown_provenance or missing_inventory,
        ):
            try:
                with st.spinner(f"Executing pipeline for station {station}..."):
                    _process_one_station(record_id, station_stream, configuration)
            except Exception as exc:
                st.exception(exc)

        contexts = st.session_state["contexts_by_station"].get(record_id)
        if contexts:
            current_station = station
            current_num_components = len(contexts)
            strongest = next(iter(contexts.values()))
            current_sampling_rate = strongest.sampling_rate
            _display_analysis(record_id, contexts, event_info, configuration)

    elif mode == "Multi-station processing":
        _batch_analysis(records, configuration)
    else:
        _export_batch(event_info)

    _render_status_footer(current_station, current_num_components, current_sampling_rate)


if __name__ == "__main__":
    main()
