"""
BMKG Strong Motion Analyzer (BSMA)
Module: utils/pdf_exporter.py

Description
-----------
Professional multi-page PDF report generator for BSMA.

The exporter generates:
    1. Station summary and strongest ground-motion component.
    2. BMKG SIG classification.
    3. Strong-motion parameter table.
    4. Response spectrum overlay.
    5. Four-panel waveform plots for every component:
       Raw waveform, acceleration, velocity, and displacement.

Python
------
>= 3.12
"""

from __future__ import annotations

import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

# Required for Streamlit/headless/server environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF

from core.types.context import ProcessingContext
from utils.logger import setup_logger


# =============================================================================
# CONSTANTS
# =============================================================================

GRAVITY = 9.80665

MM_PER_INCH = 25.4

PAGE_WIDTH_MM = 210.0
PAGE_HEIGHT_MM = 297.0

MARGIN_LEFT = 10.0
MARGIN_RIGHT = 10.0
MARGIN_TOP = 10.0
MARGIN_BOTTOM = 15.0

CONTENT_WIDTH = PAGE_WIDTH_MM - MARGIN_LEFT - MARGIN_RIGHT

logger = setup_logger(__name__)


# =============================================================================
# BMKG SIG CLASSIFICATION
# =============================================================================

def get_sig_bmkg(
    pga_gal: float,
) -> tuple[str, str, tuple[int, int, int], str, str, str]:
    """
    Classify earthquake shaking according to the BSMA BMKG SIG scale.

    Parameters
    ----------
    pga_gal:
        Peak Ground Acceleration in Gal.

    Returns
    -------
    tuple
        (
            sig_scale,
            color_name,
            rgb,
            short_description,
            mmi_range,
            detailed_description,
        )

    Notes
    -----
    PGA conversion:

        1 m/s² = 100 Gal

    The thresholds and descriptions are kept consistent with the
    classification currently used by the BSMA application.
    """

    pga_gal = float(pga_gal)

    if not math.isfinite(pga_gal):
        pga_gal = 0.0

    gravity = 9.80665
    thresholds = (
        (0.05 * gravity, ("I", "Putih", (255, 255, 255), "TIDAK DIRASAKAN", "I-II", "Tidak dirasakan atau dirasakan hanya oleh beberapa orang tetapi terekam oleh alat.")),
        (0.30 * gravity, ("II", "Hijau", (146, 208, 80), "DIRASAKAN", "III-V", "Dirasakan oleh orang banyak tetapi tidak menimbulkan kerusakan. Benda-benda ringan yang digantung bergoyang dan jendela kaca bergetar.")),
        (2.8 * gravity, ("III", "Kuning", (255, 255, 0), "KERUSAKAN RINGAN", "VI", "Bagian nonstruktur bangunan mengalami kerusakan ringan, seperti retak rambut pada dinding, atap bergeser ke bawah dan sebagian berjatuhan.")),
        (6.2 * gravity, ("IV", "Jingga", (255, 192, 0), "KERUSAKAN SEDANG", "VII-VIII", "Banyak retakan terjadi pada dinding bangunan sederhana, sebagian roboh, kaca pecah. Sebagian plester dinding lepas.")),
        (12.0 * gravity, ("V", "Merah", (255, 0, 0), "KERUSAKAN BERAT", "IX-XII", "Sebagian besar dinding bangunan permanen roboh. Struktur bangunan mengalami kerusakan berat.")),
        (22.0 * gravity, ("VI", "Merah tua", (192, 0, 0), "KERUSAKAN PARAH", "X", "Kerusakan berat pada struktur umum dan sebagian bangunan mengalami kehancuran signifikan.")),
        (40.0 * gravity, ("VII", "Merah tua", (128, 0, 0), "KERUSAKAN PARAH", "X-XI", "Bangunan umum mengalami kerusakan besar dan banyak struktur tidak layak pakai.")),
        (75.0 * gravity, ("VIII", "Cokelat", (128, 64, 0), "KERUSAKAN SANGAT PARAH", "XI-XII", "Kerusakan besar, banyak bangunan runtuh, potensi korban dan gangguan sistem utilitas tinggi.")),
        (139.0 * gravity, ("IX", "Hitam", (0, 0, 0), "KERUSAKAN EKSTREM", "XII+", "Kerusakan menyeluruh pada infrastruktur dan bangunan utama, dengan konsekuensi sangat berat.")),
    )

    if not math.isfinite(pga_gal):
        pga_gal = 0.0

    for threshold, value in thresholds:
        if pga_gal < threshold:
            return value

    return (
        "X+",
        "Hitam",
        (0, 0, 0),
        "KERUSAKAN EKSTREM",
        "XII+",
        "Kerusakan total dan konsekuensi sangat berat pada bangunan dan infrastruktur.",
    )


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to finite float safely."""

    try:
        result = float(value)

        if math.isfinite(result):
            return result

    except (TypeError, ValueError):
        pass

    return default


def _safe_array(value: Any) -> np.ndarray:
    """
    Convert input into a finite one-dimensional NumPy array.

    Invalid/non-finite values are retained as NaN for plotting.
    """

    if value is None:
        return np.array([], dtype=float)

    try:
        array = np.asarray(value, dtype=float)

        if array.ndim == 0:
            array = array.reshape(1)

        return array.ravel()

    except (TypeError, ValueError):
        return np.array([], dtype=float)


def _get_spectral_array(
    spectral_data: Any,
    *keys: str,
) -> np.ndarray:
    """Retrieve a spectral array supporting multiple key conventions."""

    if not isinstance(spectral_data, Mapping):
        return np.array([], dtype=float)

    for key in keys:
        value = spectral_data.get(key)

        if value is not None:
            array = _safe_array(value)

            if array.size > 0:
                return array

    return np.array([], dtype=float)


def _finite_max(array: np.ndarray, default: float = 0.0) -> float:
    """Return finite maximum absolute/value from an array."""

    if array.size == 0:
        return default

    finite = array[np.isfinite(array)]

    if finite.size == 0:
        return default

    return float(np.max(finite))


def _format_float(
    value: Any,
    decimals: int = 3,
) -> str:
    """Format numerical value safely for PDF output."""

    number = _safe_float(value)

    return f"{number:.{decimals}f}"


def _extract_metrics(
    context: ProcessingContext,
) -> dict[str, float]:
    """Extract standardized strong-motion metrics from ProcessingContext."""

    metrics = context.metrics or {}

    return {
        "PGA": _safe_float(metrics.get("PGA")),
        "PGV": _safe_float(metrics.get("PGV")),
        "PGD": _safe_float(metrics.get("PGD")),
        "Arias_Intensity": _safe_float(
            metrics.get("Arias_Intensity")
        ),
        "Significant_Duration_D5_95": _safe_float(
            metrics.get("Significant_Duration_D5_95")
        ),
    }


def _channel_suffix(channel: str) -> str:
    """Return the three-character channel suffix."""

    channel = str(channel).strip()

    if len(channel) >= 3:
        return channel[-3:]

    return channel


# =============================================================================
# PDF DOCUMENT
# =============================================================================

class BSMAPDFReport(FPDF):
    """
    Custom FPDF document for BSMA reports.
    """

    def __init__(
        self,
        station_code: str,
        generated_at: datetime | None = None,
    ) -> None:
        super().__init__(
            orientation="P",
            unit="mm",
            format="A4",
        )

        self.station_code = station_code

        self.generated_at = generated_at or datetime.now(
            timezone.utc
        )

        self.set_auto_page_break(
            auto=True,
            margin=MARGIN_BOTTOM,
        )

        self.set_margins(
            left=MARGIN_LEFT,
            top=MARGIN_TOP,
            right=MARGIN_RIGHT,
        )

        self.set_title(
            f"BSMA Strong Motion Report - {station_code}"
        )

        self.set_author("BSMA Development Team")

        self.set_creator("BMKG Strong Motion Analyzer")

    def header(self) -> None:
        """Render report header."""

        self.set_font(
            "helvetica",
            "B",
            13,
        )

        self.set_text_color(
            0,
            51,
            102,
        )

        self.cell(
            0,
            7,
            "BMKG STRONG MOTION ANALYZER",
            border=0,
            ln=1,
            align="C",
        )

        self.set_font(
            "helvetica",
            "",
            8,
        )

        self.set_text_color(
            80,
            80,
            80,
        )

        self.cell(
            0,
            5,
            "Laporan Analisis Strong Motion",
            border=0,
            ln=1,
            align="C",
        )

        self.set_draw_color(
            0,
            51,
            102,
        )

        self.set_line_width(0.4)

        self.line(
            MARGIN_LEFT,
            23,
            PAGE_WIDTH_MM - MARGIN_RIGHT,
            23,
        )

        self.ln(5)

    def footer(self) -> None:
        """Render report footer."""

        self.set_y(-11)

        self.set_font(
            "helvetica",
            "",
            7,
        )

        self.set_text_color(
            120,
            120,
            120,
        )

        generated = self.generated_at.strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

        self.cell(
            0,
            5,
            (
                f"BSMA | {self.station_code} | "
                f"Dibuat: {generated} | "
                f"Halaman {self.page_no()}"
            ),
            border=0,
            ln=0,
            align="C",
        )


# =============================================================================
# PDF SECTION HELPERS
# =============================================================================

def _pdf_section_title(
    pdf: FPDF,
    number: str,
    title: str,
) -> None:
    """Render standardized section heading."""

    pdf.set_font(
        "helvetica",
        "B",
        11,
    )

    pdf.set_text_color(
        0,
        51,
        102,
    )

    section_title = (
        f"{number}. {title}"
        if number and str(number).strip()
        else title
    )

    pdf.cell(
        0,
        7,
        section_title,
        border=0,
        ln=1,
    )

    pdf.set_draw_color(
        180,
        180,
        180,
    )

    pdf.set_line_width(0.2)

    pdf.line(
        MARGIN_LEFT,
        pdf.get_y(),
        PAGE_WIDTH_MM - MARGIN_RIGHT,
        pdf.get_y(),
    )

    pdf.ln(3)


def _pdf_label_value(
    pdf: FPDF,
    label: str,
    value: str,
    label_width: float = 40.0,
) -> None:
    """Render a simple label/value row."""

    pdf.set_font(
        "helvetica",
        "B",
        8.5,
    )

    pdf.set_text_color(
        50,
        50,
        50,
    )

    pdf.cell(
        label_width,
        5,
        label,
        border=0,
    )

    pdf.set_font(
        "helvetica",
        "",
        8.5,
    )

    pdf.cell(
        0,
        5,
        value,
        border=0,
        ln=1,
    )


def _pdf_add_sig_box(
    pdf: FPDF,
    sig_id: str,
    sig_color: str,
    rgb: tuple[int, int, int],
    sig_desc: str,
    mmi: str,
    sig_detail: str,
) -> None:
    """Render BMKG SIG classification box."""

    pdf.set_fill_color(*rgb)

    pdf.set_draw_color(
        70,
        70,
        70,
    )

    # Dark text is preferred for white/yellow/green.
    if sig_id in {"IV", "V"}:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)

    pdf.set_text_color(*text_color)

    pdf.set_font(
        "helvetica",
        "B",
        13,
    )

    pdf.cell(
        0,
        11,
        (
            f"SKALA {sig_id} SIG-BMKG | "
            f"{sig_desc}"
        ),
        border=1,
        ln=1,
        align="C",
        fill=True,
    )

    pdf.set_font(
        "helvetica",
        "B",
        9,
    )

    pdf.cell(
        0,
        6,
        (
            f"Warna: {sig_color} | "
            f"Rentang intensitas: {mmi}"
        ),
        border="LR",
        ln=1,
        align="C",
        fill=True,
    )

    pdf.set_font(
        "helvetica",
        "",
        8.5,
    )

    pdf.multi_cell(
        0,
        5,
        f"Dampak: {sig_detail}",
        border="LBR",
        align="L",
        fill=True,
    )

    pdf.ln(5)

    pdf.set_text_color(
        0,
        0,
        0,
    )


# =============================================================================
# RESPONSE SPECTRUM PLOT
# =============================================================================

def generate_spectrum_overlay(
    contexts: dict[str, ProcessingContext],
    output_path: str | Path,
    station_code: str,
) -> bool:
    """
    Generate response-spectrum overlay.

    Parameters
    ----------
    contexts:
        Processed station contexts indexed by channel.

    output_path:
        Destination PNG path.

    station_code:
        Station identifier.

    Returns
    -------
    bool
        True when at least one valid spectrum was plotted.
    """

    output_path = Path(output_path)

    fig, ax = plt.subplots(
        figsize=(10, 4.8),
        dpi=160,
    )

    colors = {
        "HNE": "purple",
        "HNN": "goldenrod",
        "HNZ": "darkgreen",
    }

    has_data = False

    for channel, context in sorted(contexts.items()):
        spectral_data = context.spectral_data

        periods = _get_spectral_array(
            spectral_data,
            "periods",
            "Periods",
        )

        psa = _get_spectral_array(
            spectral_data,
            "PSA",
            "psa",
        )

        if periods.size == 0 or psa.size == 0:
            continue

        n = min(
            periods.size,
            psa.size,
        )

        periods = periods[:n]
        psa = psa[:n]

        valid = (
            np.isfinite(periods)
            & np.isfinite(psa)
            & (periods > 0.0)
        )

        if not np.any(valid):
            continue

        periods = periods[valid]
        psa = psa[valid]

        psa_g = psa / GRAVITY

        max_sa = _finite_max(psa_g)

        suffix = _channel_suffix(channel)

        color = colors.get(
            suffix,
            None,
        )

        label = (
            f"{channel} "
            f"(Max Sa: {max_sa:.4f} g)"
        )

        plot_kwargs = {
            "label": label,
            "linewidth": 1.5,
        }

        if color is not None:
            plot_kwargs["color"] = color

        ax.plot(
            periods,
            psa_g,
            **plot_kwargs,
        )

        has_data = True

    ax.set_title(
        (
            "Response Spectrum Overlay "
            f"- Stasiun {station_code}"
        ),
        fontweight="bold",
        fontsize=11,
    )

    ax.set_xlabel(
        "Periode Struktur (s)",
        fontsize=9,
    )

    ax.set_ylabel(
        "Spectral Acceleration (g)",
        fontsize=9,
    )

    if has_data:
        ax.set_xscale("log")

        ax.grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.5,
            alpha=0.5,
        )

        ax.legend(
            loc="best",
            fontsize=7.5,
            frameon=True,
        )

    else:
        ax.text(
            0.5,
            0.5,
            "Response spectrum tidak tersedia",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )

        ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    return has_data


# =============================================================================
# WAVEFORM PLOT
# =============================================================================

def generate_4panel_waveform(
    context: ProcessingContext,
    channel: str,
    output_path: str | Path,
    station_code: str,
) -> bool:
    """
    Generate four-panel waveform figure.

    Panels
    ------
    1. Raw waveform.
    2. Filtered acceleration.
    3. Velocity.
    4. Displacement.

    Units
    -----
    Raw:
        Gal

    Acceleration:
        m/s²

    Velocity:
        cm/s

    Displacement:
        cm
    """

    output_path = Path(output_path)

    raw = (
        _safe_array(
            context.raw_waveform.data
        )
        if context.raw_waveform is not None
        else np.array([], dtype=float)
    )

    acc = (
        _safe_array(
            context.acceleration.data
        )
        if context.acceleration is not None
        else np.array([], dtype=float)
    )

    vel = (
        _safe_array(
            context.velocity.data
        )
        if context.velocity is not None
        else np.array([], dtype=float)
    )

    disp = (
        _safe_array(
            context.displacement.data
        )
        if context.displacement is not None
        else np.array([], dtype=float)
    )

    sampling_rate = _safe_float(
        context.sampling_rate
    )

    if sampling_rate <= 0.0:
        logger.warning(
            "Sampling rate tidak valid untuk %s/%s.",
            station_code,
            channel,
        )
        return False

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 10.5),
        sharex=False,
        dpi=160,
    )

    # -------------------------------------------------------------------------
    # RAW
    # -------------------------------------------------------------------------

    if raw.size > 0:
        time_raw = (
            np.arange(raw.size)
            / sampling_rate
        )

        axes[0].plot(
            time_raw,
            raw * 100.0,
            linewidth=0.45,
        )

    axes[0].set_title(
        f"1. Raw Waveform - {channel}",
        fontweight="bold",
        fontsize=9.5,
    )

    axes[0].set_ylabel(
        "Amplitude (Gal)",
        fontsize=8,
    )

    axes[0].grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.5,
    )

    # -------------------------------------------------------------------------
    # ACCELERATION
    # -------------------------------------------------------------------------

    if acc.size > 0:
        time_acc = (
            np.arange(acc.size)
            / sampling_rate
        )

        axes[1].plot(
            time_acc,
            acc,
            linewidth=0.45,
        )

    axes[1].set_title(
        "2. Filtered Ground Acceleration",
        fontweight="bold",
        fontsize=9.5,
    )

    axes[1].set_ylabel(
        "Acceleration (m/s²)",
        fontsize=8,
    )

    axes[1].grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.5,
    )

    # -------------------------------------------------------------------------
    # VELOCITY
    # -------------------------------------------------------------------------

    if vel.size > 0:
        time_vel = (
            np.arange(vel.size)
            / sampling_rate
        )

        axes[2].plot(
            time_vel,
            vel * 100.0,
            linewidth=0.45,
        )

    axes[2].set_title(
        "3. Ground Velocity",
        fontweight="bold",
        fontsize=9.5,
    )

    axes[2].set_ylabel(
        "Velocity (cm/s)",
        fontsize=8,
    )

    axes[2].grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.5,
    )

    # -------------------------------------------------------------------------
    # DISPLACEMENT
    # -------------------------------------------------------------------------

    if disp.size > 0:
        time_disp = (
            np.arange(disp.size)
            / sampling_rate
        )

        axes[3].plot(
            time_disp,
            disp * 100.0,
            linewidth=0.45,
        )

    axes[3].set_title(
        "4. Ground Displacement",
        fontweight="bold",
        fontsize=9.5,
    )

    axes[3].set_ylabel(
        "Displacement (cm)",
        fontsize=8,
    )

    axes[3].set_xlabel(
        "Time (s)",
        fontsize=8,
    )

    axes[3].grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.5,
    )

    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------

    metrics = _extract_metrics(context)

    # Annotate the physically meaningful timing markers on the raw and
    # processed acceleration histories.  D5/D95 are obtained from the cached
    # normalized Arias-energy curve produced by the analysis pipeline.
    marker_times: list[tuple[float, str, str, str]] = []
    if acc.size > 0:
        peak_index = int(np.argmax(np.abs(acc)))
        marker_times.append((float(peak_index / sampling_rate), "PGA", "#d62728", ":"))
    husid_curve = getattr(context.cache, "husid_curve", None)
    if husid_curve is not None and len(husid_curve) > 0:
        husid_values = np.asarray(husid_curve, dtype=float)
        for level, label, color in ((0.05, "D5", "#f39c12"), (0.95, "D95", "#2ca02c")):
            index = min(int(np.searchsorted(husid_values, level)), len(husid_values) - 1)
            marker_times.append((float(index / sampling_rate), label, color, "--"))
    for marker_time, label, color, linestyle in marker_times:
        for axis in axes[:2]:
            axis.axvline(marker_time, color=color, linestyle=linestyle, linewidth=0.85, alpha=0.9)
        axes[1].annotate(
            label,
            xy=(marker_time, 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(2, -2),
            textcoords="offset points",
            color=color,
            fontsize=6.8,
            va="top",
        )

    axes[1].text(
        0.99,
        0.92,
        (
            f"PGA = {metrics['PGA']:.6f} m/s² | "
            f"{metrics['PGA'] * 100.0:.3f} Gal"
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            alpha=0.75,
        ),
    )

    axes[2].text(
        0.99,
        0.92,
        (
            f"PGV = "
            f"{metrics['PGV'] * 100.0:.4f} cm/s"
        ),
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            alpha=0.75,
        ),
    )

    axes[3].text(
        0.99,
        0.92,
        (
            f"PGD = "
            f"{metrics['PGD'] * 100.0:.4f} cm"
        ),
        transform=axes[3].transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            alpha=0.75,
        ),
    )

    fig.suptitle(
        (
            "Analisis Kinematik Ground Motion\n"
            f"Stasiun: {station_code} | "
            f"Channel: {channel}"
        ),
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.965)
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    return True


# =============================================================================
# METADATA EXTRACTION
# =============================================================================

def _extract_station_metadata(
    contexts: dict[str, ProcessingContext],
) -> dict[str, str]:
    """Extract station metadata from the first available context."""

    metadata: dict[str, str] = {
        "Network": "-",
        "Station": "-",
        "Location": "-",
        "Sampling Rate": "-",
        "Start Time": "-",
        "End Time": "-",
        "Channels": "-",
    }

    if not contexts:
        return metadata

    first_context = next(
        iter(contexts.values())
    )

    stats = first_context.metadata or {}

    metadata["Network"] = str(
        stats.get("network", "-")
    )

    metadata["Station"] = str(
        stats.get("station", "-")
    )

    metadata["Location"] = str(
        stats.get("location", "-")
    )

    sampling_rate = stats.get(
        "sampling_rate",
        first_context.sampling_rate,
    )

    metadata["Sampling Rate"] = (
        f"{_safe_float(sampling_rate):.3f} Hz"
    )

    starttime = stats.get("starttime")

    if starttime is not None:
        metadata["Start Time"] = str(
            starttime
        )

    endtime = stats.get("endtime")

    if endtime is not None:
        metadata["End Time"] = str(
            endtime
        )

    metadata["Channels"] = ", ".join(
        sorted(contexts.keys())
    )

    return metadata


# =============================================================================
# PDF TABLE
# =============================================================================

def _add_parameter_table(
    pdf: FPDF,
    contexts: dict[str, ProcessingContext],
) -> None:
    """Render strong-motion parameter table."""

    headers = [
        "Channel",
        "PGA\n(Gal)",
        "PGV\n(cm/s)",
        "PGD\n(cm)",
        "Arias\n(m/s)",
        "D5-95\n(s)",
        "SIG",
    ]

    column_widths = [
        25,
        25,
        25,
        25,
        30,
        25,
        35,
    ]

    pdf.set_fill_color(
        220,
        225,
        230,
    )

    pdf.set_text_color(
        0,
        0,
        0,
    )

    pdf.set_font(
        "helvetica",
        "B",
        7.5,
    )

    for width, header in zip(
        column_widths,
        headers,
    ):
        # Keep the table compatible with both PyFPDF and fpdf2.  The former
        # does not accept fpdf2's ``new_x``/``new_y`` cursor arguments.
        pdf.cell(width, 10, header, border=1, align="C", fill=True)

    pdf.ln()

    pdf.set_font(
        "helvetica",
        "",
        7.5,
    )

    for channel in sorted(contexts.keys()):
        context = contexts[channel]

        metrics = _extract_metrics(
            context
        )

        pga_gal = (
            metrics["PGA"] * 100.0
        )

        sig_id, _, _, _, mmi, _ = (
            get_sig_bmkg(pga_gal)
        )

        values = [
            channel,
            f"{pga_gal:.3f}",
            f"{metrics['PGV'] * 100.0:.4f}",
            f"{metrics['PGD'] * 100.0:.4f}",
            f"{metrics['Arias_Intensity']:.6f}",
            (
                f"{metrics['Significant_Duration_D5_95']:.2f}"
            ),
            f"{sig_id} ({mmi})",
        ]

        for width, value in zip(
            column_widths,
            values,
        ):
            pdf.cell(
                width,
                7,
                value,
                border=1,
                align="C",
            )

        pdf.ln()


# =============================================================================
# MAIN EXPORT FUNCTION
# =============================================================================

def export_station_report(
    station_code: str,
    contexts: dict[str, ProcessingContext],
    output_dir: str | Path = "outputs/reports",
    output_path: str | Path | None = None,
    *,
    event_info: dict[str, Any] | None = None,
) -> Path:
    """
    Generate complete station PDF report.

    Parameters
    ----------
    station_code:
        Station identifier.

    contexts:
        Processed contexts indexed by channel.

    output_dir:
        Destination directory for the generated report when
        ``output_path`` is not specified.

    output_path:
        Optional explicit output file path for the report.

    Returns
    -------
    pathlib.Path
        Generated PDF path.

    Raises
    ------
    ValueError
        If no processed contexts are supplied.
    """

    if not contexts:
        raise ValueError(
            (
                "Tidak dapat membuat laporan PDF: "
                "contexts kosong."
            )
        )

    if output_path is not None:
        pdf_path = Path(output_path)
        output_directory = pdf_path.parent
    else:
        output_directory = Path(output_dir)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        pdf_path = output_directory / f"BSMA_Report_{station_code}_{timestamp}.pdf"

    output_directory.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Memulai pembuatan PDF stasiun %s.",
        station_code,
    )

    # -------------------------------------------------------------------------
    # Determine strongest component
    # -------------------------------------------------------------------------

    strongest_channel = None
    strongest_pga = -np.inf

    for channel, context in contexts.items():
        metrics = _extract_metrics(
            context
        )

        pga = metrics["PGA"]

        if pga > strongest_pga:
            strongest_pga = pga
            strongest_channel = channel

    if strongest_channel is None:
        strongest_channel = "-"

    if not math.isfinite(
        strongest_pga
    ):
        strongest_pga = 0.0

    strongest_pga_gal = (
        strongest_pga * 100.0
    )

    (
        sig_id,
        sig_color,
        sig_rgb,
        sig_desc,
        mmi,
        sig_detail,
    ) = get_sig_bmkg(
        strongest_pga_gal
    )

    metadata = _extract_station_metadata(
        contexts
    )

    # -------------------------------------------------------------------------
    # Create PDF
    # -------------------------------------------------------------------------

    pdf = BSMAPDFReport(
        station_code=station_code
    )

    pdf.add_page()

    # -------------------------------------------------------------------------
    # SECTION 1 - EXECUTIVE SUMMARY
    # -------------------------------------------------------------------------

    _pdf_section_title(
        pdf,
        "1",
        "RINGKASAN GUNCANGAN TERKUAT",
    )

    pdf.set_font(
        "helvetica",
        "",
        9,
    )

    summary_text = (
        f"Stasiun {station_code} mencatat "
        f"nilai percepatan tanah maksimum "
        f"(PGA) sebesar "
        f"{strongest_pga:.6f} m/s2 "
        f"atau {strongest_pga_gal:.3f} Gal "
        f"pada komponen {strongest_channel}. "
        f"Berdasarkan klasifikasi SIG-BMKG "
        f"yang digunakan oleh BSMA, hasil "
        f"tersebut termasuk Skala {sig_id} "
        f"({sig_desc}) dengan rentang "
        f"intensitas {mmi}."
    )

    pdf.multi_cell(
        0,
        5,
        summary_text,
    )

    pdf.ln(3)

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    _pdf_label_value(
        pdf,
        "Network",
        metadata["Network"],
    )

    _pdf_label_value(
        pdf,
        "Station",
        metadata["Station"],
    )

    _pdf_label_value(
        pdf,
        "Location",
        metadata["Location"],
    )

    _pdf_label_value(
        pdf,
        "Sampling Rate",
        metadata["Sampling Rate"],
    )

    _pdf_label_value(
        pdf,
        "Start Time",
        metadata["Start Time"],
    )

    _pdf_label_value(
        pdf,
        "End Time",
        metadata["End Time"],
    )

    _pdf_label_value(
        pdf,
        "Channels",
        metadata["Channels"],
    )

    pdf.ln(4)

    if event_info:
        _pdf_section_title(
            pdf,
            "",
            "INFORMASI EVENT",
        )
        event_rows = (
            ("Waktu", event_info.get("time", "-")),
            ("Koordinat", f"{event_info.get('latitude', '-')} , {event_info.get('longitude', '-')}"),
            ("Magnitudo", event_info.get("magnitude", "-")),
            ("Kedalaman", f"{event_info.get('depth_km', '-')} km"),
            ("Jarak episentral", f"{event_info.get('epicentral_distance_km', '-')} km"),
        )
        for label, value in event_rows:
            if value in (None, "", "- , -"):
                continue
            _pdf_label_value(pdf, label, value)
        pdf.ln(4)

    # -------------------------------------------------------------------------
    # SIG classification
    # -------------------------------------------------------------------------

    _pdf_add_sig_box(
        pdf=pdf,
        sig_id=sig_id,
        sig_color=sig_color,
        rgb=sig_rgb,
        sig_desc=sig_desc,
        mmi=mmi,
        sig_detail=sig_detail,
    )

    # -------------------------------------------------------------------------
    # SECTION 2 - PARAMETER TABLE
    # -------------------------------------------------------------------------

    _pdf_section_title(
        pdf,
        "2",
        "PARAMETER STRONG MOTION",
    )

    _add_parameter_table(
        pdf,
        contexts,
    )

    pdf.ln(5)

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------

    strongest_metrics = _extract_metrics(
        contexts[strongest_channel]
    )

    pdf.set_font(
        "helvetica",
        "B",
        9,
    )

    pdf.cell(
        0,
        5,
        "Parameter komponen terkuat:",
        ln=1,
    )

    pdf.set_font(
        "helvetica",
        "",
        8.5,
    )

    interpretation = (
        f"PGA = {strongest_metrics['PGA']:.6f} m/s2 "
        f"({strongest_metrics['PGA'] * 100.0:.3f} Gal); "
        f"PGV = {strongest_metrics['PGV'] * 100.0:.4f} cm/s; "
        f"PGD = {strongest_metrics['PGD'] * 100.0:.4f} cm; "
        f"Arias Intensity = "
        f"{strongest_metrics['Arias_Intensity']:.6f} m/s; "
        f"D5-95 = "
        f"{strongest_metrics['Significant_Duration_D5_95']:.2f} s."
    )

    pdf.multi_cell(
        0,
        5,
        interpretation,
    )

    # -------------------------------------------------------------------------
    # SECTION 3 - RESPONSE SPECTRUM
    # -------------------------------------------------------------------------

    pdf.add_page()

    _pdf_section_title(
        pdf,
        "3",
        "RESPONSE SPECTRUM",
    )

    pdf.set_font(
        "helvetica",
        "",
        8.5,
    )

    pdf.multi_cell(
        0,
        5,
        (
            "Kurva response spectrum menunjukkan "
            "percepatan spektral sebagai fungsi "
            "periode struktur berdasarkan hasil "
            "pemrosesan response spectrum pada "
            "pipeline BSMA."
        ),
    )

    pdf.ln(3)

    # -------------------------------------------------------------------------
    # Temporary plot directory
    # -------------------------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="bsma_pdf_"
    ) as temp_directory:

        temp_dir = Path(
            temp_directory
        )

        spectrum_path = (
            temp_dir
            / "response_spectrum.png"
        )

        try:
            spectrum_available = (
                generate_spectrum_overlay(
                    contexts=contexts,
                    output_path=spectrum_path,
                    station_code=station_code,
                )
            )

            if spectrum_available:
                pdf.image(
                    str(spectrum_path),
                    x=MARGIN_LEFT,
                    w=CONTENT_WIDTH,
                )

            else:
                pdf.set_font(
                    "helvetica",
                    "I",
                    9,
                )

                pdf.cell(
                    0,
                    8,
                    (
                        "Data response spectrum "
                        "tidak tersedia."
                    ),
                    ln=1,
                    align="C",
                )

        except Exception:
            logger.exception(
                (
                    "Gagal membuat response spectrum "
                    "untuk stasiun %s.",
                    station_code,
                )
            )

            pdf.set_font(
                "helvetica",
                "I",
                9,
            )

            pdf.cell(
                0,
                8,
                (
                    "Response spectrum gagal "
                    "dirender."
                ),
                ln=1,
                align="C",
            )

        # ---------------------------------------------------------------------
        # SECTION 4 - HUSID AND FOURIER AMPLITUDE SPECTRUM
        # ---------------------------------------------------------------------
        pdf.add_page()
        _pdf_section_title(pdf, "4", "HUSID PLOT DAN FOURIER AMPLITUDE SPECTRUM")
        husid_path = temp_dir / "husid.png"
        fas_path = temp_dir / "fas.png"
        figure, axes = plt.subplots(2, 1, figsize=(9, 8))
        for channel, context in sorted(contexts.items()):
            husid = getattr(context.cache, "husid_curve", None)
            if husid is not None:
                time = np.arange(len(husid)) / context.sampling_rate
                axes[0].plot(time, np.asarray(husid) * 100.0, label=channel)
            acceleration = context.acceleration
            if acceleration is not None:
                data = np.asarray(acceleration.data)
                freq = np.fft.rfftfreq(data.size, d=1.0 / context.sampling_rate)
                amplitude = np.abs(np.fft.rfft(data)) / data.size
                axes[1].loglog(freq[1:], amplitude[1:], label=channel)
        axes[0].set(title="Husid energy curve", xlabel="Time (s)", ylabel="Cumulative Arias energy (%)")
        axes[1].set(title="Fourier amplitude spectrum", xlabel="Frequency (Hz)", ylabel="Amplitude")
        for axis in axes:
            axis.grid(True, which="both", alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(husid_path, dpi=160)
        plt.close(figure)
        pdf.image(str(husid_path), x=MARGIN_LEFT, w=CONTENT_WIDTH)

        # ---------------------------------------------------------------------
        # SECTION 5 - WAVEFORM PER CHANNEL
        # ---------------------------------------------------------------------

        for channel in sorted(
            contexts.keys()
        ):
            pdf.add_page()

            _pdf_section_title(
                pdf,
                "5",
                f"WAVEFORM ANALYSIS - {channel}",
            )

            context = contexts[channel]

            metrics = _extract_metrics(
                context
            )

            pdf.set_font(
                "helvetica",
                "",
                8,
            )

            pdf.cell(
                0,
                5,
                (
                    f"PGA: "
                    f"{metrics['PGA'] * 100.0:.3f} Gal | "
                    f"PGV: "
                    f"{metrics['PGV'] * 100.0:.4f} cm/s | "
                    f"PGD: "
                    f"{metrics['PGD'] * 100.0:.4f} cm | "
                    f"D5-95: "
                    f"{metrics['Significant_Duration_D5_95']:.2f} s"
                ),
                ln=1,
            )

            waveform_path = (
                temp_dir
                / (
                    f"waveform_"
                    f"{channel}.png"
                )
            )

            try:
                waveform_available = (
                    generate_4panel_waveform(
                        context=context,
                        channel=channel,
                        output_path=waveform_path,
                        station_code=station_code,
                    )
                )

                if waveform_available:
                    pdf.image(
                        str(waveform_path),
                        x=MARGIN_LEFT,
                        w=CONTENT_WIDTH,
                    )

                else:
                    pdf.set_font(
                        "helvetica",
                        "I",
                        9,
                    )

                    pdf.cell(
                        0,
                        8,
                        (
                            "Data waveform "
                            "tidak tersedia."
                        ),
                        ln=1,
                        align="C",
                    )

            except Exception:
                logger.exception(
                    (
                        "Gagal membuat waveform plot "
                        "%s/%s.",
                        station_code,
                        channel,
                    )
                )

                pdf.set_font(
                    "helvetica",
                    "I",
                    9,
                )

                pdf.cell(
                    0,
                    8,
                    (
                        "Waveform gagal "
                        "dirender."
                    ),
                    ln=1,
                    align="C",
                )

    # -------------------------------------------------------------------------
    # FINALIZE PDF
    # -------------------------------------------------------------------------

    try:
        pdf.output(
            str(pdf_path)
        )

    except Exception:
        logger.exception(
            "Gagal menyimpan PDF: %s",
            pdf_path,
        )
        raise

    logger.info(
        "Laporan PDF berhasil dibuat: %s",
        pdf_path,
    )

    return pdf_path


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "BSMAPDFReport",
    "export_station_report",
    "generate_4panel_waveform",
    "generate_spectrum_overlay",
    "get_sig_bmkg",
]


# =============================================================================
# DEVELOPMENT TEST
# =============================================================================

if __name__ == "__main__":
    logger.info(
        "BSMA PDF exporter module loaded successfully."
    )
