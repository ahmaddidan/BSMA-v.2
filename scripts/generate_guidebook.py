# -*- coding: utf-8 -*-
"""
BMKG Strong Motion Analyzer (BSMA v2.0.0)
Script: scripts/generate_guidebook.py

Automated Generator for Dual-Language User Guidebooks & Technical Reference Manuals:
- outputs/BSMA_User_Guidebook_ID.pdf (Bahasa Indonesia)
- outputs/BSMA_User_Guidebook_EN.pdf (English)
- outputs/BSMA_User_Guidebook.pdf    (Default Indonesian copy for backward compatibility)

Author: Ahmad Didane Setyawan Putra (NIM: 123120094)
Department of Geophysical Engineering, Faculty of Industrial Technology, Institut Teknologi Sumatera
Academic Internship: Sleman Geophysical Station Class I, BMKG D.I. Yogyakarta
Internship Period: July 20 - August 20, 2026
"""

from __future__ import annotations

import io
import os
import re
import shutil
import sys
import textwrap
from pathlib import Path

# Force UTF-8 stdout if needed
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy import signal
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = PROJECT_ROOT / "assets"

def _resolve_asset(name: str) -> Path:
    p = ASSETS_DIR / name
    if p.is_file():
        return p
    fallback = PROJECT_ROOT / name
    return fallback if fallback.is_file() else p

LOGO_BMKG_ICON_PATH = _resolve_asset("Logo_BMKG_Icon.png")
LOGO_ITERA_ICON_PATH = _resolve_asset("Logo_ITERA_Icon.png")
LOGO_JUDUL_PATH = _resolve_asset("Logo_Judul.png")
LOGO_ITERA_PATH = _resolve_asset("Logo_ITERA.png")

# Color Palette (RGB 0.0 - 1.0)
COLOR_NAVY = (0.0, 0.176, 0.384)          # #002D62 (BMKG Navy)
COLOR_SKY = (0.008, 0.518, 0.780)          # #0284C7 (Primary Sky Blue)
COLOR_DARK = (0.059, 0.090, 0.165)         # #0F172A (Dark Slate Text)
COLOR_SLATE = (0.278, 0.333, 0.412)        # #475569 (Secondary Text)
COLOR_MUTED = (0.580, 0.639, 0.722)        # #94A3B8 (Muted Gray)
COLOR_CARD_BG = (0.973, 0.980, 0.988)      # #F8FAFC (Card Background)
COLOR_CARD_BORDER = (0.886, 0.910, 0.941)  # #E2E8F0 (Card Border)
COLOR_WHITE = (1.0, 1.0, 1.0)
COLOR_GREEN = (0.020, 0.588, 0.412)        # #059669 (Success)
COLOR_AMBER = (0.851, 0.467, 0.024)        # #D97706 (Warning)
COLOR_RED = (0.863, 0.149, 0.149)          # #DC2626 (Error)

PAGE_W = 595.3  # A4 width in pt
PAGE_H = 841.9  # A4 height in pt
LEFT_X = 50.0
RIGHT_X = 545.3
CONTENT_W = RIGHT_X - LEFT_X

FONT_ARIAL = "C:/Windows/Fonts/arial.ttf"
FONT_ARIAL_BD = "C:/Windows/Fonts/arialbd.ttf"
FONT_ARIAL_IT = "C:/Windows/Fonts/ariali.ttf"
FONT_ARIAL_BI = "C:/Windows/Fonts/arialbi.ttf"


# =============================================================================
# SCIENTIFIC LITERATURE FIGURE GENERATORS
# =============================================================================

def _fig_to_rgb_png(fig, pad=0.2, pad_inches=0.03, facecolor=None) -> bytes:
    """Render matplotlib figure to pure 24-bit RGB PNG (no alpha/SMask) for universal PDF viewer compatibility."""
    from PIL import Image
    if facecolor is None:
        facecolor = fig.get_facecolor()
    buf_raw = io.BytesIO()
    if pad is not None:
        plt.tight_layout(pad=pad)
    fig.savefig(buf_raw, format="png", bbox_inches="tight", pad_inches=pad_inches, facecolor=facecolor, edgecolor="none")
    plt.close(fig)
    buf_raw.seek(0)
    with Image.open(buf_raw) as im:
        rgb = im.convert("RGB")
        buf_out = io.BytesIO()
        rgb.save(buf_out, format="PNG", optimize=True)
        return buf_out.getvalue()


def create_pipeline_diagram(lang: str = "id") -> bytes:
    """Create 5-stage sequential processing pipeline diagram."""
    fig, ax = plt.subplots(figsize=(8.5, 2.2), dpi=220)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    stages_id = [
        ("1. INGESTI", "MiniSEED / SAC\n+ StationXML PAZ", "#0284C7"),
        ("2. PREPROCESS", "Detrend, Taper 5%\nButterworth Orde-4", "#0EA5E9"),
        ("3. INTEGRASI", "Acc -> Vel -> Disp\nKebijakan Baseline", "#10B981"),
        ("4. ANALISIS", "PGA, PGV, Arias,\nSpektrum Respons", "#F59E0B"),
        ("5. PELAPORAN", "PDF, CSV, ZIP Paket\nInst. MMI ShakeMap", "#6366F1"),
    ]
    stages_en = [
        ("1. INGESTION", "MiniSEED / SAC\n+ StationXML PAZ", "#0284C7"),
        ("2. PREPROCESS", "Detrend, Taper 5%\nButterworth 4-Pole", "#0EA5E9"),
        ("3. INTEGRATION", "Acc -> Vel -> Disp\nBaseline Policy", "#10B981"),
        ("4. ANALYSIS", "PGA, PGV, Arias,\nResponse Spectra", "#F59E0B"),
        ("5. REPORTING", "PDF, CSV, ZIP Paket\nInst. MMI ShakeMap", "#6366F1"),
    ]
    stages = stages_id if lang == "id" else stages_en

    for i, (title, desc, col) in enumerate(stages):
        x = 0.3 + i * 1.95
        y = 0.4
        w = 1.65
        h = 2.4
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.12",
            facecolor="white", edgecolor=col, linewidth=1.8,
        )
        ax.add_patch(rect)
        header_rect = patches.FancyBboxPatch(
            (x, y + 1.7), w, 0.7, boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor=col, edgecolor=col,
        )
        ax.add_patch(header_rect)
        ax.text(x + w / 2, y + 2.05, title, color="white", weight="bold", fontsize=8.0, ha="center", va="center")
        ax.text(x + w / 2, y + 0.85, desc, color="#334155", fontsize=7.0, ha="center", va="center", linespacing=1.35)

        if i < 4:
            ax.annotate(
                "", xy=(x + w + 0.28, y + 1.2), xytext=(x + w + 0.02, y + 1.2),
                arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=1.8, mutation_scale=12)
            )

    return _fig_to_rgb_png(fig, pad=0.2)


def create_filter_diagram(lang: str = "id") -> bytes:
    """Create Butterworth 4th-order Bandpass response curve."""
    sos = signal.butter(4, [0.10, 25.0], btype="bandpass", fs=100.0, output="sos")
    w, h = signal.sosfreqz(sos, worN=1024, fs=100.0)

    fig, ax = plt.subplots(figsize=(4.2, 2.1), dpi=220)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")

    mag_db = 20 * np.log10(np.maximum(np.abs(h)**2, 1e-6))
    ax.semilogx(w, mag_db, color="#0284C7", lw=2.0, label=r"Zero-Phase $|H(f)|^2$ (sosfiltfilt)")

    ax.axvline(0.10, color="#D97706", linestyle="--", lw=1.2, label=r"$f_{\mathrm{min}} = 0.10$ Hz")
    ax.axvline(25.0, color="#DC2626", linestyle="--", lw=1.2, label=r"$f_{\mathrm{max}} = 25.0$ Hz")
    ax.axvline(40.0, color="#64748B", linestyle=":", lw=1.0, label=r"Nyquist Safeguard $0.40 f_s$")

    lbl_x = "Frekuensi / Frequency (Hz)" if lang == "id" else "Frequency (Hz)"
    lbl_y = "Magnitudo / Magnitude (dB)" if lang == "id" else "Magnitude (dB)"
    ax.set_xlabel(lbl_x, fontsize=6.8, color="#334155")
    ax.set_ylabel(lbl_y, fontsize=6.8, color="#334155")
    ax.set_ylim(-65, 5)
    ax.set_xlim(0.01, 50)
    ax.tick_params(labelsize=6.2)
    ax.grid(True, which="both", linestyle=":", color="#E2E8F0", linewidth=0.7)
    ax.legend(loc="lower left", fontsize=5.6, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    return _fig_to_rgb_png(fig, pad=0.25)


def create_boore_baseline_diagram(lang: str = "id") -> bytes:
    """Create Baseline Correction & Kinematic Integration comparison diagram."""
    t = np.linspace(0, 30, 600)
    acc = np.exp(-0.18 * t) * np.sin(2 * np.pi * 1.6 * t) * (1 - np.exp(-0.8 * t))
    drift = 0.018  # DC baseline offset
    acc_raw = acc + drift

    dt = t[1] - t[0]
    vel_raw = np.cumsum(acc_raw) * dt
    disp_raw = np.cumsum(vel_raw) * dt

    vel_corr = np.cumsum(acc) * dt
    disp_corr = np.cumsum(vel_corr) * dt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4.2, 2.3), sharex=True, dpi=220)
    fig.patch.set_facecolor("#F8FAFC")
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle=":", color="#E2E8F0", linewidth=0.7)
        ax.tick_params(labelsize=5.8)

    ax1.plot(t, acc, color="#002D62", lw=1.2)
    ax1.set_ylabel(r"$a(t)$ [$\mathrm{m/s^2}$]", fontsize=6.2, color="#334155")

    ax2.plot(t, vel_raw * 100, color="#DC2626", lw=1.0, linestyle="--", label="Tanpa Koreksi" if lang == "id" else "Uncorrected")
    ax2.plot(t, vel_corr * 100, color="#0284C7", lw=1.2, label="Terkoreksi (Corrected)")
    ax2.set_ylabel(r"$v(t)$ [cm/s]", fontsize=6.2, color="#334155")
    ax2.legend(loc="upper right", fontsize=5.0, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    ax3.plot(t, disp_raw * 100, color="#DC2626", lw=1.0, linestyle="--")
    ax3.plot(t, disp_corr * 100, color="#10B981", lw=1.2)
    ax3.set_ylabel(r"$d(t)$ [cm]", fontsize=6.2, color="#334155")
    ax3.set_xlabel("Waktu / Time (s)" if lang == "id" else "Time (s)", fontsize=6.2, color="#334155")

    return _fig_to_rgb_png(fig, pad=0.2)


def create_energy_diagram(lang: str = "id") -> bytes:
    """Create Arias Intensity & Husid Curve diagram showing D5-95 significant duration."""
    t = np.linspace(0, 40, 500)
    sig_acc = np.exp(-0.15 * np.maximum(0, t - 5)) * np.sin(2 * np.pi * 1.8 * t) * (1 - 1 / (1 + np.exp(2 * (t - 6))))
    dt = t[1] - t[0]
    e_cum = np.cumsum(sig_acc**2) * dt
    e_norm = (e_cum / e_cum[-1]) * 100.0

    idx_5 = int(np.where(e_norm >= 5.0)[0][0])
    idx_95 = int(np.where(e_norm >= 95.0)[0][0])
    t_5 = t[idx_5]
    t_95 = t[idx_95]
    d_5_95 = t_95 - t_5

    fig, ax = plt.subplots(figsize=(4.2, 2.1), dpi=220)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")

    ax.plot(t, e_norm, color="#0284C7", lw=2.0, label=r"Akumulasi $I_a(t)$" if lang == "id" else r"Accumulated $I_a(t)$")
    ax.axhline(5.0, color="#D97706", linestyle=":", lw=1.0, label=r"5% $I_a$ ($t_{5\%}$)")
    ax.axhline(95.0, color="#DC2626", linestyle=":", lw=1.0, label=r"95% $I_a$ ($t_{95\%}$)")

    lbl_dur = f"$D_{{5-95}} = {d_5_95:.1f}$ s"
    ax.axvspan(t_5, t_95, color="#0EA5E9", alpha=0.15, label=lbl_dur)

    lbl_x = "Waktu / Time (s)" if lang == "id" else "Time (s)"
    lbl_y = "Akumulasi Energi (%)" if lang == "id" else "Energy Accumulation (%)"
    ax.set_xlabel(lbl_x, fontsize=6.6, color="#334155")
    ax.set_ylabel(lbl_y, fontsize=6.6, color="#334155")
    ax.set_ylim(-2, 105)
    ax.set_xlim(0, 42)
    ax.tick_params(labelsize=6.0)
    ax.grid(True, linestyle=":", color="#E2E8F0", linewidth=0.7)
    ax.legend(loc="lower right", fontsize=5.2, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    return _fig_to_rgb_png(fig, pad=0.25)


def create_gmice_diagram(lang: str = "id") -> bytes:
    """Create Worden et al. (2012) GMICE regression curves (MMI vs PGA & MMI vs PGV)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.3, 2.1), dpi=220)
    fig.patch.set_facecolor("#F8FAFC")

    pga_gal = np.logspace(-1, 3.2, 200)
    log_pga = np.log10(pga_gal)
    mmi_pga = np.where(log_pga <= 1.57, 1.78 + 1.55 * log_pga, -1.60 + 3.70 * log_pga)
    mmi_pga = np.clip(mmi_pga, 1.0, 10.0)

    ax1.set_facecolor("#FFFFFF")
    ax1.semilogx(pga_gal, mmi_pga, color="#002D62", lw=2.0, label="GMICE (PGA)")
    ax1.axvline(10**1.57, color="#D97706", linestyle=":", lw=1.0, label=r"$PGA_{\mathrm{trans}} \approx 37$ Gal")
    ax1.set_xlabel("PGA (Gal)", fontsize=6.6, color="#334155")
    ax1.set_ylabel("MMI Instrumental", fontsize=6.6, color="#334155")
    ax1.set_ylim(1, 10)
    ax1.set_xlim(0.1, 1500)
    ax1.tick_params(labelsize=5.8)
    ax1.grid(True, which="both", linestyle=":", color="#E2E8F0", linewidth=0.6)
    ax1.legend(loc="upper left", fontsize=5.2, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    pgv_cms = np.logspace(-2, 2.3, 200)
    log_pgv = np.log10(pgv_cms)
    mmi_pgv = np.where(log_pgv <= 0.53, 3.78 + 2.99 * log_pgv, 2.40 + 4.96 * log_pgv)
    mmi_pgv = np.clip(mmi_pgv, 1.0, 10.0)

    ax2.set_facecolor("#FFFFFF")
    ax2.semilogx(pgv_cms, mmi_pgv, color="#0284C7", lw=2.0, label="GMICE (PGV)")
    ax2.axvline(10**0.53, color="#DC2626", linestyle=":", lw=1.0, label=r"$PGV_{\mathrm{trans}} \approx 3.4$ cm/s")
    ax2.set_xlabel("PGV (cm/s)", fontsize=6.6, color="#334155")
    ax2.set_ylim(1, 10)
    ax2.set_xlim(0.01, 200)
    ax2.tick_params(labelsize=5.8)
    ax2.grid(True, which="both", linestyle=":", color="#E2E8F0", linewidth=0.6)
    ax2.legend(loc="upper left", fontsize=5.2, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    return _fig_to_rgb_png(fig, pad=0.25)


def create_spectrum_diagram(lang: str = "id") -> bytes:
    """Create response spectrum plot comparing Nigam-Jennings, Newmark-Beta, and SNI 1726:2019."""
    fig, ax = plt.subplots(figsize=(4.2, 2.1), dpi=220)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")

    T = np.logspace(-2, 1, 150)
    pga = 0.25
    sa_nj = pga * (1 + 1.5 * np.exp(-((np.log10(T) - np.log10(0.2))**2) / 0.35))
    sa_nb = sa_nj * (1 + 0.015 * np.sin(5 * np.log10(T)))

    sds = 0.65
    sd1 = 0.35
    t0 = 0.2 * (sd1 / sds)
    ts = sd1 / sds
    sni = np.zeros_like(T)
    for i, t_val in enumerate(T):
        if t_val < t0:
            sni[i] = sds * (0.4 + 0.6 * t_val / t0)
        elif t_val <= ts:
            sni[i] = sds
        else:
            sni[i] = sd1 / t_val

    ax.semilogx(T, sa_nj, color="#002D62", lw=2.0, label="Nigam-Jennings (1969)")
    ax.semilogx(T, sa_nb, color="#0284C7", lw=1.5, linestyle="--", label="Newmark-Beta (1959)")
    ax.semilogx(T, sni, color="#DC2626", lw=1.6, linestyle=":", label="SNI 1726:2019 Desain")

    lbl_x = "Periode Alami T (s)" if lang == "id" else "Natural Period T (s)"
    lbl_y = "Pseudo-Percepatan PSA (g)" if lang == "id" else "Pseudo-Acceleration PSA (g)"
    ax.set_xlabel(lbl_x, fontsize=6.8, color="#334155")
    ax.set_ylabel(lbl_y, fontsize=6.8, color="#334155")
    ax.set_xlim(0.01, 10.0)
    ax.tick_params(labelsize=6.2)
    ax.grid(color="#E2E8F0", linestyle=":", linewidth=0.7, which="both")
    ax.legend(loc="upper right", fontsize=5.6, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    return _fig_to_rgb_png(fig, pad=0.25)


# =============================================================================
# GUIDEBOOK BUILDER HELPER CLASS
# =============================================================================

class GuidebookBuilder:
    def __init__(self, doc: pymupdf.Document, lang: str = "id"):
        self.doc = doc
        self.lang = lang

    def init_page_fonts(self, page: pymupdf.Page) -> None:
        """Register fonts on page."""
        if os.path.exists(FONT_ARIAL) and os.path.exists(FONT_ARIAL_BD):
            page.insert_font(fontname="f_reg", fontfile=FONT_ARIAL)
            page.insert_font(fontname="f_bold", fontfile=FONT_ARIAL_BD)
            if os.path.exists(FONT_ARIAL_IT):
                page.insert_font(fontname="f_it", fontfile=FONT_ARIAL_IT)
            else:
                page.insert_font(fontname="f_it", fontfile=FONT_ARIAL)
            if os.path.exists(FONT_ARIAL_BI):
                page.insert_font(fontname="f_bi", fontfile=FONT_ARIAL_BI)
            else:
                page.insert_font(fontname="f_bi", fontfile=FONT_ARIAL_BD)
        else:
            # Fallback to standard built-in font references
            pass

    def insert_html_safe(self, page: pymupdf.Page, rect: pymupdf.Rect, html_body: str, font_size: str = "7.6pt", line_height: str = "1.32") -> None:
        """Render formatted HTML box with CSS styling, justification, links, and italic support."""
        # Clean entities that cause glyph failures in PyMuPDF HTML engine
        replacements = {
            "&sup2;": "²",
            "&frac12;": "1/2",
            "&omega;": "ω",
            "&le;": "<=",
            "&ge;": ">=",
            "&middot;": "·",
            "&xi;": "ξ",
            "&uuml;": "ü",
            "&gamma;": "γ",
            "&beta;": "β",
            "&times;": "×",
            "&bull;": "•",
        }
        for k, v in replacements.items():
            html_body = html_body.replace(k, v)

        css = f"""
        body {{
            font-family: 'Arial', sans-serif;
            color: #0f172a;
            margin: 0;
            padding: 0;
        }}
        p {{
            text-align: justify;
            text-justify: inter-word;
            line-height: {line_height};
            margin: 0 0 5px 0;
            font-size: {font_size};
        }}
        i, em {{
            font-style: italic;
        }}
        b, strong {{
            font-weight: bold;
        }}
        a {{
            color: #0284c7;
            text-decoration: underline;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 7.0pt;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 3px 5px;
        }}
        th {{
            background-color: #002d62;
            color: #ffffff;
            font-weight: bold;
            text-align: left;
        }}
        code {{
            background-color: #f1f5f9;
            color: #0f172a;
            font-family: monospace;
            padding: 1px 3px;
        }}
        """
        full_html = f"<html><head><style>{css}</style></head><body>{html_body}</body></html>"
        page.insert_htmlbox(rect, full_html)

    def add_page_header_footer(self, page: pymupdf.Page, section_badge: str, page_num_str: str) -> None:
        """Draw clean running header and academic footer on content pages."""
        header_title = "BMKG STRONG MOTION ANALYZER (BSMA v2.0.0)"
        page.insert_text((LEFT_X, 23), header_title,
                         fontsize=7.4, fontname="f_bold", color=COLOR_NAVY)
        font_bold_obj = pymupdf.Font(fontfile=FONT_ARIAL_BD)
        badge_w = font_bold_obj.text_length(section_badge.upper(), fontsize=7.2)
        page.insert_text((RIGHT_X - badge_w, 23), section_badge.upper(),
                         fontsize=7.2, fontname="f_bold", color=COLOR_SLATE)

        # Bottom Footer
        page.draw_line(pymupdf.Point(LEFT_X, 810), pymupdf.Point(RIGHT_X, 810), color=COLOR_CARD_BORDER, width=0.8)
        footer_author = (
            "Ahmad Didane Setyawan Putra - Teknik Geofisika, Institut Teknologi Sumatera"
            if self.lang == "id" else
            "Ahmad Didane Setyawan Putra - Geophysical Engineering, Institut Teknologi Sumatera"
        )
        page.insert_text((LEFT_X, 822), footer_author,
                         fontsize=7.2, fontname="f_it", color=COLOR_SLATE)
        page.insert_text((RIGHT_X - 105, 822), page_num_str,
                         fontsize=7.2, fontname="f_bold", color=COLOR_NAVY)

    def draw_card(self, page: pymupdf.Page, rect: pymupdf.Rect, bg_col=COLOR_CARD_BG, border_col=COLOR_CARD_BORDER, border_w=1.0) -> None:
        """Draw solid card box."""
        page.draw_rect(rect, color=border_col, fill=bg_col, width=border_w)

    def draw_chapter_banner(self, page: pymupdf.Page, title_str: str, y_top: float = 44.0) -> float:
        """Draw clean chapter header banner with accent bar."""
        page.draw_rect(pymupdf.Rect(LEFT_X, y_top, LEFT_X + 6, y_top + 20), color=COLOR_NAVY, fill=COLOR_NAVY)
        page.insert_text((LEFT_X + 14, y_top + 15), title_str, fontsize=11.2, fontname="f_bold", color=COLOR_NAVY)
        page.draw_line(pymupdf.Point(LEFT_X, y_top + 26), pymupdf.Point(RIGHT_X, y_top + 26), color=COLOR_CARD_BORDER, width=0.8)
        return y_top + 32.0

    def draw_callout(self, page: pymupdf.Page, rect: pymupdf.Rect, title: str, text: str, callout_type="info") -> None:
        """Draw highlighted callout box with colored left strip."""
        if callout_type == "info":
            bg = (0.941, 0.973, 1.0)
            border = (0.729, 0.855, 0.988)
            accent = COLOR_SKY
            badge = "[CATATAN TEKNIS]" if self.lang == "id" else "[TECHNICAL NOTE]"
        elif callout_type == "warning":
            bg = (1.0, 0.984, 0.922)
            border = (0.992, 0.886, 0.655)
            accent = COLOR_AMBER
            badge = "[PERHATIAN METODOLOGI]" if self.lang == "id" else "[METHODOLOGY CAUTION]"
        else:
            bg = (0.941, 0.988, 0.961)
            border = (0.655, 0.922, 0.749)
            accent = COLOR_GREEN
            badge = "[STANDAR ACUAN]" if self.lang == "id" else "[REFERENCE STANDARD]"

        self.draw_card(page, rect, bg_col=bg, border_col=border)
        page.draw_rect(pymupdf.Rect(rect.x0, rect.y0, rect.x0 + 4, rect.y1), color=accent, fill=accent)
        page.insert_text((rect.x0 + 10, rect.y0 + 12), f"{badge} {title}", fontsize=7.6, fontname="f_bold", color=accent)

        html = f"""
        <p style="font-size: 7.2pt; line-height: 1.25; margin: 0; text-align: justify; text-justify: inter-word; color: #0f172a;">
            {text}
        </p>
        """
        self.insert_html_safe(page, pymupdf.Rect(rect.x0 + 10, rect.y0 + 16, rect.x1 - 8, rect.y1 - 4), html)

    def draw_formula_card(self, page: pymupdf.Page, rect: pymupdf.Rect, formula_latex: str, title_str: str = "",
                          fontsize: float = 14.5, text_color: str = "#002D62") -> None:
        """Render publication-grade mathematical equation card via Matplotlib mathtext at 300 DPI."""
        self.draw_card(page, rect, bg_col=(0.955, 0.975, 1.0), border_col=(0.75, 0.85, 0.95), border_w=0.9)
        page.draw_rect(pymupdf.Rect(rect.x0, rect.y0, rect.x0 + 3.5, rect.y1), color=COLOR_SKY, fill=COLOR_SKY)

        y_math_top = rect.y0 + 6
        if title_str:
            page.insert_text((rect.x0 + 10, rect.y0 + 11), title_str.upper(), fontsize=7.2, fontname="f_bold", color=COLOR_SKY)
            y_math_top = rect.y0 + 16

        lines = [l.strip() for l in formula_latex.strip().split("\n") if l.strip()]
        n = len(lines)
        fig_h = max(0.55, 0.40 * n)
        card_bg_hex = "#F4F8FC"
        with plt.rc_context({"mathtext.fontset": "cm", "font.family": "serif"}):
            fig, ax = plt.subplots(figsize=(8.0, fig_h), dpi=300)
            ax.axis("off")
            fig.patch.set_facecolor(card_bg_hex)
            ax.set_facecolor(card_bg_hex)
            for idx, l in enumerate(lines):
                s = l if l.startswith("$") else f"${l}$"
                y_pos = 0.5 if n == 1 else (1.0 - (idx + 0.5) / n)
                ax.text(0.5, y_pos, s, ha="center", va="center", fontsize=fontsize, color=text_color)

            img_bytes = _fig_to_rgb_png(fig, pad=None, pad_inches=0.03, facecolor=card_bg_hex)

        page.insert_image(pymupdf.Rect(rect.x0 + 8, y_math_top, rect.x1 - 8, rect.y1 - 4), stream=img_bytes, keep_proportion=True)

    def draw_academic_bibliography(self, page: pymupdf.Page, rect: pymupdf.Rect, refs: list[tuple[str, str, str]]) -> None:
        """Render formal academic bibliography with professional hanging indent (APA 7th standard)."""
        css = """
        body {
            font-family: 'Arial', sans-serif;
            color: #0f172a;
            margin: 0;
            padding: 0;
        }
        p.ref-item {
            text-indent: -22px;
            padding-left: 22px;
            margin: 0 0 10.5px 0;
            line-height: 1.34;
            font-size: 7.5pt;
            text-align: justify;
            text-justify: inter-word;
            color: #0f172a;
        }
        i, em {
            font-style: italic;
        }
        b, strong {
            font-weight: bold;
            color: #002d62;
        }
        a {
            color: #0284c7;
            text-decoration: underline;
        }
        """
        items_html = []
        for authors_year, title_source, doi in refs:
            doi_link = f' <a href="{doi}">{doi}</a>' if doi else ""
            items_html.append(f'<p class="ref-item"><b>{authors_year}</b> {title_source}{doi_link}</p>')

        full_html = f"<html><head><style>{css}</style></head><body>{''.join(items_html)}</body></html>"
        page.insert_htmlbox(rect, full_html)

    def post_process_pdf_clean_unicode(self) -> None:
        """No-op: disabled to avoid corrupting PyMuPDF ToUnicode font streams."""
        pass

# =============================================================================
# MAIN GUIDEBOOK COMPILER FUNCTION
# =============================================================================

def build_guidebook(lang: str = "id") -> Path:
    """Build complete publication-grade User Guidebook in specified language."""
    print(f"Building BSMA User Guidebook [Language: {lang.upper()}]...")

    doc = pymupdf.open()
    builder = GuidebookBuilder(doc, lang=lang)
    TOTAL_BODY_PAGES = 13

    # -------------------------------------------------------------
    # PAGE 1 (Cover / Title Page - Roman i, suppressed)
    # -------------------------------------------------------------
    p1 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p1)

    # Executive Cover Background
    p1.draw_rect(pymupdf.Rect(0, 0, PAGE_W, 360), color=COLOR_NAVY, fill=COLOR_NAVY)
    p1.draw_rect(pymupdf.Rect(0, 355, PAGE_W, 360), color=COLOR_SKY, fill=COLOR_SKY)

    # Top Dual Logos Lockup (Directly on Navy)
    font_bold_obj = pymupdf.Font(fontfile=FONT_ARIAL_BD)
    bmkg_icon = LOGO_BMKG_ICON_PATH if LOGO_BMKG_ICON_PATH.is_file() else LOGO_JUDUL_PATH
    if bmkg_icon.is_file():
        try:
            from PIL import Image
            with Image.open(bmkg_icon) as pil_img:
                bg = Image.new("RGB", pil_img.size, (0, 45, 98))
                if pil_img.mode == "RGBA":
                    bg.paste(pil_img, mask=pil_img.split()[3])
                else:
                    bg.paste(pil_img)
                buf = io.BytesIO()
                bg.save(buf, format="PNG")
                p1.insert_image(pymupdf.Rect(50, 48, 114, 112), stream=buf.getvalue())
        except Exception:
            p1.insert_image(pymupdf.Rect(50, 48, 114, 112), filename=str(bmkg_icon))
    bmkg_tw = font_bold_obj.text_length("BMKG", fontsize=11.0)
    p1.insert_text((82 - bmkg_tw / 2, 128), "BMKG", fontsize=11.0, fontname="f_bold", color=COLOR_WHITE)

    p1.draw_line(pymupdf.Point(130, 48), pymupdf.Point(130, 130), color=(0.35, 0.60, 0.85), width=1.2)

    itera_icon = LOGO_ITERA_ICON_PATH if LOGO_ITERA_ICON_PATH.is_file() else LOGO_ITERA_PATH
    if itera_icon.is_file():
        try:
            from PIL import Image
            with Image.open(itera_icon) as pil_img:
                bg = Image.new("RGB", pil_img.size, (0, 45, 98))
                if pil_img.mode == "RGBA":
                    bg.paste(pil_img, mask=pil_img.split()[3])
                else:
                    bg.paste(pil_img)
                buf = io.BytesIO()
                bg.save(buf, format="PNG")
                p1.insert_image(pymupdf.Rect(146, 48, 210, 112), stream=buf.getvalue())
        except Exception:
            p1.insert_image(pymupdf.Rect(146, 48, 210, 112), filename=str(itera_icon))
    itera_tw = font_bold_obj.text_length("ITERA", fontsize=11.0)
    p1.insert_text((178 - itera_tw / 2, 128), "ITERA", fontsize=11.0, fontname="f_bold", color=COLOR_WHITE)

    # Institutional Running Title beside Logos
    text_header_1 = "BUKU PANDUAN PENGGUNA & REFERENSI TEKNIS SOFTWARE" if lang == "id" else "USER GUIDEBOOK & SOFTWARE TECHNICAL REFERENCE MANUAL"
    text_header_2 = "Proyek Kerja Praktik Mahasiswa Program Studi Teknik Geofisika" if lang == "id" else "Undergraduate Internship Project of Geophysical Engineering Department"
    text_header_3 = "Fakultas Teknologi Industri, Institut Teknologi Sumatera - BMKG Stasiun Geofisika Sleman" if lang == "id" else "Faculty of Industrial Technology, Institut Teknologi Sumatera - Sleman Geophysical Station BMKG"

    p1.insert_text((230, 68), text_header_1, fontsize=9.2, fontname="f_bold", color=COLOR_WHITE)
    p1.insert_text((230, 85), text_header_2, fontsize=8.2, fontname="f_reg", color=(0.85, 0.92, 1.0))
    p1.insert_text((230, 100), text_header_3, fontsize=7.4, fontname="f_it", color=(0.75, 0.85, 0.95))

    # Main Title Block
    doc_type = "DOKUMEN PANDUAN PENGGUNA & REFERENSI TEKNIS" if lang == "id" else "USER GUIDEBOOK & ENGINEERING TECHNICAL REFERENCE MANUAL"
    p1.insert_text((50, 180), doc_type, fontsize=9.5, fontname="f_bold", color=(0.75, 0.90, 1.0))
    p1.insert_text((50, 216), "BMKG Strong Motion Analyzer", fontsize=24.0, fontname="f_bold", color=COLOR_WHITE)
    p1.insert_text((50, 246), "(BSMA v2.0.0)", fontsize=18.0, fontname="f_bold", color=COLOR_SKY)

    dev_loc = "Dikembangkan di BMKG Stasiun Geofisika Sleman" if lang == "id" else "Developed at BMKG Sleman Geophysical Station"
    p1.insert_text((50, 276), dev_loc, fontsize=12.0, fontname="f_bold", color=(0.925, 0.706, 0.078))

    sub_title = (
        "Platform Komputasi Terpadu Sinyal Akselerograf, Kinematika Seismik, & Spektrum Respons"
        if lang == "id" else
        "Integrated Computational Platform for Accelerograph Signal Processing, Seismic Kinematics, & Response Spectra"
    )
    p1.insert_text((50, 298), sub_title, fontsize=9.0, fontname="f_reg", color=(0.90, 0.95, 1.0))

    # Institutional Independence Disclaimer (P0 Academic Boundary)
    inst_disclaimer = (
        "Proyek Perangkat Lunak Akademik Mandiri — Afiliasi kelembagaan tidak menyatakan bahwa BSMA merupakan perangkat lunak resmi BMKG."
        if lang == "id" else
        "Independent Academic Software Project — Institutional affiliation does not imply official BMKG software endorsement."
    )
    p1.insert_text((50, 318), inst_disclaimer, fontsize=7.6, fontname="f_it", color=(0.82, 0.90, 1.0))

    # Lower Section: Metadata Box
    builder.draw_card(p1, pymupdf.Rect(50, 385, 545, 680), bg_col=COLOR_WHITE, border_col=COLOR_CARD_BORDER)

    meta_hdr = "INFORMASI DOKUMEN & IDENTITAS PENGEMBANGAN" if lang == "id" else "DOCUMENT & DEVELOPMENT METADATA"
    p1.insert_text((70, 412), meta_hdr, fontsize=10.5, fontname="f_bold", color=COLOR_NAVY)
    p1.draw_line(pymupdf.Point(70, 420), pymupdf.Point(525, 420), color=COLOR_SKY, width=1.2)

    fields_id = [
        ("Nama Perangkat Lunak", "BMKG Strong Motion Analyzer (BSMA)"),
        ("Versi Rilis / Build", "v2.0.0 (Build 2026.08) - Edisi Rilis Akademik (Diperbarui: Agustus 2026)"),
        ("Penyusun / Pengembang", "Ahmad Didane Setyawan Putra"),
        ("Nomor Induk Mahasiswa", "123120094"),
        ("Program Studi", "Teknik Geofisika"),
        ("Fakultas", "Fakultas Teknologi Industri"),
        ("Perguruan Tinggi", "Institut Teknologi Sumatera (ITERA)"),
        ("Instansi Mitra / Lokasi", "Stasiun Geofisika Kelas I Sleman, BMKG D.I. Yogyakarta"),
        ("Periode Kerja Praktik", "20 Juli 2026 s.d. 20 Agustus 2026"),
        ("Klasifikasi Proyek", "Karya Akademik Mandiri Mahasiswa (Non-Komersial)"),
        ("URL Platform Web", "https://strong-motion.streamlit.app/"),
        ("Repositori Kode Sumber", "https://github.com/ahmaddidan/BSMA-v.2"),
    ]
    fields_en = [
        ("Software System Name", "BMKG Strong Motion Analyzer (BSMA)"),
        ("Release Version / Build", "v2.0.0 (Build 2026.08) - Academic Release Edition (Updated: August 2026)"),
        ("Author / Developer", "Ahmad Didane Setyawan Putra"),
        ("Student ID (NIM)", "123120094"),
        ("Study Program", "Geophysical Engineering"),
        ("Faculty", "Faculty of Industrial Technology"),
        ("University", "Institut Teknologi Sumatera (ITERA)"),
        ("Host Institution", "Sleman Geophysical Station Class I, BMKG D.I. Yogyakarta"),
        ("Internship Period", "July 20, 2026 - August 20, 2026"),
        ("Project Classification", "Independent Student Academic Work (Non-Commercial)"),
        ("Web Platform URL", "https://strong-motion.streamlit.app/"),
        ("Source Code Repository", "https://github.com/ahmaddidan/BSMA-v.2"),
    ]
    fields = fields_id if lang == "id" else fields_en

    cur_y = 440
    for label, val in fields:
        p1.insert_text((70, cur_y), f"{label}", fontsize=8.0, fontname="f_bold", color=COLOR_SLATE)
        if "http" in val:
            p1.insert_text((220, cur_y), f":  {val}", fontsize=8.0, fontname="f_reg", color=COLOR_SKY)
            link_w = font_bold_obj.text_length(f":  {val}", fontsize=8.0)
            p1.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(220, cur_y - 8, 220 + link_w, cur_y + 2), "uri": val})
        else:
            p1.insert_text((220, cur_y), f":  {val}", fontsize=8.0, fontname="f_reg", color=COLOR_DARK)
        cur_y += 19

    # Bottom Notice Box
    callout_cover_text = (
        "Buku panduan ini menyajikan panduan operasional langkah-demi-langkah antarmuka grafis BSMA v2.0.0, konfigurasi parameter "
        "pemrosesan sinyal digital pada sidebar, navigasi tab analisis interaktif, mode batch multi-stasiun, serta implementasi metode komputasi "
        "yang dievaluasi melalui 83 pengujian perangkat lunak otomatis dan perbandingan benchmark numerik."
        if lang == "id" else
        "This manual provides step-by-step operational workflows for the BSMA v2.0.0 graphical interface, digital signal processing "
        "parameter configurations, interactive analysis tabs, multi-station batch workflows, and computational methods "
        "evaluated through 83 automated software tests and numerical benchmark comparisons."
    )
    builder.draw_callout(p1, pymupdf.Rect(50, 700, 545, 775),
                         "Panduan Operasional & Evaluasi Numerik" if lang == "id" else "Operational Guidelines & Numerical Evaluation",
                         callout_cover_text, callout_type="success")

    # -------------------------------------------------------------
    # PAGE 2 (Executive Summary - Roman ii)
    # -------------------------------------------------------------
    p2 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p2)
    builder.add_page_header_footer(p2, "RINGKASAN EKSEKUTIF" if lang == "id" else "EXECUTIVE SUMMARY", "Halaman ii dari iv" if lang == "id" else "Page ii of iv")

    y = builder.draw_chapter_banner(p2, "RINGKASAN EKSEKUTIF & LANDASAN PENGEMBANGAN" if lang == "id" else "EXECUTIVE SUMMARY & SCIENTIFIC BACKGROUND")

    exec_summary_html = (
        """
        <p><b>1. Latar Belakang Ilmiah & Urgensi Rekayasa Kegempaan</b></p>
        <p>Analisis data rekaman akselerograf gerakan tanah kuat (<i>strong ground motion accelerograph</i>) merupakan pilar fundamental dalam 
        seismologi rekayasa (<i>engineering seismology</i>) dan rekayasa kegempaan (<i>earthquake engineering</i>). Rekaman akselerogram 
        empiris menyediakan pengukuran fisik langsung mengenai karakteristik kinematika sumber, dinamika rekahan sesar aktif, pelemahan perambatan 
        gelombang seismik, serta fenomena amplifikasi tanah lokal (<i>site effects</i>) yang menjadi dasar penentuan beban gempa rencana untuk 
        keselamatan struktur bangunan bertingkat, bendungan, jembatan bentang panjang, dan infrastruktur strategis nasional.</p>
        <p>Namun, data percepatan mentah (<i>raw accelerograms</i>) yang tercatat oleh stasiun akselerometer lapangan tidak dapat langsung diintegralkan 
        ke kecepatan dan perpindahan. Rekaman mentah umumnya mengandung konvolusi respon frekuensi instrumen (<i>instrument transfer function</i>), 
        derau seismik lingkungan (<i>ambient cultural noise</i>), serta pergeseran garis dasar (<i>baseline drift</i>) akibat rotasi lokal pondasi sensor, 
        tilt dinamik, atau histeresis mekanis transduser. Pengintegrasian langsung data mentah tanpa koreksi garis dasar yang tepat akan memunculkan hanyutan 
        garis dasar semu yang beramplifikasi menjadi divergensi kuadratik pada riwayat perpindahan (<i>displacement</i>), sehingga menghasilkan 
        estimasi perpindahan puncak (<i>peak ground displacement</i> / PGD) yang sangat bias. Oleh karena itu, diperlukan standarisasi komputasi pengolahan 
        sinyal akselerograf yang teruji secara ilmiah, memiliki transparansi matematis, serta mudah dioperasikan oleh praktisi dan peneliti.</p>
        
        <p><b>2. Solusi Terpadu: BMKG Strong Motion Analyzer (BSMA v2.0.0)</b></p>
        <p>BMKG Strong Motion Analyzer (BSMA v2.0.0) hadir sebagai platform komputasi seismologi rekayasa modular berbasis web yang mengintegrasikan seluruh 
        rantai pengolahan data rekaman akselerograf secara deterministik dan transparan:</p>
        <p>&bull; <b>Ingesti Data Seismik Standar FDSN:</b> Mendukung format MiniSEED (.mseed, .miniseed, .msd) dan SAC (.sac) biner, 
        lengkap dengan pembacaan metadata respon instrumen StationXML (.xml) standar Federasi Jaringan Seismik Digital Internasional (FDSN).</p>
        <p>&bull; <b>Kendali Mutu Sinyal (<i>Quality Control</i> Otomatis):</b> Penapisan otomatis rasio sinyal terhadap derau (<i>pre-event</i> SNR) 
        berbasis Root Mean Square (RMS), audit 6 kriteria derau fisik (<i>spikes</i>, <i>dead channel/flatline</i>, <i>clipping</i>, integritas 
        <i>bandwidth</i>, ekses <i>noise floor</i>, dan <i>baseline offset</i>), serta pemberian skor kesehatan data objektif (0-100%). Ambang batas 
        dan skema skor QC merupakan kebijakan implementasi internal BSMA.</p>
        <p>&bull; <b>Pemrosesan Sinyal Digital (DSP) Rekayasa Gempa:</b> Filter IIR Butterworth orde-4 dua-arah tanpa pergeseran fase neto 
        (<i>zero-phase forward-backward</i> via <code>scipy.signal.sosfiltfilt</code>), koreksi garis dasar polinomial Boore (2001), 
        dan proteksi batas frekuensi Nyquist adaptif (f<sub>max</sub> &le; 0.40 f<sub>s</sub>).</p>
        <p>&bull; <b>Integrasi Kinematika Bertahap:</b> Skema integrasi numerik trapesium kumulatif yang dipadukan dengan penapisan baseline bertahap, 
        mengurangi hanyutan frekuensi rendah residual berdasarkan konfigurasi pra-pemrosesan yang dipilih.</p>
        <p>&bull; <b>Karakterisasi Energi Seismik & Intensitas MMI:</b> Evaluasi Intensitas Arias (I<sub>a</sub>), <i>Cumulative Absolute Velocity</i> (CAV), 
        durasi signifikan Trifunac-Brady (D<sub>5-95</sub> dan D<sub>5-75</sub>), rasio kinematika V<sub>max</sub>/A<sub>max</sub>, serta pemetaan 
        skala intensitas makroseismik instrumental <i>Modified Mercalli Intensity</i> (MMI) berbasis regresi GMICE Worden et al. (2012). 
        Nilai MMI instrumental merupakan estimasi guncangan fisik dan bukan pengganti survei makroseismik lapangan.</p>
        <p>&bull; <b>Spektrum Respons Struktur SDOF & Desain SNI 1726:2019:</b> Pemodelan osilator linear SDOF redaman 5% 
        menggunakan solver rekursif analitik eksak Nigam-Jennings (1969) dan integrasi implisit Newmark-Beta (1959), yang dibandingkan langsung 
        terhadap spektrum desain gempa SNI 1726:2019 yang dipilih pengguna untuk kelas situs lunak (SE), sedang (SD), dan keras (SC).</p>
        <p>&bull; <b>Mode Pemrosesan Batch Regional & Ekspor Deliverables:</b> Pemrosesan sekuensial direktori multi-stasiun regional dengan proteksi 
        isolasi kegagalan per stasiun (<i>fault isolation</i>), ekspor laporan teknik PDF 2-halaman berformat terstandar, CSV kinematika dan spektral, serta arsip ZIP terpadu.</p>
        """
        if lang == "id" else
        """
        <p><b>1. Scientific Background & Engineering Seismology Urgency</b></p>
        <p>Quantitative processing of strong ground motion accelerograms constitutes a foundational prerequisite in modern engineering seismology 
        and earthquake geotechnical engineering. Empirical ground-motion records provide direct physical measurements of earthquake rupture 
        kinematics, seismic wave attenuation through crustal structures, and local geotechnical site amplification (<i>site response</i>), 
        which govern seismic design loads for high-rise buildings, critical dams, bridges, and lifeline infrastructure.</p>
        <p>However, raw accelerograms recorded by digital field instruments cannot be directly integrated into velocity and displacement. 
        Raw data routinely carry instrument frequency response convolutions (transducer transfer functions), ambient cultural noise, and non-physical 
        baseline shifts induced by dynamic sensor tilting, permanent soil rotation, or analog-to-digital converter thermal drift. Directly integrating 
        uncorrected accelerograms produces severe artificial baseline drift that amplifies into spurious quadratic displacement drift, 
        yielding heavily biased peak ground displacement (PGD) estimates. Consequently, standardized, mathematically transparent digital signal 
        processing workflows are imperative for reliable engineering practice and disaster risk reduction.</p>
        
        <p><b>2. The Integrated Solution: BMKG Strong Motion Analyzer (BSMA v2.0.0)</b></p>
        <p>The BMKG Strong Motion Analyzer (BSMA v2.0.0) provides a modular, interactive web-based scientific computing platform unifying the full 
        deterministic strong-motion data reduction and structural modeling pipeline:</p>
        <p>&bull; <b>FDSN-Standard Data Ingestion:</b> Supports binary MiniSEED (.mseed, .miniseed, .msd) and SAC (.sac) streams, seamlessly 
        coupled with FDSN StationXML (.xml) response metadata containing stage gains and pole-zero transfer functions.</p>
        <p>&bull; <b>Automated Signal Quality Control (QC):</b> Pre-event signal-to-noise ratio (SNR) estimation using Root Mean Square (RMS) 
        statistics, automated multi-class physical anomaly detection (spikes, dead channels/flatlines, sensor clipping, bandwidth integrity, 
        excess noise floor, baseline DC shifts), and diagnostic health scoring (0-100%). QC thresholds reflect BSMA implementation policies.</p>
        <p>&bull; <b>Engineering Digital Signal Processing (DSP) Pipeline:</b> 4th-order zero-phase bidirectional Butterworth filtering 
        (via <code>scipy.signal.sosfiltfilt</code> in Second-Order Sections), low-order polynomial baseline detrending (Boore, 2001), 
        and adaptive Nyquist frequency floor safeguards (f<sub>max</sub> &le; 0.40 f<sub>s</sub>).</p>
        <p>&bull; <b>Progressive Kinematic Integration:</b> Cumulative trapezoidal integration coupled with progressive highpass baseline filtering, 
        reducing residual low-frequency drift under the selected preprocessing configuration.</p>
        <p>&bull; <b>Seismic Energy & Instrumental Intensity:</b> Quantitative determination of Arias Intensity (I<sub>a</sub>), Cumulative Absolute 
        Velocity (CAV), Trifunac-Brady bracketed durations (D<sub>5-95</sub> and D<sub>5-75</sub>), V<sub>max</sub>/A<sub>max</sub> ratios, 
        and instrumental Modified Mercalli Intensity (MMI) derived from empirical GMICE regressions (Worden et al., 2012). The reported MMI is 
        an instrumental ground-motion estimate and not a substitute for observed macroseismic surveys.</p>
        <p>&bull; <b>SDOF Response Spectra & Building Code Benchmarking:</b> 5% damped elastic single-degree-of-freedom response spectra computed via 
        exact piecewise linear Nigam-Jennings (1969) and implicit Newmark-Beta (1959) algorithms, compared with user-selected Indonesian Seismic 
        Code SNI 1726:2019 target design spectra for Soft (SE), Medium (SD), and Hard (SC) site classes.</p>
        <p>&bull; <b>Regional Multi-Station Batch Engine & Reporting:</b> Sequential batch processing across regional seismic networks featuring 
        per-station exception isolation, formatted 2-page PDF engineering reports (with vector layout and high-resolution plots), full CSV datasets, and compressed ZIP archives.</p>
        """
    )
    builder.insert_html_safe(p2, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 680), exec_summary_html)

    # -------------------------------------------------------------
    # PAGE 3 (System Specifications - Roman iii)
    # -------------------------------------------------------------
    p3 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p3)
    builder.add_page_header_footer(p3, "SPESIFIKASI SISTEM" if lang == "id" else "SYSTEM SPECIFICATIONS", "Halaman iii dari iv" if lang == "id" else "Page iii of iv")

    y = builder.draw_chapter_banner(p3, "SPESIFIKASI LINGKUNGAN KOMPUTASI & DEPENDENSI" if lang == "id" else "COMPUTATIONAL ENVIRONMENT & RUNTIME DEPENDENCIES")

    spec_intro_id = (
        "Tabel di bawah ini menguraikan spesifikasi teknis arsitektur perangkat lunak, dependensi pustaka Python, "
        "format pertukaran data seismik, serta konfigurasi perangkat keras minimum untuk mengoperasikan BSMA v2.0.0:"
    )
    spec_intro_en = (
        "The following table details the technical software architecture, Python runtime dependencies, "
        "seismic data exchange formats, and minimum hardware configurations required to operate BSMA v2.0.0:"
    )
    p3.insert_text((LEFT_X, y + 5), spec_intro_id if lang == "id" else spec_intro_en, fontsize=8.0, fontname="f_reg", color=COLOR_DARK)

    table_y = y + 22
    specs_html = (
        """
        <table>
            <tr><th width="28%">Komponen Komputasi</th><th width="32%">Spesifikasi & Versi Pustaka</th><th width="40%">Peran Fungsional & Karakteristik</th></tr>
            <tr><td><b>Sistem Operasi (OS)</b></td><td>Windows 10/11, Linux, macOS</td><td>Mendukung eksekusi lokal maupun deployment di cloud server (64-bit).</td></tr>
            <tr><td><b>Lingkungan Runtime</b></td><td>Python 3.10 / 3.11 / 3.12</td><td>Bahasa komputasi ilmiah utama berbasis multi-platform.</td></tr>
            <tr><td><b>Framework Antarmuka GUI</b></td><td>Streamlit &ge; 1.30.0</td><td>Antarmuka dasbor web reaktif, modular, dan interaktif secara real-time.</td></tr>
            <tr><td><b>Pemrosesan Sinyal Seismik</b></td><td>ObsPy &ge; 1.4.0</td><td>Ingesti berkas MiniSEED/SAC dan dekonvolusi transfer function FDSN.</td></tr>
            <tr><td><b>Komputasi Numerik & DSP</b></td><td>SciPy &ge; 1.11.0, NumPy &ge; 1.24.0</td><td>Filter Butterworth SOS, koreksi baseline polinomial, dan solver SDOF.</td></tr>
            <tr><td><b>Mesin Visualisasi Grafis</b></td><td>Plotly &ge; 5.18.0, Matplotlib &ge; 3.8.0</td><td>Grafik riwayat waktu interaktif WebGL, kurva Husid, dan rendering formula.</td></tr>
            <tr><td><b>Generator Dokumen Laporan</b></td><td>PyMuPDF (fitz) &ge; 1.23.0</td><td>Kompilasi laporan rekayasa vektor 2-halaman dan buku panduan ini.</td></tr>
            <tr><td><b>Format Data Masukan</b></td><td>MiniSEED (*.mseed), SAC (*.sac), StationXML</td><td>Format baku pertukaran gelombang seismik digital standar internasional.</td></tr>
            <tr><td><b>Kebutuhan Prosesor (CPU)</b></td><td>Dual-Core 2.0 GHz (Min), Quad-Core 2.8+ GHz</td><td>Komputasi numerik integrasi kinematika dan pemrosesan batch multi-rekaman.</td></tr>
            <tr><td><b>Alokasi Memori (RAM)</b></td><td>4 GB Minimum (8 GB Disarankan)</td><td>Kebutuhan memori untuk buffering gelombang triaksial dan mode batch.</td></tr>
            <tr><td><b>Penyimpanan Disk</b></td><td>500 MB Ruang Bebas</td><td>Instalasi pustaka dependensi Python dan berkas sementara cache.</td></tr>
            <tr><td><b>Kompatibilitas Peramban</b></td><td>Google Chrome, Mozilla Firefox, Edge, Safari</td><td>Peramban modern dengan dukungan penuh akselerasi grafis WebGL.</td></tr>
        </table>
        """
        if lang == "id" else
        """
        <table>
            <tr><th width="28%">Computational Component</th><th width="32%">Specification & Package Version</th><th width="40%">Functional Role & Architecture Characteristics</th></tr>
            <tr><td><b>Operating System (OS)</b></td><td>Windows 10/11, Linux, macOS</td><td>Cross-platform 64-bit support for local workstations and cloud containers.</td></tr>
            <tr><td><b>Runtime Environment</b></td><td>Python 3.10 / 3.11 / 3.12</td><td>Primary scientific computing engine across all platforms.</td></tr>
            <tr><td><b>GUI Web Framework</b></td><td>Streamlit &ge; 1.30.0</td><td>Reactive, interactive, and modular real-time web dashboard.</td></tr>
            <tr><td><b>Seismological Engine</b></td><td>ObsPy &ge; 1.4.0</td><td>MiniSEED/SAC ingestion and FDSN StationXML instrument deconvolution.</td></tr>
            <tr><td><b>Numerical & DSP Engine</b></td><td>SciPy &ge; 1.11.0, NumPy &ge; 1.24.0</td><td>Butterworth SOS filtering, polynomial baseline detrending, and SDOF solvers.</td></tr>
            <tr><td><b>Data Visualization</b></td><td>Plotly &ge; 5.18.0, Matplotlib &ge; 3.8.0</td><td>Interactive WebGL waveform charts, Husid plots, and LaTeX math rendering.</td></tr>
            <tr><td><b>PDF Document Engine</b></td><td>PyMuPDF (fitz) &ge; 1.23.0</td><td>Vector PDF compilation of 2-page engineering reports and this manual.</td></tr>
            <tr><td><b>Input Data Formats</b></td><td>MiniSEED (*.mseed), SAC (*.sac), StationXML</td><td>Standard international FDSN seismic data exchange formats.</td></tr>
            <tr><td><b>Processor Requirements</b></td><td>Dual-Core 2.0 GHz (Min), Quad-Core 2.8+ GHz</td><td>Kinematic integration numerical processing and multi-record batch evaluation.</td></tr>
            <tr><td><b>System Memory (RAM)</b></td><td>4 GB Minimum (8 GB Recommended)</td><td>RAM overhead for triaxial waveform buffering and batch matrix execution.</td></tr>
            <tr><td><b>Storage Capacity</b></td><td>500 MB Free Disk Space</td><td>Package dependency footprint and temporary cache storage.</td></tr>
            <tr><td><b>Browser Compatibility</b></td><td>Google Chrome, Mozilla Firefox, Edge, Safari</td><td>Modern web browsers with complete WebGL hardware acceleration support.</td></tr>
        </table>
        """
    )
    builder.insert_html_safe(p3, pymupdf.Rect(LEFT_X, table_y, RIGHT_X, table_y + 360), specs_html)

    # Verification Note
    callout_p3 = (
        "Seluruh dependensi pustaka di atas telah dibekukan (frozen) dalam berkas requirements.txt pada repositori GitHub. "
        "Pengguna disarankan menggunakan lingkungan virtual Python (venv atau conda) saat mengoperasikan software secara lokal "
        "guna memastikan isolasi dependensi dan integritas komputasi yang dapat direproduksi (reproducible research)."
        if lang == "id" else
        "All software package dependencies are locked within requirements.txt in the official GitHub repository. "
        "Users are strongly encouraged to deploy within dedicated Python virtual environments (venv or conda) for local operation "
        "to guarantee dependency isolation and reproducible computational integrity."
    )
    builder.draw_callout(p3, pymupdf.Rect(LEFT_X, table_y + 375, RIGHT_X, table_y + 455),
                         "Integritas Lingkungan Komputasi & Reprodusibilitas" if lang == "id" else "Computational Environment & Reproducibility Standards",
                         callout_p3, callout_type="info")

    # -------------------------------------------------------------
    # PAGE 4 (Table of Contents - Roman iv)
    # -------------------------------------------------------------
    p4 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p4)
    builder.add_page_header_footer(p4, "DAFTAR ISI" if lang == "id" else "TABLE OF CONTENTS", "Halaman iv dari iv" if lang == "id" else "Page iv of iv")

    y = builder.draw_chapter_banner(p4, "DAFTAR ISI & STRUKTUR PANDUAN TEKNIS" if lang == "id" else "TABLE OF CONTENTS & TECHNICAL GUIDEBOOK STRUCTURE")

    builder.draw_card(p4, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 490), bg_col=COLOR_WHITE, border_col=COLOR_CARD_BORDER)
    p4.insert_text((LEFT_X + 20, y + 24), "BAGIAN DOKUMEN & URUTAN BAB" if lang == "id" else "DOCUMENT SECTIONS & CHAPTER LISTING",
                   fontsize=9.8, fontname="f_bold", color=COLOR_NAVY)
    p4.draw_line(pymupdf.Point(LEFT_X + 20, y + 30), pymupdf.Point(RIGHT_X - 20, y + 30), color=COLOR_SKY, width=1.0)

    toc_items_id = [
        ("BAGIAN AWAL (FRONT MATTER)", "", True),
        ("Halaman Sampul & Informasi Pengembang", "i", False),
        ("Ringkasan Eksekutif & Landasan Ilmiah Pengembangan", "ii", False),
        ("Spesifikasi Lingkungan Komputasi & Dependensi Sistem", "iii", False),
        ("Daftar Isi & Struktur Sistematika Panduan", "iv", False),
        ("BAGIAN I: PANDUAN OPERASIONAL & ALUR KERJA SOFTWARE", "", True),
        ("BAB I: METODOLOGI AKSES DATA DAN NAVIGASI ANTARMUKA SISTEM", "1", True),
        ("BAB II: KONFIGURASI PARAMETER PEMROSESAN SINYAL DIGITAL PADA SIDEBAR", "2", True),
        ("BAB III: PROSEDUR ANALISIS KINEMATIKA DAN KONTROL KUALITAS GELOMBANG", "3", True),
        ("BAB IV: KARAKTERISASI GERAKAN TANAH KUAT, SKALA MMI, DAN SPEKTRUM RESPONS", "4", True),
        ("BAB V: PROTOKOL PEMROSESAN BATCH MULTI-STASIUN REGIONAL", "5", True),
        ("BAGIAN II: LANDASAN TEORETIS & PEMODELAN SEISMOLOGI REKAYASA", "", True),
        ("BAB VI: TEORI DEKONVOLUSI RESPONS INSTRUMEN DAN TRANSFER FUNCTION", "6", True),
        ("BAB VII: PRINSIP PENYARINGAN DIGITAL DAN DIAGNOSTIK PRE-EVENT NOISE", "7", True),
        ("BAB VIII: FORMULASI KOREKSI BASELINE DAN INTEGRASI KINEMATIKA GELOMBANG", "8", True),
        ("BAB IX: ENERGI SEISMIK, DURASI SIGNIFIKAN, DAN INTENSITAS SKALA MMI", "9", True),
        ("BAB X: PEMODELAN DINAMIKA STRUKTUR SDOF DAN SPEKTRUM DESAIN SNI 1726:2019", "10", True),
        ("BAGIAN III: DUKUNGAN OPERASIONAL, RUJUKAN ILMIAH, & PROFIL", "", True),
        ("BAB XI: PANDUAN PEMECAHAN MASALAH DAN BATASAN OPERASIONAL SISTEM", "11", True),
        ("DAFTAR PUSTAKA", "12", False),
        ("PROFIL PENGEMBANG", "13", False),
    ]
    toc_items_en = [
        ("FRONT MATTER", "", True),
        ("Title Page & Developer Metadata", "i", False),
        ("Executive Project Summary & Scientific Background", "ii", False),
        ("System Specifications & Runtime Dependencies", "iii", False),
        ("Table of Contents & Guidebook Architecture", "iv", False),
        ("PART I: OPERATIONAL WORKFLOWS & SOFTWARE SUITE", "", True),
        ("CHAPTER I: DATA ACCESS METHODOLOGY & USER INTERFACE NAVIGATION", "1", True),
        ("CHAPTER II: SIDEBAR DSP PARAMETERS & OPERATIONAL CONFIGURATION", "2", True),
        ("CHAPTER III: KINEMATIC ANALYSIS & WAVEFORM QUALITY CONTROL PROCEDURES", "3", True),
        ("CHAPTER IV: STRONG GROUND MOTION, MMI INTENSITY, & RESPONSE SPECTRA", "4", True),
        ("CHAPTER V: REGIONAL MULTI-STATION BATCH PROCESSING PROTOCOL", "5", True),
        ("PART II: THEORETICAL FOUNDATIONS & ENGINEERING SEISMOLOGY", "", True),
        ("CHAPTER VI: INSTRUMENT RESPONSE DECONVOLUTION & TRANSFER FUNCTIONS", "6", True),
        ("CHAPTER VII: DIGITAL FILTERING PRINCIPLES & PRE-EVENT NOISE DIAGNOSTICS", "7", True),
        ("CHAPTER VIII: BASELINE CORRECTION & WAVEFORM KINEMATICS INTEGRATION", "8", True),
        ("CHAPTER IX: SEISMIC ENERGY, SIGNIFICANT DURATION, & MMI INTENSITY", "9", True),
        ("CHAPTER X: SDOF STRUCTURAL DYNAMICS & SNI 1726:2019 CODE DESIGN", "10", True),
        ("PART III: OPERATIONAL SUPPORT, BIBLIOGRAPHY, & PROFILE", "", True),
        ("CHAPTER XI: TROUBLESHOOTING GUIDE & OPERATIONAL LIMITATIONS", "11", True),
        ("REFERENCES", "12", False),
        ("DEVELOPER PROFILE", "13", False),
    ]
    toc_items = toc_items_id if lang == "id" else toc_items_en

    page_map = {
        "i": 0, "ii": 1, "iii": 2, "iv": 3,
        "1": 4, "2": 5, "3": 6, "4": 7, "5": 8, "6": 9,
        "7": 10, "8": 11, "9": 12, "10": 13, "11": 14, "12": 15, "13": 16
    }
    toc_links: list[tuple[pymupdf.Rect, int]] = []
    cur_y = y + 46
    for title, page_str, is_bold in toc_items:
        if not page_str:
            cur_y += 3
            p4.draw_rect(pymupdf.Rect(LEFT_X + 20, cur_y - 2, RIGHT_X - 20, cur_y + 12), color=COLOR_CARD_BG, fill=COLOR_CARD_BG)
            p4.insert_text((LEFT_X + 26, cur_y + 8), title, fontsize=7.2, fontname="f_bold", color=COLOR_SKY)
            cur_y += 16
            continue

        font_name = "f_bold" if is_bold else "f_reg"
        font_col = COLOR_NAVY if is_bold else COLOR_DARK
        p4.insert_text((LEFT_X + 26, cur_y), title, fontsize=7.6, fontname=font_name, color=font_col)

        text_len = font_bold_obj.text_length(title, fontsize=7.6)
        leader_start_x = LEFT_X + 32 + text_len
        leader_end_x = RIGHT_X - 52
        if leader_end_x > leader_start_x:
            dots_count = int((leader_end_x - leader_start_x) / 4.2)
            dots = ". " * (dots_count // 2)
            p4.insert_text((leader_start_x, cur_y), dots, fontsize=7.0, fontname="f_reg", color=COLOR_MUTED)

        p4.insert_text((RIGHT_X - 44, cur_y), page_str, fontsize=7.8, fontname="f_bold", color=COLOR_NAVY)
        if page_str in page_map:
            link_rect = pymupdf.Rect(LEFT_X + 20, cur_y - 10, RIGHT_X - 20, cur_y + 4)
            toc_links.append((link_rect, page_map[page_str]))
        cur_y += 15.2

    # Bottom Callout
    callout_toc_text = (
        "Buku Panduan ini menyajikan panduan pengoperasian software secara komprehensif pada Bab I s.d. V di bagian awal, "
        "dilanjutkan dengan landasan teoretis seismologi teknik dan standar ketahanan gempa pada Bab VI s.d. X, serta pemecahan masalah "
        "dan daftar pustaka ilmiah berstandar akademik pada bagian akhir."
        if lang == "id" else
        "This Guidebook presents comprehensive operational software instructions across Chapters I to V at the beginning of the volume, "
        "followed by theoretical foundations in engineering seismology across Chapters VI to X, concluding with technical troubleshooting "
        "and formal academic literature references."
    )
    builder.draw_callout(p4, pymupdf.Rect(LEFT_X, y + 505, RIGHT_X, y + 565),
                         "Petunjuk Navigasi & Struktur Dokumen" if lang == "id" else "Document Structure & Navigation Guide",
                         callout_toc_text, callout_type="info")

    # -------------------------------------------------------------
    # PAGE 5 (Halaman 1) - BAB I: MEMULAI & ALUR KERJA DASAR SISTEM
    # -------------------------------------------------------------
    p5 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p5)
    builder.add_page_header_footer(p5, "BAB I - ALUR KERJA SISTEM" if lang == "id" else "CHAPTER I - SYSTEM WORKFLOW",
                                   f"Halaman 1 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 1 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p5, "BAB I: MEMULAI & ALUR KERJA DASAR SISTEM" if lang == "id" else "CHAPTER I: GETTING STARTED & CORE SYSTEM WORKFLOW")

    c1_text = (
        """
        <p><b>1.1 Persyaratan Berkas & Spesifikasi Masukan Data</b></p>
        <p>Perangkat lunak <b>BSMA v2.0.0</b> dirancang khusus untuk memproses rekaman akselerograf gerakan tanah kuat 
        (<i>strong ground motion accelerograph</i>) dalam format standar seismologi internasional:</p>
        <p>&bull; <b>Format Gelombang (Waveforms):</b> Mendukung berkas <b>MiniSEED (.mseed, .miniseed, .msd)</b> dan <b>SAC (.sac)</b>. 
        Berkas rekaman idealnya memuat 3-komponen gerak tanah ortogonal (Vertical Z, North-South N/1, East-West E/2) dengan laju cuplik 
        seragam (<i>uniform sampling rate</i>, umumnya 50 Hz, 100 Hz, atau 200 Hz pada jaringan BMKG) serta bebas dari celah waktu (<i>data gaps</i>).</p>
        <p>&bull; <b>Metadata Respons Sensor (StationXML):</b> Berkas berformat <b>.xml (FDSN StationXML schema)</b> yang memuat kurva respons 
        instrumen, gain sensitivitas tahapan (<i>stage gain</i>), serta kutub dan nol (<i>poles and zeros</i>) untuk sensor akselerometer terkait.</p>

        <p><b>1.2 Orientasi Tata Letak Antarmuka Pengguna (GUI Streamlit)</b></p>
        <p>Antarmuka grafis BSMA v2.0.0 tersusun secara ergonomis ke dalam tiga zona fungsional utama:</p>
        <p>&bull; <b>Header Bar Navigasi:</b> Memuat identitas logo ganda BMKG & ITERA, judul versi software, serta status konektivitas komputasi.</p>
        <p>&bull; <b>Sidebar Kontrol Interaktif (Sisi Kiri):</b> Pusat kendali parameter komputasi, konfigurasi berkas, pemilihan satuan fisik, 
        parameter filter frekuensi, pemilihan solver dinamika struktur, serta entri metadata kejadian gempa.</p>
        <p>&bull; <b>Dasbor Analisis Multi-Tab (Area Utama):</b> Terdiri atas 7 tab analisis komprehensif yang menyajikan grafik riwayat waktu Plotly interaktif, 
        indikator skor mutu QC, kurva energi Husid, estimasi MMI, spektrum respons elastis SDOF, dan sarana pengunduhan luaran.</p>

        <p><b>1.3 Alur Pemrosesan 5-Tahap Terpadu (Standard Operational Workflow)</b></p>
        <p>Pengoperasian sistem mengikuti 5 tahapan berurutan: <b>(1) Ingesti Data:</b> Unggah berkas gelombang dan StationXML pada sidebar. 
        <b>(2) Parameterisasi:</b> Verifikasi satuan fisik dan frekuensi sudut filter bandpass. <b>(3) Eksekusi Sinyal:</b> Sistem secara otomatis 
        mengeksekusi QC sinyal, filter SOS zero-phase, dan integrasi kinematika ganda. <b>(4) Eksplorasi Tab:</b> Evaluasi hasil analisis 
        pada Tab 1 s.d. Tab 6 secara interaktif. <b>(5) Ekspor Luaran:</b> Unduh laporan teknik PDF ringkas, berkas data CSV, atau paket arsip ZIP pada Tab 7.</p>

        <p><b>1.4 Manajemen Sesi & Tombol Reset Data (Clear Data)</b></p>
        <p>Pada bagian bawah panel masukan data di sidebar, tersedia tombol <b>'Reset / Clear Data'</b>. Tombol ini berfungsi membersihkan 
        seluruh memori penampung sesi (<i>st.session_state</i>), mengosongkan berkas gelombang yang dimuat sebelumnya, dan mengembalikan 
        parameter filter ke nilai baku standar operasional sebelum memproses rekaman gempa bumi baru.</p>
        """
        if lang == "id" else
        """
        <p><b>1.1 Input File Requirements & Data Specifications</b></p>
        <p><b>BSMA v2.0.0</b> is engineered specifically to process strong-motion accelerograms in standard seismological formats:</p>
        <p>&bull; <b>Waveform Data Formats:</b> Fully supports <b>MiniSEED (.mseed, .miniseed, .msd)</b> and <b>SAC (.sac)</b> binary formats. 
        Recordings should ideally encompass 3 orthogonal components (Vertical Z, North-South N/1, East-West E/2) recorded at uniform 
        sampling rates (typically 50, 100, or 200 Hz across BMKG networks) free from timing discontinuities or data gaps.</p>
        <p>&bull; <b>Instrument Response Metadata (StationXML):</b> Standard <b>.xml (FDSN StationXML schema)</b> file encapsulating stage sensitivity gains, 
        digitizer bitweights, and instrument transfer function poles and zeros for raw count deconvolution.</p>

        <p><b>1.2 Streamlit User Interface Layout Orientation</b></p>
        <p>The BSMA v2.0.0 graphic interface is ergonomically partitioned into three primary functional zones:</p>
        <p>&bull; <b>Top Navigation Header:</b> Displays institutional BMKG & ITERA emblems, software title, and real-time computation status.</p>
        <p>&bull; <b>Interactive Control Sidebar (Left Panel):</b> Central command panel for dataset loading, unit selection, 
        DSP filter parameters, structural dynamic solver options, and earthquake event metadata entry.</p>
        <p>&bull; <b>Multi-Tab Analysis Dashboard (Main Area):</b> 7 comprehensive analysis tabs featuring interactive Plotly kinematic charts, 
        QC health gauges, Husid energy curves, instrumental MMI maps, SDOF elastic spectra, and downloadable deliverables.</p>

        <p><b>1.3 5-Stage Integrated Processing Workflow</b></p>
        <p>Operation follows a sequential 5-stage protocol: <b>(1) Data Ingestion:</b> Load waveform and StationXML files via the sidebar. 
        <b>(2) Parameterization:</b> Verify physical units and bandpass filter corner frequencies. <b>(3) Execution:</b> The automated engine executes 
        QC screening, zero-phase SOS filtering, and kinematic double integration. <b>(4) Interactive Exploration:</b> Inspect Tabs 1 to 6. 
        <b>(5) Deliverable Export:</b> Generate publication-grade PDF reports, numerical CSVs, or ZIP archives from Tab 7.</p>

        <p><b>1.4 Session State Management & Reset Data Button</b></p>
        <p>Beneath the data loading panel in the sidebar, the <b>'Reset / Clear Data'</b> button clears the internal cache (<i>st.session_state</i>), 
        purges loaded waveform streams from memory, and restores filter parameters to default values before ingesting subsequent event records.</p>
        """
    )
    builder.insert_html_safe(p5, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 295), c1_text)

    table_y = y + 300
    file_spec_html = (
        """
        <table>
            <tr><th width="18%">Format Berkas</th><th width="20%">Ekstensi Didukung</th><th width="28%">Persyaratan Metadata</th><th width="34%">Catatan Penanganan Teknis</th></tr>
            <tr><td><b>MiniSEED</b></td><td>.mseed, .miniseed, .msd</td><td>Header STEIM1/2, Network, Station, Channel</td><td>Format standar BMKG; diurai otomatis melalui modul ObsPy.</td></tr>
            <tr><td><b>SAC Binary</b></td><td>.sac</td><td>Header SAC (DELTA, NPTS, KZDATE, KZTIME)</td><td>Mendukung berkas SAC komponen tunggal maupun multi-kanal.</td></tr>
            <tr><td><b>StationXML</b></td><td>.xml</td><td>FDSN Schema v1.1, Poles & Zeros, Gain</td><td>Diperlukan khusus bila data rekaman berdimensi digital counts mentah.</td></tr>
            <tr><td><b>Benchmark CSV</b></td><td>.csv</td><td>Header kolom: Period_s, PSA_g / PSA_ms2</td><td>Opsional; digunakan untuk validasi silang akurasi numerik solver SDOF.</td></tr>
        </table>
        """
        if lang == "id" else
        """
        <table>
            <tr><th width="18%">Data Format</th><th width="20%">Supported Extensions</th><th width="28%">Metadata Requirements</th><th width="34%">Technical Handling Notes</th></tr>
            <tr><td><b>MiniSEED</b></td><td>.mseed, .miniseed, .msd</td><td>STEIM1/2 headers, Network, Station, Channel</td><td>BMKG standard format; parsed natively using ObsPy engine.</td></tr>
            <tr><td><b>SAC Binary</b></td><td>.sac</td><td>SAC headers (DELTA, NPTS, KZDATE, KZTIME)</td><td>Supports single-trace and multi-channel SAC binary streams.</td></tr>
            <tr><td><b>StationXML</b></td><td>.xml</td><td>FDSN Schema v1.1, Poles & Zeros, Gain</td><td>Mandatory only when input stream consists of uncalibrated digital counts.</td></tr>
            <tr><td><b>Benchmark CSV</b></td><td>.csv</td><td>Column headers: Period_s, PSA_g / PSA_ms2</td><td>Optional; utilized for cross-validating SDOF solver numerical accuracy.</td></tr>
        </table>
        """
    )
    builder.insert_html_safe(p5, pymupdf.Rect(LEFT_X, table_y, RIGHT_X, table_y + 115), file_spec_html)

    callout_c1 = (
        "Sebelum memproses berkas, pastikan rekaman akselerograf memiliki durasi pre-event (waktu hening sebelum gelombang P) "
        "minimal 5 hingga 10 detik. Jendela pre-event ini sangat krusial bagi algoritma Quality Control (QC) untuk menghitung "
        "lantai derau latar (noise floor) dan rasio sinyal terhadap derau (SNR) secara akurat."
        if lang == "id" else
        "Before processing, ensure that accelerograms include at least 5 to 10 seconds of pre-event baseline prior to the first P-wave arrival. "
        "This quiet window is essential for the automated Quality Control (QC) engine to establish pre-event noise floor levels "
        "and calculate accurate signal-to-noise ratios (SNR)."
    )
    builder.draw_callout(p5, pymupdf.Rect(LEFT_X, table_y + 125, RIGHT_X, table_y + 195),
                         "Panduan Penyiapan Sinyal & Jendela Pre-Event" if lang == "id" else "Signal Preparation & Pre-Event Window Guidelines",
                         callout_c1, callout_type="info")

    # -------------------------------------------------------------
    # PAGE 6 (Halaman 2) - BAB II: PANDUAN OPERASIONAL MENU SIDEBAR
    # -------------------------------------------------------------
    p6 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p6)
    builder.add_page_header_footer(p6, "BAB II - MENU SIDEBAR" if lang == "id" else "CHAPTER II - SIDEBAR CONTROLS",
                                   f"Halaman 2 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 2 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p6, "BAB II: PANDUAN OPERASIONAL MENU SIDEBAR & PARAMETER DSP" if lang == "id" else "CHAPTER II: SIDEBAR OPERATIONAL CONTROLS & DSP SETTINGS")

    c2_text = (
        """
        <p><b>2.1 Bagian DATA INPUT (Pengunggahan Data)</b></p>
        <p>&bull; <b>Uploader Akselerogram:</b> Area interaktif untuk memilih atau menyeret (<i>drag-and-drop</i>) berkas .mseed atau .sac. 
        Sistem secara otomatis membaca header dan mengekstrak stasiun, jaringan, tanggal kejadian, serta komponen sinyal.</p>
        <p>&bull; <b>Uploader StationXML:</b> Kolom pengunggahan berkas metadata .xml respons sensor. Jika berkas diunggah, sistem mencocokkan 
        kanal instrumen secara otomatis. Tombol <b>'Reset / Clear Data'</b> terletak di bagian bawah uploader untuk membersihkan sesi kerja.</p>

        <p><b>2.2 Bagian PROJECT (Asal-Usul Data & Satuan Fisik)</b></p>
        <p>&bull; <b>Data Provenance:</b> Pengguna dapat memilih status rekaman: <i>'Already processed physical acceleration'</i> (data telah 
        terkalibrasi dalam satuan percepatan fisik), <i>'Raw instrument counts with StationXML'</i> (data mentah hitungan digital yang 
        memerlukan dekonvolusi transfer function), atau <i>'Unknown'</i> (sistem melakukan deteksi heuristik otomatis).</p>
        <p>&bull; <b>Physical Unit Selector:</b> Memilih satuan fisik keluaran analisis: <b>m/s&sup2;</b> (standar SI internasional), 
        <b>Gal</b> (1 Gal = 1 cm/s&sup2; = 0.01 m/s&sup2;, standar operasional BMKG), atau <b>cm/s&sup2;</b>.</p>

        <p><b>2.3 Bagian PROCESSING (Konfigurasi Tapis Digital & Detrending)</b></p>
        <p>&bull; <b>Pilihan Filter:</b> Pilihan tipe filter lolos pita (<i>Bandpass</i>, sangat direkomendasikan untuk gempa kuat), 
        lolos rendah (<i>Lowpass</i>), atau lolos tinggi (<i>Highpass</i>).</p>
        <p>&bull; <b>Frekuensi Sudut (Corner Frequencies):</b> Slider interaktif untuk menentukan batas bawah <b>f<sub>min</sub></b> 
        (default: 0.05 - 0.10 Hz) guna mengeliminasi derau frekuensi rendah pemicu drift integrasi, serta batas atas <b>f<sub>max</sub></b> 
        (default: 25.0 - 40.0 Hz) guna menekan derau instrumen frekuensi tinggi.</p>
        <p>&bull; <b>Tombol Pintas 'Preset Baku Strong-Motion (0.05 - 40.0 Hz)':</b> Tombol cepat satu-klik untuk menyetel filter ke rentang baku 
        operasional pengolahan akselerogram gempa kuat di lingkungan BMKG.</p>
        <p>&bull; <b>Submenu Advanced Settings:</b> Pengaturan lanjutan mencakup opsi <i>Polynomial Detrending</i> (orde linear atau kuadratik 
        untuk koreksi offset baseline), <i>Cosine Tapering</i> (jendela Tukey 5% pada kedua ujung sinyal untuk mencegah artefak spektral), 
        serta <i>Adaptive SNR / Nyquist Floor Screening</i> untuk perlindungan batas teoritis Shannon-Nyquist (0.40 f<sub>s</sub>).</p>

        <p><b>2.4 Bagian ANALYSIS (Parameter Dinamika Struktur & Solver SDOF)</b></p>
        <p>&bull; <b>Damping Ratio (&xi;):</b> Slider rasio redaman kritis struktur SDOF dengan nilai default <b>0.05 (5%)</b> sesuai standar rekayasa gempa.</p>
        <p>&bull; <b>SDOF Solver Selection:</b> Pengguna dapat memilih antara <b>Nigam-Jennings (1969)</b> (solusi analitik eksak bagian-per-bagian 
        yang sangat stabil pada frekuensi tinggi) atau <b>Newmark-Beta (1959)</b> (integrasi implisit percepatan rata-rata linier &gamma;=0.5, &beta;=0.25).</p>

        <p><b>2.5 Bagian OUTPUT & BENCHMARK (Metadata Kejadian Gempa & Validasi)</b></p>
        <p>&bull; <b>Event Metadata Expander:</b> Memasukkan parameter gempa bumi: Origin Time (UTC), Magnitudo (M<sub>w</sub>), Kedalaman Hiposenter (km), 
        Koordinat Episenter (Lintang & Bujur), dan Jarak Episentral (km) untuk dicantumkan secara otomatis pada laporan PDF.</p>
        <p>&bull; <b>Ground Truth Benchmark Expander:</b> Mengunggah berkas CSV acuan spektrum respons eksternal guna memvalidasi akurasi numerik solver SDOF.</p>
        """
        if lang == "id" else
        """
        <p><b>2.1 DATA INPUT Section (File Uploaders)</b></p>
        <p>&bull; <b>Waveform Uploader:</b> Interactive drop zone for .mseed or .sac files. The system immediately parses trace headers 
        and displays station name, network code, event timestamp, and available components.</p>
        <p>&bull; <b>StationXML Uploader:</b> Optional upload area for .xml response metadata. The <b>'Reset / Clear Data'</b> button 
        located directly beneath purges memory buffers before processing new seismic events.</p>

        <p><b>2.2 PROJECT Section (Provenance & Physical Units)</b></p>
        <p>&bull; <b>Data Provenance:</b> Explicitly declare recording origin: <i>'Already processed physical acceleration'</i> (stream is pre-calibrated 
        in physical units), <i>'Raw instrument counts with StationXML'</i> (demands full transfer function deconvolution), or <i>'Unknown'</i> (automated heuristic detection).</p>
        <p>&bull; <b>Physical Unit Selector:</b> Choose operational output unit: <b>m/s&sup2;</b> (SI standard), <b>Gal</b> (1 Gal = 1 cm/s&sup2; = 0.01 m/s&sup2;, BMKG standard), 
        or <b>cm/s&sup2;</b>.</p>

        <p><b>2.3 PROCESSING Section (DSP Filtering & Detrending Parameters)</b></p>
        <p>&bull; <b>Filter Type:</b> Select between <i>Bandpass</i> (recommended standard for strong motion), <i>Lowpass</i>, or <i>Highpass</i>.</p>
        <p>&bull; <b>Corner Frequencies:</b> Fine-tune low-cut <b>f<sub>min</sub></b> (default: 0.05 - 0.10 Hz) to eliminate baseline drift noise, 
        and high-cut <b>f<sub>max</sub></b> (default: 25.0 - 40.0 Hz) to eliminate high-frequency cultural/instrument noise.</p>
        <p>&bull; <b>'Strong-Motion Default Preset (0.05 - 40.0 Hz)' Button:</b> One-click shortcut applying the standard operational filter band commonly adopted in strong-motion accelerograph networks.</p>
        <p>&bull; <b>Advanced Settings Expander:</b> Access low-order polynomial baseline detrending (linear/quadratic), cosine tapering (5% Tukey window), 
        and adaptive Nyquist frequency floor safeguards (enforcing cutoffs &le; 0.40 f<sub>s</sub>).</p>

        <p><b>2.4 ANALYSIS Section (Structural Dynamics & SDOF Solver)</b></p>
        <p>&bull; <b>Damping Ratio (&xi;):</b> Structural critical damping slider preset to <b>0.05 (5%)</b> per international structural codes.</p>
        <p>&bull; <b>SDOF Numerical Solver:</b> Choose between <b>Nigam-Jennings (1969)</b> (exact piecewise analytical solution highly suitable for short periods) 
        and <b>Newmark-Beta (1959)</b> (implicit average acceleration algorithm with &gamma;=0.5, &beta;=0.25).</p>

        <p><b>2.5 OUTPUT & BENCHMARK Section (Earthquake Metadata & Verification)</b></p>
        <p>&bull; <b>Event Metadata Expander:</b> Input Origin Time (UTC), Magnitude (M<sub>w</sub>), Hypocentral Depth (km), Coordinates, and Epicentral Distance (km) 
        for automated injection into standardized PDF engineering summary reports.</p>
        <p>&bull; <b>Benchmark Expander:</b> Upload external reference PSA CSV curves to benchmark and audit SDOF solver precision.</p>
        """
    )
    builder.insert_html_safe(p6, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 490), c2_text)

    callout_c2 = (
        "Rekomendasi Pemilihan Solver: Gunakan solver Nigam-Jennings (1969) jika analisis difokuskan pada spektrum respons "
        "periode pendek (T < 0.1 s) atau struktur kaku. Gunakan Newmark-Beta (1959) untuk perbandingan standar dinamika struktur. "
        "Kedua solver telah divalidasi silang pada Tab 6 dengan kesesuaian numerik tinggi (selisih relatif rata-rata < 0.02%, selisih RMS < 1.5e-4 g)."
        if lang == "id" else
        "Solver Selection Recommendation: Deploy the Nigam-Jennings (1969) solver when focusing on short-period response spectra (T < 0.1 s) "
        "or stiff masonry/concrete structures. Use Newmark-Beta (1959) for standard structural dynamic cross-checks. "
        "Both solvers exhibit rigorous agreement (mean relative difference < 0.02%, RMS difference < 1.5e-4 g across 100 periods), auditable in Tab 6."
    )
    builder.draw_callout(p6, pymupdf.Rect(LEFT_X, y + 500, RIGHT_X, y + 560),
                         "Rekomendasi Parameter Komputasi Dinamika Struktur" if lang == "id" else "Structural Dynamic Computation Recommendations",
                         callout_c2, callout_type="success")

    # -------------------------------------------------------------
    # PAGE 7 (Halaman 3) - BAB III: TAB ANALISIS INTERAKTIF (BAGIAN I)
    # -------------------------------------------------------------
    p7 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p7)
    builder.add_page_header_footer(p7, "BAB III - TAB ANALISIS I" if lang == "id" else "CHAPTER III - ANALYSIS TABS I",
                                   f"Halaman 3 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 3 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p7, "BAB III: PANDUAN PENGOPERASIAN TAB ANALISIS INTERAKTIF (BAGIAN I)" if lang == "id" else "CHAPTER III: INTERACTIVE ANALYSIS TABS OPERATION (PART I)")

    c3_text = (
        """
        <p><b>3.1 Tab 1 - Ringkasan Stasiun & Tinjauan Sinyal (Summary Tab)</b></p>
        <p>&bull; <b>Kartu Metadata & Visualizer 3-Komponen:</b> Menampilkan identitas stasiun, kode jaringan, koordinat geografis, 
        laju cuplik (<i>sampling rate</i>), durasi total rekaman, serta grafik awal sinkron 3 kanal (Z, N, E) untuk verifikasi visual fase gempa utama.</p>
        <p>&bull; <b>Indikator Skor Kesehatan QC (QC Health Score):</b> Menampilkan skor komposit mutu sinyal (Q = 100 - &sum; P<sub>i</sub>) 
        dari basis 100 poin dengan status: <b>PASS (&ge; 70)</b>, <b>WARNING (50-69)</b>, atau <b>FAIL (&lt; 50)</b>.</p>
        <p>&bull; <b>Badge Estimasi Intensitas MMI:</b> Menampilkan estimasi skala intensitas makroseismik instrumental tercepat 
        berdasarkan percepatan horizontal maksimum (<b>Max-H</b> antara komponen EW dan NS, tanpa komponen Z) mengacu formulasi Worden et al. (2012).</p>

        <p><b>3.2 Tab 2 - Riwayat Waktu Kinematika Lengkap (Waveforms Tab)</b></p>
        <p>&bull; <b>Grafik Kinematika Sinkron Tiga Baris:</b> Menyajikan riwayat waktu <b>Percepatan a(t)</b> [Gal], 
        <b>Kecepatan v(t)</b> [cm/s], dan <b>Perpindahan d(t)</b> [cm] yang sejajar terhadap sumbu waktu (<i>shared X-axis</i>).</p>
        <p>&bull; <b>Interaktivitas Plotly:</b> Fitur perbesaran (<i>zoom-in</i>), penggeseran (<i>pan</i>), kursor pembacaan fase P/S, 
        dan penanda visual nilai puncak kinematika absolut (PGA, PGV, dan PGD).</p>

        <p><b>3.3 Tab 3 - Kendali Mutu Sinyal & Diagnostik Derau (Quality Control Tab)</b></p>
        <p>Tab Quality Control menyajikan audit kesehatan sinyal otomatis dari basis 100 poin: <code>Q = max(0, min(100, 100 - &sum; P<sub>i</sub>))</code>. 
        Sinyal dievaluasi melalui matriks gerbang diagnostik fisik sebelum penapisan digital:</p>
        <table>
            <tr><th width="20%">Tingkat Keparahan</th><th width="44%">Kriteria Anomali Fisik & Parameter Gate</th><th width="16%">Bobot Penalti</th><th width="20%">Status Kelayakan</th></tr>
            <tr><td><b>Nominal / Normal</b></td><td>Sinyal bersih, noise floor rendah, pre-event SNR &ge; 10 dB, baseline stabil</td><td>0 (Basis 100)</td><td><b>PASS (&ge; 70)</b><br>Layak analisis penuh</td></tr>
            <tr><td><b>Peringatan Ringan</b><br>(Warning)</td><td>• Baseline Offset (&gt; 2% PGA)<br>• Baseline Drift kemiringan (&gt; 5%)<br>• Pre-event window &lt; 5.0 detik<br>• Pre-event SNR marginal (3 s.d. 10 dB)</td><td><b>-15 Poin</b><br>(per anomali)</td><td><b>WARNING (50-69)</b><br>Layak catatan rekayasa</td></tr>
            <tr><td><b>Kesalahan Berat</b><br>(Major Error)</td><td>• Sinyal derau tinggi parah (SNR &lt; 3 dB)<br>• Multiple impulsive spikes (MAD &ge; 6.0)</td><td><b>-40 Poin</b><br>(per anomali)</td><td>Zona degradasi mutu</td></tr>
            <tr><td><b>Anomali Kritis</b><br>(Fatal Override)</td><td>• Sensor mentok terpotong (<i>clipping</i> &ge; 0.02 s / 98% FS)<br>• Sinyal datar / hilang total (<i>flatline</i> &ge; 1.0 s) / berkas korup</td><td><b>Fatal Override<br>(Q = 0)</b></td><td><b>FAIL / REJECT (&lt; 50)</b><br>Ditolak integrasi</td></tr>
        </table>
        """
        if lang == "id" else
        """
        <p><b>3.1 Tab 1 - Station Summary & Signal Overview (Summary Tab)</b></p>
        <p>&bull; <b>Metadata Card & Triaxial Visualizer:</b> Displays station identifier, network code, coordinates, 
        sampling rate, duration, and synchronized 3-component raw waveforms (Z, N, E) for visual verification of major seismic phases.</p>
        <p>&bull; <b>QC Health Score Gauge:</b> Composite signal quality indicator (Q = 100 - &sum; P<sub>i</sub>) from a 100-point baseline 
        with calibrated status badges: <b>PASS (&ge; 70)</b>, <b>WARNING (50-69)</b>, or <b>FAIL (&lt; 50)</b>.</p>
        <p>&bull; <b>Instrumental MMI Badge:</b> Rapid macroseismic intensity badge computed from maximum horizontal ground motion 
        (<b>Max-H</b> between EW and NS components, excluding vertical Z) per Worden et al. (2012) GMICE regressions.</p>

        <p><b>3.2 Tab 2 - Full Kinematic Time Series (Waveforms Tab)</b></p>
        <p>&bull; <b>Synchronized Tri-Level Kinematic Stack:</b> Stacked synchronized plots of <b>Acceleration a(t)</b> [Gal], 
        <b>Velocity v(t)</b> [cm/s], and <b>Displacement d(t)</b> [cm] locked to a common time axis.</p>
        <p>&bull; <b>Plotly Interactivity:</b> Interactive zoom, pan, hover cursor for micro-second inspection of P/S arrivals 
        and annotated peak kinematics (PGA, PGV, and PGD).</p>

        <p><b>3.3 Tab 3 - Signal Quality Control & Diagnostics (QC Tab)</b></p>
        <p>Automated pre-processing health screening auditing raw records prior to digital filtering. The composite health score 
        is bounded on a 100-point baseline: <code>Q = max(0, min(100, 100 - &sum; P<sub>i</sub>))</code> across calibrated physical gates:</p>
        <table>
            <tr><th width="20%">Severity Level</th><th width="44%">Physical Anomaly Criteria & Gate Parameters</th><th width="16%">Point Penalty</th><th width="20%">Gate Status</th></tr>
            <tr><td><b>Nominal / Normal</b></td><td>Clean trace, low ambient floor, pre-event SNR &ge; 10 dB, stable baseline</td><td>0 (Baseline 100)</td><td><b>PASS (&ge; 70)</b><br>Full analysis approved</td></tr>
            <tr><td><b>Light Warning</b><br>(Warning Tier)</td><td>• Baseline Offset (&gt; 2% PGA)<br>• Baseline Drift thermal tilt (&gt; 5%)<br>• Pre-event noise window &lt; 5.0 s<br>• Marginal pre-event SNR (3 to 10 dB)</td><td><b>-15 Points</b><br>(per anomaly)</td><td><b>WARNING (50-69)</b><br>Usable with caution</td></tr>
            <tr><td><b>Major Error</b><br>(Error Tier)</td><td>• Severe ambient noise floor (SNR &lt; 3 dB)<br>• Multiple impulsive spikes (MAD &ge; 6.0)</td><td><b>-40 Points</b><br>(per anomaly)</td><td>Degraded quality zone</td></tr>
            <tr><td><b>Critical Violation</b><br>(Fatal Override)</td><td>• Sensor clipping / ADC saturation (&ge; 0.02 s / 98% FS)<br>• Dead channel / continuous flatline (&ge; 1.0 s) / corrupted file</td><td><b>Fatal Override<br>(Q = 0)</b></td><td><b>FAIL / REJECT (&lt; 50)</b><br>Integration rejected</td></tr>
        </table>
        """
    )
    builder.insert_html_safe(p7, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 435), c3_text)

    callout_c3_title = (
        "Penjelasan Evaluasi Skor 70 pada Data Sinyal Mentah Lapangan"
        if lang == "id" else
        "Understanding Score 70 on Raw Field Waveform Records"
    )
    callout_c3 = (
        "<b>Kondisi Sinyal Mentah di Lapangan:</b> Skor 70 bukan berarti sinyal rusak, melainkan data sensor mentah yang masih membawa "
        "karakteristik alami lingkungan sebelum disaring. Dua faktor pemicu peringatan (Warning): "
        "(1) <i>Baseline Offset</i> (Penalti -15 Poin) akibat nilai tegangan rata-rata sensor belum tepat nol, dan "
        "(2) <i>Baseline Drift</i> (Penalti -15 Poin) akibat sedikit kemiringan respon suhu alat di stasiun. "
        "<b>Evaluasi Akhir Skor Kualitas:</b> Skor Awal = 100 Poin; Total Penalti = -15 (Offset) - 15 (Drift) = -30 Poin; "
        "Skor Akhir Data Mentah = <b>70 Poin (Status Lolos / PASS)</b>. Kedua anomali non-fatal ini otomatis dibersihkan tuntas pada filter DSP BSMA!"
        if lang == "id" else
        "<b>Raw Field Waveform Conditions:</b> A QC score of 70 does not indicate corrupted data, but rather raw sensor records "
        "retaining ambient environmental effects prior to digital filtering. Two triggering warning factors: "
        "(1) <i>Baseline Offset</i> (-15 Pts penalty) from pre-event transducer mean voltage offset, and "
        "(2) <i>Baseline Drift</i> (-15 Pts penalty) caused by ambient station temperature response. "
        "<b>Final Quality Score Evaluation:</b> Initial Baseline = 100 Pts; Total Penalty = -15 (Offset) - 15 (Drift) = -30 Pts; "
        "Final Raw Score = <b>70 Pts (Nominal PASS Status)</b>. Both non-fatal anomalies are completely purged by BSMA's DSP baseline and bandpass filtering!"
    )
    builder.draw_callout(p7, pymupdf.Rect(LEFT_X, y + 442, RIGHT_X, y + 560),
                         callout_c3_title, callout_c3, callout_type="info")

    # -------------------------------------------------------------
    # PAGE 8 (Halaman 4) - BAB IV: TAB ANALISIS INTERAKTIF (BAGIAN II)
    # -------------------------------------------------------------
    p8 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p8)
    builder.add_page_header_footer(p8, "BAB IV - TAB ANALISIS II" if lang == "id" else "CHAPTER IV - ANALYSIS TABS II",
                                   f"Halaman 4 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 4 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p8, "BAB IV: PANDUAN PENGOPERASIAN TAB ANALISIS INTERAKTIF (BAGIAN II)" if lang == "id" else "CHAPTER IV: INTERACTIVE ANALYSIS TABS OPERATION (PART II)")

    c4_text = (
        """
        <p><b>4.1 Tab 4 - Parameter Gerakan Kuat & Energi Seismik (Strong Motion Tab)</b></p>
        <p>Tab Strong Motion menyajikan analisis energi getaran tanah dan potensi beban dinamik pada struktur:</p>
        <p>&bull; <b>Kurva Plot Husid:</b> Grafik integrasi kuadrat percepatan terhadap waktu ternormalisasi 0 s.d. 100%. Menggambarkan 
        laju akumulasi energi seismik selama getaran gempa berlangsung.</p>
        <p>&bull; <b>Intensitas Arias (Ia) & CAV:</b> Menghitung Intensitas Arias (Ia, m/s) dan Cumulative Absolute Velocity (CAV, g&middot;s). 
        Ambang batas <b>CAV &ge; 0.16 g&middot;s</b> (EPRI 1988) digunakan sebagai indikator skrining inspeksi potensi kerusakan infrastruktur kritis.</p>
        <p>&bull; <b>Durasi Signifikan Trifunac-Brady:</b> Menghitung interval waktu akumulasi energi <b>D<sub>5-95</sub></b> 
        (5% hingga 95% total energi Ia) dan <b>D<sub>5-75</sub></b> yang merefleksikan durasi efektif getaran kuat.</p>
        <p>&bull; <b>Rasio Kinematika V<sub>max</sub> / A<sub>max</sub>:</b> Mengindikasikan periode dominan getaran tanah dan efek tapak lokal (site effect).</p>

        <p><b>4.2 Tab 5 - Estimasi Intensitas Instrumental MMI (Intensity Tab)</b></p>
        <p>Tab Intensity memetakan parameter fisik gerakan tanah ke skala makroseismik instrumental objektif:</p>
        <p>&bull; <b>Persamaan GMICE Worden et al. (2012):</b> Mengestimasi MMI instrumental dari komponen horizontal maksimum (Max-H) PGA dan PGV.</p>
        <p>&bull; <b>Transisi Dominansi Kinematika:</b> Nilai PGA mengontrol estimasi guncangan rendah (MMI &lt; 5.0), sedangkan PGV mendominasi guncangan kuat (MMI &ge; 5.0) sesuai standar USGS ShakeMap.</p>
        <p>&bull; <b>Tabel Deskripsi Makroseismik:</b> Menguraikan dampak getaran tipikal dan reaksi persepsi manusia berdasarkan intensitas instrumental 
        (bukan pengganti survei inspeksi kerusakan struktural lapangan pascagempa).</p>

        <p><b>4.3 Tab 6 - Spektrum Respons Desain Elastis SDOF (Spectrum Tab)</b></p>
        <p>Tab Spectrum menyajikan evaluasi spektrum respon percepatan terhadap standar rancang bangunan gedung:</p>
        <p>&bull; <b>Kurva Spektrum Respons Pseudo-Acceleration (PSA 5%):</b> Menghitung respons percepatan maksimum osilator SDOF teredam 5% 
        pada rentang periode alami struktur T = 0.01 s hingga 10.0 s.</p>
        <p>&bull; <b>Overlay Target Spektrum Desain SNI 1726:2019:</b> Menampilkan kurva spektrum desain SNI 1726:2019 (Tanah Lunak SE, Sedang SD, Keras SC) 
        sebagai tolok ukur permintaan elastis guna mendeteksi potensi bahaya resonansi struktur gedung.</p>
        <p>&bull; <b>SDOF Solver Benchmark:</b> Panel validasi numerik yang membandingkan kurva spektrum Nigam-Jennings dan Newmark-Beta secara berdampingan.</p>

        <p><b>4.4 Tab 7 - Ekspor Laporan & Pengiriman Data (Report Tab)</b></p>
        <p>Tab Report menyediakan sarana pengunduhan deliverables rekayasa kegempaan yang lengkap:</p>
        <p>&bull; <b>Unduh Laporan Teknik PDF Publikasi:</b> Dokumen PDF 2-halaman berformat teknik terstandar yang memuat ringkasan metadata kejadian, 
        kartu kinematika, grafik 3-komponen, kurva Husid, parameter energi, spektrum respons, dan ringkasan QC.</p>
        <p>&bull; <b>Unduh CSV Kinematika:</b> Tabel spreadsheet deret waktu akselerasi, kecepatan, dan perpindahan terkoreksi.</p>
        <p>&bull; <b>Unduh CSV Spektrum Respons:</b> Tabel nilai diskret Periode T versus ordinat spektral PSA untuk analisis struktur sipil.</p>
        <p>&bull; <b>Unduh Paket Arsip ZIP Lengkap:</b> Berkas kompresi yang menyatukan seluruh laporan PDF, tabel CSV, dan log JSON komputasi.</p>
        """
        if lang == "id" else
        """
        <p><b>4.1 Tab 4 - Strong Ground Motion & Seismic Energy (Strong Motion Tab)</b></p>
        <p>The Strong Motion tab analyzes vibrational energy expenditure and structural demand potential:</p>
        <p>&bull; <b>Husid Energy Accumulation Plot:</b> Plots normalized squared acceleration integrals (0 to 100%) tracking energy build-up over time.</p>
        <p>&bull; <b>Arias Intensity (Ia) & CAV:</b> Quantifies total Arias Intensity (Ia, m/s) and Cumulative Absolute Velocity (CAV, g&middot;s). 
        The <b>CAV &ge; 0.16 g&middot;s</b> threshold (EPRI 1988) serves as an engineering screening criterion for potential structural damage inspection.</p>
        <p>&bull; <b>Trifunac-Brady Significant Duration:</b> Calculates bracketed energy intervals <b>D<sub>5-95</sub></b> 
        (5% to 95% total Arias energy) and <b>D<sub>5-75</sub></b> reflecting the damaging phase duration of ground motion.</p>
        <p>&bull; <b>Kinematic Ratio V<sub>max</sub> / A<sub>max</sub>:</b> Evaluates peak velocity to peak acceleration, indexing dominant periods and site effects.</p>

        <p><b>4.2 Tab 5 - Instrumental Intensity & Macroseismic Interpretation (Intensity Tab)</b></p>
        <p>The Intensity tab translates physical ground kinematics into standardized macroseismic shaking indices:</p>
        <p>&bull; <b>Worden et al. (2012) GMICE:</b> Derives instrumental MMI objectively from maximum horizontal (Max-H) PGA and PGV components.</p>
        <p>&bull; <b>Kinematic Transition Rule:</b> PGA governs lower shaking levels (MMI &lt; 5.0), whereas PGV governs higher damage levels (MMI &ge; 5.0) in adherence with USGS ShakeMap standards.</p>
        <p>&bull; <b>Macroseismic Impact Reference:</b> Details typical shaking perception and macroseismic effects 
        (objective ground shaking estimation, not a substitute for on-site post-earthquake structural building inspection).</p>

        <p><b>4.3 Tab 6 - SDOF Elastic Response Spectra & Code Compliance (Spectrum Tab)</b></p>
        <p>The Spectrum tab bridges seismology and structural building code compliance:</p>
        <p>&bull; <b>5% Damped Pseudo-Spectral Acceleration (PSA):</b> Solves peak response across natural periods T = 0.01 s to 10.0 s.</p>
        <p>&bull; <b>SNI 1726:2019 Design Code Overlay:</b> Overlays national building code target spectra for Soft Soil (SE), Medium Soil (SD), 
        and Hard Soil (SC) to benchmark elastic demand and identify potential building resonance risks.</p>
        <p>&bull; <b>SDOF Solver Benchmark:</b> Side-by-side numerical comparison auditing Nigam-Jennings vs. Newmark-Beta solver consistency.</p>

        <p><b>4.4 Tab 7 - Engineering Deliverables & Reporting (Report Tab)</b></p>
        <p>The Report tab provides automated export tools for operational and academic deliverables:</p>
        <p>&bull; <b>Publication-Grade PDF Report:</b> High-resolution 2-page vector PDF encapsulating event metadata, kinematic metrics, 
        waveforms, Husid curves, response spectra, and QC audits ready for rapid briefing.</p>
        <p>&bull; <b>Kinematic CSV Export:</b> Full corrected time series of acceleration, velocity, and displacement in tabular format.</p>
        <p>&bull; <b>Spectral Response CSV Export:</b> Discrete period T vs. PSA tables for structural engineering dynamic software ingestion.</p>
        <p>&bull; <b>Complete ZIP Archive:</b> Single-click bundle containing all PDF reports, CSV datasets, and JSON computation logs.</p>
        """
    )
    builder.insert_html_safe(p8, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 490), c4_text)

    callout_c4 = (
        "Integritas Laporan PDF: Laporan teknis yang dihasilkan pada Tab 7 dibangun secara langsung menggunakan pustaka PyMuPDF (fitz) "
        "sehingga menghasilkan grafik berbasis vektor resolusi tinggi yang tajam pada semua tingkat perbesaran. Format laporan "
        "diselaraskan dengan kaidah pelaporan akselerogram rekayasa seismologi dan koordinasi tanggap darurat bencana."
        if lang == "id" else
        "PDF Report Integrity: Technical reports compiled in Tab 7 are rendered natively using PyMuPDF (fitz) into vector-grade PDF documents "
        "that remain razor-sharp at arbitrary zoom levels. The format aligns with standard accelerometric reporting conventions adopted in "
        "engineering seismological practice and disaster mitigation coordination."
    )
    builder.draw_callout(p8, pymupdf.Rect(LEFT_X, y + 500, RIGHT_X, y + 560),
                         "Standar Laporan Teknis Seismologi Rekayasa" if lang == "id" else "Standardized Engineering Technical Deliverables",
                         callout_c4, callout_type="success")

    # -------------------------------------------------------------
    # PAGE 9 (Halaman 5) - BAB V: PANDUAN MODE BATCH MULTI-STASIUN
    # -------------------------------------------------------------
    p9 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p9)
    builder.add_page_header_footer(p9, "BAB V - MODE BATCH" if lang == "id" else "CHAPTER V - BATCH PROCESSING",
                                   f"Halaman 5 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 5 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p9, "BAB V: PANDUAN MODE PEMROSESAN BATCH MULTI-STASIUN REGIONAL" if lang == "id" else "CHAPTER V: REGIONAL MULTI-STATION BATCH PROCESSING GUIDE")

    c5_text = (
        """
        <p><b>5.1 Urgensi & Alur Pemrosesan Jaringan Gempa Regional</b></p>
        <p>Ketika terjadi gempa bumi signifikan di wilayah Indonesia, sinyal getaran terekam di puluhan stasiun 
        akselerograf BMKG yang tersebar pada radius ratusan kilometer. Pemrosesan manual satu-per-satu memakan waktu yang lama dan 
        rawan inkonsistensi parameter. <b>Mode Pemrosesan Batch Multi-Stasiun</b> pada BSMA v2.0.0 mengotomatisasi 
        pengolahan kumpulan berkas stasiun secara sekuensial, terisolasi per stasiun, dan menyusun matriks komparasi regional.</p>

        <p><b>5.2 Tata Cara Pengunggahan Massal Berkas (Bulk Upload Procedure)</b></p>
        <p>&bull; <b>Aktivasi Mode Batch:</b> Pada sidebar kontrol atau panel navigasi atas, aktifkan mode <b>'Batch Processing Mode'</b>.</p>
        <p>&bull; <b>Unggah Kumpulan Berkas:</b> Seret (<i>drag-and-drop</i>) seluruh berkas rekaman MiniSEED (.mseed) atau SAC (.sac) dari 
        berbagai stasiun sekaligus ke area uploader massal. Sistem memproses setiap rekaman dengan batas alokasi memori yang terjaga.</p>
        <p>&bull; <b>Asosiasi Metadata Bersama:</b> Pengguna dapat menetapkan parameter filter seragam (misalnya f<sub>min</sub>=0.05 Hz, 
        f<sub>max</sub>=40.0 Hz) yang diterapkan konsisten ke seluruh stasiun guna menjamin komparabilitas data.</p>

        <p><b>5.3 Eksekusi Sekuensial & Proteksi Isolasi Kegagalan (Failure Isolation Architecture)</b></p>
        <p>Arsitektur komputasi batch BSMA v2.0.0 dilengkapi sistem <b>proteksi isolasi kegagalan (failure isolation)</b>:</p>
        <p>&bull; Setiap stasiun diproses dalam blok komputasi terisolasi (<i>isolated try-except container</i>).</p>
        <p>&bull; Apabila terdapat berkas stasiun yang korup, terpotong, memiliki laju cuplik tidak seragam, atau mengalami saturasi ekstrem, 
        sistem menandai stasiun tersebut dengan status <b>FAILED</b> tanpa menghentikan pemrosesan stasiun-stasiun lainnya.</p>
        <p>&bull; Bilah kemajuan (progress bar) menampilkan status penyelesaian per stasiun secara real-time.</p>

        <p><b>5.4 Navigasi & Analisis Tabel Matriks Komparasi Regional</b></p>
        <p>Setelah komputasi selesai, sistem menampilkan <b>Tabel Matriks Komparasi Regional</b> yang komprehensif:</p>
        <p>&bull; <b>Kolom Tabel:</b> Kode Stasiun, Kode Jaringan, Komponen Terbesar, Nilai PGA (m/s&sup2; dan Gal), PGV (cm/s), PGD (cm), 
        Intensitas Arias Ia (m/s), Durasi D<sub>5-95</sub> (s), Estimasi Intensitas MMI Instrumental, Skor QC (%), dan Status Kelayakan.</p>
        <p>&bull; <b>Fitur Pengurutan Interaktif:</b> Pengguna dapat mengklik judul kolom untuk mengurutkan stasiun berdasarkan 
        nilai PGA tertinggi atau MMI terparah untuk segera mengidentifikasi zona wilayah yang mengalami guncangan terkuat.</p>

        <p><b>5.5 Ekspor Tabular Regional & Integrasi GIS / Pemetaan</b></p>
        <p>Di bawah tabel matriks komparasi, tersedia tombol <b>'Export Regional Summary CSV'</b>. Berkas CSV yang diunduh terformat rapi 
        sehingga dapat langsung diimpor ke perangkat lunak Sistem Informasi Geografis (ArcGIS, QGIS) maupun skrip analisis regional 
        untuk pembuatan peta kontur iso-seismal dan sebaran percepatan tanah secara terstruktur.</p>
        """
        if lang == "id" else
        """
        <p><b>5.1 Regional Strong-Motion Network Processing Urgency</b></p>
        <p>When a significant earthquake occurs across the Indonesian archipelago, ground motions are captured by dozens of 
        BMKG accelerograph stations across hundreds of kilometers. Manual station-by-station processing is time-prohibitive during rapid review. 
        The <b>Regional Multi-Station Batch Processing Engine</b> in BSMA v2.0.0 automates sequential ingestion, quality control, 
        and kinematic analysis across station networks with deterministic memory handling.</p>

        <p><b>5.2 Bulk Upload Operational Procedure</b></p>
        <p>&bull; <b>Activate Batch Mode:</b> Toggle the <b>'Batch Processing Mode'</b> switch on the sidebar or top command panel.</p>
        <p>&bull; <b>Multi-File Drop Zone:</b> Drag-and-drop multiple MiniSEED (.mseed) or SAC (.sac) files simultaneously into the batch uploader. 
        The system processes multi-station records sequentially with bounded memory overhead and isolated error handling.</p>
        <p>&bull; <b>Uniform DSP Configuration:</b> Apply unified filter corner frequencies (e.g., f<sub>min</sub>=0.05 Hz, f<sub>max</sub>=40.0 Hz) 
        across all records to ensure regional parameter consistency.</p>

        <p><b>5.3 Sequential Execution & Fault Isolation Architecture</b></p>
        <p>The batch pipeline incorporates rigorous <b>fault isolation boundaries</b>:</p>
        <p>&bull; Each station trace is analyzed within an independent computational container (try-except block).</p>
        <p>&bull; Corrupt, clipped, or truncated waveforms are flagged with a <b>FAILED</b> badge without interrupting or aborting execution 
        for remaining operational stations.</p>
        <p>&bull; An interactive progress bar renders live per-station completion feedback.</p>

        <p><b>5.4 Regional Comparison Matrix Navigation & Sorting</b></p>
        <p>Upon execution, the engine compiles a unified <b>Regional Comparison Matrix Table</b>:</p>
        <p>&bull; <b>Matrix Attributes:</b> Station Code, Network, Dominant Component, PGA (m/s&sup2; & Gal), PGV (cm/s), PGD (cm), 
        Arias Ia (m/s), Duration D<sub>5-95</sub> (s), Instrumental MMI Intensity, QC Score (%), and Audit Status.</p>
        <p>&bull; <b>Dynamic Column Sorting:</b> Click column headers to sort stations by peak PGA or maximum MMI intensity, 
        instantly identifying the most severely shaken geographic areas.</p>

        <p><b>5.5 Regional CSV Export & GIS Mapping Integration</b></p>
        <p>The <b>'Export Regional Summary CSV'</b> button exports the complete multi-station matrix into a structured tabular CSV. 
        The file is structured for seamless ingestion into Geographic Information Systems (QGIS, ArcGIS) and seismic contouring scripts 
        for regional ground-motion mapping.</p>
        """
    )
    builder.insert_html_safe(p9, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 490), c5_text)

    callout_c5 = (
        "Alur Kerja Pemrosesan Berkas Jamak (Batch Pipeline): Mesin batch memproses direktori akselerogram multi-stasiun secara sekuensial "
        "dengan penanganan galat mandiri (per-station fault isolation). Waktu eksekusi berlangsung cepat dan terukur sesuai panjang rekaman "
        "serta laju cuplik data, menghasilkan matriks komparasi regional terpadu guna mendukung analisis sebaran guncangan secara efisien."
        if lang == "id" else
        "High-Throughput Batch Processing Workflow: The batch engine processes multi-station accelerogram directories sequentially "
        "with per-station exception isolation. Processing runtime scales efficiently with trace duration and sampling rate, "
        "delivering a comprehensive regional comparison matrix to support rapid spatial shaking assessment."
    )
    builder.draw_callout(p9, pymupdf.Rect(LEFT_X, y + 500, RIGHT_X, y + 560),
                         "Alur Kerja Batch Multi-Stasiun Regional" if lang == "id" else "Regional Multi-Station Batch Workflow",
                         callout_c5, callout_type="success")

    # -------------------------------------------------------------
    # PAGE 10 (Halaman 6) - BAB VI: ARSITEKTUR PIPELINE & DEKONVOLUSI
    # -------------------------------------------------------------
    p10 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p10)
    builder.add_page_header_footer(p10, "BAB VI - ARSITEKTUR PIPELINE" if lang == "id" else "CHAPTER VI - PIPELINE ARCHITECTURE",
                                   f"Halaman 6 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 6 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p10, "BAB VI: ARSITEKTUR PIPELINE & DEKONVOLUSI RESPON INSTRUMEN" if lang == "id" else "CHAPTER VI: PIPELINE ARCHITECTURE & INSTRUMENT DECONVOLUTION")

    diagram_bytes = create_pipeline_diagram(lang=lang)
    p10.insert_image(pymupdf.Rect(LEFT_X + 25, y, RIGHT_X - 25, y + 115), stream=diagram_bytes)
    y_content = y + 125

    c6_text = (
        """
        <p><b>6.1 Alur Pemrosesan Seismik End-to-End</b></p>
        <p>Arsitektur komputasi BSMA v2.0.0 mengimplementasikan pipa data deterministik 5-tahap: (1) Ingesti berkas rekaman akselerograf 
        mentah dan metadata stasiun, (2) Dekonvolusi respon instrumen (koreksi transfer function sensor), (3) Penapisan sinyal digital 
        dan koreksi baseline, (4) Integrasi kinematika ganda (kecepatan & perpindahan), serta (5) Pemodelan parameter energi dan spektrum respons.</p>

        <p><b>6.2 Teori Dekonvolusi Fungsi Alih Respon Sensor (Poles and Zeros)</b></p>
        <p>Pada rekaman seismometer dan akselerometer mentah berdimensi hitungan digital (<i>raw digital counts</i>), sinyal keluaran instrumen 
        y(t) merupakan konvolusi getaran tanah sejati x(t) dengan respon impuls sensor h(t). Dalam domain Laplace, hubungan ini dinyatakan:</p>
        """
        if lang == "id" else
        """
        <p><b>6.1 End-to-End Deterministic Pipeline</b></p>
        <p>The BSMA v2.0.0 engine employs a rigorous 5-stage sequential workflow: (1) Raw waveform ingestion and StationXML parsing, 
        (2) Instrument transfer function deconvolution, (3) Digital signal filtering and baseline correction, (4) Kinematic double integration, 
        and (5) Seismic energy and elastic response spectra computation.</p>

        <p><b>6.2 Instrument Transfer Function Deconvolution (Poles and Zeros)</b></p>
        <p>For raw accelerograms recorded in digital counts, output signal y(t) represents the mathematical convolution of true ground 
        acceleration x(t) with the sensor impulse response h(t). In the Laplace domain:</p>
        """
    )
    builder.insert_html_safe(p10, pymupdf.Rect(LEFT_X, y_content, RIGHT_X, y_content + 105), c6_text)

    y_form = y_content + 110
    builder.draw_formula_card(
        p10, pymupdf.Rect(LEFT_X, y_form, RIGHT_X, y_form + 58),
        r"H(s) = \frac{Y(s)}{X(s)} = S_0 \cdot \frac{\prod_{i=1}^M (s - z_i)}{\prod_{j=1}^N (s - p_j)}",
        title_str="Fungsi Alih Respon Instrumen / Transfer Function H(s)" if lang == "id" else "Instrument Response Transfer Function H(s)",
        fontsize=13.5
    )

    y_bot = y_form + 66
    c6_text_bot = (
        """
        <p>Di mana <i>S<sub>0</sub></i> merepresentasikan faktor skala sensitivitas keseluruhan (perkalian gain sensitivitas sensor dalam V/(m/s&sup2;) 
        dengan faktor konversi digitalizer dalam counts/V), <i>z<sub>i</sub></i> adalah pembuat nol (<i>zeros</i>, M suku), dan <i>p<sub>j</sub></i> 
        adalah kutub (<i>poles</i>, N suku). Dekonvolusi instrumen dilakukan melalui domain frekuensi dengan membagi spektrum Fourier sinyal tercatat 
        terhadap fungsi alih H(&omega;). Sistem menerapkan jendela penapis <i>water-level</i> dan tapering cosinus pada batas frekuensi sudut 
        guna mencegah penguatan derau frekuensi tinggi tak terkendali saat membalikkan respon transduser.</p>

        <p><b>6.3 Kebijakan Bypass Kalibrasi untuk Data Percepatan Fisik (Physical Acceleration)</b></p>
        <p>Sebagian besar data gerakan tanah kuat operasional BMKG yang didistribusikan dari sistem pra-pengolahan stasiun lapangan telah 
        dikonversi secara otomatis ke satuan percepatan fisik (m/s&sup2; atau Gal). Pada kondisi ini, penerapan dekonvolusi instrumen 
        ulang akan menimbulkan distorsi over-correction yang merusak bentuk sinyal. Algoritma BSMA v2.0.0 secara cerdas mendeteksi 
        status provenance rekaman dan memberlakukan <i>calibration bypass protocol</i> untuk menjamin integritas saintifik data.</p>
        """
        if lang == "id" else
        """
        <p>Where <i>S<sub>0</sub></i> represents the composite sensitivity normalization factor (product of transducer generator constant 
        in V/(m/s&sup2;) and digitizer sensitivity in counts/V), <i>z<sub>i</sub></i> are transfer zeros (M terms), and <i>p<sub>j</sub></i> are 
        transfer poles (N terms). Deconvolution divides the recorded Fourier spectrum by H(&omega;) in the frequency domain. The engine 
        applies water-level cosine frequency tapering near corner boundaries to suppress high-frequency noise amplification during response inversion.</p>

        <p><b>6.3 Calibration Bypass Policy for Pre-Calibrated Physical Acceleration Records</b></p>
        <p>Many operational strong-motion records distributed across BMKG networks are pre-calibrated by dataloggers directly into physical 
        acceleration units (m/s&sup2; or Gal). Applying pole-zero deconvolution to pre-calibrated accelerograms induces severe over-correction 
        and waveform distortion. The BSMA v2.0.0 engine automatically verifies provenance metadata and applies a deterministic calibration bypass 
        protocol, preserving rigorous physical ground motion amplitude integrity.</p>
        """
    )
    builder.insert_html_safe(p10, pymupdf.Rect(LEFT_X, y_bot, RIGHT_X, y_bot + 175), c6_text_bot)

    callout_c6 = (
        "Integritas Dekonvolusi: Jika berkas masukan berdimensi digital counts mentah, unggah berkas StationXML pendamping yang valid. "
        "Sistem akan memverifikasi kesesuaian stasiun, jaringan, dan kode kanal sebelum mengeksekusi dekonvolusi pole-zero."
        if lang == "id" else
        "Deconvolution Integrity: If processing raw counts, provide a valid companion StationXML file. The engine validates matching "
        "network, station, and channel codes prior to executing pole-zero deconvolution."
    )
    builder.draw_callout(p10, pymupdf.Rect(LEFT_X, y_bot + 185, RIGHT_X, y_bot + 245),
                         "Standar Validasi Dekonvolusi Respon Sensor" if lang == "id" else "Sensor Response Deconvolution Validation Standard",
                         callout_c6, callout_type="info")

    # -------------------------------------------------------------
    # PAGE 11 (Halaman 7) - BAB VII: QUALITY CONTROL & DSP
    # -------------------------------------------------------------
    p11 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p11)
    builder.add_page_header_footer(p11, "BAB VII - DSP & FILTERING" if lang == "id" else "CHAPTER VII - DSP & FILTERING",
                                   f"Halaman 7 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 7 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p11, "BAB VII: QUALITY CONTROL & DIGITAL SIGNAL PROCESSING (DSP)" if lang == "id" else "CHAPTER VII: QUALITY CONTROL & DIGITAL SIGNAL PROCESSING (DSP)")

    diag_filter_bytes = create_filter_diagram(lang=lang)
    p11.insert_image(pymupdf.Rect(LEFT_X + 25, y, RIGHT_X - 25, y + 130), stream=diag_filter_bytes)
    y_content = y + 138

    c7_text = (
        """
        <p><b>7.1 Metodologi Pre-Event Noise Windowing & Estimasi Rasio Sinyal terhadap Derau (SNR)</b></p>
        <p>Integritas rekaman akselerograf dievaluasi secara otomatis sebelum pemrosesan utama. Algoritma QC mengidentifikasi 
        jendela waktu hening sebelum gelombang P (<i>pre-event noise window</i>, durasi 5-10 detik) dan jendela fase gempa kuat 
        (<i>strong-motion event window</i>). Nilai Root Mean Square (RMS) dari kedua jendela dihitung untuk menentukan rasio SNR dalam desibel (dB):</p>
        """
        if lang == "id" else
        """
        <p><b>7.1 Pre-Event Noise Windowing & Signal-to-Noise Ratio (SNR) Estimation</b></p>
        <p>Signal integrity is audited prior to downstream processing. The QC engine extracts a quiet pre-event noise window (5-10 seconds) 
        and the strong-motion event window. The Root Mean Square (RMS) amplitudes determine the pre-event Signal-to-Noise Ratio (SNR) in decibels (dB):</p>
        """
    )
    builder.insert_html_safe(p11, pymupdf.Rect(LEFT_X, y_content, RIGHT_X, y_content + 60), c7_text)

    y_form1 = y_content + 64
    builder.draw_formula_card(
        p11, pymupdf.Rect(LEFT_X, y_form1, RIGHT_X, y_form1 + 54),
        r"\mathrm{SNR}_{\mathrm{dB}} = 20 \log_{10} \left( \frac{\mathrm{RMS}_{\mathrm{event}}}{\mathrm{RMS}_{\mathrm{noise}}} \right)",
        title_str="Rasio Sinyal terhadap Derau / Signal-to-Noise Ratio (SNR)" if lang == "id" else "Signal-to-Noise Ratio (SNR)",
        fontsize=13.5
    )

    y_content2 = y_form1 + 62
    c7_text_mid = (
        """
        <p><b>7.2 Desain Filter IIR Butterworth Orde-4 Zero-Phase Second-Order Sections (SOS)</b></p>
        <p>Filter IIR Butterworth dipilih karena memiliki karakteristik respons magnitudo yang <i>maximally flat</i> pada passband 
        (tanpa riak/ripple) serta atenuasi monotonik di luar frekuensi sudut. Untuk meniadakan distorsi fase (<i>phase distortion</i>) 
        yang dapat menggeser waktu puncak sinyal, BSMA mengimplementasikan penapisan dua arah maju-mundur (<i>forward-backward</i> via 
        <code>scipy.signal.sosfiltfilt</code>) berbasis Second-Order Sections (SOS) orde-4 numerik stabil dengan respon magnitudo:</p>
        """
        if lang == "id" else
        """
        <p><b>7.2 Zero-Phase 4th-Order Butterworth IIR Second-Order Sections (SOS) Filter</b></p>
        <p>The Butterworth IIR architecture is selected for its maximally flat passband magnitude response (free of passband ripples) 
        and smooth monotonic roll-off. To eliminate phase distortion that would shift kinematic peak timings, BSMA implements 
        two-pass forward-backward filtering (<code>scipy.signal.sosfiltfilt</code>) configured in Second-Order Sections (SOS) 
        with magnitude squared response:</p>
        """
    )
    builder.insert_html_safe(p11, pymupdf.Rect(LEFT_X, y_content2, RIGHT_X, y_content2 + 55), c7_text_mid)

    y_form2 = y_content2 + 58
    builder.draw_formula_card(
        p11, pymupdf.Rect(LEFT_X, y_form2, RIGHT_X, y_form2 + 54),
        r"|H(f)|^2 = \frac{1}{1 + \left( \frac{f}{f_c} \right)^{2n}}, \quad n = 4 \quad (\text{two-pass order 8})",
        title_str="Respon Frekuensi Magnitudo Filter Butterworth / Frequency Response" if lang == "id" else "Butterworth Magnitude Response",
        fontsize=13.0
    )

    y_bot7 = y_form2 + 62
    c7_text_bot = (
        """
        <p><b>7.3 Perlindungan Frekuensi Nyquist Adaptif (Nyquist Floor Safeguard)</b></p>
        <p>Sesuai teorema Shannon-Nyquist, batas atas pemrosesan digital dibatasi oleh frekuensi Nyquist f<sub>nyq</sub> = f<sub>s</sub> / 2. 
        Untuk menghindari distorsi aliasing dan resonansi kutub numerik di dekat batas pita cuplik, sistem memberlakukan perlindungan 
        otomatis di mana batas atas frekuensi filter lolos rendah dipangkas maksimal pada <b>f<sub>max</sub> &le; 0.40 f<sub>s</sub></b>.</p>
        """
        if lang == "id" else
        """
        <p><b>7.3 Adaptive Nyquist Frequency Floor Safeguard</b></p>
        <p>Per the Shannon-Nyquist sampling theorem, digital signal processing is strictly bounded by f<sub>nyq</sub> = f<sub>s</sub> / 2. 
        To prevent aliasing distortion and high-frequency numerical pole instabilities near the sampling boundary, BSMA enforces 
        an automatic cutoff ceiling of <b>f<sub>max</sub> &le; 0.40 f<sub>s</sub></b>.</p>
        """
    )
    builder.insert_html_safe(p11, pymupdf.Rect(LEFT_X, y_bot7, RIGHT_X, y_bot7 + 60), c7_text_bot)

    callout_c7 = (
        "Prinsip Penapisan Bebas Distorsi Fase: Penapisan dua arah maju-mundur (forward-backward) menghasilkan pergeseran fase neto nol "
        "(zero net phase distortion, theta(f) = 0). Waktu kedatangan fase seismik dan simetri bentuk gelombang tetap terjaga tanpa "
        "distorsi perlambatan grup (group delay). Operator disarankan memilih frekuensi sudut lolos-tinggi di bawah frekuensi pulsa gempa "
        "guna meminimalkan potensi osilasi awal semu (acausal pre-cursor ringing) pada rekaman dekat-sumber."
        if lang == "id" else
        "Zero-Phase Filtering Principles: Forward-backward bidirectional filtering eliminates net phase lag identically "
        "(zero net phase distortion, theta(f) = 0). Seismic phase arrival times and waveform symmetry are preserved without group-delay "
        "distortion. Operators should select high-pass corner frequencies safely below dominant pulse frequencies to minimize potential "
        "acausal pre-cursor ringing on impulsive near-fault records."
    )
    builder.draw_callout(p11, pymupdf.Rect(LEFT_X, y_bot7 + 68, RIGHT_X, y_bot7 + 138),
                         "Prinsip Penapisan Bebas Distorsi Fase (Zero-Phase SOS)" if lang == "id" else "Zero-Phase SOS Filtering Principles",
                         callout_c7, callout_type="success")

    # -------------------------------------------------------------
    # PAGE 12 (Halaman 8) - BAB VIII: INTEGRASI KINEMATIKA & BASELINE
    # -------------------------------------------------------------
    p12 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p12)
    builder.add_page_header_footer(p12, "BAB VIII - INTEGRASI KINEMATIKA" if lang == "id" else "CHAPTER VIII - KINEMATICS",
                                   f"Halaman 8 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 8 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p12, "BAB VIII: INTEGRASI KINEMATIKA & KEBIJAKAN GARIS DASAR" if lang == "id" else "CHAPTER VIII: KINEMATIC INTEGRATION & BASELINE POLICY")

    diag_boore_bytes = create_boore_baseline_diagram(lang=lang)
    p12.insert_image(pymupdf.Rect(LEFT_X + 25, y, RIGHT_X - 25, y + 140), stream=diag_boore_bytes)
    y_content = y + 148

    c8_text = (
        """
        <p><b>8.1 Masalah Pergeseran Garis Dasar (Baseline Drift) pada Akselerogram</b></p>
        <p>Integrasi langsung dari deret waktu percepatan a(t) menghasilkan riwayat kecepatan v(t) dan perpindahan d(t). Namun, rekaman 
        akselerograf mentah hampir selalu mengandung offset arus searah (DC bias) kecil dan derau frekuensi rendah akibat kemiringan 
        dinamik sensor (<i>sensor tilt</i>), rotasi tanah lokal pada pondasi instrumen, histeresis mekanis pegas penyeimbang, atau fluktuasi 
        termal konverter analog-ke-digital (ADC). Saat diintegralkan dua kali, galat offset konstan a<sub>0</sub> terakumulasi menjadi 
        penyimpangan kuadratik ekstrem pada perpindahan (&frac12; a<sub>0</sub> t&sup2;), sehingga kurva perpindahan melengkung tak terhingga 
        dan kehilangan makna fisiknya (Boore 2001; Boore & Bommer 2005).</p>

        <p><b>8.2 Skema Integrasi Trapesium Kumulatif Terkoreksi</b></p>
        <p>BSMA v2.0.0 menerapkan skema integrasi trapesium kumulatif bertahap yang diawali dengan pengurangan nilai rata-rata derau awal 
        (<i>pre-event mean subtraction</i>) dan detrending polinomial orde rendah, dilanjutkan penapisan lolos tinggi f<sub>min</sub> &ge; 0.05 Hz:</p>
        """
        if lang == "id" else
        """
        <p><b>8.1 The Baseline Drift Phenomenon in Accelerograms</b></p>
        <p>Integrating ground acceleration a(t) yields ground velocity v(t) and displacement d(t). However, raw accelerograms inevitably 
        contain subtle DC offsets and long-period transducer noise induced by dynamic sensor tilting, local soil rotations beneath instrument 
        pads, mechanical suspension hysteresis, or thermal drift in analog-to-digital converters (ADCs). Upon double integration, an infinitesimal 
        acceleration offset a<sub>0</sub> compounds into an unphysical quadratic parabolic divergence in displacement (&frac12; a<sub>0</sub> t&sup2;), 
        obscuring true ground displacement physics (Boore 2001; Boore & Bommer 2005).</p>

        <p><b>8.2 Corrected Cumulative Trapezoidal Integration Workflow</b></p>
        <p>BSMA v2.0.0 implements a robust multi-stage cumulative trapezoidal integration protocol combined with pre-event mean subtraction, 
        low-order polynomial baseline detrending, and zero-phase highpass filtering bounded by f<sub>min</sub> &ge; 0.05 Hz:</p>
        """
    )
    builder.insert_html_safe(p12, pymupdf.Rect(LEFT_X, y_content, RIGHT_X, y_content + 105), c8_text)

    y_form8 = y_content + 110
    builder.draw_formula_card(
        p12, pymupdf.Rect(LEFT_X, y_form8, RIGHT_X, y_form8 + 56),
        r"v(t) = v_0 + \int_0^t a(\tau) \, d\tau, \quad d(t) = d_0 + \int_0^t v(\tau) \, d\tau",
        title_str="Persamaan Integrasi Kinematika Tanah / Kinematic Double Integration" if lang == "id" else "Kinematic Double Integration",
        fontsize=13.5
    )

    y_bot8 = y_form8 + 64
    c8_text_bot = (
        """
        <p><b>8.3 Pengendalian Galat Periode Panjang & Spektrum Perpindahan</b></p>
        <p>Kombinasi detrending dan filter lolos tinggi menjamin bahwa riwayat kecepatan dan perpindahan kembali berosilasi di sekitar garis 
        nol setelah getaran gempa mereda. Hal ini krusial untuk mencegah distorsi artifisial pada ordinat spektral periode panjang (T &gt; 2.0 s).</p>
        """
        if lang == "id" else
        """
        <p><b>8.3 Long-Period Error Control & Displacement Spectra</b></p>
        <p>Coupled baseline detrending and highpass filtering ensure velocity and displacement traces decay smoothly to zero after shaking terminates, 
        preventing artificial distortion in long-period spectral ordinates (T &gt; 2.0 s).</p>
        """
    )
    builder.insert_html_safe(p12, pymupdf.Rect(LEFT_X, y_bot8, RIGHT_X, y_bot8 + 50), c8_text_bot)

    callout_c8 = (
        "Batas Metodologi Saintifik: Nilai PGD (Peak Ground Displacement) yang dihasilkan melalui integrasi rekaman akselerograf "
        "merupakan perpindahan dinamik transient akibat perambatan gelombang seismik, BUKAN deformasi tektonik statis permanen "
        "(permanent tectonic fling/offset). Estimasi pergeseran statis permanen sesar membutuhkan pengamatan GNSS/GPS geodesi presisi tinggi."
        if lang == "id" else
        "Scientific Methodology Boundary: Peak Ground Displacement (PGD) derived via accelerogram integration represents transient dynamic "
        "wave vibrations, NOT permanent static tectonic fault offsets (fling-step). Estimating permanent tectonic ground deformation "
        "requires continuous high-rate geodetic GNSS/GPS observations."
    )
    builder.draw_callout(p12, pymupdf.Rect(LEFT_X, y_bot8 + 56, RIGHT_X, y_bot8 + 130),
                         "Batasan Metodologi: PGD Dinamik vs Deformasi Statis" if lang == "id" else "Methodology Boundary: Dynamic PGD vs Static Fling",
                         callout_c8, callout_type="warning")

    # -------------------------------------------------------------
    # PAGE 13 (Halaman 9) - BAB IX: PARAMETER GERAKAN KUAT & INTENSITAS
    # -------------------------------------------------------------
    p13 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p13)
    builder.add_page_header_footer(p13, "BAB IX - PARAMETER & ENERGI" if lang == "id" else "CHAPTER IX - KINEMATICS & ENERGY",
                                   f"Halaman 9 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 9 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p13, "BAB IX: PARAMETER KINEMATIKA, ENERGI SEISMIK, & INTENSITAS MMI" if lang == "id" else "CHAPTER IX: KINEMATIC PARAMETERS, SEISMIC ENERGY, & MMI")

    diag_gmice_bytes = create_gmice_diagram(lang=lang)
    p13.insert_image(pymupdf.Rect(LEFT_X + 25, y, RIGHT_X - 25, y + 130), stream=diag_gmice_bytes)
    y_content = y + 138

    c9_text = (
        """
        <p><b>9.1 Intensitas Arias (Ia), CAV, dan Kurva Husid Akumulasi Energi</b></p>
        <p>Kerusakan infrastruktur tidak hanya ditentukan oleh amplitudo percepatan puncak (PGA), melainkan juga durasi dan kandungan 
        energi getaran. <b>Intensitas Arias (Ia)</b> mengukur total energi getaran yang diserap struktur per satuan massa (Arias 1970). 
        <b>Cumulative Absolute Velocity (CAV)</b> merepresentasikan integral absolut percepatan yang digunakan oleh EPRI (1988) untuk kriteria 
        ambang batas operasi fasilitas rekayasa kritis nuklir dan industri (ambang batas kerusakan: CAV &ge; 0.16 g&middot;s):</p>
        """
        if lang == "id" else
        """
        <p><b>9.1 Arias Intensity (Ia), CAV, and Husid Energy Accumulation</b></p>
        <p>Structural damage correlates not only with peak ground acceleration (PGA) but also with vibrational energy and shaking duration. 
        <b>Arias Intensity (Ia)</b> quantifies total seismic energy absorbed per unit mass (Arias 1970). <b>Cumulative Absolute Velocity (CAV)</b> 
        represents the integral of absolute acceleration, employed by EPRI (1988) for damage threshold auditing across critical facilities (threshold: CAV &ge; 0.16 g&middot;s):</p>
        """
    )
    builder.insert_html_safe(p13, pymupdf.Rect(LEFT_X, y_content, RIGHT_X, y_content + 75), c9_text)

    y_form9 = y_content + 80
    builder.draw_formula_card(
        p13, pymupdf.Rect(LEFT_X, y_form9, RIGHT_X, y_form9 + 56),
        r"I_a = \frac{\pi}{2g} \int_0^{t_{\max}} [a(t)]^2 \, dt, \quad \mathrm{CAV} = \int_0^{t_{\max}} |a(t)| \, dt",
        title_str="Formulasi Intensitas Arias & CAV / Arias Intensity & CAV Formulations" if lang == "id" else "Arias Intensity & CAV Formulations",
        fontsize=13.5
    )

    y_mid9 = y_form9 + 64
    c9_text_mid = (
        """
        <p><b>9.2 Persamaan Konversi Intensitas Makroseismik Instrumental (GMICE)</b></p>
        <p>Untuk memetakan kinematika tanah ke skala Modified Mercalli Intensity (MMI), BSMA mengevaluasi komponen horisontal maksimum (Max-H) 
        mengacu formulasi bilinear piecewise Worden et al. (2012): PGA mengatur intensitas rendah (MMI &le; V), sedangkan PGV mendominasi intensitas tinggi (MMI &ge; VI):</p>
        """
        if lang == "id" else
        """
        <p><b>9.2 Ground-Motion Intensity Conversion Equations (GMICE)</b></p>
        <p>To map ground kinematics to Modified Mercalli Intensity (MMI), BSMA evaluates maximum horizontal motion (Max-H) 
        using Worden et al. (2012) bilinear piecewise GMICE: PGA governs lower intensities (MMI &le; V), while PGV dominates higher damage-correlated levels (MMI &ge; VI):</p>
        """
    )
    builder.insert_html_safe(p13, pymupdf.Rect(LEFT_X, y_mid9, RIGHT_X, y_mid9 + 45), c9_text_mid)

    y_form9b = y_mid9 + 48
    form_worden = (
        r"\mathrm{MMI}_{\mathrm{PGA}} = 1.78 + 1.55 \log_{10}(\mathrm{PGA})\ [\leq 37.15\ \mathrm{Gal}] \quad \vert \quad -1.60 + 3.70 \log_{10}(\mathrm{PGA})\ [> 37.15\ \mathrm{Gal}]" + "\n" +
        r"\mathrm{MMI}_{\mathrm{PGV}} = 3.78 + 2.99 \log_{10}(\mathrm{PGV})\ [\leq 3.39\ \mathrm{cm/s}] \quad \vert \quad 2.40 + 4.96 \log_{10}(\mathrm{PGV})\ [> 3.39\ \mathrm{cm/s}]"
    )
    builder.draw_formula_card(
        p13, pymupdf.Rect(LEFT_X, y_form9b, RIGHT_X, y_form9b + 66),
        form_worden,
        title_str="Regresi Bilinear GMICE Worden et al. (2012) Max-H (PGA & PGV)" if lang == "id" else "Bilinear GMICE Max-H Regressions (Worden et al. 2012)",
        fontsize=11.5
    )

    y_bot9 = y_form9b + 74
    callout_c9 = (
        "Aturan Transisi GMICE Skala MMI: Komputasi MMI menggunakan komponen horizontal terbesar (Max-H antara EW dan NS). "
        "Untuk intensitas rendah (MMI < 5.0), nilai ditentukan oleh PGA; untuk intensitas tinggi (MMI >= 5.0), sistem beralih ke PGV "
        "karena lebih berkorelasi dengan potensi regangan dan kerusakan struktur bangunan."
        if lang == "id" else
        "GMICE Transition Rule for Instrumental MMI: MMI is evaluated from the maximum horizontal component (Max-H between EW and NS). "
        "PGA governs lower intensities (MMI < 5.0), whereas PGV governs higher damage levels (MMI >= 5.0) reflecting structural strain."
    )
    builder.draw_callout(p13, pymupdf.Rect(LEFT_X, y_bot9, RIGHT_X, y_bot9 + 64),
                         "Klasifikasi Intensitas MMI Instrumental" if lang == "id" else "Instrumental MMI Classification",
                         callout_c9, callout_type="info")

    # -------------------------------------------------------------
    # PAGE 14 (Halaman 10) - BAB X: SPEKTRUM RESPONS SDOF & SNI 1726
    # -------------------------------------------------------------
    p14 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p14)
    builder.add_page_header_footer(p14, "BAB X - SPEKTRUM RESPONS" if lang == "id" else "CHAPTER X - RESPONSE SPECTRA",
                                   f"Halaman 10 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 10 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p14, "BAB X: PEMODELAN SPEKTRUM RESPONS SDOF & STANDAR SNI 1726:2019" if lang == "id" else "CHAPTER X: SDOF RESPONSE SPECTRA & SNI 1726:2019 CODE")

    diag_spec_bytes = create_spectrum_diagram(lang=lang)
    p14.insert_image(pymupdf.Rect(LEFT_X + 25, y, RIGHT_X - 25, y + 135), stream=diag_spec_bytes)
    y_content = y + 142

    c10_text = (
        """
        <p><b>10.1 Persamaan Diferensial Osilator SDOF Teredam 5%</b></p>
        <p>Spektrum respons merepresentasikan respon percepatan puncak dari serangkaian osilator linier berderajat kebebasan tunggal 
        (<i>Single Degree of Freedom - SDOF</i>) dengan redaman kritis &xi; = 0.05 (5%) yang mengalami eksitasi percepatan tanah &uuml;<sub>g</sub>(t):</p>
        """
        if lang == "id" else
        """
        <p><b>10.1 5% Damped SDOF Dynamic Equation of Motion</b></p>
        <p>Response spectra quantify peak structural acceleration across single-degree-of-freedom (SDOF) linear elastic oscillators 
        having critical damping ratio &xi; = 0.05 (5%) subjected to base acceleration &uuml;<sub>g</sub>(t):</p>
        """
    )
    builder.insert_html_safe(p14, pymupdf.Rect(LEFT_X, y_content, RIGHT_X, y_content + 45), c10_text)

    y_form10 = y_content + 48
    builder.draw_formula_card(
        p14, pymupdf.Rect(LEFT_X, y_form10, RIGHT_X, y_form10 + 58),
        r"\ddot{u}(t) + 2\xi\omega_n \dot{u}(t) + \omega_n^2 u(t) = -\ddot{u}_g(t), \quad \mathrm{PSA}(T, \xi) = \omega_n^2 \max_t |u(t)|",
        title_str="Persamaan Gerak Osilator SDOF & Pseudo-Spectral Acceleration" if lang == "id" else "SDOF Equation of Motion & PSA Formulation",
        fontsize=13.5
    )

    y_mid10 = y_form10 + 66
    c10_text_mid = (
        """
        <p><b>10.2 Komparasi Algoritma Numerik: Nigam-Jennings (1969) vs Newmark-Beta (1959)</b></p>
        <p>BSMA v2.0.0 menyediakan dua pilihan solver standar industri rekayasa gempa internasional:</p>
        <p>&bull; <b>Nigam-Jennings (1969):</b> Solusi rekursif analitik eksak dengan asumsi percepatan masukan linier bagian-per-bagian 
        (<i>piecewise linear</i>). Sangat unggul pada periode pendek (T &lt; 0.1 s) tanpa mengalami dispersi frekuensi buatan.</p>
        <p>&bull; <b>Newmark-Beta (1959):</b> Integrasi implisit percepatan rata-rata (&gamma; = 0.5, &beta; = 0.25). 
        Stabil tanpa syarat (unconditionally stable) dan menjadi tolok ukur klasik analisis dinamika struktur.</p>
        <p>&bull; <b>Benchmark Konkordansi:</b> Kedua solver divalidasi silang pada 100 titik periode (T = 0.01 - 10.0 s) dengan kesesuaian sangat tinggi: 
        selisih relatif rata-rata &lt; 0.02%, selisih relatif maks &lt; 0.08%, dan RMS selisih &lt; 1.5 &times; 10<sup>-4</sup> g.</p>

        <p><b>10.3 Tolok Ukur Spektrum Desain SNI 1726:2019</b></p>
        <p>Kurva spektrum respons yang dihasilkan di-overlay langsung terhadap kurva spektrum desain standar nasional Indonesia <b>SNI 1726:2019</b> 
        (dihitung dari parameter S<sub>DS</sub>, S<sub>D1</sub>, T<sub>0</sub>, T<sub>s</sub>, T<sub>L</sub>) untuk kelas situs Tanah Lunak (SE), 
        Sedang (SD), dan Keras (SC) sebagai tolok ukur permintaan elastis guna mendeteksi spektrum gempa yang melampaui kapasitas elastis.</p>
        """
        if lang == "id" else
        """
        <p><b>10.2 Solver Algorithm Auditing: Nigam-Jennings (1969) vs. Newmark-Beta (1959)</b></p>
        <p>BSMA incorporates two industry-standard numerical algorithms for structural dynamics:</p>
        <p>&bull; <b>Nigam-Jennings (1969):</b> Exact piecewise-linear recursive analytical formulation, maintaining absolute stability 
        in the high-frequency/short-period regime (T &lt; 0.1 s) without artificial numerical dispersion.</p>
        <p>&bull; <b>Newmark-Beta (1959):</b> Unconditionally stable implicit average acceleration algorithm (&gamma; = 0.5, &beta; = 0.25), 
        the classic benchmark in structural dynamic finite element computation.</p>
        <p>&bull; <b>Concordance Benchmarks:</b> Rigorously cross-validated across 100 period points (T = 0.01 - 10.0 s): 
        mean relative difference &lt; 0.02%, maximum relative difference &lt; 0.08%, and RMS difference &lt; 1.5 &times; 10<sup>-4</sup> g.</p>

        <p><b>10.3 SNI 1726:2019 National Design Spectra Benchmark</b></p>
        <p>Calculated PSA curves are benchmarked against Indonesia's national building design code <b>SNI 1726:2019</b> 
        (governed by S<sub>DS</sub>, S<sub>D1</sub>, T<sub>0</sub>, T<sub>s</sub>, T<sub>L</sub>) across site classes (Soft Soil SE, Medium Soil SD, 
        Hard Soil SC), providing an immediate elastic demand benchmark against codified design thresholds.</p>
        """
    )
    builder.insert_html_safe(p14, pymupdf.Rect(LEFT_X, y_mid10, RIGHT_X, y_mid10 + 175), c10_text_mid)

    callout_c10 = (
        "Pembandingan Spektrum Desain SNI 1726:2019: Spektrum respons elastis 5% redaman yang dihasilkan BSMA menyediakan tolok ukur "
        "permintaan seismik elastis (elastic demand benchmark) terhadap spektrum desain SNI 1726:2019. Catatan Rekayasa: Pembandingan ini "
        "memberikan indikasi awal potensi eksitasi resonansi pada periode alami struktur, namun BUKAN merupakan verifikasi desain struktural, "
        "analisis non-linier terperinci, maupun asesmen kerusakan gedung secara definitif."
        if lang == "id" else
        "SNI 1726:2019 Design Code Demand Benchmark: The 5% damped elastic response spectra computed by BSMA provide an initial elastic "
        "demand benchmark against SNI 1726:2019 target design spectra. Engineering Disclaimer: This comparison provides preliminary "
        "indication of potential resonant excitation near structural natural periods, but does NOT constitute formal structural design "
        "verification, non-linear inelastic modeling, or definitive post-earthquake damage assessment."
    )
    builder.draw_callout(p14, pymupdf.Rect(LEFT_X, y_mid10 + 185, RIGHT_X, y_mid10 + 255),
                         "Tolok Ukur Spektrum Desain SNI 1726:2019" if lang == "id" else "SNI 1726:2019 Design Code Demand Benchmark",
                         callout_c10, callout_type="info")

    # -------------------------------------------------------------
    # PAGE 15 (Halaman 11) - BAB XI: PANDUAN PEMECAHAN MASALAH (FAQ)
    # -------------------------------------------------------------
    p15 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p15)
    builder.add_page_header_footer(p15, "BAB XI - PEMECAHAN MASALAH" if lang == "id" else "CHAPTER XI - TROUBLESHOOTING",
                                   f"Halaman 11 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 11 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p15, "BAB XI: PANDUAN PEMECAHAN MASALAH (FAQ) & BATASAN SISTEM" if lang == "id" else "CHAPTER XI: TROUBLESHOOTING GUIDE (FAQ) & OPERATIONAL LIMITATIONS")

    c11_text = (
        """
        <p><b>11.1 Pemecahan Masalah Umum Operasional (Troubleshooting FAQ)</b></p>
        <p><b>Q1: Mengapa kurva perpindahan (displacement) melengkung drastis ke atas atau ke bawah secara kuadratik?</b><br>
        <i>A:</i> Terjadi akibat residual baseline drift pada integrasi ganda derau frekuensi rendah sisa (Boore 2001). Solusi: Naikkan frekuensi 
        cutoff bawah (f<sub>min</sub>) dari 0.05 Hz menjadi 0.10 Hz atau 0.20 Hz pada panel filter PROCESSING di sidebar kontrol.</p>
        <p><b>Q2: Muncul peringatan 'StationXML Response Correction Bypassed'. Apa artinya bagi analisis?</b><br>
        <i>A:</i> Berkas StationXML tidak cocok dengan metadata kanal rekaman, atau sinyal masukan telah terdeteksi berdimensi percepatan fisik 
        (m/s&sup2;). Sistem secara aman mengalihkan ke mode Physical Acceleration. Solusi: Pastikan kode stasiun, jaringan, dan kanal pada berkas .xml dan .mseed identik.</p>
        <p><b>Q3: Mengapa nilai PGA akselerograf berbeda dengan sensor broadband di stasiun yang sama?</b><br>
        <i>A:</i> Akselerograf dioptimalkan merekam percepatan gempa kuat tanpa kliping (dinamika tinggi), sedangkan sensor broadband dioptimalkan 
        untuk kecepatan getaran lemah periode panjang. Untuk rekayasa struktur dan estimasi MMI getaran kuat, akselerograf merupakan acuan otoritatif utama.</p>
        <p><b>Q4: Mengapa status Quality Control menunjukkan WARNING atau FAIL meski bentuk gelombang tampak jelas?</b><br>
        <i>A:</i> Sistem QC mengevaluasi 6 kriteria diagnostik (SNR, clipping sensor, spikes, flatline, pre-event noise, dan DC offset). Status WARNING 
        umumnya dipicu durasi jendela pre-event noise &lt; 5 detik atau SNR &lt; 10 dB. Buka Tab 3 (Quality Control) untuk memeriksa tabel penalti.</p>
        <p><b>Q5: Kapan sebaiknya memilih solver Nigam-Jennings dibandingkan Newmark-Beta?</b><br>
        <i>A:</i> Formulasi analitik eksak linier per-segmen Nigam-Jennings (1969) sangat tepat (<i>highly suitable</i>) untuk spektrum respons 
        periode pendek (T &lt; 0.1 s) dan struktur kaku karena mengevaluasi matriks eksponensial eksak tanpa redaman semu numerik. 
        Newmark-Beta (1959) menggunakan integrasi implisit percepatan rata-rata. Kedua solver divalidasi silang pada Tab 6 dengan 
        kesesuaian sangat tinggi: selisih relatif rata-rata &lt; 0.02% dan selisih maks &lt; 0.08%.</p>
        <p><b>Q6: Mengapa proses pembuatan laporan PDF pada Tab 7 memerlukan beberapa detik?</b><br>
        <i>A:</i> Generator PDF menyusun dokumen beresolusi vektor tinggi secara real-time, merender grafik kinematika 3-komponen, kurva Husid, 
        dan spektrum respons langsung ke dalam dokumen biner. Proses kompilasi grafis vektor intensif ini membutuhkan waktu beberapa detik 
        hingga &lt; 35 detik bergantung pada jumlah stasiun rekaman dan spesifikasi perangkat keras (hardware).</p>

        <p><b>11.2 Batasan Metodologis & Lingkup Operasional Software</b></p>
        <p>&bull; <b>Batasan Data & Sensor:</b> Rekaman dengan kliping sensor ekstrem (&gt; 2g) atau SNR &lt; 3 dB tidak layak untuk analisis spektral kuantitatif; dekonvolusi menuntut StationXML valid.</p>
        <p>&bull; <b>Batasan Pemrosesan Numerik:</b> Integrasi ganda dengan filter AC menghasilkan PGD dinamik gelombang transien, bukan perpindahan statis permanen sesar (fling-step).</p>
        <p>&bull; <b>Batasan Rekayasa Struktur:</b> Spektrum SDOF mengasumsikan osilator linier elastis tunggal (5% damping), bukan respon inelastis non-linier multi-derajat-kebebasan gedung riil.</p>

        <p><b>11.3 Tautan Akses Cloud Web & Repositori GitHub</b></p>
        <p>&bull; <b>Aplikasi Cloud Web Publik:</b> <a href="https://strong-motion.streamlit.app/">https://strong-motion.streamlit.app/</a> (Dapat diakses instan tanpa instalasi).</p>
        <p>&bull; <b>Repositori Kode Sumber:</b> <a href="https://github.com/ahmaddidan/BSMA-v.2">https://github.com/ahmaddidan/BSMA-v.2</a> (Kode sumber terbuka, test suite, & dokumentasi).</p>
        """
        if lang == "id" else
        """
        <p><b>11.1 Frequently Asked Questions (Troubleshooting FAQ)</b></p>
        <p><b>Q1: Why does the displacement curve drift upward or downward quadratically?</b><br>
        <i>A:</i> Caused by residual low-frequency noise drift during double integration (Boore 2001). Solution: Increase the highpass cutoff 
        frequency (f<sub>min</sub>) from 0.05 Hz to 0.10 Hz or 0.20 Hz in the PROCESSING sidebar filter panel.</p>
        <p><b>Q2: Why does the warning 'StationXML Response Correction Bypassed' appear?</b><br>
        <i>A:</i> StationXML metadata does not match waveform trace headers, or the trace is already calibrated in physical acceleration (m/s&sup2;). 
        The system automatically falls back to Physical Acceleration mode. Solution: Verify that network, station, and channel codes match between .xml and .mseed files.</p>
        <p><b>Q3: Why does accelerograph PGA differ from a collocated broadband seismometer?</b><br>
        <i>A:</i> Accelerographs are engineered for high-amplitude strong ground motion without saturation, while broadband velocimeters optimize weak-motion 
        long-period sensitivity. For structural engineering and instrumental MMI, the accelerograph is the authoritative standard.</p>
        <p><b>Q4: Why does Quality Control show WARNING or FAIL when the waveform visually appears clear?</b><br>
        <i>A:</i> The QC engine audits 6 diagnostic criteria (SNR, sensor clipping, spikes, flatline, pre-event noise, and DC offset). WARNING is frequently 
        triggered by pre-event windows &lt; 5 seconds or SNR &lt; 10 dB. Inspect Tab 3 for detailed diagnostic penalties.</p>
        <p><b>Q5: When should Nigam-Jennings be chosen over Newmark-Beta?</b><br>
        <i>A:</i> The Nigam-Jennings (1969) exact analytical solver is analytically exact for piecewise-linear acceleration excitation and highly 
        suitable for stiff, short-period systems (T &lt; 0.1 s). Newmark-Beta (1959) uses implicit average acceleration. 
        Both solvers agree within &lt; 0.02% mean relative difference, verified in Tab 6.</p>
        <p><b>Q6: Why does generating the PDF report in Tab 7 take a few seconds?</b><br>
        <i>A:</i> The engine dynamically compiles a publication-grade vector PDF, rasterizing kinematic traces, Husid curves, and response 
        spectra directly into the document stream. This intensive vector compilation takes from a few seconds up to &lt; 35 seconds depending on 
        station count and hardware specifications.</p>

        <p><b>11.2 Methodological Limitations & Operational Scope</b></p>
        <p>&bull; <b>Input Data & Sensors:</b> Traces exhibiting severe sensor clipping (&gt; 2g) or SNR &lt; 3 dB are unsuitable for spectral analysis; response deconvolution requires valid StationXML.</p>
        <p>&bull; <b>DSP & Numerical Integration:</b> AC-coupled bandpass integration yields transient dynamic PGD, not permanent tectonic surface ruptures (fling-step).</p>
        <p>&bull; <b>Structural Dynamics & Engineering:</b> SDOF spectra model single-degree-of-freedom linear elastic oscillators (5% damping), not non-linear multi-story inelastic behavior.</p>

        <p><b>11.3 Cloud Web Access & GitHub Repository</b></p>
        <p>&bull; <b>Cloud Web Platform:</b> <a href="https://strong-motion.streamlit.app/">https://strong-motion.streamlit.app/</a> (Zero installation required).</p>
        <p>&bull; <b>GitHub Repository:</b> <a href="https://github.com/ahmaddidan/BSMA-v.2">https://github.com/ahmaddidan/BSMA-v.2</a> (Open-source code, test suite, & docs).</p>
        """
    )
    builder.insert_html_safe(p15, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 540), c11_text)

    # -------------------------------------------------------------
    # PAGE 16 (Halaman 12) - DAFTAR PUSTAKA
    # -------------------------------------------------------------
    p16 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p16)
    builder.add_page_header_footer(p16, "DAFTAR PUSTAKA" if lang == "id" else "REFERENCES",
                                   f"Halaman 12 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 12 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p16, "DAFTAR PUSTAKA" if lang == "id" else "REFERENCES")

    refs_all = [
        ("Arias, A. (1970).",
         "A measure of earthquake intensity. In R. J. Hansen (Ed.), <i>Seismic Design for Nuclear Power Plants</i> (pp. 438-483). Cambridge, MA: MIT Press.",
         ""),
        ("Badan Meteorologi, Klimatologi, dan Geofisika. (2019).",
         "<i>Pedoman Pengolahan dan Analisis Data Akselerograf Gempabumi Kuat di Lingkungan BMKG</i>. Jakarta: Kedeputian Bidang Geofisika BMKG.",
         ""),
        ("Badan Standardisasi Nasional. (2019).",
         "<i>SNI 1726:2019: Tata cara perencanaan ketahanan gempa untuk struktur bangunan gedung dan non gedung</i>. Jakarta: Badan Standardisasi Nasional.",
         ""),
        ("Beyreuther, M., Barsch, R., Krischer, L., Megies, T., Behr, Y., & Wassermann, J. (2010).",
         "ObsPy: A Python toolbox for seismology. <i>Seismological Research Letters</i>, 81(3), 530-533.",
         "https://doi.org/10.1785/gssrl.81.3.530"),
        ("Boore, D. M. (2001).",
         "Effect of baseline corrections on displacements and response spectra for several recordings of the 1999 Chi-Chi, Taiwan, earthquake. <i>Bulletin of the Seismological Society of America</i>, 91(5), 1199-1211.",
         "https://doi.org/10.1785/0120000703"),
        ("Boore, D. M., & Bommer, J. J. (2005).",
         "Processing of strong-motion accelerograms: Needs, options and consequences. <i>Soil Dynamics and Earthquake Engineering</i>, 25(2), 93-115.",
         "https://doi.org/10.1016/j.soildyn.2004.10.007"),
        ("Chopra, A. K. (2017).",
         "<i>Dynamics of Structures: Theory and Applications to Earthquake Engineering</i> (5th ed.). Upper Saddle River, NJ: Pearson Education.",
         ""),
        ("Douglas, J. (2003).",
         "Earthquake ground motion estimation using strong-motion records: a review of equations for the estimation of peak ground acceleration and response spectral ordinates. <i>Earth-Science Reviews</i>, 61(1-2), 43-104.",
         "https://doi.org/10.1016/S0012-8252(02)00112-5"),
        ("Electric Power Research Institute. (1988).",
         "<i>A criterion for determining exceedance of the Operating Basis Earthquake</i> (EPRI Report NP-5930). Palo Alto, CA: Electric Power Research Institute.",
         ""),
        ("Harris, F. J. (1978).",
         "On the use of windows for harmonic analysis with the discrete Fourier transform. <i>Proceedings of the IEEE</i>, 66(1), 51-83.",
         "https://doi.org/10.1109/PROC.1978.10837"),
        ("Kempton, J. J., & Stewart, J. P. (2006).",
         "Prediction equations for significant duration of earthquake ground motions. <i>Bulletin of the Seismological Society of America</i>, 96(5), 1831-1846.",
         "https://doi.org/10.1785/0120050202"),
        ("Newmark, N. M. (1959).",
         "A method of computation for structural dynamics. <i>Journal of the Engineering Mechanics Division, ASCE</i>, 85(3), 67-94.",
         "https://doi.org/10.1061/JMCEA3.0000098"),
        ("Nigam, N. C., & Jennings, P. C. (1969).",
         "Calculation of response spectra from strong-motion earthquake records. <i>Bulletin of the Seismological Society of America</i>, 59(2), 909-922.",
         "https://doi.org/10.1785/BSSA0590020909"),
        ("Trifunac, M. D., & Brady, A. G. (1975).",
         "A study on the duration of strong earthquake ground motion. <i>Bulletin of the Seismological Society of America</i>, 65(3), 581-626.",
         "https://doi.org/10.1785/BSSA0650030581"),
        ("Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., & van der Walt, S. J. (2020).",
         "SciPy 1.0: Fundamental algorithms for scientific computing in Python. <i>Nature Methods</i>, 17(3), 261-272.",
         "https://doi.org/10.1038/s41592-019-0686-2"),
        ("Worden, C. B., Gerstenberger, M. C., Rhoades, D. A., & Wald, D. J. (2012).",
         "Probabilistic relationships between ground-motion parameters and MMI. <i>Bulletin of the Seismological Society of America</i>, 102(1), 204-221.",
         "https://doi.org/10.1785/0120110156"),
    ]

    builder.draw_card(p16, pymupdf.Rect(LEFT_X, y + 4, RIGHT_X, y + 710), bg_col=COLOR_WHITE, border_col=COLOR_CARD_BORDER)
    builder.draw_academic_bibliography(p16, pymupdf.Rect(LEFT_X + 16, y + 16, RIGHT_X - 16, y + 700), refs_all)

    # -------------------------------------------------------------
    # PAGE 17 (Halaman 13) - PROFIL PENGEMBANG
    # -------------------------------------------------------------
    p17 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p17)
    builder.add_page_header_footer(p17, "PROFIL PENGEMBANG" if lang == "id" else "DEVELOPER PROFILE",
                                   f"Halaman 13 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 13 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p17, "PROFIL PENGEMBANG" if lang == "id" else "DEVELOPER PROFILE")

    # Developer Profile Card (Table matching exact structure)
    builder.draw_card(p17, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 295), bg_col=COLOR_WHITE, border_col=COLOR_CARD_BORDER)
    p17.insert_text((LEFT_X + 16, y + 22), "DATA PROFIL PENGEMBANG & SISTEM" if lang == "id" else "DEVELOPER PROFILE & SYSTEM METADATA",
                    fontsize=10.0, fontname="f_bold", color=COLOR_NAVY)
    p17.draw_line(pymupdf.Point(LEFT_X + 16, y + 28), pymupdf.Point(RIGHT_X - 16, y + 28), color=COLOR_SKY, width=1.0)

    profile_fields_id = [
        ("Penyusun / Pengembang", "Ahmad Didane Setyawan Putra"),
        ("Nomor Induk Mahasiswa (NIM)", "123120094"),
        ("Program Studi", "Teknik Geofisika"),
        ("Fakultas", "Fakultas Teknologi Industri"),
        ("Perguruan Tinggi", "Institut Teknologi Sumatera (ITERA)"),
        ("Instansi Mitra / Lokasi", "Stasiun Geofisika Kelas I Sleman, BMKG D.I. Yogyakarta"),
        ("Periode Kerja Praktik", "20 Juli 2026 s.d. 20 Agustus 2026"),
        ("Klasifikasi Proyek", "Karya Akademik Mandiri Mahasiswa (Non-Komersial)"),
        ("Edisi Dokumen", "v2.0.0 (Build 2026.08) - Edisi Rilis Akademik"),
        ("URL Platform Web", "https://strong-motion.streamlit.app/"),
        ("Repositori Kode Sumber", "https://github.com/ahmaddidan/BSMA-v.2"),
    ]
    profile_fields_en = [
        ("Author / Developer", "Ahmad Didane Setyawan Putra"),
        ("Student ID (NIM)", "123120094"),
        ("Study Program", "Geophysical Engineering"),
        ("Faculty", "Faculty of Industrial Technology"),
        ("University", "Institut Teknologi Sumatera (ITERA)"),
        ("Host Institution", "Sleman Geophysical Station Class I, BMKG D.I. Yogyakarta"),
        ("Internship Period", "July 20, 2026 - August 20, 2026"),
        ("Project Classification", "Independent Student Academic Work (Non-Commercial)"),
        ("Document Edition", "v2.0.0 (Build 2026.08) - Academic Release Edition"),
        ("Web Platform URL", "https://strong-motion.streamlit.app/"),
        ("Source Code Repository", "https://github.com/ahmaddidan/BSMA-v.2"),
    ]
    p_fields = profile_fields_id if lang == "id" else profile_fields_en

    cur_py = y + 48
    for label, val in p_fields:
        p17.insert_text((LEFT_X + 20, cur_py), label, fontsize=8.0, fontname="f_bold", color=COLOR_SLATE)
        if "http" in val:
            p17.insert_text((LEFT_X + 175, cur_py), f":  {val}", fontsize=8.0, fontname="f_reg", color=COLOR_SKY)
            link_w = font_bold_obj.text_length(f":  {val}", fontsize=8.0)
            p17.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(LEFT_X + 175, cur_py - 8, LEFT_X + 175 + link_w, cur_py + 2), "uri": val})
        else:
            p17.insert_text((LEFT_X + 175, cur_py), f":  {val}", fontsize=8.0, fontname="f_reg", color=COLOR_DARK)
        p17.draw_line(pymupdf.Point(LEFT_X + 16, cur_py + 5), pymupdf.Point(RIGHT_X - 16, cur_py + 5), color=COLOR_CARD_BG, width=0.6)
        cur_py += 22

    # Academic & Operational Disclaimer
    y_disclaimer = y + 315
    disclaimer_title = "BATASAN TANGGUNG JAWAB & PERNYATAAN PENGGUNAAN (DISCLAIMER)" if lang == "id" else "ACADEMIC & OPERATIONAL DISCLAIMER"
    disclaimer_body = (
        "Perangkat lunak BMKG Strong Motion Analyzer (BSMA v2.0.0) beserta Buku Panduan Pengguna ini disusun secara mandiri "
        "sebagai karya akademik mahasiswa dalam rangka penyelesaian Kerja Praktik di BMKG Stasiun Geofisika Kelas I Sleman. "
        "Perangkat lunak ini disediakan 'sebagaimana adanya' (as-is) untuk tujuan pendidikan, penelitian ilmiah, dan evaluasi teknis "
        "seismologi rekayasa. Pengembang tidak bertanggung jawab atas kerugian langsung maupun tidak langsung, kegagalan struktur, "
        "atau keputusan rekayasa yang timbul akibat penggunaan atau interpretasi hasil komputasi perangkat lunak ini tanpa verifikasi silang "
        "independen oleh tenaga ahli rekayasa geoteknik dan kegempaan yang bersertifikasi."
        if lang == "id" else
        "The BMKG Strong Motion Analyzer (BSMA v2.0.0) software and this User Guidebook have been developed independently as an "
        "academic student work in partial fulfillment of the undergraduate internship at BMKG Sleman Geophysical Station Class I. "
        "This software is provided 'as-is' for educational, academic research, and technical evaluation in engineering seismology. "
        "The author assumes no liability or legal responsibility for direct, indirect, or consequential damages, structural failures, "
        "or engineering decisions resulting from the use or interpretation of computational outputs without independent cross-validation "
        "by certified professional geotechnical and earthquake structural engineers."
    )
    builder.draw_callout(p17, pymupdf.Rect(LEFT_X, y_disclaimer, RIGHT_X, y_disclaimer + 110),
                         disclaimer_title, disclaimer_body, callout_type="warning")

    # Native PDF Bookmarks Outline (TOC)
    toc_id = [
        [1, "Halaman Sampul", 1],
        [1, "Ringkasan Eksekutif & Landasan Ilmiah", 2],
        [1, "Spesifikasi Lingkungan Komputasi & Dependensi", 3],
        [1, "Daftar Isi & Struktur Panduan", 4],
        [1, "Bagian I: Panduan Operasional & Alur Kerja Software", 5],
        [2, "Bab I: Memulai & Alur Kerja Dasar Sistem", 5],
        [2, "Bab II: Panduan Operasional Menu Sidebar & Parameter DSP", 6],
        [2, "Bab III: Prosedur Analisis Kinematika & Kontrol Kualitas (QC)", 7],
        [2, "Bab IV: Karakterisasi Gerakan Kuat, Skala MMI & Spektrum Respons", 8],
        [2, "Bab V: Panduan Mode Pemrosesan Batch Multi-Stasiun Regional", 9],
        [1, "Bagian II: Landasan Teoretis & Pemodelan Seismologi Rekayasa", 10],
        [2, "Bab VI: Arsitektur Pipeline & Dekonvolusi Respon Instrumen", 10],
        [2, "Bab VII: Quality Control & Digital Signal Processing (DSP)", 11],
        [2, "Bab VIII: Integrasi Kinematika & Kebijakan Garis Dasar", 12],
        [2, "Bab IX: Parameter Kinematika, Energi Seismik & Intensitas MMI", 13],
        [2, "Bab X: Pemodelan Spektrum Respons SDOF & Standar SNI 1726:2019", 14],
        [1, "Bagian III: Penjaminan Mutu & Dokumentasi Teknis", 15],
        [2, "Bab XI: Panduan Pemecahan Masalah (FAQ) & Batasan Sistem", 15],
        [2, "Daftar Pustaka Ilmiah & Standar Acuan", 16],
        [2, "Profil Pengembang & Penafian Tanggung Jawab", 17],
    ]
    toc_en = [
        [1, "Title Page & Metadata", 1],
        [1, "Executive Summary & Scientific Background", 2],
        [1, "Computational Environment & Runtime Dependencies", 3],
        [1, "Table of Contents & Guidebook Architecture", 4],
        [1, "Part I: Operational Workflows & Software Suite", 5],
        [2, "Chapter I: Getting Started & Core System Workflow", 5],
        [2, "Chapter II: Sidebar Operational Controls & DSP Settings", 6],
        [2, "Chapter III: Kinematic Analysis & Waveform Quality Control", 7],
        [2, "Chapter IV: Strong Ground Motion, MMI Intensity & Spectra", 8],
        [2, "Chapter V: Regional Multi-Station Batch Processing Guide", 9],
        [1, "Part II: Theoretical Foundations & Engineering Seismology", 10],
        [2, "Chapter VI: Pipeline Architecture & Instrument Deconvolution", 10],
        [2, "Chapter VII: Quality Control & Digital Signal Processing (DSP)", 11],
        [2, "Chapter VIII: Kinematic Integration & Baseline Policy", 12],
        [2, "Chapter IX: Kinematic Parameters, Seismic Energy & MMI Intensity", 13],
        [2, "Chapter X: SDOF Structural Dynamics & SNI 1726:2019 Code", 14],
        [1, "Part III: Quality Assurance & Technical Documentation", 15],
        [2, "Chapter XI: Troubleshooting Guide (FAQ) & System Limitations", 15],
        [2, "Academic References & Standards", 16],
        [2, "Developer Profile & Disclaimer", 17],
    ]
    doc.set_toc(toc_id if lang == "id" else toc_en)

    # Insert internal clickable Table of Contents links onto Page 4 (0-indexed page 3)
    p4_page = doc[3]
    for link_rect, target_idx in toc_links:
        p4_page.insert_link({"kind": pymupdf.LINK_GOTO, "page": target_idx, "from": link_rect})

    # Post-process ToUnicode streams to eliminate ligatures, non-breaking spaces, and soft hyphens
    builder.post_process_pdf_clean_unicode()

    # Final Output PDF Saving
    suffix = "ID" if lang == "id" else "EN"
    out_name = f"BSMA_Scientific_Guidebook_{suffix}.pdf"
    out_path = OUTPUT_DIR / out_name
    doc.save(str(out_path), garbage=4, deflate=True)
    doc.close()
    print(f"Scientific Guidebook [{lang.upper()}] successfully generated: {out_path} ({out_path.stat().st_size / 1024:.1f} KB, Total Pages: 17)")
    return out_path


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    print("=" * 72)
    print("BMKG Strong Motion Analyzer (BSMA v2.0.0) - Scientific Guidebook Generator")
    print("=" * 72)

    pdf_id = build_guidebook(lang="id")
    pdf_en = build_guidebook(lang="en")

    print("\nScientific Guidebook generation completed successfully!")
    print(f"  - Indonesian Edition: {pdf_id}")
    print(f"  - English Edition:    {pdf_en}")
    print("=" * 72)


if __name__ == "__main__":
    main()
