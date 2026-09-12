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
Internship Period: July 20 – August 20, 2026
"""

from __future__ import annotations

import io
import os
import shutil
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
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

FONT_ARIAL = r"C:\Windows\Fonts\arial.ttf"
FONT_ARIAL_BD = r"C:\Windows\Fonts\arialbd.ttf"
FONT_ARIAL_IT = r"C:\Windows\Fonts\ariali.ttf"
FONT_ARIAL_BI = r"C:\Windows\Fonts\arialbi.ttf"


# =============================================================================
# DIAGRAM GENERATORS
# =============================================================================

def create_pipeline_diagram(lang: str = "id") -> bytes:
    """Create high-resolution 5-stage processing pipeline diagram."""
    fig, ax = plt.subplots(figsize=(8.5, 2.3), dpi=220)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    stages_id = [
        ("1. INGESTI", "MiniSEED / SAC\n+ StationXML PAZ", "#0284C7"),
        ("2. PREPROCESS", "Detrend, Taper 5%\nButterworth Orde-4", "#0EA5E9"),
        ("3. INTEGRASI", "Acc -> Vel -> Disp\nMitigasi Baseline", "#10B981"),
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
        ax.text(x + w / 2, y + 2.05, title, color="white", weight="bold", fontsize=8.2, ha="center", va="center")
        ax.text(x + w / 2, y + 0.85, desc, color="#334155", fontsize=7.2, ha="center", va="center", linespacing=1.35)

        if i < 4:
            ax.annotate(
                "", xy=(x + w + 0.28, y + 1.2), xytext=(x + w + 0.02, y + 1.2),
                arrowprops=dict(arrowstyle="-|>", color="#94A3B8", lw=1.8, mutation_scale=12),
            )

    buf = io.BytesIO()
    plt.tight_layout(pad=0.2)
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def create_qc_spider_diagram(lang: str = "id") -> bytes:
    """Create publication-grade polar QC radar chart."""
    labels_id = ["SNR Pre-Event", "Garis Dasar (Drift)", "Kebersihan Spikes", "Margin Kliping", "Spektrum Nyquist", "Kontinuitas Sinyal"]
    labels_en = ["Pre-Event SNR", "Baseline (Drift)", "Spike Cleanliness", "Clipping Margin", "Nyquist Spectrum", "Signal Continuity"]
    labels = labels_id if lang == "id" else labels_en

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    values_pass = [92, 88, 95, 98, 90, 100]
    values_pass += values_pass[:1]
    values_warn = [62, 55, 70, 85, 75, 90]
    values_warn += values_warn[:1]

    fig, ax = plt.subplots(figsize=(4.0, 2.6), subplot_kw=dict(polar=True), dpi=220)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")

    ax.plot(angles, values_pass, color="#059669", linewidth=2.0, label="Nominal (PASS ≥ 70)")
    ax.fill(angles, values_pass, color="#059669", alpha=0.20)

    ax.plot(angles, values_warn, color="#D97706", linewidth=1.8, linestyle="--", label="Marginal (WARN 50-69)")
    ax.fill(angles, values_warn, color="#D97706", alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=6.6, color="#1E293B", weight="semibold")
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], size=5.8, color="#94A3B8")
    ax.set_ylim(0, 105)
    ax.grid(color="#E2E8F0", linestyle=":", linewidth=0.8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.15), fontsize=6.5, frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")

    buf = io.BytesIO()
    plt.tight_layout(pad=0.2)
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def create_husid_diagram(lang: str = "id") -> bytes:
    """Create Arias Intensity / Husid accumulation curve diagram."""
    fig, ax = plt.subplots(figsize=(4.0, 2.4), dpi=220)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")

    t = np.linspace(0, 40, 400)
    energy = 1.0 / (1.0 + np.exp(-0.35 * (t - 16)))
    energy = (energy - energy[0]) / (energy[-1] - energy[0])

    ax.plot(t, energy * 100, color="#0284C7", lw=2.2, label=r"Akumulasi $I_a(t)$" if lang == "id" else r"Accumulated $I_a(t)$")
    ax.axhline(5, color="#D97706", linestyle=":", lw=1.2, label="5% $I_a$ ($t_5$)")
    ax.axhline(95, color="#DC2626", linestyle=":", lw=1.2, label="95% $I_a$ ($t_{95}$)")

    t5 = t[np.where(energy >= 0.05)[0][0]]
    t95 = t[np.where(energy >= 0.95)[0][0]]
    ax.axvspan(t5, t95, color="#0284C7", alpha=0.12, label=f"$D_{{5-95}} = {t95-t5:.1f}$ s")

    lbl_x = "Waktu / Time (s)" if lang == "id" else "Time (s)"
    lbl_y = "Akumulasi Energi (%)" if lang == "id" else "Accumulated Energy (%)"
    ax.set_xlabel(lbl_x, fontsize=7.2, color="#334155")
    ax.set_ylabel(lbl_y, fontsize=7.2, color="#334155")
    ax.tick_params(labelsize=6.5)
    ax.grid(color="#E2E8F0", linestyle=":", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=6.2, frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")

    buf = io.BytesIO()
    plt.tight_layout(pad=0.3)
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def create_spectrum_diagram(lang: str = "id") -> bytes:
    """Create response spectrum plot comparing Nigam-Jennings, Newmark-Beta, and SNI 1726."""
    fig, ax = plt.subplots(figsize=(4.0, 2.4), dpi=220)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")

    T = np.logspace(-2, 1, 150)
    pga = 0.25
    sa_nj = pga * (1 + 1.5 * np.exp(-((np.log10(T) - np.log10(0.2))**2) / 0.35))
    sa_nb = sa_nj * (1 + 0.015 * np.sin(5 * np.log10(T)))

    # SNI Design spectrum
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
    ax.semilogx(T, sni, color="#DC2626", lw=1.6, linestyle=":", label="SNI 1726:2019 Design")

    lbl_x = "Periode Alami T (s)" if lang == "id" else "Natural Period T (s)"
    lbl_y = "Pseudo-Percepatan PSA (g)" if lang == "id" else "Pseudo-Acceleration PSA (g)"
    ax.set_xlabel(lbl_x, fontsize=7.2, color="#334155")
    ax.set_ylabel(lbl_y, fontsize=7.2, color="#334155")
    ax.set_xlim(0.01, 10.0)
    ax.tick_params(labelsize=6.5)
    ax.grid(color="#E2E8F0", linestyle=":", linewidth=0.8, which="both")
    ax.legend(loc="upper right", fontsize=6.0, frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")

    buf = io.BytesIO()
    plt.tight_layout(pad=0.3)
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# GUIDEBOOK BUILDER HELPER CLASS
# =============================================================================

class GuidebookBuilder:
    def __init__(self, doc: pymupdf.Document, lang: str = "id"):
        self.doc = doc
        self.lang = lang

    def init_page_fonts(self, page: pymupdf.Page) -> None:
        """Register true fonts on page."""
        if os.path.exists(FONT_ARIAL):
            page.insert_font(fontname="f_reg", fontfile=FONT_ARIAL)
            page.insert_font(fontname="f_bold", fontfile=FONT_ARIAL_BD)
            page.insert_font(fontname="f_it", fontfile=FONT_ARIAL_IT)
            page.insert_font(fontname="f_bi", fontfile=FONT_ARIAL_BI)
        else:
            page.insert_font(fontname="f_reg", fontname_builtin="helv")
            page.insert_font(fontname="f_bold", fontname_builtin="hebo")
            page.insert_font(fontname="f_it", fontname_builtin="heit")
            page.insert_font(fontname="f_bi", fontname_builtin="hebi")

    def insert_html_safe(self, page: pymupdf.Page, rect: pymupdf.Rect, html_body: str, font_size: str = "7.8pt", line_height: str = "1.34") -> None:
        """Render formatted HTML box with CSS styling, justification, links, and italic support."""
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
            margin: 0 0 6px 0;
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
            font-size: 7.2pt;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 4px 6px;
        }}
        th {{
            background-color: #002d62;
            color: #ffffff;
            font-weight: bold;
            text-align: left;
        }}
        """
        full_html = f"<html><head><style>{css}</style></head><body>{html_body}</body></html>"
        page.insert_htmlbox(rect, full_html)

    def add_page_header_footer(self, page: pymupdf.Page, section_badge: str, page_num_str: str) -> None:
        """Draw clean running header and academic footer on content pages."""
        # Top Header
        header_title = "BMKG STRONG MOTION ANALYZER (BSMA v2.0.0)"
        page.insert_text((LEFT_X, 23), header_title,
                         fontsize=7.4, fontname="f_bold", color=COLOR_NAVY)
        font_bold_obj = pymupdf.Font(fontfile=FONT_ARIAL_BD)
        badge_w = font_bold_obj.text_length(section_badge.upper(), fontsize=7.2)
        page.insert_text((RIGHT_X - badge_w, 23), section_badge.upper(),
                         fontsize=7.2, fontname="f_bold", color=COLOR_SLATE)

        # Bottom Footer (Strict academic footer)
        page.draw_line(pymupdf.Point(LEFT_X, 810), pymupdf.Point(RIGHT_X, 810), color=COLOR_CARD_BORDER, width=0.8)
        footer_author = (
            "Ahmad Didane Setyawan Putra · Teknik Geofisika, Institut Teknologi Sumatera"
            if self.lang == "id" else
            "Ahmad Didane Setyawan Putra · Geophysical Engineering, Institut Teknologi Sumatera"
        )
        page.insert_text((LEFT_X, 822), footer_author,
                         fontsize=7.2, fontname="f_it", color=COLOR_SLATE)
        page.insert_text((RIGHT_X - 95, 822), page_num_str,
                         fontsize=7.2, fontname="f_bold", color=COLOR_NAVY)

    def draw_card(self, page: pymupdf.Page, rect: pymupdf.Rect, bg_col=COLOR_CARD_BG, border_col=COLOR_CARD_BORDER, border_w=1.0) -> None:
        """Draw solid card box."""
        page.draw_rect(rect, color=border_col, fill=bg_col, width=border_w)

    def draw_chapter_banner(self, page: pymupdf.Page, title_str: str, y_top: float = 48.0) -> float:
        """Draw clean chapter header banner with accent bar."""
        page.draw_rect(pymupdf.Rect(LEFT_X, y_top, LEFT_X + 6, y_top + 20), color=COLOR_NAVY, fill=COLOR_NAVY)
        page.insert_text((LEFT_X + 14, y_top + 15), title_str, fontsize=11.6, fontname="f_bold", color=COLOR_NAVY)
        page.draw_line(pymupdf.Point(LEFT_X, y_top + 26), pymupdf.Point(RIGHT_X, y_top + 26), color=COLOR_CARD_BORDER, width=0.8)
        return y_top + 34.0

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
        page.insert_text((rect.x0 + 12, rect.y0 + 13), f"{badge} {title}", fontsize=7.8, fontname="f_bold", color=accent)

        html = f"""
        <p style="font-size: 7.2pt; line-height: 1.25; margin: 0; text-align: justify; text-justify: inter-word; color: #0f172a;">
            {text}
        </p>
        """
        self.insert_html_safe(page, pymupdf.Rect(rect.x0 + 12, rect.y0 + 18, rect.x1 - 10, rect.y1 - 6), html)


# =============================================================================
# MAIN GUIDEBOOK COMPILER FUNCTION
# =============================================================================

def build_guidebook(lang: str = "id") -> Path:
    """Build complete publication-grade User Guidebook in specified language."""
    print(f"Building BSMA User Guidebook [Language: {lang.upper()}]...")

    doc = pymupdf.open()
    builder = GuidebookBuilder(doc, lang=lang)
    TOTAL_BODY_PAGES = 12

    # -------------------------------------------------------------
    # PAGE 1 (Cover / Title Page - Roman i, suppressed)
    # -------------------------------------------------------------
    p1 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p1)

    # Full Executive Cover Background
    p1.draw_rect(pymupdf.Rect(0, 0, PAGE_W, 360), color=COLOR_NAVY, fill=COLOR_NAVY)
    p1.draw_rect(pymupdf.Rect(0, 355, PAGE_W, 360), color=COLOR_SKY, fill=COLOR_SKY)

    # Top Dual Logos Lockup (Directly on Navy)
    font_bold_obj = pymupdf.Font(fontfile=FONT_ARIAL_BD)
    bmkg_icon = LOGO_BMKG_ICON_PATH if LOGO_BMKG_ICON_PATH.is_file() else LOGO_JUDUL_PATH
    if bmkg_icon.is_file():
        p1.insert_image(pymupdf.Rect(50, 48, 114, 112), filename=str(bmkg_icon))
    bmkg_tw = font_bold_obj.text_length("BMKG", fontsize=11.0)
    p1.insert_text((82 - bmkg_tw / 2, 128), "BMKG", fontsize=11.0, fontname="f_bold", color=COLOR_WHITE)

    p1.draw_line(pymupdf.Point(130, 48), pymupdf.Point(130, 130), color=(0.35, 0.60, 0.85), width=1.2)

    itera_icon = LOGO_ITERA_ICON_PATH if LOGO_ITERA_ICON_PATH.is_file() else LOGO_ITERA_PATH
    if itera_icon.is_file():
        p1.insert_image(pymupdf.Rect(146, 48, 210, 112), filename=str(itera_icon))
    itera_tw = font_bold_obj.text_length("ITERA", fontsize=11.0)
    p1.insert_text((178 - itera_tw / 2, 128), "ITERA", fontsize=11.0, fontname="f_bold", color=COLOR_WHITE)

    # Institutional Running Title beside Logos
    text_header_1 = "BUKU PANDUAN PENGGUNA & REFERENSI TEKNIS SOFTWARE" if lang == "id" else "USER GUIDEBOOK & SOFTWARE TECHNICAL REFERENCE MANUAL"
    text_header_2 = "Proyek Kerja Praktik Mahasiswa Program Studi Teknik Geofisika" if lang == "id" else "Undergraduate Internship Project of Geophysical Engineering Department"
    text_header_3 = "Fakultas Teknik Industri, Institut Teknologi Sumatera · BMKG Stasiun Geofisika Sleman" if lang == "id" else "Faculty of Industrial Technology, Institut Teknologi Sumatera · Sleman Geophysical Station BMKG"

    p1.insert_text((230, 68), text_header_1, fontsize=9.2, fontname="f_bold", color=COLOR_WHITE)
    p1.insert_text((230, 85), text_header_2, fontsize=8.2, fontname="f_reg", color=(0.85, 0.92, 1.0))
    p1.insert_text((230, 100), text_header_3, fontsize=7.4, fontname="f_it", color=(0.75, 0.85, 0.95))

    # Main Title Block
    doc_type = "DOKUMEN PANDUAN PENGGUNA & REFERENSI TEKNIS (USER GUIDEBOOK)" if lang == "id" else "OFFICIAL USER GUIDEBOOK & ENGINEERING TECHNICAL REFERENCE"
    p1.insert_text((50, 195), doc_type, fontsize=9.2, fontname="f_bold", color=(0.75, 0.90, 1.0))
    p1.insert_text((50, 230), "BMKG Strong Motion Analyzer", fontsize=23.0, fontname="f_bold", color=COLOR_WHITE)
    p1.insert_text((50, 258), "(BSMA v2.0.0)", fontsize=18.0, fontname="f_bold", color=COLOR_SKY)

    sub_title = (
        "Platform Komputasi Terpadu Sinyal Akselerograf, Kinematika Seismik, & Spektrum Respons"
        if lang == "id" else
        "Integrated Computational Platform for Accelerograph Signal Processing, Seismic Kinematics, & Response Spectra"
    )
    p1.insert_text((50, 285), sub_title, fontsize=9.5, fontname="f_reg", color=(0.90, 0.95, 1.0))

    # Decorative Division Line
    p1.draw_line(pymupdf.Point(50, 310), pymupdf.Point(PAGE_W - 50, 310), color=(0.20, 0.45, 0.70), width=1.0)
    sub_desc = (
        "Edisi Rilis Operasional & Referensi Akademik Seismologi Rekayasa"
        if lang == "id" else
        "Operational Release Edition & Engineering Seismology Academic Reference"
    )
    p1.insert_text((50, 326), sub_desc, fontsize=8.2, fontname="f_it", color=(0.80, 0.90, 1.0))

    # Bottom Half: Author Credentials Card
    builder.draw_card(p1, pymupdf.Rect(50, 410, PAGE_W - 50, 770), bg_col=COLOR_CARD_BG, border_col=COLOR_CARD_BORDER)
    p1.draw_rect(pymupdf.Rect(50, 410, PAGE_W - 50, 440), color=COLOR_NAVY, fill=COLOR_NAVY)
    auth_card_title = "INFORMASI PENYUSUN & PENGESAHAN KERJA PRAKTIK" if lang == "id" else "AUTHOR METADATA & ACADEMIC INTERNSHIP CREDENTIALS"
    p1.insert_text((68, 430), auth_card_title, fontsize=10.2, fontname="f_bold", color=COLOR_WHITE)

    info_html = (
        """
        <table style="font-size:8.4pt; line-height:1.5;">
            <tr><td width="30%"><b>Nama Pengembang</b></td><td>: Ahmad Didane Setyawan Putra</td></tr>
            <tr><td><b>Nomor Induk Mahasiswa (NIM)</b></td><td>: 123120094</td></tr>
            <tr><td><b>Program Studi</b></td><td>: Teknik Geofisika</td></tr>
            <tr><td><b>Fakultas / Jurusan</b></td><td>: Fakultas Teknik Industri / Teknologi Produksi & Industri</td></tr>
            <tr><td><b>Perguruan Tinggi</b></td><td>: Institut Teknologi Sumatera (ITERA)</td></tr>
            <tr><td><b>Instansi Pelaksana KP</b></td><td>: Stasiun Geofisika Kelas I Sleman, BMKG D.I. Yogyakarta</td></tr>
            <tr><td><b>Periode Pelaksanaan</b></td><td>: 20 Juli 2026 – 20 Agustus 2026</td></tr>
            <tr><td><b>Status Proyek</b></td><td>: Karya Ilmiah Mandiri Mahasiswa (Bukan Produk Resmi Komersial BMKG)</td></tr>
            <tr><td><b>Versi Dokumen</b></td><td>: v2.0.0 (Build 2026.08) — Edisi Rilis Resmi</td></tr>
            <tr><td><b>Akses Daring Web</b></td><td>: <a href="https://strong-motion.streamlit.app/">https://strong-motion.streamlit.app/</a></td></tr>
            <tr><td><b>Repositori Kode Sumber</b></td><td>: <a href="https://github.com/ahmaddidan/BSMA-v.2">https://github.com/ahmaddidan/BSMA-v.2</a></td></tr>
        </table>
        """
        if lang == "id" else
        """
        <table style="font-size:8.4pt; line-height:1.5;">
            <tr><td width="32%"><b>Author / Developer</b></td><td>: Ahmad Didane Setyawan Putra</td></tr>
            <tr><td><b>Student ID (NIM)</b></td><td>: 123120094</td></tr>
            <tr><td><b>Study Program</b></td><td>: Geophysical Engineering</td></tr>
            <tr><td><b>Faculty / Division</b></td><td>: Faculty of Industrial Technology</td></tr>
            <tr><td><b>University</b></td><td>: Institut Teknologi Sumatera (ITERA)</td></tr>
            <tr><td><b>Host Institution</b></td><td>: Sleman Geophysical Station Class I, BMKG D.I. Yogyakarta</td></tr>
            <tr><td><b>Internship Period</b></td><td>: July 20, 2026 – August 20, 2026</td></tr>
            <tr><td><b>Project Classification</b></td><td>: Independent Student Academic Work (Non-Commercial)</td></tr>
            <tr><td><b>Document Edition</b></td><td>: v2.0.0 (Build 2026.08) — Official Release Edition</td></tr>
            <tr><td><b>Live Cloud Web Application</b></td><td>: <a href="https://strong-motion.streamlit.app/">https://strong-motion.streamlit.app/</a></td></tr>
            <tr><td><b>GitHub Source Repository</b></td><td>: <a href="https://github.com/ahmaddidan/BSMA-v.2">https://github.com/ahmaddidan/BSMA-v.2</a></td></tr>
        </table>
        """
    )
    builder.insert_html_safe(p1, pymupdf.Rect(65, 455, PAGE_W - 65, 755), info_html)

    # -------------------------------------------------------------
    # PAGE 2 (Executive Summary - Roman ii)
    # -------------------------------------------------------------
    p2 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p2)
    builder.add_page_header_footer(p2, "RINGKASAN EKSEKUTIF" if lang == "id" else "EXECUTIVE SUMMARY", "Halaman ii" if lang == "id" else "Page ii")

    y = builder.draw_chapter_banner(p2, "RINGKASAN EKSEKUTIF & SPESIFIKASI SISTEM" if lang == "id" else "EXECUTIVE PROJECT SUMMARY & SYSTEM SPECIFICATIONS")

    # Card 1: Overview
    builder.draw_card(p2, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 265), bg_col=COLOR_WHITE, border_col=COLOR_CARD_BORDER)
    p2.insert_text((LEFT_X + 16, y + 20), "1. RINGKASAN PROYEK & KAPABILITAS SOFTWARE" if lang == "id" else "1. PROJECT OVERVIEW & SOFTWARE CAPABILITIES",
                   fontsize=9.8, fontname="f_bold", color=COLOR_NAVY)
    p2.draw_line(pymupdf.Point(LEFT_X + 16, y + 26), pymupdf.Point(RIGHT_X - 16, y + 26), color=COLOR_SKY, width=1.0)

    summary_text = (
        """
        <p><b>BMKG Strong Motion Analyzer (BSMA v2.0.0)</b> merupakan perangkat lunak analisis sinyal akselerograf terintegrasi 
        yang dirancang dan dikembangkan secara mandiri dalam rangka pelaksanaan kegiatan Kerja Praktik (KP) mahasiswa 
        Program Studi Teknik Geofisika, Fakultas Teknik Industri, Institut Teknologi Sumatera (ITERA) di Stasiun Geofisika Kelas I Sleman, 
        Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) D.I. Yogyakarta (periode 20 Juli – 20 Agustus 2026). Perangkat lunak ini 
        merupakan karya akademik independen dan bukan merupakan sistem operasional resmi atau produk komersial BMKG.</p>
        <p>Aplikasi ini mengotomatisasi alur kerja pengolahan rekaman getaran tanah kuat (<i>strong ground motion</i>) secara terpadu—mulai 
        dari pembacaan data mentah (<i>raw counts</i>), dekonvolusi fungsi transfer instrumen StationXML, evaluasi integritas kualitas sinyal 
        (Quality Control), pemrosesan sinyal digital (DSP), integrasi kinematika puncak (PGA, PGV, PGD), estimasi intensitas instrumental 
        Skala MMI berbasis perumusan empiris GMICE Worden et al. (2012), hingga pemodelan Spektrum Respons Pseudo-Spectral Acceleration (PSA) 
        elastis redaman 5% dengan opsi solver analitik Nigam–Jennings (1969) dan integrasi implisit Newmark-Beta (1959) yang dapat diperbandingkan 
        langsung dengan standar ketahanan gempa SNI 1726:2019.</p>
        <p><b>Keunggulan Utama Platform:</b> Kompatibilitas dwibahasa, dukungan mode pemrosesan stasiun tunggal dan multi-stasiun (batch mode), 
        alat uji verifikasi silang solver numerik (SDOF Cross-Validation Benchmark), ekspor multi-format (laporan teknis PDF, CSV kinematika, 
        matriks respons spektral, dan paket arsip ZIP), serta jejak audit komputasi (provenance tracking) untuk transparansi saintifik penuh.</p>
        """
        if lang == "id" else
        """
        <p><b>BMKG Strong Motion Analyzer (BSMA v2.0.0)</b> is an integrated accelerograph signal analysis software suite independently 
        designed and developed as part of an undergraduate Academic Internship by a student of the Department of Geophysical Engineering, 
        Faculty of Industrial Technology, Institut Teknologi Sumatera (ITERA) at the Sleman Geophysical Station Class I, Meteorology, 
        Climatology, and Geophysical Agency (BMKG) D.I. Yogyakarta (conducted July 20 – August 20, 2026). This software is an independent 
        academic contribution and does not constitute an official operational system or commercial software product of BMKG.</p>
        <p>The application automates the end-to-end processing workflow of strong ground motion records—spanning raw count ingestion, 
        StationXML instrument transfer function deconvolution, multi-tier signal quality control (QC), digital signal processing (DSP), 
        numerical kinematic integration (PGA, PGV, PGD), instrumental Modified Mercalli Intensity (MMI) estimation based on Worden et al. (2012) 
        GMICE formulations, and 5% damped elastic Single-Degree-of-Freedom (SDOF) Pseudo-Spectral Acceleration (PSA) response spectra modeling 
        featuring both the Nigam–Jennings (1969) analytical solver and the Newmark-Beta (1959) implicit integration solver, benchmarked against 
        the Indonesian building code standard SNI 1726:2019.</p>
        <p><b>Key Innovations:</b> Fully documented open-source architecture, single-station and multi-station batch processing, interactive 
        numerical SDOF solver cross-validation benchmarking, publication-grade multi-format reporting (PDF engineering reports, kinematic CSVs, 
        spectral response matrices, and ZIP archives), and comprehensive execution provenance tracking ensuring scientific reproducibility.</p>
        """
    )
    builder.insert_html_safe(p2, pymupdf.Rect(LEFT_X + 16, y + 34, RIGHT_X - 16, y + 258), summary_text)

    # Card 2: Environment Specifications Table
    y_card2 = y + 280
    builder.draw_card(p2, pymupdf.Rect(LEFT_X, y_card2, RIGHT_X, y_card2 + 200), bg_col=COLOR_CARD_BG, border_col=COLOR_CARD_BORDER)
    p2.insert_text((LEFT_X + 16, y_card2 + 20), "2. SPESIFIKASI LINGKUNGAN PENGEMBANGAN & DEPENDENSI" if lang == "id" else "2. SYSTEM SPECIFICATIONS & RUNTIME DEPENDENCIES",
                   fontsize=9.8, fontname="f_bold", color=COLOR_NAVY)
    p2.draw_line(pymupdf.Point(LEFT_X + 16, y_card2 + 26), pymupdf.Point(RIGHT_X - 16, y_card2 + 26), color=COLOR_SKY, width=1.0)

    specs_html = (
        """
        <table style="font-size:7.4pt; line-height:1.45;">
            <tr><td width="30%"><b>Sistem Operasi</b></td><td>Windows 10/11, Linux (Ubuntu 20.04+), macOS 12+</td></tr>
            <tr><td><b>Lingkungan Python</b></td><td>Python 3.10 s.d. 3.13 (Lingkungan pengujian utama: Python 3.13.2 64-bit)</td></tr>
            <tr><td><b>Pustaka GUI & Visualisasi</b></td><td>Streamlit (≥1.35.0), Plotly (≥5.20.0), Matplotlib (≥3.8.0)</td></tr>
            <tr><td><b>Pustaka Sains & DSP</b></td><td>ObsPy (≥1.4.0), NumPy (≥1.26.0), SciPy (≥1.12.0), Pandas (≥2.0.0)</td></tr>
            <tr><td><b>Pustaka Laporan PDF</b></td><td>PyMuPDF / Fitz (≥1.24.0), FPDF (≥1.7.2)</td></tr>
            <tr><td><b>Kerangka Uji Otomatis</b></td><td>Pytest (≥8.0.0) — 83 unit & regression benchmark tests (100% lulus)</td></tr>
            <tr><td><b>Kebutuhan Memori (RAM)</b></td><td>Minimal 4 GB (disarankan 8 GB untuk batch processing rekaman multi-stasiun)</td></tr>
            <tr><td><b>Deployment Target</b></td><td>Streamlit Community Cloud & Local Workstation Desktop Server</td></tr>
        </table>
        """
        if lang == "id" else
        """
        <table style="font-size:7.4pt; line-height:1.45;">
            <tr><td width="32%"><b>Operating System</b></td><td>Windows 10/11, Linux (Ubuntu 20.04+), macOS 12+</td></tr>
            <tr><td><b>Python Runtime</b></td><td>Python 3.10 to 3.13 (Primary benchmark environment: Python 3.13.2 64-bit)</td></tr>
            <tr><td><b>GUI & Visualization Stack</b></td><td>Streamlit (≥1.35.0), Plotly (≥5.20.0), Matplotlib (≥3.8.0)</td></tr>
            <tr><td><b>Scientific Computing & DSP</b></td><td>ObsPy (≥1.4.0), NumPy (≥1.26.0), SciPy (≥1.12.0), Pandas (≥2.0.0)</td></tr>
            <tr><td><b>Document Generation</b></td><td>PyMuPDF / Fitz (≥1.24.0), FPDF (≥1.7.2)</td></tr>
            <tr><td><b>Automated Test Suite</b></td><td>Pytest (≥8.0.0) — 83 unit and regression benchmark tests (100% passing)</td></tr>
            <tr><td><b>Memory Requirements</b></td><td>Minimum 4 GB RAM (8 GB recommended for multi-station batch workflows)</td></tr>
            <tr><td><b>Deployment Options</b></td><td>Streamlit Community Cloud & Local Workstation Desktop Server</td></tr>
        </table>
        """
    )
    builder.insert_html_safe(p2, pymupdf.Rect(LEFT_X + 16, y_card2 + 34, RIGHT_X - 16, y_card2 + 192), specs_html)

    # -------------------------------------------------------------
    # PAGE 3 (Table of Contents - Roman iii)
    # -------------------------------------------------------------
    p3 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p3)
    builder.add_page_header_footer(p3, "DAFTAR ISI" if lang == "id" else "TABLE OF CONTENTS", "Halaman iii" if lang == "id" else "Page iii")

    y = builder.draw_chapter_banner(p3, "DAFTAR ISI & STRUKTUR PANDUAN TEKNIS" if lang == "id" else "TABLE OF CONTENTS & GUIDEBOOK STRUCTURE")

    builder.draw_card(p3, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 490), bg_col=COLOR_WHITE, border_col=COLOR_CARD_BORDER)
    p3.insert_text((LEFT_X + 20, y + 24), "BAGIAN DOKUMEN & URUTAN BAB" if lang == "id" else "DOCUMENT SECTIONS & CHAPTER LISTING",
                   fontsize=9.8, fontname="f_bold", color=COLOR_NAVY)
    p3.draw_line(pymupdf.Point(LEFT_X + 20, y + 30), pymupdf.Point(RIGHT_X - 20, y + 30), color=COLOR_SKY, width=1.0)

    toc_items_id = [
        ("BAGIAN AWAL (FRONT MATTER)", "", True),
        ("Halaman Judul & Sampul Resmi", "i", False),
        ("Ringkasan Eksekutif & Spesifikasi Sistem", "ii", False),
        ("Daftar Isi & Struktur Panduan Teknis", "iii", False),
        ("BAB I: PENDAHULUAN & RUANG LINGKUP", "1", True),
        ("BAB II: ARSITEKTUR PIPELINE & INGESTI DATA", "2", True),
        ("BAB III: QUALITY CONTROL (QC) & INTEGRITAS SINYAL", "3", True),
        ("BAB IV: DIGITAL SIGNAL PROCESSING (DSP) & FILTERING", "4", True),
        ("BAB V: INTEGRASI KINEMATIKA & KEBIJAKAN BASELINE", "5", True),
        ("BAB VI: PARAMETER KINEMATIKA & ENERGI SEISMIK", "6", True),
        ("BAB VII: ESTIMASI INTENSITAS INSTRUMENTAL SKALA MMI", "7", True),
        ("BAB VIII: SPEKTRUM RESPONS SDOF & STANDAR SNI 1726:2019", "8", True),
        ("BAB IX: PANDUAN OPERASIONAL GUI & WORKFLOW ANALISIS", "9", True),
        ("BAB X: PEMECAHAN MASALAH & BATASAN OPERASIONAL", "10", True),
        ("BAGIAN AKHIR (BACK MATTER)", "", True),
        ("DAFTAR PUSTAKA & RUJUKAN ILMIAH", "11", False),
        ("PROFIL PENGEMBANG & PENGESAHAN AKADEMIK", "12", False),
    ]
    toc_items_en = [
        ("FRONT MATTER", "", True),
        ("Official Title & Cover Page", "i", False),
        ("Executive Project Summary & System Specifications", "ii", False),
        ("Table of Contents & Guidebook Structure", "iii", False),
        ("CHAPTER I: INTRODUCTION & TECHNICAL SCOPE", "1", True),
        ("CHAPTER II: PIPELINE ARCHITECTURE & DATA INGESTION", "2", True),
        ("CHAPTER III: QUALITY CONTROL (QC) & SIGNAL INTEGRITY", "3", True),
        ("CHAPTER IV: DIGITAL SIGNAL PROCESSING (DSP) & FILTERING", "4", True),
        ("CHAPTER V: KINEMATIC INTEGRATION & BASELINE POLICY", "5", True),
        ("CHAPTER VI: KINEMATIC PARAMETERS & SEISMIC ENERGY", "6", True),
        ("CHAPTER VII: INSTRUMENTAL MMI INTENSITY ESTIMATION", "7", True),
        ("CHAPTER VIII: SDOF RESPONSE SPECTRA & SNI 1726:2019 CODE", "8", True),
        ("CHAPTER IX: GUI OPERATIONAL GUIDE & ANALYSIS WORKFLOW", "9", True),
        ("CHAPTER X: TROUBLESHOOTING & OPERATIONAL LIMITATIONS", "10", True),
        ("BACK MATTER", "", True),
        ("REFERENCES & SCIENTIFIC BIBLIOGRAPHY", "11", False),
        ("DEVELOPER PROFILE & ACADEMIC ENDORSEMENT", "12", False),
    ]
    toc_items = toc_items_id if lang == "id" else toc_items_en

    cur_y = y + 50
    for title, page_str, is_bold in toc_items:
        if not page_str:
            # Section header
            cur_y += 6
            p3.draw_rect(pymupdf.Rect(LEFT_X + 20, cur_y - 2, RIGHT_X - 20, cur_y + 14), color=COLOR_CARD_BG, fill=COLOR_CARD_BG)
            p3.insert_text((LEFT_X + 26, cur_y + 9), title, fontsize=7.8, fontname="f_bold", color=COLOR_SKY)
            cur_y += 20
            continue

        font_name = "f_bold" if is_bold else "f_reg"
        font_col = COLOR_NAVY if is_bold else COLOR_DARK
        p3.insert_text((LEFT_X + 26, cur_y), title, fontsize=8.0, fontname=font_name, color=font_col)

        # Leaders
        text_len = font_bold_obj.text_length(title, fontsize=8.0)
        leader_start_x = LEFT_X + 32 + text_len
        leader_end_x = RIGHT_X - 52
        if leader_end_x > leader_start_x:
            dots_count = int((leader_end_x - leader_start_x) / 4.2)
            dots = ". " * (dots_count // 2)
            p3.insert_text((leader_start_x, cur_y), dots, fontsize=7.0, fontname="f_reg", color=COLOR_MUTED)

        p3.insert_text((RIGHT_X - 44, cur_y), page_str, fontsize=8.2, fontname="f_bold", color=COLOR_NAVY)
        cur_y += 18

    # -------------------------------------------------------------
    # BODY PAGE 1 (Halaman 1) - BAB I: PENDAHULUAN
    # -------------------------------------------------------------
    p4 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p4)
    builder.add_page_header_footer(p4, "BAB I — PENDAHULUAN" if lang == "id" else "CHAPTER I — INTRODUCTION",
                                   f"Halaman 1 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 1 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p4, "BAB I: PENDAHULUAN & RUANG LINGKUP" if lang == "id" else "CHAPTER I: INTRODUCTION & TECHNICAL SCOPE")

    b1_html = (
        """
        <p><b>1.1 Latar Belakang & Urgensi Akselerograf</b></p>
        <p>Indonesia merupakan salah satu wilayah seismotektonik paling aktif di dunia akibat pertemuan empat lempeng utama 
        (Eurasia, Indo-Australia, Pasifik, dan Laut Filipina). Rekaman getaran tanah kuat (<i>strong ground motion</i>) yang dicatat 
        oleh akselerograf memberikan informasi fisik krusial percepatan tanah absolut tanpa saturasi instrumen, yang sangat penting 
        untuk mitigasi bahaya gempa bumi, estimasi guncangan instan (USGS ShakeMap), serta evaluasi ketahanan struktur gedung dan infrastruktur vital.</p>
        
        <p><b>1.2 Tujuan Pengembangan & Filosofi Desain</b></p>
        <p>Perangkat lunak <b>BSMA v2.0.0</b> dikembangkan untuk menjawab kebutuhan operasional dan riset seismologi rekayasa:</p>
        <p>• <b>Otomatisasi Penuh</b>: Mengintegrasikan seluruh tahapan dekonvolusi instrumen, penyaringan derau, integrasi kinematika, dan perhitungan spektral dalam satu dasbor terpadu.<br>
        • <b>Integritas Ilmiah & Transparansi</b>: Setiap parameter, formula, dan asumsi komputasi mengacu pada literatur baku (Boore & Bommer 2005, Worden et al. 2012, Nigam & Jennings 1969), serta dicatat dalam jejak audit (<i>provenance tracking</i>).<br>
        • <b>Aksesibilitas Ganda</b>: Tersedia sebagai aplikasi cloud web publik tanpa instalasi (<a href="https://strong-motion.streamlit.app/">strong-motion.streamlit.app</a>) maupun eksekusi lokal pada stasiun kerja geofisika.</p>

        <p><b>1.3 Ruang Lingkup Fungsional & Matriks Fitur</b></p>
        <p>Platform BSMA v2.0.0 mencakup 7 modul analisis utama yang disusun secara terintegrasi:</p>
        <table style="font-size:7.2pt; line-height:1.35;">
            <tr><th width="20%">Modul / Tab</th><th width="35%">Fungsi Utama</th><th>Luaran Ilmiah Utama</th></tr>
            <tr><td><b>1. Summary</b></td><td>Dasbor eksekutif stasiun & waveform preview</td><td>Metadata stasiun, puncak 3-komponen, status QC, badge MMI</td></tr>
            <tr><td><b>2. Waveforms</b></td><td>Visualisasi interaktif kinematika lengkap (a, v, d)</td><td>Grafik triaksial percepatan (m/s²), kecepatan (cm/s), perpindahan (cm)</td></tr>
            <tr><td><b>3. Quality Control</b></td><td>Diagnostik integritas sinyal & anomali fisik</td><td>Skor kualitas (0–100), polar radar, estimasi SNR (dB), status PASS/WARN/FAIL</td></tr>
            <tr><td><b>4. Strong Motion</b></td><td>Analisis energi getaran & durasi signifikan</td><td>Plot Husid akumulasi Intensitas Arias (Ia), durasi D5-95 dan D5-75</td></tr>
            <tr><td><b>5. Intensity</b></td><td>Klasifikasi tingkat guncangan instrumental</td><td>Skala MMI USGS ShakeMap (Worden et al., 2012), kartu dampak & kerusakan</td></tr>
            <tr><td><b>6. Spectrum</b></td><td>Spektrum respons elastis SDOF redaman 5%</td><td>Kurva PSA (T = 0.01–10 s), overlay SNI 1726:2019, SDOF Solver Benchmark</td></tr>
            <tr><td><b>7. Report</b></td><td>Ekspor dokumen teknis & data tabular</td><td>Laporan teknis PDF komprehensif, CSV parameter, CSV matriks spektrum, ZIP</td></tr>
        </table>
        """
        if lang == "id" else
        """
        <p><b>1.1 Background & Critical Need for Strong-Motion Analysis</b></p>
        <p>Indonesia is located within one of the most seismotectonically complex regions in the world, formed by the convergence 
        of four major tectonic plates (Eurasia, Indo-Australia, Pacific, and Philippine Sea). Accelerograph records of strong ground motion 
        provide indispensable, unsaturated records of ground acceleration during moderate-to-large earthquakes, serving as the cornerstone 
        for seismic hazard mitigation, rapid shake mapping (USGS ShakeMap), and structural earthquake engineering.</p>
        
        <p><b>1.2 Development Objectives & Design Philosophy</b></p>
        <p>The <b>BSMA v2.0.0</b> platform was designed to address key engineering and research challenges:</p>
        <p>• <b>End-to-End Automation</b>: Seamlessly ingests raw records, performs instrument response correction, evaluates signal quality, computes kinematics, and calculates response spectra within a unified interface.<br>
        • <b>Scientific Provenance & Reproducibility</b>: Every algorithm and parameter conforms to established literature (Boore & Bommer 2005, Worden et al. 2012, Nigam & Jennings 1969) and is logged into an execution audit trail.<br>
        • <b>Dual Accessibility</b>: Accessible both as a zero-installation cloud web application (<a href="https://strong-motion.streamlit.app/">strong-motion.streamlit.app</a>) and as a local high-performance workstation suite.</p>

        <p><b>1.3 Functional Scope & Module Matrix</b></p>
        <p>The BSMA v2.0.0 suite is organized into 7 primary functional analysis modules:</p>
        <table style="font-size:7.2pt; line-height:1.35;">
            <tr><th width="20%">Module / Tab</th><th width="35%">Primary Scope</th><th>Key Scientific Deliverables</th></tr>
            <tr><td><b>1. Summary</b></td><td>Executive station overview & waveform preview</td><td>Station coordinates, triaxial peaks, QC status badge, MMI badge</td></tr>
            <tr><td><b>2. Waveforms</b></td><td>Interactive 3-component kinematics (a, v, d)</td><td>Synchronized plots for Acceleration (m/s²), Velocity (cm/s), Displacement (cm)</td></tr>
            <tr><td><b>3. Quality Control</b></td><td>Signal integrity diagnostics & anomaly screening</td><td>Quality Score (0–100), polar metric radar, SNR (dB), PASS/WARN/FAIL status</td></tr>
            <tr><td><b>4. Strong Motion</b></td><td>Seismic energy accumulation & duration</td><td>Husid plot of Arias Intensity (Ia), Significant Duration intervals D5-95 / D5-75</td></tr>
            <tr><td><b>5. Intensity</b></td><td>Instrumental ground shaking classification</td><td>USGS ShakeMap MMI (Worden et al., 2012), perceived shaking & damage cards</td></tr>
            <tr><td><b>6. Spectrum</b></td><td>Elastic SDOF 5% damped response spectra</td><td>PSA curves (T = 0.01–10 s), SNI 1726:2019 design overlay, SDOF Benchmark</td></tr>
            <tr><td><b>7. Report</b></td><td>Publication-grade reporting & multi-format export</td><td>Comprehensive technical PDF report, summary CSV, spectral matrix CSV, ZIP</td></tr>
        </table>
        """
    )
    builder.insert_html_safe(p4, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 430), b1_html)

    builder.draw_callout(p4, pymupdf.Rect(LEFT_X, y + 440, RIGHT_X, y + 510),
                         "Standar Integritas Akademik" if lang == "id" else "Academic Integrity Standard",
                         "Perangkat lunak ini dikembangkan secara independen sebagai karya inovasi mahasiswa Kerja Praktik Teknik Geofisika ITERA di BMKG Stasiun Geofisika Sleman. Seluruh formula matematis, kode sumber, dan visualisasi telah divalidasi dengan test suite otomatis 83 pengujian unit berbasis rekaman aktual operasional BMKG."
                         if lang == "id" else
                         "This software was independently developed as an undergraduate innovation project during an academic internship at Sleman Geophysical Station BMKG. All mathematical formulations and visualization routines have been thoroughly verified against an 83-test automated test suite using real operational BMKG accelerograms.",
                         callout_type="success")

    # -------------------------------------------------------------
    # BODY PAGE 2 (Halaman 2) - BAB II: ARSITEKTUR PIPELINE
    # -------------------------------------------------------------
    p5 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p5)
    builder.add_page_header_footer(p5, "BAB II — ARSITEKTUR PIPELINE" if lang == "id" else "CHAPTER II — PIPELINE ARCHITECTURE",
                                   f"Halaman 2 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 2 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p5, "BAB II: ARSITEKTUR PIPELINE & INGESTI DATA" if lang == "id" else "CHAPTER II: PIPELINE ARCHITECTURE & DATA INGESTION")

    pipe_img = create_pipeline_diagram(lang=lang)
    p5.insert_image(pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 130), stream=pipe_img)

    b2_html = (
        """
        <p><b>2.1 Alur Pemrosesan Sinyal Sekuensial</b></p>
        <p>Arsitektur sistem BSMA menerapkan pipa pemrosesan sekuensial 5-tahap yang ketat guna menjamin transparansi (*provenance tracking*) 
        dan reproduksibilitas ilmiah. Setiap tahapan merekam parameter pemrosesan, versi pustaka, dan stempel waktu ke dalam jejak audit.</p>
        
        <p><b>2.2 Format Berkas Gelombang (MiniSEED & SAC)</b></p>
        <p>BSMA mendukung dua format berkas seismologi standar global:</p>
        <p>• <b>MiniSEED (.mseed)</b>: Format standar FDSN (International Federation of Digital Seismograph Networks). Mendukung rekaman triaksial multi-kanal terkompresi dengan penanda waktu UTC berpresisi tinggi.<br>
        • <b>SAC (.sac)</b>: Format Seismic Analysis Code standar IRIS. Mendukung pembacaan header seismologi lengkap (koordinat stasiun, komponen kanal, dan waktu mulai absolut).</p>

        <p><b>2.3 Dekonvolusi Transfer Function StationXML (PAZ & Stage Gain)</b></p>
        <p>Untuk rekaman mentah (<i>raw counts</i>), sistem melakukan dekonvolusi fungsi transfer instrumen berbasis berkas metadata <b>StationXML (.xml)</b>:</p>
        <p>• Ekstraksi kutub dan nol (<i>poles-zeros</i> / PAZ) serta faktor penguatan bertingkat (<i>stage gain</i>) menggunakan kerangka kerja ObsPy.<br>
        • Konversi otomatis satuan digitasi sensor (*counts*) menjadi percepatan fisis baku dalam meter per detik kuadrat ($m/s^2$) atau Gal ($cm/s^2$).<br>
        • Fitur <b>Physical Acceleration Bypass</b>: Jika data yang diunggah telah berdimensi percepatan fisik (misalnya telah dikoreksi sebelumnya), sistem secara cerdas melewati tahap dekonvolusi untuk mencegah distorsi ganda (*double deconvolution error*).</p>
        """
        if lang == "id" else
        """
        <p><b>2.1 Sequential Signal Processing Architecture</b></p>
        <p>The BSMA architecture enforces a rigorous 5-stage sequential execution pipeline designed for full data provenance tracking 
        and scientific reproducibility. Every processing step records filter parameters, algorithm choices, and execution timestamps into an audit log.</p>
        
        <p><b>2.2 Supported Waveform Data Formats (MiniSEED & SAC)</b></p>
        <p>BSMA natively ingests the two dominant formats in international earthquake seismology:</p>
        <p>• <b>MiniSEED (.mseed)</b>: The FDSN standard format for compressed digital seismic time series, supporting multi-channel triaxial streams with microsecond UTC timing precision.<br>
        • <b>SAC (.sac)</b>: The IRIS Seismic Analysis Code format, supporting detailed header metadata (station coordinates, channel orientation, and sampling rate).</p>

        <p><b>2.3 StationXML Instrument Response Deconvolution</b></p>
        <p>When input waveforms are recorded in raw digital counts, BSMA applies instrument transfer function deconvolution via <b>StationXML (.xml)</b>:</p>
        <p>• Parses poles, zeros (PAZ), and stage sensitivity gains via ObsPy.<br>
        • Converts digital counts into true physical ground acceleration in $m/s^2$ or Gal ($cm/s^2$).<br>
        • <b>Physical Acceleration Bypass Mode</b>: If uploaded records are already calibrated in physical acceleration units, the system automatically detects this and bypasses deconvolution, preventing double-correction distortion.</p>
        """
    )
    builder.insert_html_safe(p5, pymupdf.Rect(LEFT_X, y + 140, RIGHT_X, y + 490), b2_html)

    # -------------------------------------------------------------
    # BODY PAGE 3 (Halaman 3) - BAB III: QUALITY CONTROL
    # -------------------------------------------------------------
    p6 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p6)
    builder.add_page_header_footer(p6, "BAB III — QUALITY CONTROL" if lang == "id" else "CHAPTER III — QUALITY CONTROL",
                                   f"Halaman 3 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 3 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p6, "BAB III: QUALITY CONTROL (QC) & INTEGRITAS SINYAL" if lang == "id" else "CHAPTER III: QUALITY CONTROL & SIGNAL INTEGRITY")

    qc_img = create_qc_spider_diagram(lang=lang)
    p6.insert_image(pymupdf.Rect(RIGHT_X - 195, y, RIGHT_X, y + 150), stream=qc_img)

    b3_left = (
        """
        <p><b>3.1 Kerangka Evaluasi 3-Tingkat</b></p>
        <p>BSMA mengimplementasikan sistem screening kualitas sinyal komprehensif dengan skor kuantitatif (0–100):</p>
        <p>• <b>QC PASS (Skor ≥ 70)</b>: Sinyal bersih, bebas anomali fatal, andal untuk analisis rekayasa dan spektrum.<br>
        • <b>QC WARNING (Skor 50–69)</b>: Anomali non-fatal (lonjakan terisolasi, SNR marginal 10–20 dB, atau drift minor). Rekomendasi: periksa jendela filter.<br>
        • <b>QC FAIL (Skor &lt; 50)</b>: Mengalami anomali fatal (sensor <i>clipping</i>, saturasi ADC, sensor mati / <i>flatline</i>, atau data gap).</p>
        """
        if lang == "id" else
        """
        <p><b>3.1 Three-Tier Quality Evaluation</b></p>
        <p>BSMA provides an automated signal screening framework producing an objective Quality Score (0–100):</p>
        <p>• <b>QC PASS (Score ≥ 70)</b>: High-fidelity record suitable for structural engineering and spectra.<br>
        • <b>QC WARNING (Score 50–69)</b>: Non-fatal anomalies (marginal SNR 10–20 dB, isolated spikes, mild baseline tilt). Filtering adjustments advised.<br>
        • <b>QC FAIL (Score &lt; 50)</b>: Fatal anomalies detected (sensor clipping, ADC saturation, flatlines, or corrupted channels).</p>
        """
    )
    builder.insert_html_safe(p6, pymupdf.Rect(LEFT_X, y, RIGHT_X - 205, y + 160), b3_left)

    b3_bottom = (
        """
        <p><b>3.2 Kriteria Penalti & Anomali Fatal (Clipping & ADC Saturation)</b></p>
        <p>Jika terdeteksi fenomena sensor <i>clipping</i> (pemotongan puncak amplitudo akibat guncangan melebihi kapasitas dinamik sensor) 
        atau saturasi ADC (amplitudo konstan berulang pada nilai bit maksimum), sistem secara otomatis memberlakukan <b>Fatal Override</b> 
        sehingga status QC langsung menjadi <b>FAIL</b> terlepas dari skor numerik lainnya. Data yang terkliping dinyatakan tidak valid 
        untuk analisis parameter puncak karena nilai PGA terpotong dan estimasi MMI akan mengalami <i>underestimate</i>.</p>

        <p><b>3.3 Taksonomi Diagnostik 6-Kelas</b></p>
        <table style="font-size:7.0pt; line-height:1.35;">
            <tr><th width="15%">Kelas</th><th width="25%">Karakteristik Fisik</th><th width="20%">Status QC</th><th>Implikasi Operasional</th></tr>
            <tr><td><b>Class 1</b></td><td>Nominal Strong Motion</td><td>PASS (≥70)</td><td>Kualitas prima; hasil PGA, PGV, MMI, dan spektrum respons 100% valid.</td></tr>
            <tr><td><b>Class 2</b></td><td>Low SNR / Weak Motion</td><td>WARNING (50-69)</td><td>Sinyal lemah; filter frekuensi rendah otomatis dinaikkan (0.20–0.40 Hz).</td></tr>
            <tr><td><b>Class 3</b></td><td>Baseline Offset / Drift</td><td>WARNING (50-69)</td><td>Pergeseran DC terdeteksi; koreksi polinomial diterapkan sebelum integrasi.</td></tr>
            <tr><td><b>Class 4</b></td><td>Glitches & Spikes</td><td>WARNING / FAIL</td><td>Lonjakan non-seismik terdeteksi; verifikasi manual dianjurkan.</td></tr>
            <tr><td><b>Class 5</b></td><td>Sensor Clipping / Saturation</td><td>FAIL (&lt;50 Fatal)</td><td>Puncak terpotong; tidak boleh dijadikan acuan perancangan struktur.</td></tr>
            <tr><td><b>Class 6</b></td><td>Sensor Flatline / Dead</td><td>FAIL (&lt;50 Fatal)</td><td>Sensor mati atau kanal terputus; data didiskualifikasi secara otomatis.</td></tr>
        </table>
        """
        if lang == "id" else
        """
        <p><b>3.2 Fatal Overrides: Sensor Clipping & ADC Saturation</b></p>
        <p>When amplitude truncation (sensor clipping exceeding full-scale range) or bit saturation (repeated identical maximum integer counts) 
        is detected, the system triggers a mandatory <b>Fatal Override</b>, forcing the status to <b>FAIL</b> regardless of other metrics. 
        Clipped records are strictly invalidated for peak ground motion analysis because PGA is truncated and MMI estimation becomes severely underestimated.</p>

        <p><b>3.3 Standard Six-Class Diagnostic Taxonomy</b></p>
        <table style="font-size:7.0pt; line-height:1.35;">
            <tr><th width="15%">Class</th><th width="25%">Physical Characteristics</th><th width="20%">QC Status</th><th>Operational Implication</th></tr>
            <tr><td><b>Class 1</b></td><td>Nominal Strong Motion</td><td>PASS (≥70)</td><td>Clean record; PGA, PGV, PGD, MMI, and response spectra fully reliable.</td></tr>
            <tr><td><b>Class 2</b></td><td>Low SNR / Weak Motion</td><td>WARNING (50-69)</td><td>Weak shaking; highpass filter corner adaptively raised (0.20–0.40 Hz).</td></tr>
            <tr><td><b>Class 3</b></td><td>Baseline Offset / Drift</td><td>WARNING (50-69)</td><td>DC shift detected; polynomial detrending applied prior to integration.</td></tr>
            <tr><td><b>Class 4</b></td><td>Glitches & Spikes</td><td>WARNING / FAIL</td><td>Transient electrical/mechanical spike; manual verification advised.</td></tr>
            <tr><td><b>Class 5</b></td><td>Sensor Clipping / Saturation</td><td>FAIL (&lt;50 Fatal)</td><td>Peaks clipped; strictly invalid for engineering design spectra.</td></tr>
            <tr><td><b>Class 6</b></td><td>Sensor Flatline / Dead Channel</td><td>FAIL (&lt;50 Fatal)</td><td>Dead sensor or broken transmission link; record disqualified.</td></tr>
        </table>
        """
    )
    builder.insert_html_safe(p6, pymupdf.Rect(LEFT_X, y + 165, RIGHT_X, y + 510), b3_bottom)

    # -------------------------------------------------------------
    # BODY PAGE 4 (Halaman 4) - BAB IV: DSP & FILTERING
    # -------------------------------------------------------------
    p7 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p7)
    builder.add_page_header_footer(p7, "BAB IV — DSP & FILTERING" if lang == "id" else "CHAPTER IV — DSP & FILTERING",
                                   f"Halaman 4 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 4 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p7, "BAB IV: DIGITAL SIGNAL PROCESSING (DSP) & FILTERING" if lang == "id" else "CHAPTER IV: DIGITAL SIGNAL PROCESSING & FILTERING")

    b4_html = (
        r"""
        <p><b>4.1 Koreksi Garis Dasar Awal (Mean & Polynomial Detrending)</b></p>
        <p>Sebelum penyaringan, deret waktu percepatan dikoreksi dari pergeseran DC (<i>DC offset</i>) dan tren kemiringan instrumen 
        (<i>linear / polynomial drift</i>) sesuai rekomendasi <b>Boore & Bommer (2005)</b>. Tanpa detrending awal, derau frekuensi nol 
        akan teramplifikasi secara kuadratik saat integrasi ke perpindahan, menyebabkan kurva perpindahan melengkung tak berhingga.</p>

        <p><b>4.2 Jendela Kosinus Tukey 5% (Tapering & Reduksi Kebocoran Spektral)</b></p>
        <p>Penyaringan konvolusi pada deret waktu berdurasi terhingga dapat menimbulkan diskontinuitas amplitudo pada ujung rekaman. 
        BSMA menerapkan jendela <b>Tukey cosine taper 5%</b> pada awal dan akhir rekaman sesuai formulasi <b>Harris (1978)</b>:</p>
        <p>• Mengurangi kebocoran spektral (<i>spectral leakage</i>) tanpa memodifikasi amplitudo fase gelombang gempa utama.<br>
        • Transisi kosinus halus menjamin kestabilan respon impuls filter digital.</p>

        <p><b>4.3 Zero-Phase Butterworth Bandpass Filter Orde-4 (Dua Arah sosfiltfilt)</b></p>
        <p>Penyaringan frekuensi dilakukan menggunakan filter <b>Butterworth bandpass orde-4</b> yang dieksekusi secara maju-mundur 
        (<i>forward-backward filtering</i> melalui <code>scipy.signal.sosfiltfilt</code>):</p>
        <p>• <b>Karakteristik Zero-Phase ($\Delta \phi = 0$)</b>: Fase gelombang tidak mengalami pergeseran waktu neto, sehingga waktu tiba fase P dan puncak amplitudo tetap berada pada koordinat waktu yang eksak.<br>
        • <b>Laju Atenuasi Efektif 48 dB/oktaf</b>: Pemrosesan dua arah mengkuadratkan fungsi transfer magnitudo ($|H(f)|^2$), menghasilkan kemiringan atenuasi setara orde-8 pada pita henti, dengan tetap menjaga kestabilan numerik prototipe orde-4 (Virtanen et al., 2020).<br>
        • <b>Representasi SOS (Second-Order Sections)</b>: Mencegah galat pembulatan numerik (<i>numerical underflow/overflow</i>) yang kerap terjadi pada representasi transfer function rasional konvensional.</p>

        <p><b>4.4 Batas Nyquist Safeguard (0.40 fs) & Floor Adaptif Frekuensi Rendah</b></p>
        <p>• <b>Batas Atas Nyquist</b>: Frekuensi sudut atas dibatasi secara ketat pada $f_{\max} \le 0.80 f_{\mathrm{Nyquist}} = 0.40 f_s$ untuk mengeliminasi distorsi aliasing.<br>
        • <b>Floor Adaptif SNR</b>: Pada rekaman dengan rasio sinyal-derau marginal (SNR &lt; 20 dB atau &lt; 10 dB), frekuensi cutoff bawah $f_{\min}$ secara otomatis dinaikkan dari 0.10 Hz ke 0.20 Hz atau 0.40 Hz guna menekan akumulasi derau periode panjang saat integrasi.</p>
        """
        if lang == "id" else
        r"""
        <p><b>4.1 Baseline Detrending (Mean & Polynomial Drift Removal)</b></p>
        <p>Prior to filtering, acceleration time series are cleansed of zero-frequency offsets (DC bias) and mechanical drift 
        following the benchmark guidelines of <b>Boore & Bommer (2005)</b>. Without initial detrending, uncorrected baseline shifts 
        propagate quadratically through double integration, producing catastrophic parabolic displacement trajectories.</p>

        <p><b>4.2 5% Tukey Cosine Window (Tapering & Spectral Leakage Mitigation)</b></p>
        <p>Applying digital filters to finite-length time series induces edge discontinuities. BSMA applies a <b>5% Tukey window</b> 
        to both record boundaries in accordance with <b>Harris (1978)</b>:</p>
        <p>• Mitigates high-frequency spectral leakage across the discrete Fourier transform.<br>
        • Ensures smooth asymptotic transitions to zero at the record edges without altering the primary strong-motion phase amplitudes.</p>

        <p><b>4.3 Zero-Phase 4th-Order Butterworth Bandpass Filtering (Forward-Backward sosfiltfilt)</b></p>
        <p>Filtering is performed using a <b>4th-order Butterworth bandpass filter</b> implemented via two-pass forward-backward processing 
        (<code>scipy.signal.sosfiltfilt</code>):</p>
        <p>• <b>Zero Net Phase Lag ($\Delta \phi = 0$)</b>: Eliminates phase distortion entirely, guaranteeing that phase arrival times and peak accelerations remain strictly locked to their exact physical timestamps.<br>
        • <b>Effective 48 dB/octave Attenuation Rate</b>: Forward-backward filtering squares the magnitude transfer function ($|H(f)|^2$), achieving an 8th-order stopband rolloff while preserving the intrinsic numerical stability of the 4th-order prototype (Virtanen et al., 2020).<br>
        • <b>Second-Order Sections (SOS)</b>: Prevents numerical precision loss and pole sensitivity issues inherent in high-order polynomial representations.</p>

        <p><b>4.4 Nyquist Safeguard Limit (0.40 fs) & Adaptive Low-Frequency SNR Floor</b></p>
        <p>• <b>Nyquist Safeguard</b>: The upper cutoff frequency is strictly capped at $f_{\max} \le 0.80 f_{\mathrm{Nyquist}} = 0.40 f_s$ to safeguard against digital aliasing artifacts.<br>
        • <b>Adaptive SNR Floor</b>: For records with marginal pre-event signal-to-noise ratios (SNR &lt; 20 dB or &lt; 10 dB), the highpass corner $f_{\min}$ is automatically elevated from 0.10 Hz to 0.20 Hz or 0.40 Hz, shielding subsequent kinematic integrations from low-frequency drift.</p>
        """
    )
    builder.insert_html_safe(p7, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 510), b4_html)

    # -------------------------------------------------------------
    # BODY PAGE 5 (Halaman 5) - BAB V: INTEGRASI KINEMATIKA
    # -------------------------------------------------------------
    p8 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p8)
    builder.add_page_header_footer(p8, "BAB V — INTEGRASI KINEMATIKA" if lang == "id" else "CHAPTER V — KINEMATIC INTEGRATION",
                                   f"Halaman 5 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 5 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p8, "BAB V: INTEGRASI KINEMATIKA & KEBIJAKAN BASELINE" if lang == "id" else "CHAPTER V: KINEMATIC INTEGRATION & BASELINE POLICY")

    b5_html = (
        """
        <p><b>5.1 Integrasi Numerik Kumulatif Trapesium (a(t) -> v(t) -> d(t))</b></p>
        <p>Perhitungan riwayat waktu kecepatan $v(t)$ dan perpindahan $d(t)$ dari riwayat percepatan $a(t)$ dilakukan melalui 
        aturan trapesium kumulatif bertahap (<i>cumulative trapezoidal numerical integration</i>):</p>
        <p>$$v(t) = \\int_0^t a(\\tau) \\, d\\tau \\approx \\sum_{i=1}^{k} \\frac{a(t_i) + a(t_{i-1})}{2} \\Delta t$$</p>
        <p>$$d(t) = \\int_0^t v(\\tau) \\, d\\tau \\approx \\sum_{i=1}^{k} \\frac{v(t_i) + v(t_{i-1})}{2} \\Delta t$$</p>

        <p><b>5.2 Kebijakan Saintifik Garis Dasar (Strict Acceleration-Only Detrending)</b></p>
        <p>Sesuai dengan kaidah kalkulus kinematika murni dan standar pemrosesan akselerograf modern, BSMA menerapkan 
        <b>Kebijakan Garis Dasar yang Ketat</b>:</p>
        <p>• <b>Koreksi Garis Dasar Hanya pada Percepatan</b>: Detrending dan filtering diaplikasikan secara penuh dan tuntas pada deret waktu percepatan <i>sebelum</i> proses integrasi dilakukan.<br>
        • <b>Tanpa Pemaksaan Detrending Pasca-Integrasi</b>: BSMA secara sengaja <b>tidak melakukan modifikasi detrending buatan</b> pada kecepatan atau perpindahan hasil integrasi guna menjaga konsistensi turunan fisik fungsional ($a = \\dot{v} = \\ddot{d}$). Pemaksaan polinomial pada kecepatan atau perpindahan pasca-integrasi dapat mendistorsi amplitudo puncak riil gelombang seismik.</p>

        <p><b>5.3 Interpretasi Fisis PGD (Perpindahan Dinamik Transien vs Fling-Step)</b></p>
        <p>Nilai Peak Ground Displacement (PGD) yang dihasilkan oleh BSMA merepresentasikan <b>perpindahan puncak dinamik transien</b> 
        dalam batas pita frekuensi filter ($f_{\\min} - f_{\\max}$), bukan deformasi tektonik statis permanen (<i>static fling-step</i>):</p>
        <p>• Dalam seismologi gempa bumi, deformasi statis permanen di dekat sesar membutuhkan koreksi garis dasar nonlinear khusus (misal metode baseline piecewise Iwan et al. 1985) atau data pengukuran geodetik GPS laju-tinggi (<i>high-rate GNSS</i>).<br>
        • PGD transien BSMA sangat andal dan sesuai untuk evaluasi simpangan maksimum struktur rekayasa teknik sipil dan respon gedung bertingkat.</p>
        """
        if lang == "id" else
        """
        <p><b>5.1 Cumulative Trapezoidal Numerical Integration (a(t) -> v(t) -> d(t))</b></p>
        <p>Velocity time histories $v(t)$ and displacement time histories $d(t)$ are evaluated from the corrected acceleration $a(t)$ 
        via sequential cumulative trapezoidal numerical integration:</p>
        <p>$$v(t) = \\int_0^t a(\\tau) \\, d\\tau \\approx \\sum_{i=1}^{k} \\frac{a(t_i) + a(t_{i-1})}{2} \\Delta t$$</p>
        <p>$$d(t) = \\int_0^t v(\\tau) \\, d\\tau \\approx \\sum_{i=1}^{k} \\frac{v(t_i) + v(t_{i-1})}{2} \\Delta t$$</p>

        <p><b>5.2 Strict Scientific Baseline Policy (Acceleration-Only Correction)</b></p>
        <p>In strict adherence to the principles of kinematic calculus and modern earthquake engineering standards, BSMA enforces 
        a <b>Strict Baseline Correction Policy</b>:</p>
        <p>• <b>Exclusively Applied to Acceleration</b>: Detrending, tapering, and bandpass filtering are completed entirely on acceleration <i>prior</i> to integration.<br>
        • <b>No Post-Integration Baseline Forcing</b>: BSMA deliberately avoids applying artificial polynomial or highpass filtering post-integration to $v(t)$ or $d(t)$, strictly preserving true derivative consistency ($a = \\dot{v} = \\ddot{d}$). Post-integration baseline forcing introduces non-physical phase distortions and artificially diminishes actual seismic peak amplitudes.</p>

        <p><b>5.3 Physical Interpretation of PGD (Transient Dynamic Peak vs. Static Fling-Step)</b></p>
        <p>The Peak Ground Displacement (PGD) output by BSMA represents the <b>transient dynamic peak displacement</b> within the passband 
        of the digital filter ($f_{\\min} - f_{\\max}$), rather than the permanent tectonic static dislocation (<i>static fling-step</i>):</p>
        <p>• Resolving permanent near-fault tectonic displacement seismologically requires specialized nonlinear baseline correction (e.g., Iwan et al. 1985) or co-located high-rate GNSS instrumentation.<br>
        • Transient dynamic PGD is the engineering metric governing peak structural drift, inter-story shear strain, and building damage.</p>
        """
    )
    builder.insert_html_safe(p8, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 510), b5_html)

    # -------------------------------------------------------------
    # BODY PAGE 6 (Halaman 6) - BAB VI: PARAMETER KINEMATIKA & ENERGI
    # -------------------------------------------------------------
    p9 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p9)
    builder.add_page_header_footer(p9, "BAB VI — PARAMETER & ENERGI" if lang == "id" else "CHAPTER VI — KINEMATICS & ENERGY",
                                   f"Halaman 6 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 6 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p9, "BAB VI: PARAMETER KINEMATIKA & ENERGI SEISMIK" if lang == "id" else "CHAPTER VI: KINEMATIC PARAMETERS & SEISMIC ENERGY")

    husid_img = create_husid_diagram(lang=lang)
    p9.insert_image(pymupdf.Rect(RIGHT_X - 195, y, RIGHT_X, y + 145), stream=husid_img)

    b6_top = (
        """
        <p><b>6.1 Ekstraksi Nilai Puncak Kinematika</b></p>
        <p>BSMA mengekstrak parameter gerakan tanah puncak absolut:</p>
        <p>• <b>PGA</b>: $\\max_t |a(t)|$ dalam $m/s^2$ atau Gal ($cm/s^2$).<br>
        • <b>PGV</b>: $\\max_t |v(t)|$ dalam $cm/s$.<br>
        • <b>PGD</b>: $\\max_t |d(t)|$ dalam $cm$.<br>
        • <b>Rasio $V_{\\max}/A_{\\max}$</b>: Indikator kandungan frekuensi dominan dan jarak sumber gempa bumi.</p>
        """
        if lang == "id" else
        """
        <p><b>6.1 Peak Ground Motion Extraction</b></p>
        <p>BSMA extracts absolute peak ground motion parameters:</p>
        <p>• <b>PGA</b>: $\\max_t |a(t)|$ in $m/s^2$ or Gal ($cm/s^2$).<br>
        • <b>PGV</b>: $\\max_t |v(t)|$ in $cm/s$.<br>
        • <b>PGD</b>: $\\max_t |d(t)|$ in $cm$.<br>
        • <b>$V_{\\max}/A_{\\max}$ Ratio</b>: Indicator of predominant spectral period and source-to-site distance.</p>
        """
    )
    builder.insert_html_safe(p9, pymupdf.Rect(LEFT_X, y, RIGHT_X - 205, y + 155), b6_top)

    b6_bottom = (
        """
        <p><b>6.2 Akumulasi Energi Seismik Intensitas Arias (Ia) & Plot Husid</b></p>
        <p><b>Intensitas Arias ($I_a$)</b> mengukur total energi guncangan yang dilepaskan gempa bumi dan diserap oleh osilator struktur 
        sepanjang durasi rekaman, diformulasikan oleh <b>Arias (1970)</b>:</p>
        <p>$$I_a = \\frac{\\pi}{2g} \\int_0^{t_{\\max}} [a(t)]^2 \\, dt \\quad [m/s]$$</p>
        <p>Kurva <b>Plot Husid</b> menggambarkan akumulasi persentase energi $I_a$ terhadap waktu (0% hingga 100%), memperlihatkan 
        laju pelepasan energi seismik secara visual.</p>

        <p><b>6.3 Interval Durasi Signifikan (D5-95 & D5-75)</b></p>
        <p>Durasi getaran kuat dievaluasi berdasarkan interval waktu akumulasi energi Husid sesuai standar <b>Trifunac & Brady (1975)</b>:</p>
        <p>• <b>Durasi Signifikan $D_{5-95}$</b>: $t_{95} - t_5$ (selisih waktu antara pencapaian 5% hingga 95% total energi Arias). Merepresentasikan durasi guncangan efektif yang berpotensi menyebabkan kelelahan struktur (<i>structural fatigue</i>) dan likuefaksi tanah.<br>
        • <b>Durasi Signifikan $D_{5-75}$</b>: $t_{75} - t_5$ (selisih waktu antara 5% hingga 75% total energi Arias), umum digunakan dalam studi perambatan retakan dan disipasi histeresis energi gempa bumi.</p>
        """
        if lang == "id" else
        """
        <p><b>6.2 Cumulative Arias Intensity (Ia) & Husid Plot</b></p>
        <p><b>Arias Intensity ($I_a$)</b> quantifies the total seismic shaking energy dissipated into a population of structural oscillators 
        over the entire duration of ground motion, formulated by <b>Arias (1970)</b>:</p>
        <p>$$I_a = \\frac{\\pi}{2g} \\int_0^{t_{\\max}} [a(t)]^2 \\, dt \\quad [m/s]$$</p>
        <p>The <b>Husid Plot</b> displays the normalized cumulative buildup of Arias Intensity over time (0% to 100%), visualizing the instantaneous 
        energy arrival rate and the most destructive shaking pulses.</p>

        <p><b>6.3 Significant Duration Intervals (D5-95 & D5-75)</b></p>
        <p>Strong motion duration is evaluated using the energy accumulation boundaries defined by <b>Trifunac & Brady (1975)</b>:</p>
        <p>• <b>Significant Duration $D_{5-95}$</b>: $t_{95} - t_5$ (time interval between 5% and 95% cumulative Arias energy). This metric governs cyclic structural fatigue degradation, pore-water pressure generation, and soil liquefaction triggering.<br>
        • <b>Significant Duration $D_{5-75}$</b>: $t_{75} - t_5$ (time interval between 5% and 75% cumulative Arias energy), widely applied in nonlinear hysteretic dissipation and ductility demand assessments.</p>
        """
    )
    builder.insert_html_safe(p9, pymupdf.Rect(LEFT_X, y + 160, RIGHT_X, y + 510), b6_bottom)

    # -------------------------------------------------------------
    # BODY PAGE 7 (Halaman 7) - BAB VII: ESTIMASI MMI
    # -------------------------------------------------------------
    p10 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p10)
    builder.add_page_header_footer(p10, "BAB VII — INTENSITAS MMI" if lang == "id" else "CHAPTER VII — MMI INTENSITY",
                                   f"Halaman 7 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 7 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p10, "BAB VII: ESTIMASI INTENSITAS INSTRUMENTAL SKALA MMI" if lang == "id" else "CHAPTER VII: INSTRUMENTAL MMI INTENSITY ESTIMATION")

    b7_html = (
        """
        <p><b>7.1 Formulasi GMICE Worden et al. (2012) USGS ShakeMap</b></p>
        <p>Estimasi intensitas guncangan tanah Modified Mercalli Intensity (MMI) instrumental dihitung menggunakan relasi empiris 
        Ground-Motion Intensity Conversion Equations (GMICE) standar global <b>Worden et al. (2012)</b> yang digunakan secara resmi 
        dalam sistem USGS ShakeMap.</p>

        <p><b>Formulasi Berdasarkan PGA (Gal):</b></p>
        <p>$$\\text{MMI}_{\\text{PGA}} = \\begin{cases} 1.78 + 1.55 \\log_{10}(\\text{PGA}), & \\log_{10}(\\text{PGA}) \\le 1.57 \\\\ -1.60 + 3.70 \\log_{10}(\\text{PGA}), & \\log_{10}(\\text{PGA}) > 1.57 \\end{cases}$$</p>

        <p><b>Formulasi Berdasarkan PGV (cm/s):</b></p>
        <p>$$\\text{MMI}_{\\text{PGV}} = \\begin{cases} 3.78 + 2.99 \\log_{10}(\\text{PGV}), & \\log_{10}(\\text{PGV}) \\le 0.53 \\\\ 2.40 + 4.96 \\log_{10}(\\text{PGV}), & \\log_{10}(\\text{PGV}) > 0.53 \\end{cases}$$</p>

        <p><b>7.2 Evaluasi Komponen Horizontal Maksimum (Max-H) & Transisi Dominansi PGV</b></p>
        <p>• <b>Komponen Acuan</b>: Nilai PGA dan PGV dievaluasi secara baku dari <b>Komponen Horizontal Maksimum (Max-H)</b> (antara kanal Utara-Selatan dan Timur-Barat), mengecualikan kanal vertikal Z/U sesuai konvensi seismologi rekayasa gempa bumi.<br>
        • <b>Transisi Dominansi PGV</b>: Pada guncangan kuat ($I_{\\text{MMI}} \\ge 5.0$), perumusan berbasis PGV secara otomatis mendominasi penentuan nilai MMI akhir karena kecepatan getaran tanah memiliki korelasi fisik yang jauh lebih erat terhadap potensi kerusakan bangunan dibandingkan percepatan puncak frekuensi tinggi.</p>

        <p><b>7.3 Tabel Korelasi Skala MMI, Deskripsi Guncangan, & Dampak Kerusakan</b></p>
        <table style="font-size:7.0pt; line-height:1.35;">
            <tr><th width="10%">MMI</th><th width="15%">PGA (Gal)</th><th width="15%">PGV (cm/s)</th><th width="20%">Guncangan (Perceived)</th><th>Potensi Kerusakan Fisik (Potential Damage)</th></tr>
            <tr><td><b>I</b></td><td>&lt; 0.17</td><td>&lt; 0.1</td><td>Tidak Terasa</td><td>Tidak ada kerusakan sama sekali.</td></tr>
            <tr><td><b>II - III</b></td><td>0.17 - 1.4</td><td>0.1 - 1.1</td><td>Sangat Lemah / Lemah</td><td>Terasa oleh beberapa orang dalam keadaan diam; tidak ada kerusakan.</td></tr>
            <tr><td><b>IV</b></td><td>1.4 - 3.9</td><td>1.1 - 3.4</td><td>Ringan</td><td>Pintu dan jendela berderik, gerabah pecah; tidak ada kerusakan struktur.</td></tr>
            <tr><td><b>V</b></td><td>3.9 - 9.2</td><td>3.4 - 8.1</td><td>Sedang</td><td>Terasa oleh hampir semua orang; plester dinding retak ringan.</td></tr>
            <tr><td><b>VI</b></td><td>9.2 - 18</td><td>8.1 - 16</td><td>Kuat</td><td>Kerusakan non-struktural ringan; cerobong dan plester jatuh.</td></tr>
            <tr><td><b>VII</b></td><td>18 - 34</td><td>16 - 31</td><td>Sangat Kuat</td><td>Kerusakan ringan pada bangunan berdesain baik; kerusakan sedang pada bangunan biasa.</td></tr>
            <tr><td><b>VIII</b></td><td>34 - 65</td><td>31 - 60</td><td>Hebat</td><td>Kerusakan parah pada bangunan biasa; beberapa struktur mengalami keruntuhan parsial.</td></tr>
            <tr><td><b>IX+</b></td><td>&gt; 65</td><td>&gt; 60</td><td>Dahsyat / Ekstrem</td><td>Kerusakan struktural meluas; pergeseran pondasi dan keruntuhan total.</td></tr>
        </table>
        """
        if lang == "id" else
        """
        <p><b>7.1 Worden et al. (2012) GMICE Formulations (USGS ShakeMap)</b></p>
        <p>Instrumental Modified Mercalli Intensity (MMI) is computed using the global standard Ground-Motion Intensity Conversion 
        Equations (GMICE) established by <b>Worden et al. (2012)</b>, which forms the core of the USGS ShakeMap operational framework.</p>

        <p><b>Formulation Based on PGA (Gal):</b></p>
        <p>$$\\text{MMI}_{\\text{PGA}} = \\begin{cases} 1.78 + 1.55 \\log_{10}(\\text{PGA}), & \\log_{10}(\\text{PGA}) \\le 1.57 \\\\ -1.60 + 3.70 \\log_{10}(\\text{PGA}), & \\log_{10}(\\text{PGA}) > 1.57 \\end{cases}$$</p>

        <p><b>Formulation Based on PGV (cm/s):</b></p>
        <p>$$\\text{MMI}_{\\text{PGV}} = \\begin{cases} 3.78 + 2.99 \\log_{10}(\\text{PGV}), & \\log_{10}(\\text{PGV}) \\le 0.53 \\\\ 2.40 + 4.96 \\log_{10}(\\text{PGV}), & \\log_{10}(\\text{PGV}) > 0.53 \\end{cases}$$</p>

        <p><b>7.2 Maximum Horizontal Component (Max-H) & PGV Dominance Transition</b></p>
        <p>• <b>Reference Component</b>: Peak ground motions are evaluated strictly on the <b>Maximum Horizontal Component (Max-H)</b> (envelope of North-South and East-West), excluding the vertical channel Z/U in accordance with earthquake engineering convention.<br>
        • <b>PGV Dominance at Strong Shaking</b>: At moderate-to-severe ground shaking levels ($I_{\\text{MMI}} \\ge 5.0$), the PGV relationship automatically governs final intensity because ground velocity correlates directly with kinetic energy flux and building structural damage.</p>

        <p><b>7.3 MMI Scale Correlation, Perceived Shaking, & Structural Damage Matrix</b></p>
        <table style="font-size:7.0pt; line-height:1.35;">
            <tr><th width="10%">MMI</th><th width="15%">PGA (Gal)</th><th width="15%">PGV (cm/s)</th><th width="20%">Perceived Shaking</th><th>Potential Physical Structural Damage</th></tr>
            <tr><td><b>I</b></td><td>&lt; 0.17</td><td>&lt; 0.1</td><td>Not Felt</td><td>No damage whatsoever.</td></tr>
            <tr><td><b>II - III</b></td><td>0.17 - 1.4</td><td>0.1 - 1.1</td><td>Weak / Light</td><td>Felt by resting persons indoors; no structural damage.</td></tr>
            <tr><td><b>IV</b></td><td>1.4 - 3.9</td><td>1.1 - 3.4</td><td>Moderate</td><td>Dishes and windows rattle; walls crack slightly; no structural failure.</td></tr>
            <tr><td><b>V</b></td><td>3.9 - 9.2</td><td>3.4 - 8.1</td><td>Strong</td><td>Felt by nearly everyone; small unstable objects overturned.</td></tr>
            <tr><td><b>VI</b></td><td>9.2 - 18</td><td>8.1 - 16</td><td>Very Strong</td><td>Minor non-structural damage; fallen plaster and masonry cracks.</td></tr>
            <tr><td><b>VII</b></td><td>18 - 34</td><td>16 - 31</td><td>Severe</td><td>Moderate damage in standard masonry structures; slight in well-designed buildings.</td></tr>
            <tr><td><b>VIII</b></td><td>34 - 65</td><td>31 - 60</td><td>Violent</td><td>Heavy structural damage; partial collapse in unreinforced masonry.</td></tr>
            <tr><td><b>IX+</b></td><td>&gt; 65</td><td>&gt; 60</td><td>Extreme</td><td>Catastrophic structural failure; widespread building collapses and foundation shifts.</td></tr>
        </table>
        """
    )
    builder.insert_html_safe(p10, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 510), b7_html)

    # -------------------------------------------------------------
    # BODY PAGE 8 (Halaman 8) - BAB VIII: SPEKTRUM RESPONS SDOF
    # -------------------------------------------------------------
    p11 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p11)
    builder.add_page_header_footer(p11, "BAB VIII — SPEKTRUM RESPONS" if lang == "id" else "CHAPTER VIII — RESPONSE SPECTRA",
                                   f"Halaman 8 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 8 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p11, "BAB VIII: SPEKTRUM RESPONS SDOF & STANDAR SNI 1726:2019" if lang == "id" else "CHAPTER VIII: SDOF RESPONSE SPECTRA & SNI 1726:2019 CODE")

    spec_img = create_spectrum_diagram(lang=lang)
    p11.insert_image(pymupdf.Rect(RIGHT_X - 195, y, RIGHT_X, y + 145), stream=spec_img)

    b8_top = (
        """
        <p><b>8.1 Pemodelan Osilator Elastis SDOF Redaman 5%</b></p>
        <p>Spektrum Respons Pseudo-Percepatan (PSA) memodelkan respon dinamik simpangan maksimum $u(t)$ dari rangkaian osilator 
        derajat kebebasan tunggal (Single-Degree-of-Freedom / SDOF) dengan rasio redaman kritis 5% ($\\xi = 0.05$):</p>
        <p>$$\\text{PSA}(T, \\xi) = \\omega^2 S_d(T, \\xi) = \\omega^2 \\max_t |u(t)| \\quad [g \\text{ atau } m/s^2]$$</p>
        <p>Dievaluasi pada 150 titik periode alami logaritmik $T = 0.01 - 10.0$ s.</p>
        """
        if lang == "id" else
        """
        <p><b>8.1 Elastic SDOF 5% Damped Oscillator Modeling</b></p>
        <p>The Pseudo-Spectral Acceleration (PSA) spectrum models the peak dynamic displacement response $u(t)$ of an array 
        of Single-Degree-of-Freedom (SDOF) linear elastic oscillators with 5% critical damping ($\\xi = 0.05$):</p>
        <p>$$\\text{PSA}(T, \\xi) = \\omega^2 S_d(T, \\xi) = \\omega^2 \\max_t |u(t)| \\quad [g \\text{ or } m/s^2]$$</p>
        <p>Evaluated across 150 logarithmically spaced natural periods $T = 0.01 - 10.0$ s.</p>
        """
    )
    builder.insert_html_safe(p11, pymupdf.Rect(LEFT_X, y, RIGHT_X - 205, y + 155), b8_top)

    b8_bottom = (
        """
        <p><b>8.2 Dual Engine Solver: Nigam-Jennings (1969) vs. Newmark-Beta (1959)</b></p>
        <p>BSMA menyediakan dua algoritma pemecah numerik independen:</p>
        <p>• <b>Nigam–Jennings (1969)</b>: Formulasi analitik eksak berbasis matriks transisi status yang mengasumsikan percepatan tanah linier bagian-per-bagian (<i>piecewise-linear</i>). Sangat stabil dan presisi tinggi untuk periode pendek ($T &lt; 0.1$ s).<br>
        • <b>Newmark-Beta (1959)</b>: Skema integrasi implisit beda hingga langkah-waktu dengan parameter percepatan rata-rata konstan ($\\gamma = 1/2, \\beta = 1/4$). Stabil tanpa syarat (<i>unconditionally stable</i>) untuk seluruh rentang periode.</p>

        <p><b>8.3 Fitur SDOF Solver Cross-Validation & Benchmark Presisi</b></p>
        <p>BSMA menghadirkan alat uji presisi numerik interaktif yang menghitung deviasi relatif maksimal, rata-rata, dan selisih RMS antara kedua solver secara langsung. Rata-rata deviasi relatif kedua solver konsisten $&lt; 5\\%$, serta membuktikan batas kaku (<i>rigid limit anchor</i>) $\\lim_{T \\to 0} \\text{PSA}(T) = \\text{PGA}$.</p>

        <p><b>8.4 Spektrum Desain Gempa SNI 1726:2019 (Design Code Reference Overlay)</b></p>
        <p>Aplikasi memfasilitasi perbandingan langsung (<i>overlay</i>) spektrum respons aktual rekaman gempa terhadap kurva spektrum desain <b>SNI 1726:2019</b> ($S_{DS}, S_{D1}, T_0, T_s$) untuk berbagai kelas situs tanah (Tanah Keras SA/SB, Tanah Sedang SD, hingga Tanah Lunak SE). Fitur ini berfungsi sebagai acuan rekayasa kegempaan (<i>design code reference overlay</i>) untuk mengidentifikasi apakah energi gempa melampaui kapasitas desain gedung.</p>
        """
        if lang == "id" else
        """
        <p><b>8.2 Dual Numerical Solver Engines: Nigam-Jennings vs. Newmark-Beta</b></p>
        <p>BSMA equips researchers with two independent numerical engines:</p>
        <p>• <b>Nigam–Jennings (1969)</b>: Exact analytical state-transition matrix formulation assuming piecewise-linear ground excitation. Highly superior in computational precision for short periods ($T &lt; 0.1$ s).<br>
        • <b>Newmark-Beta (1959)</b>: Step-by-step implicit time integration using the average acceleration scheme ($\\gamma = 1/2, \\beta = 1/4$), providing unconditional numerical stability across all periods.</p>

        <p><b>8.3 SDOF Solver Cross-Validation & Numerical Benchmarking</b></p>
        <p>An interactive verification tool computes maximum relative deviation, mean relative deviation, and RMS error between solvers in real time. Mean solver deviation remains strictly $&lt; 5\\%$, rigorously confirming the high-frequency rigid limit anchor $\\lim_{T \\to 0} \\text{PSA}(T) = \\text{PGA}$.</p>

        <p><b>8.4 SNI 1726:2019 Building Code Design Spectrum Overlay</b></p>
        <p>BSMA enables direct overlay comparison against the Indonesian building code standard <b>SNI 1726:2019</b> design spectra ($S_{DS}, S_{D1}, T_0, T_s$) across soil site classes (Hard Rock SA/SB, Medium Soil SD, and Soft Soil SE). This functions as an engineering reference overlay to verify whether observed shaking demands exceed standard building design capacities.</p>
        """
    )
    builder.insert_html_safe(p11, pymupdf.Rect(LEFT_X, y + 160, RIGHT_X, y + 510), b8_bottom)

    # -------------------------------------------------------------
    # BODY PAGE 9 (Halaman 9) - BAB IX: PANDUAN OPERASIONAL GUI
    # -------------------------------------------------------------
    p12 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p12)
    builder.add_page_header_footer(p12, "BAB IX — PANDUAN GUI" if lang == "id" else "CHAPTER IX — GUI GUIDE",
                                   f"Halaman 9 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 9 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p12, "BAB IX: PANDUAN OPERASIONAL GUI & WORKFLOW ANALISIS" if lang == "id" else "CHAPTER IX: GUI OPERATIONAL GUIDE & ANALYSIS WORKFLOW")

    b9_html = (
        """
        <p><b>9.1 Alur Kerja Mode Analisis Stasiun Tunggal (Single-Station Mode)</b></p>
        <p>1. <b>Unggah Data (Sidebar)</b>: Pilih berkas triaksial MiniSEED (.mseed) atau SAC (.sac). Unggah berkas StationXML (.xml) pendamping bila kalibrasi respons instrumen diperlukan.<br>
        2. <b>Konfigurasi Pemrosesan</b>: Atur frekuensi cutoff bandpass filter ($f_{\\min} = 0.10$ Hz, $f_{\\max} = 25.0$ Hz), opsi detrending, dan algoritma solver SDOF pada panel kontrol.<br>
        3. <b>Navigasi Tab Interaktif</b>:
           • <i>Tab 1 (Summary)</i>: Periksa metadata stasiun, pratinjau gelombang 3-kanal, dan status Quality Control.<br>
           • <i>Tab 2 (Waveforms)</i>: Analisis riwayat waktu percepatan, kecepatan, dan perpindahan secara interaktif (zoom, pan, titik puncak).<br>
           • <i>Tab 3 (Quality Control)</i>: Evaluasi radar diagnostik derau, Power Spectral Density (PSD), dan estimasi SNR (dB).<br>
           • <i>Tab 4 (Strong Motion)</i>: Pantau akumulasi kurva energi Plot Husid, durasi signifikan $D_{5-95}$, dan rasio $V_{\\max}/A_{\\max}$.<br>
           • <i>Tab 5 (Intensity)</i>: Lihat estimasi MMI instrumental Worden et al. (2012) dan deskripsi potensi dampak struktural.<br>
           • <i>Tab 6 (Spectrum)</i>: Tampilkan kurva Spektrum Respons PSA, overlay spektrum desain SNI 1726:2019, dan uji SDOF Benchmark.</p>

        <p><b>9.2 Alur Kerja Pemrosesan Batch Multi-Stasiun (Batch Processing Mode)</b></p>
        <p>Untuk gempa regional dengan puluhan rekaman stasiun:
        • Unggah seluruh berkas .mseed atau .sac secara simultan melalui menu Batch Upload.<br>
        • Sistem mengeksekusi pipeline komputasi terisolasi untuk masing-masing stasiun dengan isolasi galat (<i>failure isolation</i>).<br>
        • Menghasilkan <b>Tabel Matriks Komparasi Regional</b> terpadu yang dapat diurutkan berdasarkan stasiun dengan nilai PGA tertinggi atau estimasi MMI terbesar.</p>

        <p><b>9.3 Opsi Pelaporan & Ekspor Data (Tab 7 Report)</b></p>
        <p>BSMA menyediakan fasilitas ekspor lengkap untuk diseminasi dan riset lanjutan:</p>
        <p>• <b>Laporan Teknis PDF</b>: Dokumen PDF resmi berformat publikasi teknis dengan visualisasi seismogram, parameter, dan audit trail.<br>
        • <b>Ringkasan Kinematika CSV</b>: Tabel parameter puncak (PGA, PGV, PGD, MMI, Durasi, $I_a$) siap olah di spreadsheet atau GIS.<br>
        • <b>Matriks Spektrum Respons CSV</b>: Nilai diskret $T$ vs PSA untuk ketiga komponen gelombang.<br>
        • <b>Paket Arsip ZIP</b>: Paket kompresi menyeluruh berisi laporan PDF, seluruh berkas CSV, dan berkas seismogram terproses.</p>
        """
        if lang == "id" else
        """
        <p><b>9.1 Single-Station Analysis Workflow</b></p>
        <p>1. <b>Data Ingestion (Sidebar)</b>: Upload triaxial MiniSEED (.mseed) or SAC (.sac) waveform files. Optionally attach StationXML (.xml) metadata for instrument deconvolution.<br>
        2. <b>Processing Configuration</b>: Define bandpass filter corners ($f_{\\min} = 0.10$ Hz, $f_{\\max} = 25.0$ Hz), detrending polynomial order, and SDOF response spectrum solver engine.<br>
        3. <b>Interactive Tab Exploration</b>:
           • <i>Tab 1 (Summary)</i>: Review station geographic coordinates, 3-component waveform preview, and QC health badges.<br>
           • <i>Tab 2 (Waveforms)</i>: Interactively explore synchronized time histories for acceleration, velocity, and displacement.<br>
           • <i>Tab 3 (Quality Control)</i>: Inspect noise metric radar charts, Power Spectral Density (PSD), and SNR diagnostics.<br>
           • <i>Tab 4 (Strong Motion)</i>: Monitor Arias Intensity Husid accumulation curves, $D_{5-95}$ duration, and $V_{\\max}/A_{\\max}$.<br>
           • <i>Tab 5 (Intensity)</i>: Examine instrumental MMI classifications (Worden et al., 2012) and perceived damage cards.<br>
           • <i>Tab 6 (Spectrum)</i>: Visualize elastic 5% damped PSA response curves, SNI 1726:2019 overlays, and solver benchmarks.</p>

        <p><b>9.2 Multi-Station Batch Processing Workflow</b></p>
        <p>For regional earthquake events recorded across extensive accelerograph networks:
        • Upload multiple station files simultaneously via the batch ingestion panel.<br>
        • The system executes an asynchronous, isolated pipeline loop with automatic failure isolation.<br>
        • Generates a unified <b>Regional Comparison Matrix</b> sortable by highest PGA or maximum instrumental MMI.</p>

        <p><b>9.3 Reporting & Multi-Format Data Export (Tab 7 Report)</b></p>
        <p>BSMA provides comprehensive data products for post-earthquake reporting and research:</p>
        <p>• <b>Technical PDF Report</b>: Publication-grade engineering report featuring full seismograms, kinematic tables, and audit logs.<br>
        • <b>Kinematics Summary CSV</b>: Tabular spreadsheet containing peak kinematics (PGA, PGV, PGD, MMI, $D_{5-95}$, $I_a$).<br>
        • <b>Spectral Response Matrix CSV</b>: Discrete period ($T$) versus PSA values for all three orthogonal components.<br>
        • <b>Full Export ZIP Archive</b>: Bundled archive containing the PDF report, all CSV tables, and processed waveform series.</p>
        """
    )
    builder.insert_html_safe(p12, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 510), b9_html)

    # -------------------------------------------------------------
    # BODY PAGE 10 (Halaman 10) - BAB X: TROUBLESHOOTING
    # -------------------------------------------------------------
    p13 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p13)
    builder.add_page_header_footer(p13, "BAB X — PEMECAHAN MASALAH" if lang == "id" else "CHAPTER X — TROUBLESHOOTING",
                                   f"Halaman 10 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 10 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p13, "BAB X: PANDUAN PEMECAHAN MASALAH & BATASAN OPERASIONAL" if lang == "id" else "CHAPTER X: TROUBLESHOOTING GUIDE & OPERATIONAL LIMITATIONS")

    b10_html = (
        """
        <p><b>10.1 Pemecahan Masalah Umum (Troubleshooting FAQ)</b></p>
        <p>• <b>Q: Mengapa kurva perpindahan (displacement) melengkung drastis ke atas atau ke bawah?</b><br>
        <i>A: Terjadi akibat residual baseline drift pada integrasi ganda derau frekuensi rendah. Solusi: Naikkan frekuensi cutoff bawah ($f_{\\min}$) dari 0.05 Hz menjadi 0.10 Hz atau 0.20 Hz pada panel filter.</i></p>
        <p>• <b>Q: Muncul peringatan 'StationXML Response Correction Bypassed'. Apa artinya?</b><br>
        <i>A: Berkas StationXML tidak cocok dengan metadata stasiun/kanal rekaman, atau sinyal masukan telah terdeteksi berdimensi percepatan fisik ($m/s^2$). Sistem otomatis menggunakan mode Physical Acceleration. Solusi: Pastikan kode stasiun dan jaringan pada berkas .xml dan .mseed identik.</i></p>
        <p>• <b>Q: Mengapa nilai PGA akselerograf berbeda dengan sensor broadband di stasiun yang sama?</b><br>
        <i>A: Akselerograf dioptimalkan merekam percepatan gempa kuat tanpa clipping, sedangkan sensor broadband dioptimalkan untuk kecepatan getaran lemah periode panjang. Untuk rekayasa struktur dan estimasi MMI, akselerograf merupakan acuan utama.</i></p>
        <p>• <b>Q: Mengapa status Quality Control menunjukkan WARNING atau FAIL meski sinyal tampak jelas?</b><br>
        <i>A: Sistem QC mengevaluasi 6 kriteria diagnostik (SNR, sensor clipping, spikes, flatline, pre-event noise, dan DC offset). Status WARNING umumnya dipicu durasi jendela pre-event noise &lt; 5 detik atau SNR &lt; 10 dB. Periksa Tab 3 untuk rincian penalti.</i></p>
        <p>• <b>Q: Kapan sebaiknya memilih solver Nigam-Jennings dibandingkan Newmark-Beta?</b><br>
        <i>A: Formulasi analitik rekursif Nigam-Jennings (1969) sangat direkomendasikan untuk spektrum periode pendek ($T &lt; 0.1$ s) karena mengasumsikan eksitasi percepatan linier bagian-per-bagian. Newmark-Beta (1959) menggunakan integrasi implisit percepatan rata-rata. Keduanya memiliki deviasi rata-rata &lt; 5% yang dapat diverifikasi pada Tab 6.</i></p>

        <p><b>10.2 Batasan Metodologis Perangkat Lunak</b></p>
        <p>• <b>Sensitivitas PGD</b>: Integrasi ganda numerik sangat sensitif terhadap derau frekuensi rendah sisa.<br>
        • <b>Dependensi StationXML</b>: Akurasi dekonvolusi menuntut keabsahan transfer function PAZ & stage gain sensor.<br>
        • <b>Rekaman Derau Tinggi</b>: Sinyal dengan estimasi SNR &lt; 3 dB tidak direkomendasikan untuk analisis kuantitatif.<br>
        • <b>Estimasi MMI Instrumental</b>: Merupakan representasi fisik empiris, bukan pengganti survei makroseismik lapangan.</p>

        <p><b>10.3 Tautan Akses Cloud Web & Repositori GitHub</b></p>
        <p>• <b>Aplikasi Cloud Web</b>: <a href="https://strong-motion.streamlit.app/">https://strong-motion.streamlit.app/</a> (Siap pakai tanpa instalasi).<br>
        • <b>Repositori GitHub</b>: <a href="https://github.com/ahmaddidan/BSMA-v.2">https://github.com/ahmaddidan/BSMA-v.2</a> (Kode sumber terbuka & test suite).</p>
        """
        if lang == "id" else
        """
        <p><b>10.1 Frequently Asked Questions (Troubleshooting FAQ)</b></p>
        <p>• <b>Q: Why does the displacement trajectory curve drastically upward or downward?</b><br>
        <i>A: Caused by residual baseline drift during double numerical integration of low-frequency noise. Solution: Increase the highpass cutoff frequency ($f_{\\min}$) from 0.05 Hz to 0.10 Hz or 0.20 Hz in the filter panel.</i></p>
        <p>• <b>Q: Why do I receive the notification 'StationXML Response Correction Bypassed'?</b><br>
        <i>A: The StationXML file does not match the station/network code of the record, or input waveforms are already calibrated in physical acceleration ($m/s^2$). The system safely defaults to Physical Acceleration mode. Solution: Ensure station/channel metadata match exactly.</i></p>
        <p>• <b>Q: Why does the accelerograph PGA differ from a broadband seismometer at the same site?</b><br>
        <i>A: Accelerographs are designed for high-amplitude strong ground motions without clipping, whereas broadband sensors record weak-motion velocities. For structural engineering and MMI estimation, accelerographs are the standard reference.</i></p>
        <p>• <b>Q: Why does Quality Control report WARNING or FAIL even when the waveform looks clear?</b><br>
        <i>A: The QC engine evaluates 6 distinct diagnostic metrics (clipping, spikes, flatlines, pre-event noise, SNR, DC offset). WARNING is typically triggered by a pre-event window &lt; 5 seconds or SNR &lt; 10 dB. Inspect Tab 3 for detailed penalty breakdowns.</i></p>
        <p>• <b>Q: When should I select Nigam-Jennings instead of Newmark-Beta?</b><br>
        <i>A: Nigam-Jennings (1969) is strongly recommended for short-period spectra ($T &lt; 0.1$ s) due to its exact analytical piecewise-linear formulation. Newmark-Beta (1959) uses implicit average acceleration integration. Both exhibit mean relative discrepancy &lt; 5% verified in Tab 6.</i></p>

        <p><b>10.2 Methodological & Operational Limitations</b></p>
        <p>• <b>PGD Sensitivity</b>: Double numerical integration is inherently sensitive to residual low-frequency baseline drift.<br>
        • <b>StationXML Dependency</b>: Deconvolution accuracy is strictly bounded by the validity of sensor PAZ and sensitivity gains.<br>
        • <b>High Noise Records</b>: Records with estimated SNR &lt; 3 dB are not recommended for quantitative kinematic modeling.<br>
        • <b>Instrumental MMI</b>: Represents an empirical kinematic approximation and does not replace field macroseismic surveys.</p>

        <p><b>10.3 Cloud Web Access & GitHub Source Code</b></p>
        <p>• <b>Live Cloud Web Application</b>: <a href="https://strong-motion.streamlit.app/">https://strong-motion.streamlit.app/</a> (Zero-installation cloud access).<br>
        • <b>GitHub Source Repository</b>: <a href="https://github.com/ahmaddidan/BSMA-v.2">https://github.com/ahmaddidan/BSMA-v.2</a> (Full open-source suite & automated tests).</p>
        """
    )
    builder.insert_html_safe(p13, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 510), b10_html)

    # -------------------------------------------------------------
    # BODY PAGE 11 (Halaman 11) - DAFTAR PUSTAKA (UNNUMBERED)
    # -------------------------------------------------------------
    p14 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p14)
    builder.add_page_header_footer(p14, "DAFTAR PUSTAKA" if lang == "id" else "REFERENCES",
                                   f"Halaman 11 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 11 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p14, "DAFTAR PUSTAKA & RUJUKAN ILMIAH" if lang == "id" else "REFERENCES & SCIENTIFIC BIBLIOGRAPHY")

    ref_desc = (
        "Daftar pustaka di bawah ini memuat landasan teoretis, standar ketahanan gempa nasional (SNI), algoritma dinamika struktur, "
        "serta metodologi pemrosesan sinyal seismologi rekayasa yang diimplementasikan secara langsung pada perangkat lunak BMKG Strong Motion Analyzer (BSMA v2.0.0):"
        if lang == "id" else
        "The following bibliography documents the theoretical foundations, national building design codes (SNI), structural dynamics algorithms, "
        "and engineering seismology signal processing methodologies directly implemented within the BMKG Strong Motion Analyzer (BSMA v2.0.0) computational engine:"
    )
    builder.draw_callout(p14, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 55),
                         "Landasan Teoretis & Kerangka Kerja Komputasi" if lang == "id" else "Theoretical Foundations & Computational Framework",
                         ref_desc, callout_type="info")

    references = [
        ("1. Arias, A. (1970).", "A measure of earthquake intensity. In R. J. Hansen (Ed.), Seismic design for nuclear power plants (pp. 438–483). Cambridge: MIT Press.", ""),
        ("2. Badan Standardisasi Nasional. (2019).", "SNI 1726:2019: Tata cara perencanaan ketahanan gempa untuk struktur bangunan gedung dan non gedung. Jakarta: Badan Standardisasi Nasional.", ""),
        ("3. Beyreuther, M., Barsch, R., Krischer, L., Megies, T., Behr, Y., & Wassermann, J. (2010).", "ObsPy: A Python toolbox for seismology. Seismological Research Letters, 81(3), 530–533.", "https://doi.org/10.1785/gssrl.81.3.530"),
        ("4. Boore, D. M., & Bommer, J. J. (2005).", "Processing of strong-motion accelerograms: Needs, options and consequences. Soil Dynamics and Earthquake Engineering, 25(2), 93–115.", "https://doi.org/10.1016/j.soildyn.2004.10.007"),
        ("5. Harris, F. J. (1978).", "On the use of windows for harmonic analysis with the discrete Fourier transform. Proceedings of the IEEE, 66(1), 51–83.", "https://doi.org/10.1109/PROC.1978.10837"),
        ("6. Newmark, N. M. (1959).", "A method of computation for structural dynamics. Journal of the Engineering Mechanics Division, ASCE, 85(3), 67–94.", "https://doi.org/10.1061/JMCEA3.0000098"),
        ("7. Nigam, N. C., & Jennings, P. C. (1969).", "Calculation of response spectra from strong-motion earthquake records. Bulletin of the Seismological Society of America, 59(2), 909–922.", "https://doi.org/10.1785/BSSA0590020909"),
        ("8. Trifunac, M. D., & Brady, A. G. (1975).", "A study on the duration of strong earthquake ground motion. Bulletin of the Seismological Society of America, 65(3), 581–626.", "https://doi.org/10.1785/BSSA0650030581"),
        ("9. Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., & van der Walt, S. J. (2020).", "SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods, 17(3), 261–272.", "https://doi.org/10.1038/s41592-019-0686-2"),
        ("10. Worden, C. B., Gerstenberger, M. C., Rhoades, D. A., & Wald, D. J. (2012).", "Probabilistic relationships between ground-motion parameters and MMI. Bulletin of the Seismological Society of America, 102(1), 204–221.", "https://doi.org/10.1785/0120110156"),
    ]

    cur_y = y + 68
    for authors, citation, doi in references:
        p14.insert_text((LEFT_X, cur_y), authors, fontsize=8.0, fontname="f_bold", color=COLOR_NAVY)
        cur_y += 11
        # Wrapped citation
        wrapped_lines = textwrap.wrap(citation, width=105)
        for line in wrapped_lines:
            p14.insert_text((LEFT_X + 10, cur_y), line, fontsize=7.4, fontname="f_reg", color=COLOR_DARK)
            cur_y += 10
        if doi:
            p14.insert_text((LEFT_X + 10, cur_y), doi, fontsize=7.2, fontname="f_it", color=COLOR_SKY)
            p14.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(LEFT_X + 10, cur_y - 8, RIGHT_X, cur_y + 2), "uri": doi})
            cur_y += 12
        else:
            cur_y += 4

    # -------------------------------------------------------------
    # BODY PAGE 12 (Halaman 12) - PROFIL PENGEMBANG (UNNUMBERED)
    # -------------------------------------------------------------
    p15 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p15)
    builder.add_page_header_footer(p15, "PROFIL PENGEMBANG" if lang == "id" else "DEVELOPER PROFILE",
                                   f"Halaman 12 dari {TOTAL_BODY_PAGES}" if lang == "id" else f"Page 12 of {TOTAL_BODY_PAGES}")

    y = builder.draw_chapter_banner(p15, "PROFIL PENGEMBANG & PENGESAHAN AKADEMIK" if lang == "id" else "DEVELOPER PROFILE & ACADEMIC ENDORSEMENT")

    # Card 1: Author Profile
    builder.draw_card(p15, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 210), bg_col=COLOR_WHITE, border_col=COLOR_CARD_BORDER)
    p15.insert_text((LEFT_X + 16, y + 22), "1. PROFIL PENGEMBANG SOFTWARE" if lang == "id" else "1. SOFTWARE DEVELOPER PROFILE",
                   fontsize=9.8, fontname="f_bold", color=COLOR_NAVY)
    p15.draw_line(pymupdf.Point(LEFT_X + 16, y + 28), pymupdf.Point(RIGHT_X - 16, y + 28), color=COLOR_SKY, width=1.0)

    author_profile_html = (
        """
        <table style="font-size:7.8pt; line-height:1.5;">
            <tr><td width="30%"><b>Nama Lengkap</b></td><td>: Ahmad Didane Setyawan Putra</td></tr>
            <tr><td><b>Nomor Induk Mahasiswa (NIM)</b></td><td>: 123120094</td></tr>
            <tr><td><b>Program Studi</b></td><td>: Teknik Geofisika (Geophysical Engineering)</td></tr>
            <tr><td><b>Fakultas / Jurusan</b></td><td>: Fakultas Teknik Industri (FTI) / Jurusan Teknologi Produksi dan Industri</td></tr>
            <tr><td><b>Perguruan Tinggi</b></td><td>: Institut Teknologi Sumatera (ITERA), Lampung Selatan, Indonesia</td></tr>
            <tr><td><b>Surel Akademik</b></td><td>: <a href="mailto:ahmad.123120094@student.itera.ac.id">ahmad.123120094@student.itera.ac.id</a></td></tr>
            <tr><td><b>Repositori Proyek</b></td><td>: <a href="https://github.com/ahmaddidan/BSMA-v.2">https://github.com/ahmaddidan/BSMA-v.2</a></td></tr>
            <tr><td><b>Instansi Magang KP</b></td><td>: Stasiun Geofisika Kelas I Sleman, BMKG D.I. Yogyakarta</td></tr>
            <tr><td><b>Periode Kegiatan KP</b></td><td>: 20 Juli 2026 – 20 Agustus 2026</td></tr>
        </table>
        """
        if lang == "id" else
        """
        <table style="font-size:7.8pt; line-height:1.5;">
            <tr><td width="32%"><b>Full Name</b></td><td>: Ahmad Didane Setyawan Putra</td></tr>
            <tr><td><b>Student ID (NIM)</b></td><td>: 123120094</td></tr>
            <tr><td><b>Academic Program</b></td><td>: Department of Geophysical Engineering</td></tr>
            <tr><td><b>Faculty / Division</b></td><td>: Faculty of Industrial Technology (FTI)</td></tr>
            <tr><td><b>University</b></td><td>: Institut Teknologi Sumatera (ITERA), South Lampung, Indonesia</td></tr>
            <tr><td><b>Academic Email</b></td><td>: <a href="mailto:ahmad.123120094@student.itera.ac.id">ahmad.123120094@student.itera.ac.id</a></td></tr>
            <tr><td><b>Project Repository</b></td><td>: <a href="https://github.com/ahmaddidan/BSMA-v.2">https://github.com/ahmaddidan/BSMA-v.2</a></td></tr>
            <tr><td><b>Internship Host Institution</b></td><td>: Sleman Geophysical Station Class I, BMKG D.I. Yogyakarta</td></tr>
            <tr><td><b>Internship Period</b></td><td>: July 20, 2026 – August 20, 2026</td></tr>
        </table>
        """
    )
    builder.insert_html_safe(p15, pymupdf.Rect(LEFT_X + 16, y + 36, RIGHT_X - 16, y + 202), author_profile_html)

    # Card 2: Academic Integrity Statement & Legal Disclaimer
    y_card2 = y + 230
    builder.draw_card(p15, pymupdf.Rect(LEFT_X, y_card2, RIGHT_X, y_card2 + 250), bg_col=COLOR_CARD_BG, border_col=COLOR_CARD_BORDER)
    p15.insert_text((LEFT_X + 16, y_card2 + 22), "2. PERNYATAAN INTEGRITAS AKADEMIK & PENGESAHAN DOKUMEN" if lang == "id" else "2. ACADEMIC INTEGRITY STATEMENT & DOCUMENT ENDORSEMENT",
                   fontsize=9.8, fontname="f_bold", color=COLOR_NAVY)
    p15.draw_line(pymupdf.Point(LEFT_X + 16, y_card2 + 28), pymupdf.Point(RIGHT_X - 16, y_card2 + 28), color=COLOR_SKY, width=1.0)

    endorsement_html = (
        """
        <p><b>Pernyataan Integritas Ilmiah:</b></p>
        <p>Buku Panduan Pengguna dan Referensi Teknis ini disusun sebagai dokumentasi resmi luaran perangkat lunak 
        <b>BMKG Strong Motion Analyzer (BSMA v2.0.0)</b> dalam rangka pemenuhan kewajiban kegiatan Kerja Praktik mahasiswa 
        Program Studi Teknik Geofisika ITERA di Stasiun Geofisika Kelas I Sleman, BMKG D.I. Yogyakarta. Seluruh formulasi matematis, 
        kode sumber komputasi, algoritma solver dinamika struktur, dan rancangan antarmuka visual telah diuji secara komprehensif 
        menggunakan data akselerogram riil operasional BMKG serta diverifikasi terhadap solusi analitik sintetik eksak.</p>
        <p><b>Batasan Pertanggungjawaban (Disclaimer):</b></p>
        <p>Perangkat lunak ini merupakan karya akademik mandiri dan bukan merupakan sistem diseminasi resmi atau perangkat lunak 
        komersial milik BMKG. Penulis dan instansi pelaksana tidak bertanggung jawab atas kerugian langsung maupun tidak langsung 
        yang timbul akibat penggunaan luaran perangkat lunak ini di luar konteks penelitian akademis dan evaluasi ilmiah.</p>
        <p><b>Hak Cipta & Lisensi:</b></p>
        <p>Ahmad Didane Setyawan Putra © 2026. Didistribusikan secara terbuka untuk komunitas ilmiah seismologi dan rekayasa kegempaan 
        di bawah lisensi akademik terbuka.</p>
        """
        if lang == "id" else
        """
        <p><b>Statement of Scientific Integrity:</b></p>
        <p>This User Guidebook and Technical Reference Manual is compiled as the official documentation for the 
        <b>BMKG Strong Motion Analyzer (BSMA v2.0.0)</b> software suite in fulfillment of the undergraduate Academic Internship 
        requirements of the Geophysical Engineering Study Program ITERA at Sleman Geophysical Station Class I, BMKG D.I. Yogyakarta. 
        All mathematical formulations, numerical integration routines, structural dynamics solvers, and graphical interfaces have been 
        rigorously validated against actual BMKG accelerograms and verified against closed-form analytical synthetic solutions.</p>
        <p><b>Legal & Operational Disclaimer:</b></p>
        <p>This software represents an independent academic contribution and does not constitute an official operational system 
        or commercial product of BMKG. Neither the author nor the host institution assumes any liability for direct or indirect damages 
        arising from the application of this software outside the scope of academic research and scientific evaluation.</p>
        <p><b>Copyright & Open Academic License:</b></p>
        <p>Ahmad Didane Setyawan Putra © 2026. Distributed openly for the international seismological and earthquake engineering 
        community under an open academic research license.</p>
        """
    )
    builder.insert_html_safe(p15, pymupdf.Rect(LEFT_X + 16, y_card2 + 36, RIGHT_X - 16, y_card2 + 242), endorsement_html)

    # Save Document
    target_filename = f"BSMA_User_Guidebook_{lang.upper()}.pdf"
    target_path = OUTPUT_DIR / target_filename
    total_page_count = len(doc)
    doc.save(str(target_path), garbage=4, deflate=True)
    doc.close()
    print(f"Successfully generated: {target_path} (Total Pages: {total_page_count})")
    return target_path


def main():
    print("===============================================================")
    print("BSMA Dual-Language User Guidebook Generator")
    print("===============================================================")

    # 1. Generate Indonesian Guidebook
    id_pdf = build_guidebook(lang="id")

    # 2. Generate English Guidebook
    en_pdf = build_guidebook(lang="en")

    # 3. Create default copy BSMA_User_Guidebook.pdf matching Indonesian version
    default_pdf = OUTPUT_DIR / "BSMA_User_Guidebook.pdf"
    shutil.copyfile(id_pdf, default_pdf)
    print(f"Default guidebook updated: {default_pdf}")

    print("===============================================================")
    print("All guidebooks successfully built and ready for release!")
    print("===============================================================")


if __name__ == "__main__":
    main()
