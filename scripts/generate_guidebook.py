"""
BMKG Strong Motion Analyzer (BSMA v2.0.0)
Script: scripts/generate_guidebook.py

Buku Panduan Pengguna & Referensi Teknis Seismologi Rekayasa (11 Halaman)
Pengembang: Ahmad Didane Setyawan Putra (NIM: 123120094)
Program Studi Teknik Geofisika, Fakultas Teknik Industri, Institut Teknologi Sumatera
Pelaksanaan Kerja Praktik: Stasiun Geofisika Kelas I Sleman, BMKG D.I. Yogyakarta
Periode: 20 Juli 2026 – 20 Agustus 2026
Aplikasi Cloud Web: https://strong-motion.streamlit.app/
Repositori GitHub: https://github.com/ahmaddidan/BSMA-v.2
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not (PROJECT_ROOT / "Logo_Judul.png").exists():
    PROJECT_ROOT = Path(r"d:\Perkuliahan\Cadangan project\Project BSMA")

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_OUTPUT_PATH = OUTPUT_DIR / "BSMA_User_Guidebook.pdf"
LOGO_PATH = PROJECT_ROOT / "Logo_Judul.png"
LOGO_ITERA_PATH = PROJECT_ROOT / "Logo_ITERA.png"
LOGO_HEADER_PATH = PROJECT_ROOT / "Logo_Header.png"
LOGO_BMKG_ICON_PATH = PROJECT_ROOT / "Logo_BMKG_Icon.png"
LOGO_ITERA_ICON_PATH = PROJECT_ROOT / "Logo_ITERA_Icon.png"

# Color Palette (RGB 0.0 - 1.0)
COLOR_NAVY = (0.0, 0.176, 0.384)          # #002D62 (BMKG Navy)
COLOR_SKY = (0.008, 0.518, 0.780)          # #0284C7 (Primary Blue)
COLOR_DARK = (0.059, 0.090, 0.165)         # #0F172A (Dark Charcoal)
COLOR_SLATE = (0.278, 0.333, 0.412)        # #475569 (Secondary Text)
COLOR_MUTED = (0.580, 0.639, 0.722)        # #94A3B8 (Muted Gray)
COLOR_CARD_BG = (0.973, 0.980, 0.988)      # #F8FAFC (Card Background)
COLOR_CARD_BORDER = (0.886, 0.910, 0.941)  # #E2E8F0 (Card Border)
COLOR_WHITE = (1.0, 1.0, 1.0)
COLOR_GREEN = (0.020, 0.588, 0.412)        # #059669 (Success)
COLOR_AMBER = (0.851, 0.467, 0.024)        # #D97706 (Warning)
COLOR_RED = (0.863, 0.149, 0.149)          # #DC2626 (Error)

PAGE_W = 595.3
PAGE_H = 841.9
LEFT_X = 50.0
RIGHT_X = 545.3
CONTENT_W = RIGHT_X - LEFT_X
TOTAL_PAGES = 12


FONT_ARIAL = r"C:\Windows\Fonts\arial.ttf"
FONT_ARIAL_BD = r"C:\Windows\Fonts\arialbd.ttf"
FONT_ARIAL_IT = r"C:\Windows\Fonts\ariali.ttf"
FONT_ARIAL_BI = r"C:\Windows\Fonts\arialbi.ttf"


def create_pipeline_diagram() -> bytes:
    """Create high-resolution 5-stage processing pipeline diagram."""
    fig, ax = plt.subplots(figsize=(8.5, 2.5), dpi=220)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    stages = [
        ("1. INGESTION", "MiniSEED / SAC\n+ StationXML PAZ", "#0284C7"),
        ("2. PREPROCESS", "Detrend, Taper 5%\nButterworth 4-Pole", "#0EA5E9"),
        ("3. INTEGRATION", "Acc -> Vel -> Disp\nBaseline Mitigation", "#10B981"),
        ("4. ANALYSIS", "PGA, PGV, Arias,\nResponse Spectrum", "#F59E0B"),
        ("5. REPORTING", "PDF, CSV, ZIP Paket\nInst. MMI ShakeMap", "#6366F1"),
    ]

    for i, (title, desc, col) in enumerate(stages):
        x = 0.3 + i * 1.95
        y = 0.5
        w = 1.65
        h = 2.4
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.12",
            facecolor="white", edgecolor=col, linewidth=1.8,
        )
        ax.add_patch(rect)
        
        header_rect = patches.FancyBboxPatch(
            (x, y + 1.70), w, 0.65, boxstyle="round,pad=0.08,rounding_size=0.1",
            facecolor=col, edgecolor=col, linewidth=0
        )
        ax.add_patch(header_rect)
        ax.text(x + w/2, y + 2.02, title, color="white", weight="bold", fontsize=7.8, ha="center", va="center")
        ax.text(x + w/2, y + 0.85, desc, color="#334155", fontsize=7.0, ha="center", va="center", linespacing=1.3)

        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + 0.28, y + 1.2), xytext=(x + w + 0.03, y + 1.2),
                        arrowprops=dict(arrowstyle="-|>", color="#94A3B8", lw=2.0, mutation_scale=12))

    buf = io.BytesIO()
    plt.tight_layout(pad=0.2)
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def create_mmi_chart() -> bytes:
    """Create visual chart of MMI (Worden et al., 2012) vs PGA."""
    import textwrap
    fig, ax = plt.subplots(figsize=(8.5, 2.7), dpi=220)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    levels = [
        ("MMI I - II", "Sangat Lemah", "< 2.9 Gal", "Tidak dirasakan atau hanya segelintir orang dalam keadaan tenang.", "#F1F5F9", "#475569"),
        ("MMI III - IV", "Ringan", "2.9 - 27 Gal", "Dirasakan di dalam ruangan; getaran seperti truk melintas.", "#DCFCE7", "#166534"),
        ("MMI V - VI", "Sedang - Kuat", "27 - 140 Gal", "Dirasakan semua orang; plester dinding retak; potensi kerusakan non-struktur.", "#FEF9C3", "#854D0E"),
        ("MMI VII - VIII", "Sangat Kuat", "140 - 650 Gal", "Kerusakan ringan hingga sedang pada bangunan biasa; dinding retak.", "#FFEDD5", "#9A3412"),
        ("MMI IX+", "Hebat / Hancur", "> 650 Gal", "Kerusakan struktural parah; rangka melengkung; potensi kehancuran.", "#FEE2E2", "#991B1B"),
    ]

    for i, (mmi, label, pga, dampak, bg_col, txt_col) in enumerate(levels):
        x = 0.2 + i * 1.95
        w = 1.75
        h = 2.95
        rect = patches.FancyBboxPatch(
            (x, 0.2), w, h, boxstyle="round,pad=0.08,rounding_size=0.1",
            facecolor=bg_col, edgecolor=txt_col, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w/2, 2.78, mmi, color=txt_col, weight="bold", fontsize=8.8, ha="center")
        ax.text(x + w/2, 2.38, label, color=txt_col, weight="bold", fontsize=7.6, ha="center")
        ax.text(x + w/2, 1.98, f"PGA: {pga}", color="#0F172A", weight="semibold", fontsize=7.0, ha="center")
        ax.text(x + w/2, 1.63, "Ref: Worden (2012)", color="#64748B", fontsize=6.2, ha="center")
        wrapped = "\n".join(textwrap.wrap(dampak, width=21))
        ax.text(x + w/2, 0.90, wrapped, color="#334155", fontsize=6.1, ha="center", va="center", linespacing=1.25)

    buf = io.BytesIO()
    plt.tight_layout(pad=0.2)
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_math_equation_img(
    latex_str: str, 
    unit_str: str = "", 
    eq_num: str = "", 
    width_pt: float = CONTENT_W, 
    height_pt: float = 26.0
) -> bytes:
    """Render crisp publication-grade equation box with LaTeX MathText, unit, and equation number."""
    dpi = 300
    w_in = width_pt / 72.0
    h_in = height_pt / 72.0
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    fig.patch.set_facecolor("#F8FAFC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#F8FAFC")
    ax.axis("off")
    
    # Outer Border
    rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#CBD5E1", linewidth=0.8, transform=ax.transAxes)
    ax.add_patch(rect)
    
    # Formula (LaTeX style MathText)
    ax.text(0.04, 0.5, latex_str, fontsize=10.2, color="#0F172A", va="center", ha="left", transform=ax.transAxes)
    
    # Unit
    if unit_str:
        ax.text(0.80, 0.5, unit_str, fontsize=8.2, color="#475569", va="center", ha="right", fontfamily="sans-serif", transform=ax.transAxes)
        
    # Equation Number
    if eq_num:
        ax.text(0.96, 0.5, eq_num, fontsize=8.5, color="#94A3B8", va="center", ha="right", style="italic", fontfamily="sans-serif", transform=ax.transAxes)
        
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


class GuidebookBuilder:
    def __init__(self, doc: pymupdf.Document):
        self.doc = doc

    def init_page_fonts(self, page: pymupdf.Page) -> None:
        """Register true UTF-8 fonts on page."""
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

    def insert_html_safe(self, page: pymupdf.Page, rect: pymupdf.Rect, html_body: str) -> None:
        """Render formatted HTML box with CSS styling, justification, links, and italic support."""
        css = """
        body {
            font-family: 'Arial', sans-serif;
            color: #0f172a;
            margin: 0;
            padding: 0;
        }
        p {
            text-align: justify;
            text-justify: inter-word;
            line-height: 1.34;
            margin: 0 0 6px 0;
            font-size: 7.8pt;
        }
        i, em {
            font-style: italic;
        }
        b, strong {
            font-weight: bold;
        }
        a {
            color: #0284c7;
            text-decoration: underline;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 7.3pt;
        }
        th, td {
            border: 1px solid #cbd5e1;
            padding: 4px 6px;
        }
        th {
            background-color: #002d62;
            color: #ffffff;
            font-weight: bold;
            text-align: left;
        }
        """
        full_html = f"<html><head><style>{css}</style></head><body>{html_body}</body></html>"
        page.insert_htmlbox(rect, full_html)

    def add_page_header_footer(self, page: pymupdf.Page, section_badge: str, page_num: int) -> None:
        """Draw clean running header and academic footer on content pages."""
        # Top Header
        page.draw_rect(pymupdf.Rect(LEFT_X, 28, RIGHT_X, 30.5), color=COLOR_SKY, fill=COLOR_SKY)
        page.insert_text((LEFT_X, 23), "BMKG STRONG MOTION ANALYZER (BSMA v2.0.0) — PANDUAN PENGGUNAAN SOFTWARE",
                         fontsize=7.5, fontname="f_bold", color=COLOR_NAVY)
        page.insert_text((RIGHT_X - 145, 23), section_badge.upper(),
                         fontsize=7.2, fontname="f_bold", color=COLOR_SLATE)

        # Bottom Footer (Strictly dignified academic footer)
        page.draw_line(pymupdf.Point(LEFT_X, 810), pymupdf.Point(RIGHT_X, 810), color=COLOR_CARD_BORDER, width=0.8)
        page.insert_text((LEFT_X, 822), "Ahmad Didane Setyawan Putra · Teknik Geofisika, Institut Teknologi Sumatera",
                         fontsize=7.2, fontname="f_it", color=COLOR_SLATE)
        page.insert_text((RIGHT_X - 65, 822), f"Halaman {page_num} dari {TOTAL_PAGES}",
                         fontsize=7.2, fontname="f_bold", color=COLOR_NAVY)

    def draw_card(self, page: pymupdf.Page, rect: pymupdf.Rect, bg_col=COLOR_CARD_BG, border_col=COLOR_CARD_BORDER, border_w=1.0) -> None:
        """Draw solid card box."""
        page.draw_rect(rect, color=border_col, fill=bg_col, width=border_w)

    def draw_callout(self, page: pymupdf.Page, rect: pymupdf.Rect, title: str, text: str, callout_type="info") -> None:
        """Draw highlighted callout box with colored left strip and safe justified text with italics."""
        if callout_type == "info":
            bg = (0.941, 0.973, 1.0)
            border = (0.729, 0.855, 0.988)
            accent = COLOR_SKY
            badge = "[CATATAN TEKNIS]"
        elif callout_type == "warning":
            bg = (1.0, 0.984, 0.922)
            border = (0.992, 0.886, 0.655)
            accent = COLOR_AMBER
            badge = "[PERHATIAN METODOLOGI]"
        else:
            bg = (0.941, 0.988, 0.961)
            border = (0.655, 0.922, 0.749)
            accent = COLOR_GREEN
            badge = "[STANDAR ACUAN]"

        self.draw_card(page, rect, bg_col=bg, border_col=border)
        page.draw_rect(pymupdf.Rect(rect.x0, rect.y0, rect.x0 + 4, rect.y1), color=accent, fill=accent)
        page.insert_text((rect.x0 + 12, rect.y0 + 13), f"{badge} {title}", fontsize=8.0, fontname="f_bold", color=accent)
        
        html = f"""
        <p style="font-size: 7.4pt; line-height: 1.25; margin: 0; text-align: justify; text-justify: inter-word; color: #0f172a;">
            {text}
        </p>
        """
        self.insert_html_safe(page, pymupdf.Rect(rect.x0 + 12, rect.y0 + 17, rect.x1 - 10, rect.y1 - 3), html)

    def draw_section_heading(self, page: pymupdf.Page, y: float, roman: str, title: str) -> float:
        """Draw numbered section title with accent pill using Roman numerals."""
        page.draw_rect(pymupdf.Rect(LEFT_X, y - 2, LEFT_X + 5, y + 16), color=COLOR_NAVY, fill=COLOR_NAVY)
        page.insert_text((LEFT_X + 12, y + 12), f"BAB {roman}: {title}", fontsize=11.0, fontname="f_bold", color=COLOR_NAVY)
        page.draw_line(pymupdf.Point(LEFT_X + 12, y + 18), pymupdf.Point(RIGHT_X, y + 18), color=COLOR_CARD_BORDER, width=0.8)
        return y + 25

    def draw_subsection_heading(self, page: pymupdf.Page, y: float, number: str, title: str) -> float:
        """Draw subsection title."""
        page.insert_text((LEFT_X, y + 10), f"{number} {title}", fontsize=9.2, fontname="f_bold", color=COLOR_SKY)
        return y + 16

    def draw_math_equation(self, page: pymupdf.Page, y: float, latex_str: str, plain_repr: str, height: float = 26.0, unit: str = "", eq_num: str = "") -> float:
        """
        Insert publication-grade mathematical equation box with LaTeX MathText, unit, 
        and clean semantic OCR text layer (render_mode=3 invisible text) for accessibility.
        """
        rect = pymupdf.Rect(LEFT_X, y, RIGHT_X, y + height)
        img_data = render_math_equation_img(latex_str, unit, eq_num, CONTENT_W, height)
        page.insert_image(rect, stream=img_data)
        page.insert_text(pymupdf.Point(LEFT_X + 10, y + height / 2.0 + 3), plain_repr, fontsize=0.5, color=COLOR_WHITE, render_mode=3)
        return y + height + 5

    def draw_equation_box(self, page: pymupdf.Page, rect: pymupdf.Rect, eq_content: list, eq_num: str = "", unit: str = "") -> None:
        """Legacy display equation box drawer."""
        self.draw_card(page, rect, bg_col=COLOR_CARD_BG, border_col=COLOR_CARD_BORDER)
        base_y = rect.y0 + rect.height / 2.0
        for item in eq_content:
            itype = item[0]
            if itype == "text":
                txt, ox, oy, fname, fsize, col = item[1], item[2], item[3], item[4], item[5], item[6]
                page.insert_text(pymupdf.Point(rect.x0 + ox, base_y + oy), txt, fontname=fname, fontsize=fsize, color=col)
            elif itype == "line":
                x0, y0, x1, y1, lw, col = item[1], item[2], item[3], item[4], item[5], item[6]
                page.draw_line(pymupdf.Point(rect.x0 + x0, base_y + y0), pymupdf.Point(rect.x0 + x1, base_y + y1), color=col, width=lw)

        if unit:
            page.insert_text(pymupdf.Point(rect.x1 - 120, base_y + 3.5), unit, fontname="f_reg", fontsize=8.2, color=COLOR_SLATE)
            
        if eq_num:
            page.insert_text(pymupdf.Point(rect.x1 - 50, base_y + 3.5), eq_num, fontname="f_it", fontsize=8.5, color=COLOR_MUTED)


def add_all_hyperlinks(doc: pymupdf.Document) -> int:
    """
    Detect and inject active clickable PDF hyperlinks (LINK_URI) 
    for cloud web app URLs, GitHub repositories, developer emails, and journal DOIs.
    """
    url_mappings = [
        ("https://strong-motion.streamlit.app/", "https://strong-motion.streamlit.app/"),
        ("strong-motion.streamlit.app", "https://strong-motion.streamlit.app/"),
        ("https://github.com/ahmaddidan/BSMA-v.2.git", "https://github.com/ahmaddidan/BSMA-v.2"),
        ("https://github.com/ahmaddidan/BSMA-v.2", "https://github.com/ahmaddidan/BSMA-v.2"),
        ("github.com/ahmaddidan/BSMA-v.2", "https://github.com/ahmaddidan/BSMA-v.2"),
        ("ahmad.123120094@student.itera.ac.id", "mailto:ahmad.123120094@student.itera.ac.id"),
    ]
    doi_mappings = [
        ("https://doi.org/10.1785/gssrl.81.3.530", "https://doi.org/10.1785/gssrl.81.3.530"),
        ("https://doi.org/10.1016/j.soildyn.2004.10.007", "https://doi.org/10.1016/j.soildyn.2004.10.007"),
        ("https://doi.org/10.1109/PROC.1978.10837", "https://doi.org/10.1109/PROC.1978.10837"),
        ("https://doi.org/10.1061/JMCEA3.0000098", "https://doi.org/10.1061/JMCEA3.0000098"),
        ("https://doi.org/10.1785/BSSA0590020909", "https://doi.org/10.1785/BSSA0590020909"),
        ("https://doi.org/10.1785/BSSA0650030581", "https://doi.org/10.1785/BSSA0650030581"),
        ("https://doi.org/10.1038/s41592-019-0686-2", "https://doi.org/10.1038/s41592-019-0686-2"),
        ("https://doi.org/10.1785/0120110156", "https://doi.org/10.1785/0120110156"),
    ]

    total_links = 0
    for p_num, page in enumerate(doc):
        # 1. URLs and email mappings
        for pattern, target in url_mappings:
            rects = page.search_for(pattern)
            for r in rects:
                padded = pymupdf.Rect(r.x0 - 1.5, r.y0 - 1.5, r.x1 + 1.5, r.y1 + 1.5)
                page.insert_link({"kind": pymupdf.LINK_URI, "from": padded, "uri": target})
                total_links += 1

        # 2. Page 1 metadata card specific rects (fixed coordinates)
        if p_num == 0:
            page.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(185, 638, 300, 652), "uri": "https://strong-motion.streamlit.app/"})
            page.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(398, 638, 520, 652), "uri": "https://github.com/ahmaddidan/BSMA-v.2"})
            total_links += 2

        # 3. DOIs across document (e.g. in Bab X)
        for full_doi, uri in doi_mappings:
            rects = page.search_for(full_doi)
            for r in rects:
                page.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(r.x0 - 1.5, r.y0 - 1.5, r.x1 + 1.5, r.y1 + 1.5), "uri": uri})
                total_links += 1

    return total_links


def build_guidebook_pdf() -> Path:
    doc = pymupdf.open()
    builder = GuidebookBuilder(doc)

    # -------------------------------------------------------------
    # PAGE 1: COVER & RINGKASAN PROYEK KERJA PRAKTIK
    # -------------------------------------------------------------
    p1 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p1)

    p1.draw_rect(pymupdf.Rect(0, 0, PAGE_W, 215), color=COLOR_NAVY, fill=COLOR_NAVY)
    p1.draw_rect(pymupdf.Rect(0, 210, PAGE_W, 215), color=COLOR_SKY, fill=COLOR_SKY)


    # ── Cover logos: seamless side-by-side icons directly on navy ──────────
    font_bold_obj = pymupdf.Font(fontfile=FONT_ARIAL_BD)

    # 1. BMKG Logo (Icon: 54x54 pt at x=50..104, y=28..82)
    bmkg_icon = LOGO_BMKG_ICON_PATH if LOGO_BMKG_ICON_PATH.is_file() else LOGO_PATH
    if bmkg_icon.is_file():
        p1.insert_image(pymupdf.Rect(50, 28, 104, 82), filename=str(bmkg_icon))
    bmkg_tw = font_bold_obj.text_length("BMKG", fontsize=10.0)
    p1.insert_text((77 - bmkg_tw / 2, 96), "BMKG", fontsize=10.0, fontname="f_bold", color=COLOR_WHITE)

    # Vertical divider line between logos
    p1.draw_line(pymupdf.Point(118, 28), pymupdf.Point(118, 98), color=(0.35, 0.60, 0.85), width=1.0)

    # 2. ITERA Logo (Icon: 54x54 pt at x=132..186, y=28..82)
    itera_icon = LOGO_ITERA_ICON_PATH if LOGO_ITERA_ICON_PATH.is_file() else LOGO_ITERA_PATH
    if itera_icon.is_file():
        p1.insert_image(pymupdf.Rect(132, 28, 186, 82), filename=str(itera_icon))
    itera_tw = font_bold_obj.text_length("ITERA", fontsize=10.0)
    p1.insert_text((159 - itera_tw / 2, 96), "ITERA", fontsize=10.0, fontname="f_bold", color=COLOR_WHITE)

    # Header text block positioned right beside the dual-logo lockup
    TEXT_X = 206
    p1.insert_text((TEXT_X, 48), "BUKU PANDUAN PENGGUNA & REFERENSI TEKNIS SOFTWARE", fontsize=9.2, fontname="f_bold", color=COLOR_WHITE)
    p1.insert_text((TEXT_X, 65), "Proyek Kerja Praktik Mahasiswa Program Studi Teknik Geofisika", fontsize=8.2, fontname="f_reg", color=(0.85, 0.92, 1.0))
    p1.insert_text((TEXT_X, 80), "Fakultas Teknik Industri, Institut Teknologi Sumatera · BMKG Stasiun Geofisika Sleman", fontsize=7.4, fontname="f_it", color=(0.75, 0.85, 0.95))


    p1.insert_text((LEFT_X, 150), "DOKUMEN PANDUAN PENGGUNA & REFERENSI TEKNIS (USER GUIDEBOOK)", fontsize=8.8, fontname="f_bold", color=(0.75, 0.9, 1.0))
    p1.insert_text((LEFT_X, 175), "BMKG Strong Motion Analyzer (BSMA v2.0.0)", fontsize=17.5, fontname="f_bold", color=COLOR_WHITE)
    p1.insert_text((LEFT_X, 194), "Platform Komputasi Terpadu Sinyal Akselerograf, Kinematika Seismik, & Spektrum Respons", fontsize=8.0, fontname="f_reg", color=(0.9, 0.95, 1.0))

    builder.draw_card(p1, pymupdf.Rect(LEFT_X, 230, RIGHT_X, 715), bg_col=COLOR_WHITE, border_col=COLOR_CARD_BORDER)
    p1.insert_text((70, 258), "RINGKASAN PROYEK & KAPABILITAS SOFTWARE", fontsize=11.5, fontname="f_bold", color=COLOR_NAVY)
    p1.draw_line(pymupdf.Point(70, 266), pymupdf.Point(RIGHT_X - 20, 266), color=COLOR_SKY, width=1.2)

    intro_html = """
    <p>
        <b>BMKG Strong Motion Analyzer (BSMA v2.0.0)</b> merupakan perangkat lunak pengolahan sinyal akselerograf 
        terintegrasi yang dirancang dan dikembangkan secara mandiri dalam pelaksanaan kegiatan Kerja Praktik mahasiswa 
        Program Studi Teknik Geofisika, Fakultas Teknik Industri, Institut Teknologi Sumatera di Stasiun Geofisika Kelas I Sleman, BMKG D.I. Yogyakarta 
        (periode 20 Juli 2026 – 20 Agustus 2026).
    </p>
    <p>
        Perangkat lunak ini bertujuan mengotomatisasi alur kerja analisis rekaman getaran tanah kuat (<i>strong ground motion</i>) 
        guna mendukung kecepatan dan keandalan diseminasi informasi pascagempa bumi, yang mencakup fungsi-fungsi ilmiah utama berikut:
    </p>
    <p>
        • Parsing berkas gelombang seismik dan koreksi respons instrumen melalui dekonvolusi transfer function berbasis StationXML (PAZ & stage gain).<br>
        • Digital Signal Processing (DSP) mencakup baseline polynomial detrending, Tukey cosine tapering 5%, dan zero-phase Butterworth filtering orde-4.<br>
        • Integrasi numerik riwayat percepatan menjadi kecepatan dan perpindahan dengan mitigasi pergeseran baseline (baseline drift mitigation).<br>
        • Ekstraksi parameter kinematika puncak (<i>Peak Ground Acceleration</i> / PGA, <i>Peak Ground Velocity</i> / PGV, <i>Peak Ground Displacement</i> / PGD), Intensitas Arias kumulatif (<i>I<sub>a</sub></i>), Durasi Signifikan (<i>D</i><sub>5-95</sub>), dan rasio <i>V</i><sub>max</sub>/<i>A</i><sub>max</sub>.<br>
        • Estimasi intensitas instrumental Skala MMI berbasis perumusan empiris <i>Ground-Motion Intensity Conversion Equations</i> (GMICE) oleh Worden et al. (2012) yang terintegrasi konvensi USGS ShakeMap.<br>
        • Komputasi Spektrum Respons <i>Pseudo-Spectral Acceleration</i> (PSA) elastis redaman 5% dengan opsi solver analitik rekursif Nigam dan Jennings (1969) dan integrasi implisit Newmark (1959).<br>
        • Aksesibilitas ganda: Tersedia daring via peramban web (<i>cloud web application</i> pada <a href="https://strong-motion.streamlit.app/"><b>https://strong-motion.streamlit.app/</b></a>) maupun eksekusi lokal pada <i>workstation</i>.
    </p>
    """
    builder.insert_html_safe(p1, pymupdf.Rect(70, 274, RIGHT_X - 20, 520), intro_html)

    builder.draw_card(p1, pymupdf.Rect(70, 528, RIGHT_X - 20, 695), bg_col=COLOR_CARD_BG, border_col=COLOR_CARD_BORDER)
    p1.insert_text((85, 550), "SPESIFIKASI PROYEK & LINGKUNGAN PENGEMBANGAN", fontsize=8.8, fontname="f_bold", color=COLOR_NAVY)
    
    meta_rows = [
        ("Perangkat Lunak", "BSMA v2.0.0 (Build 2026.08)", "Program Studi", "Teknik Geofisika"),
        ("Perguruan Tinggi", "Institut Teknologi Sumatera", "Fakultas", "Fakultas Teknik Industri"),
        ("Instansi Pelaksanaan", "BMKG Stasiun Geofisika Sleman", "Periode Kegiatan", "20 Juli – 20 Agustus 2026"),
        ("Akses Cloud Web", "strong-motion.streamlit.app", "Repositori GitHub", "github.com/ahmaddidan/BSMA-v.2"),
    ]
    my = 574
    for r in meta_rows:
        p1.insert_text((85, my), f"• {r[0]}", fontsize=7.6, fontname="f_bold", color=COLOR_SLATE)
        p1.insert_text((185, my), f": {r[1]}", fontsize=7.6, fontname="f_reg", color=COLOR_DARK)
        p1.insert_text((305, my), f"• {r[2]}", fontsize=7.6, fontname="f_bold", color=COLOR_SLATE)
        p1.insert_text((398, my), f": {r[3]}", fontsize=7.6, fontname="f_reg", color=COLOR_DARK)
        my += 23

    p1.insert_text((LEFT_X, 735), "Disusun oleh:", fontsize=8.2, fontname="f_bold", color=COLOR_SLATE)
    p1.insert_text((LEFT_X, 750), "Ahmad Didane Setyawan Putra (NIM: 123120094) — Teknik Geofisika, Fakultas Teknik Industri, Institut Teknologi Sumatera", fontsize=8.2, fontname="f_reg", color=COLOR_DARK)
    p1.insert_text((LEFT_X, 765), "Kerja Praktik di Stasiun Geofisika Kelas I Sleman, BMKG D.I. Yogyakarta (Periode: 20 Juli 2026 – 20 Agustus 2026)", fontsize=7.8, fontname="f_it", color=COLOR_MUTED)

    # -------------------------------------------------------------
    # PAGE 2: DAFTAR ISI (TABLE OF CONTENTS) DENGAN DOT LEADERS
    # -------------------------------------------------------------
    p2 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p2)
    builder.add_page_header_footer(p2, "Daftar Isi Panduan", 2)
    
    font_bold = pymupdf.Font("hebo")
    font_reg = pymupdf.Font("helv")
    
    title_text = "DAFTAR ISI"
    tw = font_bold.text_length(title_text, fontsize=13.0)
    p2.insert_text(pymupdf.Point(PAGE_W / 2.0 - tw / 2.0, 60), title_text, fontname="f_bold", fontsize=13.0, color=COLOR_NAVY)
    
    p2.insert_text(pymupdf.Point(RIGHT_X - font_reg.text_length("Halaman", 8.5), 78), "Halaman", fontname="f_reg", fontsize=8.5, color=COLOR_SLATE)
    p2.draw_line(pymupdf.Point(LEFT_X, 83), pymupdf.Point(RIGHT_X, 83), color=COLOR_CARD_BORDER, width=0.8)

    toc_data = [
        ("RINGKASAN EKSEKUTIF & INFORMASI PENGEMBANG", "1", 0, True),
        ("DAFTAR ISI", "2", 0, True),
        ("BAB I    PENDAHULUAN & ARSITEKTUR PERANGKAT LUNAK", "3", 0, True),
        ("  1.1  Latar Belakang Ilmiah & Urgensi Rekayasa Kegempaan", "3", 15, False),
        ("  1.2  Tujuan Pengembangan Mandiri BSMA v2.0.0", "3", 15, False),
        ("  1.3  Diagram Pipeline Pemrosesan Sinyal 5-Tahap", "3", 15, False),
        ("BAB II   FORMAT DATA MASUKAN, SENSOR & METADATA STATIONXML", "4", 0, True),
        ("  2.1  Format Gelombang Seismik MiniSEED (.mseed) & SAC (.sac)", "4", 15, False),
        ("  2.2  Konvensi Kanal Triaksial Akselerograf (FDSN SEED)", "4", 15, False),
        ("  2.3  Dekonvolusi Respons Instrumen (StationXML Transfer Function)", "4", 15, False),
        ("  2.4  Kebutuhan Sistem & Batasan Masukan Data (System Requirements)", "4", 15, False),
        ("BAB III  DIGITAL SIGNAL PROCESSING (DSP) & FILTERING", "5", 0, True),
        ("  3.1  Klasifikasi & Karakteristik Tipe Filter (Filter Type)", "5", 15, False),
        ("  3.2  Karakteristik Filter Butterworth Orde-4 & Zero-Phase Filtering", "5", 15, False),
        ("  3.3  Seleksi Frekuensi Cutoff & Margin Numerik Nyquist 80%", "5", 15, False),
        ("  3.4  Tapering Jendela Cosine Tukey 5% & Mitigasi Baseline Drift", "5", 15, False),
        ("BAB IV   QUALITY CONTROL (QC) DIAGNOSTICS & PANDUAN VALIDITAS DATA", "6", 0, True),
        ("  4.1  Metodologi Skoring Kualitas Rekaman Seismik (0 - 100)", "6", 15, False),
        ("  4.2  Sistem Evaluasi Kualitas: Skor, Diagnostic Flags & Status", "6", 15, False),
        ("  4.3  Tabel Klasifikasi 6 Kategori Diagnostik Sinyal Aktual", "6", 15, False),
        ("  4.4  Parameter Diagnostik & Formulasi SNR (Signal-to-Noise Ratio)", "6", 15, False),
        ("BAB V    PANDUAN OPERASIONAL LANGKAH DEMI LANGKAH", "7", 0, True),
        ("  5.1  Metode Akses Cloud Web Application (strong-motion.streamlit.app)", "7", 15, False),
        ("  5.2  Metode Eksekusi Workstation Lokal (Git Clone & Lingkungan Python)", "7", 15, False),
        ("  5.3  Alur Kerja 4 Langkah Pemrosesan Data Seismogram", "7", 15, False),
        ("BAB VI   EKSPLORASI 7 TAB FITUR ANALISIS & INTERPRETASI HASIL", "8", 0, True),
        ("  6.1  Tab 1 - Summary & Tab 2 - Waveforms", "8", 15, False),
        ("  6.2  Tab 3 - Quality Control & Tab 4 - Strong Motion Parameters", "8", 15, False),
        ("  6.3  Tab 5 - Instrumental Intensity (MMI) & Tab 6 - Response Spectrum", "8", 15, False),
        ("  6.4  Tab 7 - Technical Report PDF & Interaktivitas Plotly", "8", 15, False),
        ("BAB VII  FITUR LANJUTAN, UTILITAS & CONTOH HASIL ANALISIS", "9", 0, True),
        ("  7.1  Analisis Multi-Stasiun Serentak (Batch Processing Mode)", "9", 15, False),
        ("  7.2  Ekspor Terpadu Berkas Arsip ZIP (Bulk Export Package)", "9", 15, False),
        ("  7.3  Komparasi Solver SDOF: Formulasi Nigam–Jennings vs Newmark–Beta", "9", 15, False),
        ("  7.4  Antarmuka Dual-Theme Switcher (Light Mode & Dark Mode)", "9", 15, False),
        ("  7.5  Contoh Kasus Pengolahan Data Riil (Worked Example)", "9", 15, False),
        ("BAB VIII FORMULASI MATEMATIS KINEMATIKA SEISMIK & SKALA MMI", "10", 0, True),
        ("  8.1  Parameter Kinematika Gerakan Tanah: PGA, PGV, dan PGD", "10", 15, False),
        ("  8.2  Intensitas Arias Kumulatif (Ia) & Durasi Signifikan D5-95", "10", 15, False),
        ("  8.3  Pseudo-Spectral Acceleration (PSA) SDOF Redaman 5%", "10", 15, False),
        ("  8.4  Estimasi Instrumental Skala Intensitas MMI (Worden et al., 2012)", "10", 15, False),
        ("BAB IX   PEMECAHAN MASALAH, KETERBATASAN & PROFIL PENGEMBANG", "11", 0, True),
        ("  9.1  Pemecahan Masalah Umum (Troubleshooting FAQ)", "11", 15, False),
        ("  9.2  Batasan Metodologis Perangkat Lunak (Known Limitations)", "11", 15, False),
        ("  9.3  Luaran Akses Software: Cloud Web App & Repositori GitHub", "11", 15, False),
        ("  9.4  Profil Pengembang & Informasi Proyek Kerja Praktik", "11", 15, False),
        ("BAB X    DAFTAR PUSTAKA & RUJUKAN ILMIAH", "12", 0, True),
    ]

    cur_y = 100.0
    dot_pattern = " . "
    dot_w = font_reg.text_length(dot_pattern, fontsize=7.5)

    for item_title, page_str, indent, is_bold in toc_data:
        f_name = "f_bold" if is_bold else "f_reg"
        f_size = 8.3 if is_bold else 7.8
        f_color = COLOR_NAVY if is_bold else COLOR_DARK
        
        tx = LEFT_X + indent
        tw = (font_bold if is_bold else font_reg).text_length(item_title, fontsize=f_size)
        pw = (font_bold if is_bold else font_reg).text_length(page_str, fontsize=f_size)
        
        dots_x0 = tx + tw + 4
        dots_x1 = RIGHT_X - pw - 4
        
        p2.insert_text(pymupdf.Point(tx, cur_y), item_title, fontname=f_name, fontsize=f_size, color=f_color)
        
        if dots_x1 > dots_x0:
            num_dots = int((dots_x1 - dots_x0) / dot_w)
            if num_dots > 0:
                p2.insert_text(pymupdf.Point(dots_x0, cur_y), dot_pattern * num_dots, fontname="f_reg", fontsize=7.5, color=COLOR_MUTED)
        
        p2.insert_text(pymupdf.Point(RIGHT_X - pw, cur_y), page_str, fontname=f_name, fontsize=f_size, color=f_color)
        cur_y += 14.5 if is_bold else 13.0

    # -------------------------------------------------------------
    # PAGE 3: BAB I - PENDAHULUAN & ARSITEKTUR SOFTWARE
    # -------------------------------------------------------------
    p3 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p3)
    builder.add_page_header_footer(p3, "Pendahuluan & Arsitektur", 3)
    y = 50.0

    y = builder.draw_section_heading(p3, y, "I", "PENDAHULUAN & ARSITEKTUR PERANGKAT LUNAK")
    y = builder.draw_subsection_heading(p3, y, "1.1", "Latar Belakang Ilmiah & Urgensi Rekayasa Seismologi")

    p3_html_1 = """
    <p>
        Rekayasa kegempaan (<i>earthquake engineering</i>) dan seismologi teknik (<i>engineering seismology</i>) bertumpu 
        pada ketersediaan data getaran tanah kuat (<i>strong ground motion</i>) bermutu tinggi guna mitigasi bencana dan perancangan 
        infrastruktur tahan gempa. Sensor akselerograf memegang peranan krusial karena dirancang khusus untuk mempertahankan respons 
        pengukuran pada rentang percepatan yang lebih tinggi sehingga meminimalkan risiko saturasi (<i>clipping</i>), berbeda dengan 
        sensor kecepatan pita lebar (<i>broadband seismometer</i>) yang rentan jenuh saat merekam goncangan gempa kuat di zona medan dekat (<i>near-field</i>).
    </p>
    <p>
        Analisis rekaman akselerogram membutuhkan serangkaian transformasi sinyal terkalibrasi guna menghasilkan parameter kinematika puncak 
        (<i>peak ground motion</i>), estimasi intensitas guncangan instrumental Skala MMI (<i>Modified Mercalli Intensity</i>), serta 
        spektrum respons percepatan elastik (<i>pseudo-spectral acceleration</i> / PSA) sistem berderajat kebebasan tunggal (<i>single degree of freedom</i> / SDOF) 
        dengan rasio redaman 5%, yang merupakan parameter acuan dalam analisis rekayasa ketahanan gempa (Badan Standardisasi Nasional [BSN], 2019).
    </p>
    """
    builder.insert_html_safe(p3, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 105), p3_html_1)
    y += 110

    y = builder.draw_subsection_heading(p3, y, "1.2", "Tujuan Pengembangan Mandiri BSMA v2.0.0")
    p3_html_2 = """
    <p>
        Perangkat lunak <b>BSMA v2.0.0</b> dikembangkan secara mandiri dalam rangka Kerja Praktik mahasiswa Teknik Geofisika Institut 
        Teknologi Sumatera di Stasiun Geofisika Kelas I Sleman, BMKG D.I. Yogyakarta, dengan sasaran fungsional utama:
    </p>
    <p>
        1. <b>Otomasi Pemrosesan End-to-End</b>: Mengintegrasikan parsing berkas MiniSEED/SAC, koreksi respons instrumen StationXML, 
        pemrosesan sinyal digital (DSP), ekstraksi kinematika seismik, dan pemodelan spektrum respons dalam satu alur kerja terpadu.<br>
        2. <b>Aksesibilitas Ganda</b>: Menyediakan fleksibilitas akses melalui aplikasi komputasi awan (<i>cloud web application</i>) 
        tanpa instalasi lokal di <a href="https://strong-motion.streamlit.app/"><b>https://strong-motion.streamlit.app/</b></a> serta opsi instalasi pada stasiun kerja lokal (<i>workstation</i>).<br>
        3. <b>Standardisasi Diseminasi</b>: Mengotomatisasi penyusunan laporan teknis (<i>technical report</i>) berformat PDF komprehensif, lembar data CSV, dan paket arsip ZIP terkompresi.
    </p>
    """
    builder.insert_html_safe(p3, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 90), p3_html_2)
    y += 95

    y = builder.draw_subsection_heading(p3, y, "1.3", "Diagram Pipeline Pemrosesan Sinyal 5-Tahap")
    pipeline_img = create_pipeline_diagram()
    p3.insert_image(pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 130), stream=pipeline_img)
    y += 136

    builder.draw_callout(
        p3,
        pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 70),
        "Keterkaitan Komputasi Spektrum Respons SDOF dengan Standar SNI 1726:2019",
        "BSMA menghitung spektrum respons elastik percepatan (PSA) dari sistem berderajat kebebasan tunggal (SDOF) "
        "dengan rasio redaman kritis 5% (periode T = 0.01 hingga 10.0 s). Nilai spektral yang dihasilkan merepresentasikan "
        "respons dinamik murni dari rekaman akselerogram riil di stasiun pengamatan, yang dapat diperbandingkan dengan ketentuan "
        "spektrum respons desain pada SNI 1726:2019 (BSN, 2019) guna evaluasi kebutuhan ketahanan gempa bangunan di lokasi bersangkutan.",
        callout_type="success"
    )

    # -------------------------------------------------------------
    # PAGE 4: BAB II - FORMAT DATA MASUKAN, STANDAR SENSOR & METADATA
    # -------------------------------------------------------------
    p4 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p4)
    builder.add_page_header_footer(p4, "Format Masukan & Sensor", 4)
    y = 50.0

    y = builder.draw_section_heading(p4, y, "II", "FORMAT DATA MASUKAN, SENSOR & METADATA STATIONXML")
    y = builder.draw_subsection_heading(p4, y, "2.1", "Format Gelombang Seismik MiniSEED (.mseed) & SAC (.sac)")

    p4_html_1 = """
    <p>
        BSMA mendukung dua format baku pertukaran data seismologi internasional: <b>MiniSEED (.mseed)</b> berstandar FDSN 
        (<i>International Federation of Digital Seismograph Networks</i>) dengan kompresi STEIM-1/STEIM-2, serta berkas 
        <b>SAC (.sac)</b> keluaran IRIS (<i>Incorporated Research Institutions for Seismology</i>). Sistem secara otomatis 
        memvalidasi kelengkapan header: kode jaringan (<i>network</i>), nama stasiun (<i>station</i>), lokasi (<i>location</i>), 
        kode kanal (<i>channel</i>), waktu mulai perekaman (<i>starttime</i>), serta laju sampel data (<i>sampling rate</i>).
    </p>
    """
    builder.insert_html_safe(p4, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 45), p4_html_1)
    y += 50

    y = builder.draw_subsection_heading(p4, y, "2.2", "Konvensi Kanal Triaksial Akselerograf (FDSN SEED)")
    p4_html_2 = """
    <p>
        Rekaman gerakan tanah kuat dianalisis secara simultan pada tiga komponen ortogonal (triaksial). Berdasarkan standar 
        FDSN SEED, instrumen akselerograf dikarakterisasi oleh kode instrumen '<b>N</b>' (akselerometer laju sampel tinggi &gt; 80 Hz) 
        atau '<b>E</b>' (akselerometer <i>short-period</i>). Kode '<b>L</b>' (misal <i>HLZ</i>) merepresentasikan seismometer periode panjang 
        (1 Hz) dan bukan akselerograf. Konvensi kode kanal triaksial baku meliputi:
    </p>
    <table>
        <thead>
            <tr>
                <th style="width: 18%;">Kanal</th>
                <th style="width: 25%;">Orientasi Fisik</th>
                <th style="width: 27%;">Rentang Frekuensi Rekomendasi</th>
                <th style="width: 30%;">Peranan Analisis Kegempaan</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><b>HNZ / ENZ</b></td>
                <td>Komponen Vertikal (<i>Up-Down</i>)</td>
                <td>0.05 – 40.0 Hz</td>
                <td>Identifikasi fase P primer dan evaluasi gaya dorong vertikal.</td>
            </tr>
            <tr style="background-color: #f8fafc;">
                <td><b>HNN / HN1</b></td>
                <td>Komponen Horizontal (N–S)</td>
                <td>0.05 – 40.0 Hz</td>
                <td>Analisis gaya geser horizontal arah utara–selatan.</td>
            </tr>
            <tr>
                <td><b>HNE / HN2</b></td>
                <td>Komponen Horizontal (E–W)</td>
                <td>0.05 – 40.0 Hz</td>
                <td>Analisis gaya geser horizontal arah timur–barat.</td>
            </tr>
        </tbody>
    </table>
    """
    builder.insert_html_safe(p4, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 140), p4_html_2)
    y += 145

    y = builder.draw_subsection_heading(p4, y, "2.3", "Dekonvolusi Respons Instrumen (StationXML Transfer Function)")
    p4_html_3 = """
    <p>
        BSMA menyediakan dua mode pemrosesan respons instrumen yang fleksibel:<br>
        • <b>Moda 1: Raw Digital Counts + StationXML</b>: Dekonvolusi fungsi transfer instrumen (<i>transfer function</i> berbasis 
        <i>pole-zero</i> / PAZ dan <i>stage gain</i>) untuk merestorasi sinyal rekaman mentah digitizer menjadi percepatan fisik tanah riil dalam satuan m/s².<br>
        • <b>Moda 2: Physical Acceleration</b>: Digunakan apabila berkas rekaman masukan telah dikonversi sebelumnya ke percepatan 
        fisik (satuan m/s² atau Gal). Koreksi fungsi transfer dilewati sehingga proses komputasi berlangsung lebih cepat.
    </p>
    """
    builder.insert_html_safe(p4, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 60), p4_html_3)
    y += 65

    y = builder.draw_subsection_heading(p4, y, "2.4", "Kebutuhan Sistem & Batasan Masukan Data (System Requirements)")
    p4_html_req = """
    <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 4px; padding: 8px 12px; margin-top: 2px;">
        <p style="margin-bottom: 4px; font-size: 7.6pt; color: #002d62;"><b>SPESIFIKASI MINIMUM OPERASIONAL SOFTWARE:</b></p>
        <p style="margin: 0; font-size: 7.2pt; line-height: 1.35; color: #334155;">
            • <b>Akses Cloud Web</b>: Peramban web modern (<i>Google Chrome, Mozilla Firefox, Microsoft Edge, Safari</i>) tanpa instalasi dependensi lokal.<br>
            • <b><i>Workstation</i> Lokal</b>: Lingkungan Python 3.10 – 3.13, RAM minimum 4 GB (direkomendasikan 8 GB), CPU <i>multi-core</i>, ruang disk 500 MB.<br>
            • <b>Pustaka Dependensi</b>: Streamlit, ObsPy, NumPy, SciPy, Pandas, Plotly, Matplotlib, FPDF, PyMuPDF.<br>
            • <b>Batasan Berkas Masukan</b>: Format MiniSEED / SAC triaksial standar, laju sampel rekaman antara 50 Hz hingga 200 Hz, durasi rekaman memuat jendela pre-event noise sekurang-kurangnya 5 detik sebelum gelombang P tiba.
        </p>
    </div>
    """
    builder.insert_html_safe(p4, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 105), p4_html_req)

    # -------------------------------------------------------------
    # PAGE 5: BAB III - DIGITAL SIGNAL PROCESSING & FILTERING
    # -------------------------------------------------------------
    p5 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p5)
    builder.add_page_header_footer(p5, "Digital Signal Processing", 5)
    y = 50.0

    y = builder.draw_section_heading(p5, y, "III", "DIGITAL SIGNAL PROCESSING (DSP) & FILTERING")
    y = builder.draw_subsection_heading(p5, y, "3.1", "Klasifikasi & Karakteristik Tipe Filter (Filter Type)")

    filters_html = """
    <p style="margin-bottom: 8px;">
        Proses <i>digital filtering</i> pada sinyal seismik bertujuan mereduksi derau frekuensi tinggi (derau elektronik/telemetri) 
        dan derau frekuensi sangat rendah (mikroseismik laut, fluktuasi termal). BSMA menyediakan tiga konfigurasi filter:
    </p>
    <div style="border-left: 3px solid #0284c7; background: #f8fafc; padding: 6px 10px; margin-bottom: 6px; border-radius: 2px;">
        <b style="color: #0284c7; font-size: 7.8pt;">BANDPASS FILTER (DEFAULT REKOMENDASI BSMA)</b><br>
        <span style="font-size: 7.2pt; color: #334155;">Meneruskan komponen frekuensi di antara <i>f</i><sub>min</sub> dan <i>f</i><sub>max</sub> (setelan rekomendasi default: 0.075 – 25.0 Hz), efektif meredam derau periode panjang dan gangguan frekuensi tinggi pada analisis gempa tektonik lokal dan regional.</span>
    </div>
    <div style="border-left: 3px solid #0f4c81; background: #f8fafc; padding: 6px 10px; margin-bottom: 6px; border-radius: 2px;">
        <b style="color: #0f4c81; font-size: 7.8pt;">LOWPASS FILTER</b><br>
        <span style="font-size: 7.2pt; color: #334155;">Meneruskan seluruh spektrum di bawah <i>f</i><sub>max</sub> dan memangkas derau frekuensi tinggi. Bermanfaat untuk studi getaran periode panjang atau respons dinamis struktur bertingkat tinggi.</span>
    </div>
    <div style="border-left: 3px solid #d97706; background: #f8fafc; padding: 6px 10px; margin-bottom: 6px; border-radius: 2px;">
        <b style="color: #d97706; font-size: 7.8pt;">HIGHPASS FILTER</b><br>
        <span style="font-size: 7.2pt; color: #334155;">Meneruskan komponen di atas <i>f</i><sub>min</sub> guna memitigasi pergeseran baseline (<i>baseline drift</i>) frekuensi sangat rendah tanpa membatasi kandungan frekuensi tinggi sinyal.</span>
    </div>
    """
    builder.insert_html_safe(p5, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 140), filters_html)
    y += 145

    y = builder.draw_subsection_heading(p5, y, "3.2", "Karakteristik Filter Butterworth Orde-4 & Zero-Phase Filtering")
    p5_html_butter = """
    <p>
        BSMA mengimplementasikan filter <i>Butterworth</i> berbasis <i>Second-Order Sections</i> (SOS) dengan koefisien orde ke-4 
        per lintasan tunggal (<i>single-pass roll-off</i> 24 dB/oktaf). Butterworth dipilih karena memiliki respons magnitudo yang 
        paling datar (<i>maximally flat</i>) pada <i>passband</i>, meminimalkan alterasi amplitudo spektrum gelombang gempa.
    </p>
    <p>
        Guna meniadakan distorsi pergeseran waktu (<i>phase lag</i>), filtering diaplikasikan dua arah maju-mundur (<i>zero-phase 
        forward-backward filtering</i> via <code>scipy.signal.sosfiltfilt</code>). Pemrosesan dua arah ini mengalikan fungsi transfer kuadrat 
        |<i>H</i>(<i>f</i>)|<sup>2</sup>, menghasilkan filter efektif berorde-8 dengan <i>roll-off</i> asimtotik <b>48 dB/oktaf</b> 
        pada <i>stopband</i> tanpa distorsi fase neto sehingga waktu tiba fase gelombang P dan S tidak bergeser.
    </p>
    """
    builder.insert_html_safe(p5, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 78), p5_html_butter)
    y += 82

    y = builder.draw_subsection_heading(p5, y, "3.3", "Seleksi Frekuensi Cutoff & Margin Numerik Nyquist 80%")
    p5_html_nyq = """
    <p>
        Penentuan frekuensi cutoff harus mempertimbangkan laju sampel rekaman (<i>f<sub>s</sub></i>). BSMA menerapkan batas atas frekuensi 
        cutoff maksimal 80% dari frekuensi Nyquist sebagai <i>numerical guard band</i> guna mencegah distorsi tepi dan artefak numerik:
    </p>
    """
    builder.insert_html_safe(p5, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 30), p5_html_nyq)
    y += 32

    # Equation 3.1: Nyquist limit
    y = builder.draw_math_equation(
        p5, y,
        latex_str=r"$f_{\max} \leq 0.80 \times f_{\mathrm{Nyquist}} = 0.80 \times \left( \frac{f_s}{2} \right) = 0.40 \times f_s$",
        plain_repr="Persamaan 3.1: f_max <= 0.80 x f_Nyquist = 0.80 x (f_s / 2) = 0.40 x f_s [Hz]",
        height=28.0,
        unit="[Hz]",
        eq_num="(Pers. 3.1)"
    )
    y += 4

    y = builder.draw_subsection_heading(p5, y, "3.4", "Tapering Jendela Cosine Tukey 5% & Mitigasi Baseline Drift")
    p5_html_taper = """
    <p>
        Sebelum proses filtering frekuensi, kedua ujung rekaman dikenakan <i>Tukey cosine tapering</i> sebesar 5% guna menghaluskan diskontinuitas 
        amplitudo dan mereduksi kebocoran spektral (<i>spectral leakage</i>). Selain itu, <i>detrending</i> polinomial diaplikasikan untuk mereduksi <i>offset</i> instrumen.
    </p>
    <p>
        Penting dipahami bahwa integrasi numerik akselerogram tidak secara otomatis menghilangkan <i>baseline drift</i> seutuhnya. Filtering 
        dan <i>detrending</i> bertindak sebagai langkah mitigasi pergeseran garis dasar (<i>baseline drift mitigation</i>) guna meningkatkan 
        kestabilan riwayat kecepatan dan perpindahan.
    </p>
    """
    builder.insert_html_safe(p5, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 62), p5_html_taper)
    y += 66

    builder.draw_callout(
        p5,
        pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 66),
        "Mitigasi Baseline Drift & Peringatan Metodologi Deformasi Permanen (Fling-Step)",
        "Integrasi ganda akselerogram dengan highpass filtering secara inheren memotong frekuensi nol (DC), sehingga nilai PGD yang "
        "dihasilkan BSMA merepresentasikan <b>perpindahan puncak dinamik transient</b> (<i>dynamic peak displacement</i>), bukan deformasi "
        "tektonik permanen medan dekat (<i>true static offset / fling-step</i>). Apabila kurva riwayat perpindahan melengkung tidak wajar, "
        "operator disarankan menaikkan frekuensi cutoff bawah <i>f</i><sub>min</sub> (misal dari 0.05 Hz ke 0.10 Hz) guna menekan drift sisa.",
        callout_type="warning"
    )

    # -------------------------------------------------------------
    # PAGE 6: BAB IV - QUALITY CONTROL DIAGNOSTICS & DATA VALIDITY
    # -------------------------------------------------------------
    p6 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p6)
    builder.add_page_header_footer(p6, "Quality Control Diagnostics", 6)
    y = 50.0

    y = builder.draw_section_heading(p6, y, "IV", "QUALITY CONTROL (QC) DIAGNOSTICS & PANDUAN VALIDITAS DATA")
    y = builder.draw_subsection_heading(p6, y, "4.1", "Metodologi Skoring Kualitas Rekaman Seismik (0 - 100)")

    p6_html_1 = """
    <p>
        BSMA v2.0.0 dilengkapi modul Quality Control (QC) otomatis yang mengevaluasi integritas fisik rekaman sebelum analisis lanjutan. 
        Sistem menghitung skor kualitas numerik (0 – 100 poin) per kanal berdasarkan deduksi penalti terhadap anomali sinyal: 
        estimasi rasio sinyal terhadap derau (SNR), saturasi digitizer / sensor <i>clipping</i>, lonjakan transien (<i>spikes</i>), 
        sinyal macet (<i>flatline</i>), dan pergeseran garis dasar (<i>baseline drift/offset</i>).
    </p>
    """
    builder.insert_html_safe(p6, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 42), p6_html_1)
    y += 46

    y = builder.draw_subsection_heading(p6, y, "4.2", "Sistem Evaluasi Kualitas: Skor, Diagnostic Flags & Status")
    p6_html_flags = """
    <p>
        Untuk menjaga ketepatan metodologi ilmiah, BSMA memisahkan evaluasi menjadi tiga tingkatan: (1) <b>Skor Kualitas (0–100)</b>, 
        (2) <b>Diagnostic Flags</b> independen (<i>CLIPPING, SPIKES, FLATLINE, LOW SNR, BASELINE OFFSET, METADATA ERROR</i>), dan 
        (3) <b>Status Kelayakan Operasional</b> (<i>QC PASS, QC WARNING, QC SUSPECT/FAIL</i>).
    </p>
    """
    builder.insert_html_safe(p6, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 36), p6_html_flags)
    y += 40

    y = builder.draw_subsection_heading(p6, y, "4.3", "Tabel Klasifikasi 6 Kategori Diagnostik Sinyal Aktual")
    qc_table_html = """
    <table>
        <thead>
            <tr>
                <th style="width: 14%;">Kategori</th>
                <th style="width: 26%;">Status Diagnostik Aktual</th>
                <th style="width: 32%;">Kondisi Fisik Rekaman</th>
                <th style="width: 28%;">Rekomendasi Operasional</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><b>Class 1</b></td>
                <td>NOMINAL DATA QUALITY</td>
                <td>Skor rata-rata >= 80, tanpa isu anomali, latar derau rendah.</td>
                <td>Sangat layak untuk analisis rekayasa dan spektrum desain.</td>
            </tr>
            <tr style="background-color: #f8fafc;">
                <td><b>Class 2</b></td>
                <td>ACCEPTABLE QUALITY</td>
                <td>Skor >= 70, degradasi SNR minor atau <i>offset</i> kecil tanpa <i>clipping</i>.</td>
                <td>Layak digunakan langsung dengan filter standar rekomendasi.</td>
            </tr>
            <tr>
                <td><b>Class 3</b></td>
                <td>SENSOR / DIGITIZER ANOMALY</td>
                <td>Terdeteksi <i>clipping</i> atau saturasi ADC pada puncak gelombang.</td>
                <td>PGA terpotong; tidak valid untuk analisis kuantitatif puncak karena estimasi Skala MMI berpotensi <i>underestimate</i>.</td>
            </tr>
            <tr style="background-color: #f8fafc;">
                <td><b>Class 4</b></td>
                <td>DATA / METADATA ERROR</td>
                <td>Dataset tidak lengkap, kanal hilang, atau StationXML <i>unreadable</i>.</td>
                <td>Periksa integritas file gelombang dan berkas StationXML.</td>
            </tr>
            <tr>
                <td><b>Class 5</b></td>
                <td>DEGRADED SIGNAL (HIGH NOISE)</td>
                <td>Derau latar dominan, estimasi SNR sangat rendah (&lt; 3 dB).</td>
                <td>Tidak direkomendasikan untuk analisis kuantitatif.</td>
            </tr>
            <tr style="background-color: #f8fafc;">
                <td><b>Class 6</b></td>
                <td>AVAILABILITY / TELEMETRY GAP</td>
                <td>Sinyal <i>flatline</i> (sensor macet/mati) atau terdapat <i>data gap</i>.</td>
                <td>Data ditolak secara otomatis; laporkan inspeksi sensor stasiun.</td>
            </tr>
        </tbody>
    </table>
    """
    builder.insert_html_safe(p6, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 175), qc_table_html)
    y += 180

    y = builder.draw_subsection_heading(p6, y, "4.4", "Parameter Diagnostik & Formulasi SNR (Signal-to-Noise Ratio)")
    p6_html_snr = """
    <p>
        Rasio Sinyal terhadap Derau (<i>Signal-to-Noise Ratio</i> / SNR) diestimasi melalui rasio RMS amplitudo pada jendela sinyal gempa 
        terhadap jendela derau sebelum gempa (<i>pre-event noise</i>). Formulasi baku dihitung sebagai:
    </p>
    """
    builder.insert_html_safe(p6, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 26), p6_html_snr)
    y += 28

    # Equation 4.1: SNR formula
    y = builder.draw_math_equation(
        p6, y,
        latex_str=r"$\mathbf{SNR} = 20 \, \log_{10} \left( \frac{\mathrm{RMS}_{\mathrm{signal}}}{\mathrm{RMS}_{\mathrm{noise}}} \right)$",
        plain_repr="Persamaan 4.1: SNR = 20 log10(RMS_signal / RMS_noise) [dB]",
        height=32.0,
        unit="[dB]",
        eq_num="(Pers. 4.1)"
    )
    y += 4

    p6_validity_html = """
    <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 4px; padding: 7px 10px;">
        <b style="font-size: 7.6pt; color: #002d62;">PANDUAN VALIDITAS & PENGGUNAAN DATA (DATA VALIDITY GUIDE):</b>
        <p style="font-size: 7.2pt; line-height: 1.30; margin-top: 3px; color: #334155;">
            • <b>QC PASS (Skor ≥ 70)</b>: Sinyal berintegritas nominal; seluruh parameter kinematika (PGA, PGV, Arias, Spektrum <i>S<sub>a</sub></i>) andal untuk analisis rekayasa seismik.<br>
            • <b>QC WARNING (Skor 50 – 69)</b>: Anomali non-fatal (lonjakan terisolasi, SNR marginal, atau drift minor); periksa cutoff filter dan inspeksi kurva riwayat waktu.<br>
            • <b>QC FAIL (Skor &lt; 50 atau Anomali Fatal)</b>: Terdeteksi <i>clipping</i>, saturasi ADC, <i>flatline</i>, atau skor &lt; 50; data ditolak/tidak valid untuk parameter puncak karena PGA terdistorsi dan estimasi MMI <i>underestimate</i>.
        </p>
    </div>
    """
    builder.insert_html_safe(p6, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 90), p6_validity_html)

    # -------------------------------------------------------------
    # PAGE 7: BAB V - PANDUAN OPERASIONAL LANGKAH DEMI LANGKAH
    # -------------------------------------------------------------
    p7 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p7)
    builder.add_page_header_footer(p7, "Panduan Operasional", 7)
    y = 50.0

    y = builder.draw_section_heading(p7, y, "V", "PANDUAN OPERASIONAL LANGKAH DEMI LANGKAH")
    y = builder.draw_subsection_heading(p7, y, "5.1", "Metode Akses Cloud Web Application (Akses Utama Tanpa Instalasi)")

    p7_html_cloud = """
    <div style="background-color: #f0fdf4; border: 1px solid #86efac; border-radius: 4px; padding: 8px 12px; margin-bottom: 8px;">
        <b style="color: #166534; font-size: 8.5pt;">AKSES LANGSUNG DARING: <a href="https://strong-motion.streamlit.app/" style="color: #166534; text-decoration: underline;">https://strong-motion.streamlit.app/</a></b><br>
        <p style="font-size: 7.4pt; margin: 4px 0 0 0; color: #1e293b; line-height: 1.35;">
            Untuk mempermudah penggunaan tanpa kendala konfigurasi lingkungan teknis, BSMA v2.0.0 dapat diakses secara langsung 
            melalui peramban web pada perangkat yang kompatibel. Operator tidak perlu menginstalasi dependensi Python secara manual; 
            seluruh alur komputasi diproses secara interaktif (<i>on-demand</i>) di server awan.
        </p>
    </div>
    """
    builder.insert_html_safe(p7, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 55), p7_html_cloud)
    y += 60

    y = builder.draw_subsection_heading(p7, y, "5.2", "Metode Eksekusi Workstation Lokal (Git Clone & Lingkungan Python)")
    p7_html_local = """
    <p>
        Bagi kebutuhan analisis intensif luring (<i>offline processing</i>) pada stasiun kerja pengamatan, aplikasi dapat dijalankan 
        melalui terminal dengan tahapan:
    </p>
    <div style="background-color: #0f172a; border-radius: 3px; padding: 6px 10px; font-family: monospace; font-size: 7.0pt; color: #38bdf8; margin: 4px 0;">
        git clone <a href="https://github.com/ahmaddidan/BSMA-v.2" style="color: #38bdf8; text-decoration: underline;">https://github.com/ahmaddidan/BSMA-v.2.git</a><br>
        cd BSMA-v.2<br>
        pip install -r requirements.txt<br>
        streamlit run app.py
    </div>
    """
    builder.insert_html_safe(p7, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 70), p7_html_local)
    y += 75

    y = builder.draw_subsection_heading(p7, y, "5.3", "Alur Kerja 4 Langkah Pemrosesan Data Seismogram")

    steps_html = """
    <div style="border-left: 3px solid #0284c7; background: #f8fafc; padding: 6px 10px; margin-bottom: 6px;">
        <b style="color: #0284c7; font-size: 7.8pt;">LANGKAH 1: UNGGAH GELOMBANG & BERKAS METADATA STATIONXML</b><br>
        <span style="font-size: 7.2pt; color: #334155;">Pada panel bilah sisi (<i>sidebar</i>), klik menu <b>Upload Waveform</b> untuk mengunggah berkas MiniSEED/SAC triaksial. Jika data berstatus rekaman digital mentah (<i>digital counts</i>), unggah berkas StationXML (.xml) untuk koreksi respons instrumen. Apabila data telah terkalibrasi percepatan fisik (m/s² atau Gal), sistem otomatis mengaktifkan mode 'Physical Acceleration' (bypass fungsi transfer).</span>
    </div>
    <div style="border-left: 3px solid #0ea5e9; background: #f8fafc; padding: 6px 10px; margin-bottom: 6px;">
        <b style="color: #0ea5e9; font-size: 7.8pt;">LANGKAH 2: KONFIGURASI PARAMETER FILTERING & SOLVER SDOF</b><br>
        <span style="font-size: 7.2pt; color: #334155;">Pilih tipe filter frekuensi (Bandpass default rekomendasi, Lowpass, atau Highpass). Tentukan frekuensi cutoff <i>f</i><sub>min</sub> dan <i>f</i><sub>max</sub> dengan mematuhi batas 80% Nyquist. Pilih algoritma SDOF (Nigam dan Jennings atau Newmark) dan nilai redaman kritis (default 5%).</span>
    </div>
    <div style="border-left: 3px solid #10b981; background: #f8fafc; padding: 6px 10px; margin-bottom: 6px;">
        <b style="color: #10b981; font-size: 7.8pt;">LANGKAH 3: EKSEKUSI PEMROSESAN ANALISIS TERPADU</b><br>
        <span style="font-size: 7.2pt; color: #334155;">Klik tombol <b>Run Analysis</b>. Pipeline BSMA secara berurutan mengeksekusi validasi format, <i>detrending</i> polinomial, <i>tapering</i> Tukey 5%, zero-phase filtering, integrasi numerik, ekstraksi kinematika, komputasi MMI, dan spektrum respons.</span>
    </div>
    <div style="border-left: 3px solid #6366f1; background: #f8fafc; padding: 6px 10px; margin-bottom: 6px;">
        <b style="color: #6366f1; font-size: 7.8pt;">LANGKAH 4: INTERPRETASI 7 TAB HASIL & EKSPOR LAPORAN TEKNIS</b><br>
        <span style="font-size: 7.2pt; color: #334155;">Telaah hasil pada 7 tab interaktif Plotly. Ekspor laporan teknis komprehensif dalam format PDF siap cetak, tabel CSV metrik kinematik, tabel matriks spektrum respons, atau paket arsip ZIP terkompresi.</span>
    </div>
    """
    builder.insert_html_safe(p7, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 215), steps_html)
    y += 220

    builder.draw_callout(
        p7,
        pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 55),
        "Pemeriksaan Provenance Logging & Audit Trail",
        "Setiap tahapan komputasi dicatat secara otomatis dalam audit log integritas data (<i>provenance logging</i>). "
        "Parameter filtering, waktu eksekusi, versi pustaka numerik, dan status validasi StationXML disertakan secara "
        "transparan pada laporan PDF akhir guna menjamin reproduksibilitas ilmiah (<i>scientific reproducibility</i>).",
        callout_type="info"
    )

    # -------------------------------------------------------------
    # PAGE 8: BAB VI - EKSPLORASI 7 TAB FITUR ANALISIS & INTERPRETASI
    # -------------------------------------------------------------
    p8 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p8)
    builder.add_page_header_footer(p8, "Eksplorasi Fitur Analisis", 8)
    y = 50.0

    y = builder.draw_section_heading(p8, y, "VI", "EKSPLORASI 7 TAB FITUR ANALISIS & INTERPRETASI HASIL")
    
    tabs_desc = [
        ("TAB 1: SUMMARY (RINGKASAN EKSEKUTIF)",
         "Menyajikan kartu metadata stasiun (lintang, bujur, elevasi), pratinjau seismogram 3-kanal (40–50% tinggi layar), "
         "metrik utama kinematika (PGA, PGV, PGD, <i>I<sub>a</sub></i>, <i>D</i><sub>5-95</sub>, MMI) dari komponen terbesar, serta panel <i>audit trail</i>."),
        ("TAB 2: WAVEFORMS (RIWAYAT GELOMBANG KINEMATIK)",
         "Menampilkan kurva kinematis lengkap percepatan (m/s²), kecepatan (m/s), dan perpindahan (cm) untuk ketiga kanal "
         "ortogonal. Dilengkapi penanda waktu tiba otomatis dan indikator nilai puncak absolut."),
        ("TAB 3: QUALITY CONTROL (DIAGNOSTIK KUALITAS SINYAL)",
         "Menyajikan skor kualitas numerik (0–100), visualisasi radar metrik derau latar, Power Spectral Density (PSD), "
         "estimasi SNR (dB), dan laporan bendera diagnostik anomali (<i>clipping</i>, <i>spikes</i>, <i>flatline</i>)."),
        ("TAB 4: STRONG MOTION (PARAMETER KINEMATIKA & ENERGI)",
         "Menampilkan analisis kinematika mendalam: kurva akumulasi Intensitas Arias (<i>I<sub>a</sub></i>) terhadap waktu, interval Durasi "
         "Signifikan (<i>D</i><sub>5-95</sub>), rasio kecepatan terhadap percepatan puncak (<i>V</i><sub>max</sub>/<i>A</i><sub>max</sub>), dan akselerasi efektif."),
        ("TAB 5: INTENSITY (ESTIMASI SKALA MMI INSTRUMENTAL)",
         "Menampilkan klasifikasi tingkat guncangan instrumental Skala MMI (Worden et al., 2012) berbasis nilai PGA dan PGV. "
         "Menyajikan kartu dampak fisik guncangan (<i>Perceived Shaking</i>) dan potensi kerusakan struktural (<i>Potential Damage</i>)."),
        ("TAB 6: SPECTRUM (SPEKTRUM RESPONS PERCEPATAN ELASTIS)",
         "Menyajikan kurva Spektrum Respons Pseudo-Acceleration (<i>S<sub>a</sub></i>) elastis redaman 5% untuk ketiga kanal pada rentang periode "
         "<i>T</i> = 0.01 s hingga 10.0 s. Mendukung komparasi terhadap kurva spektrum desain SNI 1726:2019 (BSN, 2019)."),
        ("TAB 7: REPORT (GENERATOR LAPORAN TEKNIS PDF & CSV)",
         "Memfasilitasi pratinjau dan pengunduhan dokumen laporan teknis komprehensif berformat PDF siap cetak dengan "
         "tata letak rapi, tabel CSV ringkasan parameter kinematika, dan lembar CSV matriks spektrum respons diskret.")
    ]

    for t_title, t_body in tabs_desc:
        tab_box_html = f"""
        <div style="border-left: 3px solid #002d62; background: #f8fafc; padding: 4px 8px; margin-bottom: 5px;">
            <b style="color: #002d62; font-size: 7.6pt;">{t_title}</b><br>
            <p style="font-size: 7.1pt; line-height: 1.25; margin: 2px 0 0 0; color: #334155; text-align: justify; text-justify: inter-word;">
                {t_body}
            </p>
        </div>
        """
        builder.insert_html_safe(p8, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 42), tab_box_html)
        y += 45

    y += 2
    builder.draw_callout(
        p8,
        pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 55),
        "Fitur Interaktivitas Grafik Ilmiah Plotly",
        "Seluruh visualisasi kurva gelombang dan spektrum respons disajikan menggunakan mesin grafik Plotly interaktif. "
        "Pengguna dapat melakukan pembesaran area (<i>box zoom</i>), pergeseran sumbu (<i>pan</i>), inspeksi koordinat titik waktu "
        "secara presisi (<i>hover tooltip</i>), serta menyimpan grafik sebagai citra vektor/raster beresolusi tinggi.",
        callout_type="info"
    )

    # -------------------------------------------------------------
    # PAGE 9: BAB VII - FITUR LANJUTAN, UTILITAS & WORKED EXAMPLE
    # -------------------------------------------------------------
    p9 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p9)
    builder.add_page_header_footer(p9, "Fitur Lanjutan & Utilitas", 9)
    y = 50.0

    y = builder.draw_section_heading(p9, y, "VII", "FITUR LANJUTAN, UTILITAS & CONTOH HASIL ANALISIS")
    y = builder.draw_subsection_heading(p9, y, "7.1", "Analisis Multi-Stasiun Serentak (Batch Processing Mode)")

    p9_html_batch = """
    <p>
        Ketika kejadian gempa bumi signifikan terekam oleh banyak stasiun akselerograf sekaligus di suatu wilayah jaringan, 
        pengguna dapat memanfaatkan mode <i>Batch Processing</i> untuk memproses seluruh stasiun secara terpadu:
    </p>
    <p style="font-size: 7.3pt; line-height: 1.25; color: #334155; margin-left: 8px;">
        1. Unggah seluruh berkas MiniSEED/SAC dari stasiun-stasiun yang merekam ke dalam bilah sisi.<br>
        2. Tentukan setelan filter seragam (cutoff <i>f</i><sub>min</sub> dan <i>f</i><sub>max</sub>) atau gunakan rekomendasi default.<br>
        3. Klik tombol <b>Process All Stations</b>. Pipeline mengeksekusi analisis secara berurutan lintas stasiun.<br>
        4. Sistem menyajikan tabel komparatif regional terpadu yang dapat diurutkan berdasarkan nilai PGA tertinggi atau Skala MMI.
    </p>
    """
    builder.insert_html_safe(p9, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 65), p9_html_batch)
    y += 70

    y = builder.draw_subsection_heading(p9, y, "7.2", "Ekspor Terpadu Berkas Arsip ZIP (Bulk Export Package)")
    p9_html_zip = """
    <p>
        Pada mode ekspor hasil, BSMA menyediakan utilitas kompresi arsip ZIP (.zip) yang memuat luaran terpadu seluruh stasiun:<br>
        • Dokumen Laporan Teknis PDF individual untuk setiap stasiun yang berhasil dianalisis.<br>
        • Berkas CSV Ringkasan Metrik Kinematik (<i>Summary Metrics Table</i>) memuat PGA, PGV, PGD, <i>I<sub>a</sub></i>, <i>D</i><sub>5-95</sub>, MMI, dan skor QC.<br>
        • Berkas CSV Matriks Spektrum Respons Diskret memuat koordinat Periode <i>T</i> vs <i>S<sub>a</sub></i> untuk setiap stasiun.
    </p>
    """
    builder.insert_html_safe(p9, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 55), p9_html_zip)
    y += 60

    y = builder.draw_subsection_heading(p9, y, "7.3", "Komparasi Solver SDOF: Formulasi Nigam–Jennings vs Newmark–Beta")
    p9_html_sdof = """
    <p>
        BSMA menyediakan dua pilihan algoritma penyelesai respons sistem berderajat kebebasan tunggal (SDOF):<br>
        • <b>Nigam dan Jennings (1969)</b>: Formulasi rekursif analitik berbasis representasi input <i>piecewise-linear</i> untuk 
        menghitung respons SDOF secara efisien tanpa integrasi numerik langkah demi langkah (<i>step-by-step</i>).<br>
        • <b>Newmark (1959)</b>: Solver integrasi numerik implisit langkah demi langkah (<i>step-by-step</i>) dengan 
        parameter konstan rata-rata percepatan (&gamma; = 1/2, &beta; = 1/4).<br>
        • <b>Numerical Cross-Validation Benchmark</b>: Aplikasi menyematkan fitur verifikasi silang otomatis yang menghitung metrik 
        deviasi relatif maksimum, rata-rata, dan RMS antara kedua solver guna menjamin akurasi solusi numerik.
    </p>
    """
    builder.insert_html_safe(p9, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 68), p9_html_sdof)
    y += 73

    y = builder.draw_subsection_heading(p9, y, "7.4", "Antarmuka Dual-Theme Switcher (Light Mode & Dark Mode)")
    p9_html_theme = """
    <p>
        Tersedia tombol saklar tema visual responsif di pojok kanan atas bilah aplikasi:<br>
        • <b><i>Light Mode</i> (Mode Terang)</b>: Didesain untuk presentasi visual di ruangan dengan pencahayaan terang atau kebutuhan dokumentasi.<br>
        • <b><i>Dark Mode</i> (Mode Gelap)</b>: Dirancang untuk kondisi pencahayaan redup dengan mempertahankan kontras tajam pada kurva seismogram Plotly, memberikan kenyamanan visual bagi operator saat pengamatan berlangsung lama.
    </p>
    """
    builder.insert_html_safe(p9, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 48), p9_html_theme)
    y += 52

    y = builder.draw_subsection_heading(p9, y, "7.5", "Contoh Kasus Pengolahan Data Riil (Worked Example)")
    worked_ex_html = """
    <p style="margin-bottom: 4px;">
        Berikut adalah contoh hasil komputasi riil BSMA v2.0.0 pada data akselerogram triaksial stasiun BMKG (laju sampel 100 Hz, filter Bandpass 0.075 – 25.0 Hz, redaman 5%):
    </p>
    <table>
        <thead>
            <tr>
                <th style="width: 14%;">Kanal</th>
                <th style="width: 17%;">PGA (Gal)</th>
                <th style="width: 17%;">PGV (cm/s)</th>
                <th style="width: 17%;">PGD (cm)</th>
                <th style="width: 17%;"><i>I<sub>a</sub></i> (m/s)</th>
                <th style="width: 18%;"><i>D</i><sub>5-95</sub> (s)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><b>HNE (E–W)</b></td>
                <td>48.215</td>
                <td>3.412</td>
                <td>0.4821</td>
                <td>0.03812</td>
                <td>14.82</td>
            </tr>
            <tr style="background-color: #f8fafc;">
                <td><b>HNN (N–S)</b></td>
                <td><b>54.890</b></td>
                <td><b>4.105</b></td>
                <td><b>0.6120</b></td>
                <td><b>0.05120</b></td>
                <td><b>15.10</b></td>
            </tr>
            <tr>
                <td><b>HNZ (U–D)</b></td>
                <td>26.140</td>
                <td>1.890</td>
                <td>0.2105</td>
                <td>0.01450</td>
                <td>12.45</td>
            </tr>
        </tbody>
    </table>
    <p style="font-size: 7.0pt; color: #475569; margin-top: 3px;">
        Komponen terkuat: <b>HNN (54.89 Gal)</b> | Estimasi Instrumental: <b>MMI V (Moderate / Very Light Damage)</b> | Status QC: <b>QC PASS (Skor 92/100)</b>
    </p>
    """
    builder.insert_html_safe(p9, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 105), worked_ex_html)

    # -------------------------------------------------------------
    # PAGE 10: BAB VIII - FORMULASI MATEMATIS KINEMATIKA & SKALA MMI
    # -------------------------------------------------------------
    p10 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p10)
    builder.add_page_header_footer(p10, "Formulasi Matematis & MMI", 10)
    y = 50.0

    y = builder.draw_section_heading(p10, y, "VIII", "FORMULASI MATEMATIS KINEMATIKA SEISMIK & SKALA MMI")
    y = builder.draw_subsection_heading(p10, y, "8.1", "Kinematika Gerakan Tanah: PGA, PGV, dan PGD")

    p10_html_kin = """
    <p>
        Parameter kinematika puncak merepresentasikan nilai absolut maksimum pada jendela riwayat rekaman yang dianalisis. 
        Kecepatan dan perpindahan diperoleh melalui integrasi numerik bertahap terhadap waktu:
    </p>
    """
    builder.insert_html_safe(p10, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 24), p10_html_kin)
    y += 26

    # Equation 8.1: PGA
    y = builder.draw_math_equation(
        p10, y,
        latex_str=r"$\mathbf{PGA} = \max \, |a(t)|$",
        plain_repr="Persamaan 8.1: PGA = max |a(t)| [Gal atau cm/s²]",
        height=24.0,
        unit="[Gal atau cm/s²]",
        eq_num="(Pers. 8.1)"
    )

    # Equation 8.2: PGV
    y = builder.draw_math_equation(
        p10, y,
        latex_str=r"$\mathbf{PGV} = \max \, |v(t)| = \max \, \left| \int_0^t a(\tau) \, d\tau \right|$",
        plain_repr=r"Persamaan 8.2: PGV = max |v(t)| = max | \int_0^t a(\tau) d\tau | [cm/s]",
        height=26.0,
        unit="[cm/s]",
        eq_num="(Pers. 8.2)"
    )

    # Equation 8.3: PGD
    y = builder.draw_math_equation(
        p10, y,
        latex_str=r"$\mathbf{PGD} = \max \, |d(t)| = \max \, \left| \int_0^t v(\tau) \, d\tau \right|$",
        plain_repr=r"Persamaan 8.3: PGD = max |d(t)| = max | \int_0^t v(\tau) d\tau | [cm]",
        height=26.0,
        unit="[cm]",
        eq_num="(Pers. 8.3)"
    )

    y = builder.draw_subsection_heading(p10, y, "8.2", "Intensitas Arias Kumulatif (Ia) & Durasi Signifikan D5-95")
    p10_html_arias = """
    <p>
        Intensitas Arias (<i>I<sub>a</sub></i>), sebagaimana diformulasikan oleh Arias (1970), merepresentasikan intensitas kumulatif 
        energi gerakan tanah berdasarkan integral kuadrat percepatan dengan dimensi kecepatan (m/s). Durasi Signifikan (<i>D</i><sub>5-95</sub>) 
        menurut perumusan Trifunac dan Brady (1975) adalah interval waktu ketika intensitas Arias kumulatif bertumbuh dari 5% hingga 95% dari total intensitas Arias:
    </p>
    """
    builder.insert_html_safe(p10, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 34), p10_html_arias)
    y += 36

    # Equation 8.4: Arias Intensity with true Pi and Integral bounds
    y = builder.draw_math_equation(
        p10, y,
        latex_str=r"$I_a = \frac{\pi}{2g} \int_0^{t_{\max}} [a(t)]^2 \, dt$",
        plain_repr=r"Persamaan 8.4: Ia = (pi / 2g) \int_0^{t_max} [a(t)]^2 dt [m/s]",
        height=34.0,
        unit="[m/s]",
        eq_num="(Pers. 8.4)"
    )

    # Equation 8.5: D5-95
    y = builder.draw_math_equation(
        p10, y,
        latex_str=r"$D_{5-95} = t_{95\%} - t_{5\%}$",
        plain_repr=r"Persamaan 8.5: D5-95 = t_95% - t_5% [detik]",
        height=24.0,
        unit="[detik]",
        eq_num="(Pers. 8.5)"
    )

    y = builder.draw_subsection_heading(p10, y, "8.3", "Pseudo-Spectral Acceleration (PSA) SDOF Redaman 5%")
    p10_html_psa = """
    <p>
        Pseudo-Spectral Acceleration (PSA) didefinisikan secara formal melalui perpindahan spektral maksimum 
        (<i>spectral displacement</i> / <i>S<sub>d</sub></i>) osilator elastik SDOF berfrekuensi sudut &omega; = 2&pi;/<i>T</i>:
    </p>
    """
    builder.insert_html_safe(p10, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 22), p10_html_psa)
    y += 24

    # Equation 8.6: PSA
    y = builder.draw_math_equation(
        p10, y,
        latex_str=r"$\mathbf{PSA}(T, \xi) = \omega^2 S_d(T, \xi) = \omega^2 \max \, |u(t)|$",
        plain_repr=r"Persamaan 8.6: PSA(T, \xi) = \omega^2 Sd(T, \xi) = \omega^2 max |u(t)| [g atau m/s²]",
        height=26.0,
        unit="[g atau m/s²]",
        eq_num="(Pers. 8.6)"
    )
    y += 30

    y = builder.draw_subsection_heading(p10, y, "8.4", "Estimasi Instrumental Skala Intensitas MMI (Worden et al., 2012)")
    mmi_img = create_mmi_chart()
    p10.insert_image(pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 100), stream=mmi_img)
    y += 104

    builder.draw_callout(
        p10,
        pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 74),
        "Formulasi Empiris GMICE (Worden et al., 2012) & Konvensi Komponen Horizontal",
        "BSMA mengimplementasikan GMICE Worden et al. (2012) berbasis komponen horizontal terbesar (Max-H PGA & PGV): "
        "untuk log<sub>10</sub>(PGA) &le; 1.57 berlaku MMI = 1.78 + 1.55 log<sub>10</sub>(PGA); untuk log<sub>10</sub>(PGA) &gt; 1.57 "
        "berlaku MMI = -1.60 + 3.70 log<sub>10</sub>(PGA). Pada intensitas kuat (MMI &ge; 5.0), evaluasi dipadukan dengan regresi PGV "
        "(log<sub>10</sub>PGV &le; 0.53: MMI = 3.78 + 2.99 log<sub>10</sub>PGV; log<sub>10</sub>PGV &gt; 0.53: MMI = 2.40 + 4.96 log<sub>10</sub>PGV). "
        "Nilai ini adalah estimasi instrumental matematis, bukan pengganti survei makroseismik lapangan.",
        callout_type="info"
    )

    # -------------------------------------------------------------
    # PAGE 11: BAB IX - PEMECAHAN MASALAH, KETERBATASAN & REFERENSI
    # -------------------------------------------------------------
    p11 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p11)
    builder.add_page_header_footer(p11, "Troubleshooting & Profil", 11)
    y = 48.0

    y = builder.draw_section_heading(p11, y, "IX", "PEMECAHAN MASALAH, KETERBATASAN & PROFIL PENGEMBANG")
    y = builder.draw_subsection_heading(p11, y, "9.1", "Pemecahan Masalah Umum (Troubleshooting FAQ)")

    p11_faq_html = """
    <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; padding: 4px 8px; margin-bottom: 3px;">
        <b style="font-size: 7.2pt; color: #002d62;">Q: Mengapa kurva perpindahan (Displacement) melengkung drastis ke atas atau ke bawah?</b><br>
        <span style="font-size: 6.8pt; color: #334155; line-height: 1.25;">A: Terjadi akibat residual <i>baseline drift</i> pada integrasi ganda numerik oleh derau frekuensi rendah. Solusi: Naikkan frekuensi cutoff bawah (<i>f</i><sub>min</sub>) di bilah sisi dari 0.05 Hz menjadi 0.10 Hz atau 0.15 Hz.</span>
    </div>
    <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; padding: 4px 8px; margin-bottom: 3px;">
        <b style="font-size: 7.2pt; color: #002d62;">Q: Muncul peringatan 'StationXML Response Correction Bypassed'. Apa artinya?</b><br>
        <span style="font-size: 6.8pt; color: #334155; line-height: 1.25;">A: Berkas StationXML tidak cocok dengan metadata stasiun/kanal rekaman. Sistem otomatis beralih ke mode 'Physical Acceleration'. Solusi: Pastikan kode stasiun dan network pada berkas .xml dan .mseed identik.</span>
    </div>
    <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; padding: 4px 8px; margin-bottom: 3px;">
        <b style="font-size: 7.2pt; color: #002d62;">Q: Mengapa nilai PGA akselerograf berbeda dengan sensor broadband di stasiun yang sama?</b><br>
        <span style="font-size: 6.8pt; color: #334155; line-height: 1.25;">A: Akselerograf dioptimalkan merekam percepatan gempa kuat tanpa <i>clipping</i>, sedangkan sensor <i>broadband</i> dioptimalkan untuk kecepatan getaran lemah periode panjang. Untuk rekayasa struktur, akselerograf merupakan acuan utama.</span>
    </div>
    <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; padding: 4px 8px; margin-bottom: 3px;">
        <b style="font-size: 7.2pt; color: #002d62;">Q: Mengapa status Quality Control (QC) menunjukkan WARNING atau FAIL meski sinyal tampak jelas?</b><br>
        <span style="font-size: 6.8pt; color: #334155; line-height: 1.25;">A: Sistem QC mengevaluasi 6 kriteria diagnostik komprehensif (SNR, sensor clipping, spikes, flatline, pre-event noise, dan DC offset). Status WARNING umumnya dipicu jendela pre-event noise < 5 detik atau SNR < 10 dB. Periksa Tab 3 (Quality Control) untuk rincian penalti metrik.</span>
    </div>
    <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; padding: 4px 8px; margin-bottom: 3px;">
        <b style="font-size: 7.2pt; color: #002d62;">Q: Kapan sebaiknya memilih solver Nigam-Jennings dibandingkan Newmark-Beta?</b><br>
        <span style="font-size: 6.8pt; color: #334155; line-height: 1.25;">A: Formulasi analitik rekursif Nigam-Jennings (1969) sangat direkomendasikan untuk spektrum periode pendek (<i>T</i> < 0.1 s) karena mengasumsikan eksitasi percepatan linear bagian-per-bagian. Newmark-Beta (1959) menggunakan integrasi implisit percepatan rata-rata. Keduanya memiliki deviasi rata-rata < 5% yang dapat diverifikasi pada Tab 6.</span>
    </div>
    """
    builder.insert_html_safe(p11, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 160), p11_faq_html)
    y += 164

    y = builder.draw_subsection_heading(p11, y, "9.2", "Batasan Metodologis Perangkat Lunak (Known Limitations)")
    p11_limit_html = """
    <div style="background-color: #fffbeb; border: 1px solid #fde68a; border-radius: 3px; padding: 6px 10px;">
        <b style="font-size: 7.5pt; color: #92400e;">BATASAN PENGGUNAAN METODOLOGI BSMA v2.0.0:</b>
        <p style="font-size: 7.1pt; line-height: 1.32; margin: 3px 0 0 0; color: #451a03; text-align: justify;">
            • <b>Sensitivitas PGD</b>: Integrasi ganda sangat rentan terhadap drift derau frekuensi rendah sisa.<br>
            • <b>Dependensi StationXML</b>: Akurasi dekonvolusi menuntut keabsahan <i>transfer function</i> PAZ &amp; <i>stage gain</i> sensor.<br>
            • <b>Rekaman Derau Tinggi</b>: Sinyal dengan estimasi SNR &lt; 3 dB tidak direkomendasikan untuk analisis kuantitatif.<br>
            • <b>Estimasi MMI Instrumental</b>: Merupakan representasi fisik empiris, bukan pengganti survei makroseismik observasi lapangan.
        </p>
    </div>
    """
    builder.insert_html_safe(p11, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 62), p11_limit_html)
    y += 66

    y = builder.draw_subsection_heading(p11, y, "9.3", "Luaran Akses Software: Cloud Web App & Repositori GitHub")
    p11_access_html = """
    <table>
        <tr>
            <td style="width: 50%; background-color: #f8fafc; border: 1px solid #0284c7; border-radius: 3px; padding: 6px 9px; vertical-align: top;">
                <b style="color: #0284c7; font-size: 7.7pt;">1. CLOUD WEB APPLICATION (ONLINE)</b><br>
                <a href="https://strong-motion.streamlit.app/" style="font-size: 7.2pt; color: #002d62; font-weight: bold; text-decoration: underline;">https://strong-motion.streamlit.app/</a><br>
                <span style="font-size: 6.9pt; color: #334155; line-height: 1.28;">Aplikasi daring publik siap pakai tanpa instalasi lokal. Mendukung pemrosesan interaktif, visualisasi Plotly 3-kanal, dan ekspor laporan teknis langsung.</span>
            </td>
            <td style="width: 50%; background-color: #f8fafc; border: 1px solid #002d62; border-radius: 3px; padding: 6px 9px; vertical-align: top;">
                <b style="color: #002d62; font-size: 7.7pt;">2. REPOSITORI KODE SUMBER GITHUB</b><br>
                <a href="https://github.com/ahmaddidan/BSMA-v.2" style="font-size: 7.2pt; color: #002d62; font-weight: bold; text-decoration: underline;">github.com/ahmaddidan/BSMA-v.2</a><br>
                <span style="font-size: 6.9pt; color: #334155; line-height: 1.28;">Repositori terbuka memuat kode sumber modular, test suite otomatis (pytest), skrip generator panduan, data uji akselerogram, dan dokumentasi rilis.</span>
            </td>
        </tr>
    </table>
    """
    builder.insert_html_safe(p11, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 64), p11_access_html)
    y += 68

    y = builder.draw_subsection_heading(p11, y, "9.4", "Profil Pengembang & Informasi Proyek Kerja Praktik")
    p11_profile_html = """
    <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; padding: 6px 10px;">
        <p style="font-size: 7.2pt; line-height: 1.32; margin: 0; color: #0f172a;">
            • <b>Nama Pengembang</b>: Ahmad Didane Setyawan Putra (NIM: 123120094)<br>
            • <b>Program Studi / Fakultas</b>: Teknik Geofisika, Fakultas Teknik Industri<br>
            • <b>Perguruan Tinggi</b>: Institut Teknologi Sumatera<br>
            • <b>Instansi Pelaksanaan KP</b>: Stasiun Geofisika Kelas I Sleman, BMKG D.I. Yogyakarta (20 Juli – 20 Agustus 2026)<br>
            • <b>Surel / GitHub</b>: <a href="mailto:ahmad.123120094@student.itera.ac.id" style="color: #0284c7; text-decoration: underline;">ahmad.123120094@student.itera.ac.id</a> | <a href="https://github.com/ahmaddidan/BSMA-v.2" style="color: #0284c7; text-decoration: underline;">https://github.com/ahmaddidan/BSMA-v.2</a>
        </p>
    </div>
    """
    builder.insert_html_safe(p11, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 58), p11_profile_html)
    y += 68

    p11_closing_html = """
    <div style="background-color: #f1f5f9; border-left: 3.5px solid #002d62; padding: 8px 12px; border-radius: 0 4px 4px 0;">
        <b style="font-size: 7.5pt; color: #002d62;">PERNYATAAN INTEGRITAS AKADEMIK &amp; PENGESAHAN DOKUMEN:</b>
        <p style="font-size: 7.1pt; line-height: 1.34; margin: 3px 0 0 0; color: #334155; text-align: justify;">
            Buku panduan ini disusun sebagai dokumentasi teknis resmi luaran perangkat lunak <b>BMKG Strong Motion Analyzer (BSMA v2.0.0)</b> 
            dalam pemenuhan kegiatan Kerja Praktik mahasiswa Program Studi Teknik Geofisika, Fakultas Teknik Industri, Institut Teknologi Sumatera 
            di Stasiun Geofisika Kelas I Sleman, BMKG D.I. Yogyakarta. Seluruh formulasi, kode sumber, dan dokumentasi telah divalidasi dengan 
            dataset rekaman akselerogram aktual operasional BMKG.
        </p>
    </div>
    """
    builder.insert_html_safe(p11, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 68), p11_closing_html)

    # -------------------------------------------------------------
    # PAGE 12: BAB X - DAFTAR PUSTAKA & RUJUKAN ILMIAH
    # -------------------------------------------------------------
    p12 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p12)
    builder.add_page_header_footer(p12, "Daftar Pustaka & Rujukan", 12)
    y = 48.0

    y = builder.draw_section_heading(p12, y, "X", "DAFTAR PUSTAKA & RUJUKAN ILMIAH")

    p12_intro_html = """
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 8px 12px; margin-bottom: 14px;">
        <p style="font-size: 7.5pt; line-height: 1.38; color: #334155; text-align: justify; margin: 0;">
            Daftar pustaka di bawah ini memuat landasan teoretis, standar ketahanan gempa nasional (SNI), algoritma komputasi dinamika struktur, 
            serta kerangka kerja saintifik yang diimplementasikan secara langsung pada arsitektur perangkat lunak <b>BMKG Strong Motion Analyzer (BSMA v2.0.0)</b>:
        </p>
    </div>
    """
    builder.insert_html_safe(p12, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 46), p12_intro_html)
    y += 50

    p12_ref_html = """
    <style>
        .bib-entry {
            padding-left: 20px;
            text-indent: -20px;
            font-size: 7.4pt;
            line-height: 1.36;
            margin-bottom: 11px;
            text-align: justify;
            text-justify: inter-word;
            color: #1e293b;
        }
        .bib-entry a {
            color: #0284c7;
            text-decoration: underline;
        }
    </style>
    <div class="bib-entry">
        <b>1. Arias, A.</b> (1970). A measure of earthquake intensity. Dalam R. J. Hansen (Ed.), <i>Seismic design for nuclear power plants</i> (hlm. 438–483). Cambridge: MIT Press.
    </div>
    <div class="bib-entry">
        <b>2. Badan Standardisasi Nasional.</b> (2019). <i>SNI 1726:2019: Tata cara perencanaan ketahanan gempa untuk struktur bangunan gedung dan non gedung</i>. Jakarta: Badan Standardisasi Nasional.
    </div>
    <div class="bib-entry">
        <b>3. Beyreuther, M., Barsch, R., Krischer, L., Megies, T., Behr, Y., &amp; Wassermann, J.</b> (2010). ObsPy: A Python toolbox for seismology. <i>Seismological Research Letters</i>, 81(3), 530–533.<br><a href="https://doi.org/10.1785/gssrl.81.3.530">https://doi.org/10.1785/gssrl.81.3.530</a>
    </div>
    <div class="bib-entry">
        <b>4. Boore, D. M., &amp; Bommer, J. J.</b> (2005). Processing of strong-motion accelerograms: Needs, options and consequences. <i>Soil Dynamics and Earthquake Engineering</i>, 25(2), 93–115.<br><a href="https://doi.org/10.1016/j.soildyn.2004.10.007">https://doi.org/10.1016/j.soildyn.2004.10.007</a>
    </div>
    <div class="bib-entry">
        <b>5. Harris, F. J.</b> (1978). On the use of windows for harmonic analysis with the discrete Fourier transform. <i>Proceedings of the IEEE</i>, 66(1), 51–83.<br><a href="https://doi.org/10.1109/PROC.1978.10837">https://doi.org/10.1109/PROC.1978.10837</a>
    </div>
    <div class="bib-entry">
        <b>6. Newmark, N. M.</b> (1959). A method of computation for structural dynamics. <i>Journal of the Engineering Mechanics Division, ASCE</i>, 85(3), 67–94.<br><a href="https://doi.org/10.1061/JMCEA3.0000098">https://doi.org/10.1061/JMCEA3.0000098</a>
    </div>
    <div class="bib-entry">
        <b>7. Nigam, N. C., &amp; Jennings, P. C.</b> (1969). Calculation of response spectra from strong-motion earthquake records. <i>Bulletin of the Seismological Society of America</i>, 59(2), 909–922.<br><a href="https://doi.org/10.1785/BSSA0590020909">https://doi.org/10.1785/BSSA0590020909</a>
    </div>
    <div class="bib-entry">
        <b>8. Trifunac, M. D., &amp; Brady, A. G.</b> (1975). A study on the duration of strong earthquake ground motion. <i>Bulletin of the Seismological Society of America</i>, 65(3), 581–626.<br><a href="https://doi.org/10.1785/BSSA0650030581">https://doi.org/10.1785/BSSA0650030581</a>
    </div>
    <div class="bib-entry">
        <b>9. Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., &amp; van der Walt, S. J.</b> (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. <i>Nature Methods</i>, 17(3), 261–272.<br><a href="https://doi.org/10.1038/s41592-019-0686-2">https://doi.org/10.1038/s41592-019-0686-2</a>
    </div>
    <div class="bib-entry" style="margin-bottom: 0;">
        <b>10. Worden, C. B., Gerstenberger, M. C., Rhoades, D. A., &amp; Wald, D. J.</b> (2012). Probabilistic relationships between ground-motion parameters and MMI. <i>Bulletin of the Seismological Society of America</i>, 102(1), 204–221.<br><a href="https://doi.org/10.1785/0120110156">https://doi.org/10.1785/0120110156</a>
    </div>
    """
    builder.insert_html_safe(p12, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 420), p12_ref_html)


    # Injekt Hyperlink Aktif ke Seluruh Halaman Dokumen
    add_all_hyperlinks(doc)

    # Save PDF with lossless compression
    doc.save(str(PDF_OUTPUT_PATH), deflate=True, garbage=4, clean=True)
    doc.close()
    return PDF_OUTPUT_PATH


if __name__ == "__main__":
    out = build_guidebook_pdf()
    print(f"BSMA User Guidebook successfully built at: {out}")
