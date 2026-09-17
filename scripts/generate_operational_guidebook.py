# -*- coding: utf-8 -*-
"""
BMKG Strong Motion Analyzer (BSMA v2.0.0)
Script: scripts/generate_operational_guidebook.py

Bilingual Dedicated Operational Technical Manual & User Guidebook:
- outputs/BSMA_Panduan_Teknis_Operasional_ID.pdf (Bahasa Indonesia)
- outputs/BSMA_Panduan_Teknis_Operasional_EN.pdf (English Edition)

Features:
1. Dynamic Proportional Screenshot Cards: Automatically reads the true aspect ratio
   of each of the UI screenshots from FOTO UNTUK GUIDEBOOK.
2. Snug, elegant image framing with subtle borders, padding, and bottom caption bars.
3. 18 dedicated pages (3 Front Matter + 15 Body Pages) per language.
4. Dual numbering: Roman numerals (i, ii, iii) for Front Matter, Arabic numerals (1..15) for Body Chapters.
5. Strictly focused on practical software operation and engineering workflows.
6. Does NOT overwrite old legacy files; writes strictly to new designated files.

Author: Ahmad Didane Setyawan Putra (NIM: 123120094)
Department of Geophysical Engineering, Institut Teknologi Sumatera (ITERA)
Host Institution: BMKG Stasiun Geofisika Kelas I Sleman, D.I. Yogyakarta
"""

from __future__ import annotations

import io
import os
import re
import shutil
import sys
from pathlib import Path

# Force UTF-8 stdout if needed
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = PROJECT_ROOT / "assets"
SCREENSHOTS_DIR = Path(r"D:\Perkuliahan\Semester 7\Magang\Laporan\FOTO UNTUK GUIDEBOOK")

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
COLOR_GOLD = (0.925, 0.706, 0.078)         # #EAB308 (Accent Gold)

PAGE_W = 595.3  # A4 width in pt
PAGE_H = 841.9  # A4 height in pt
LEFT_X = 50.0
RIGHT_X = 545.3
CONTENT_W = RIGHT_X - LEFT_X

FONT_REG = "helv"
FONT_BOLD = "hebo"
FONT_IT = "heit"
FONT_BI = "hebi"


class OperationalGuidebookBuilder:
    """Publication-grade Operational Guidebook Builder using PyMuPDF."""

    def __init__(self, doc: pymupdf.Document, lang: str = "id"):
        self.doc = doc
        self.lang = lang.lower()

    def init_page_fonts(self, page: pymupdf.Page) -> None:
        """Base-14 Helvetica fonts are natively built-in across all PDF viewers."""
        pass

    def insert_html_safe(self, page: pymupdf.Page, rect: pymupdf.Rect, html_body: str,
                         font_size="7.5pt", line_height="1.30") -> None:
        """Insert HTML body with robust styling and typography."""
        css = f"""
        body {{
            font-family: 'Helvetica', 'Arial', sans-serif;
            color: #0f172a;
            margin: 0;
            padding: 0;
        }}
        p {{
            text-align: justify;
            text-justify: inter-word;
            line-height: {line_height};
            margin: 0 0 4.5px 0;
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
            margin-top: 3px;
            margin-bottom: 5px;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 3.5px 5.5px;
            line-height: 1.25;
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
        if self.lang == "id":
            header_title = "BMKG STRONG MOTION ANALYZER (BSMA v2.0.0) - PANDUAN PENGOPERASIAN"
            footer_author = "Ahmad Didane Setyawan Putra - Teknik Geofisika, Institut Teknologi Sumatera (ITERA)"
        else:
            header_title = "BMKG STRONG MOTION ANALYZER (BSMA v2.0.0) - OPERATIONAL TECHNICAL MANUAL"
            footer_author = "Ahmad Didane Setyawan Putra - Geophysical Engineering, Institut Teknologi Sumatera (ITERA)"

        page.insert_text((LEFT_X, 23), header_title,
                         fontsize=7.4, fontname=FONT_BOLD, color=COLOR_NAVY)
        font_bold_obj = pymupdf.Font(FONT_BOLD)
        badge_w = font_bold_obj.text_length(section_badge.upper(), fontsize=7.2)
        page.insert_text((RIGHT_X - badge_w, 23), section_badge.upper(),
                         fontsize=7.2, fontname=FONT_BOLD, color=COLOR_SLATE)

        # Bottom Footer
        page.draw_line(pymupdf.Point(LEFT_X, 810), pymupdf.Point(RIGHT_X, 810), color=COLOR_CARD_BORDER, width=0.8)
        page.insert_text((LEFT_X, 822), footer_author,
                         fontsize=7.2, fontname=FONT_IT, color=COLOR_SLATE)
        page.insert_text((RIGHT_X - 105, 822), page_num_str,
                         fontsize=7.2, fontname=FONT_BOLD, color=COLOR_NAVY)

    def draw_card(self, page: pymupdf.Page, rect: pymupdf.Rect, bg_col=COLOR_CARD_BG, border_col=COLOR_CARD_BORDER, border_w=0.8) -> None:
        """Draw solid card box."""
        page.draw_rect(rect, color=border_col, fill=bg_col, width=border_w)

    def draw_chapter_banner(self, page: pymupdf.Page, title_str: str, y_top: float = 38.0) -> float:
        """Draw clean chapter header banner with accent bar."""
        page.draw_rect(pymupdf.Rect(LEFT_X, y_top, LEFT_X + 6, y_top + 20), color=COLOR_NAVY, fill=COLOR_NAVY)
        page.insert_text((LEFT_X + 14, y_top + 15), title_str, fontsize=10.5, fontname=FONT_BOLD, color=COLOR_NAVY)
        page.draw_line(pymupdf.Point(LEFT_X, y_top + 25), pymupdf.Point(RIGHT_X, y_top + 25), color=COLOR_CARD_BORDER, width=0.8)
        return y_top + 31.0

    def draw_callout(self, page: pymupdf.Page, rect: pymupdf.Rect, title: str, text: str, callout_type="info") -> None:
        """Draw highlighted callout box with colored left strip."""
        is_id = (self.lang == "id")
        if callout_type == "info":
            bg = (0.941, 0.973, 1.0)
            border = (0.729, 0.855, 0.988)
            accent = COLOR_SKY
            badge = "[PETUNJUK PRAKTIS]" if is_id else "[PRACTICAL TIP]"
        elif callout_type == "warning":
            bg = (1.0, 0.984, 0.922)
            border = (0.992, 0.886, 0.655)
            accent = COLOR_AMBER
            badge = "[PERHATIAN OPERASIONAL]" if is_id else "[OPERATIONAL ADVISORY]"
        else:
            bg = (0.941, 0.988, 0.961)
            border = (0.655, 0.922, 0.749)
            accent = COLOR_GREEN
            badge = "[REKOMENDASI BAKU]" if is_id else "[STANDARD PRACTICE]"

        self.draw_card(page, rect, bg_col=bg, border_col=border)
        page.draw_rect(pymupdf.Rect(rect.x0, rect.y0, rect.x0 + 4, rect.y1), color=accent, fill=accent)
        page.insert_text((rect.x0 + 10, rect.y0 + 11.5), f"{badge} {title}", fontsize=7.5, fontname=FONT_BOLD, color=accent)

        html = f"""
        <p style="font-size: 7.0pt; line-height: 1.25; margin: 0; text-align: justify; text-justify: inter-word; color: #0f172a;">
            {text}
        </p>
        """
        self.insert_html_safe(page, pymupdf.Rect(rect.x0 + 10, rect.y0 + 15.5, rect.x1 - 8, rect.y1 - 3.5), html)

    def insert_screenshot_card(self, page: pymupdf.Page, x0: float, y0: float, w: float,
                                img_name: str, caption: str, bg_col=COLOR_WHITE,
                                forced_h: float = None, pad: float = 3.5, caption_h: float = 15.0) -> pymupdf.Rect:
        """
        Insert a framed screenshot card with an exact proportional height matching the image aspect ratio.
        Eliminates awkward top/bottom white space gaps completely.
        Returns the bounding Rect of the rendered card.
        """
        img_path = SCREENSHOTS_DIR / img_name
        if not img_path.is_file():
            img_path = ASSETS_DIR / img_name
        if not img_path.is_file():
            for f in SCREENSHOTS_DIR.glob(f"*{img_name.strip()}*"):
                img_path = f
                break

        aspect = 16.0 / 9.0  # fallback
        if img_path.is_file():
            try:
                temp_doc = pymupdf.open(img_path)
                t_rect = temp_doc[0].rect
                if t_rect.height > 0:
                    aspect = t_rect.width / t_rect.height
                temp_doc.close()
            except Exception:
                pass

        inner_w = w - (2 * pad)
        if forced_h is not None:
            inner_h = forced_h
        else:
            inner_h = inner_w / aspect

        card_h = inner_h + caption_h + (2 * pad)
        card_rect = pymupdf.Rect(x0, y0, x0 + w, y0 + card_h)

        # Draw outer container card
        self.draw_card(page, card_rect, bg_col=bg_col, border_col=COLOR_CARD_BORDER, border_w=0.8)

        img_rect = pymupdf.Rect(x0 + pad, y0 + pad, x0 + w - pad, y0 + pad + inner_h)
        if img_path.is_file():
            page.insert_image(img_rect, filename=str(img_path), keep_proportion=True)
        else:
            page.insert_text((x0 + 10, y0 + 20), f"[Screenshot: {img_name} not found]",
                             fontsize=8.0, fontname=FONT_IT, color=COLOR_RED)

        # Caption Bar at bottom
        caption_rect = pymupdf.Rect(x0, y0 + card_h - caption_h, x0 + w, y0 + card_h)
        page.draw_rect(caption_rect, color=COLOR_CARD_BORDER, fill=(0.95, 0.96, 0.98), width=0.5)
        page.insert_text((x0 + 8, y0 + card_h - 4.5), caption,
                         fontsize=6.8, fontname=FONT_BOLD, color=COLOR_NAVY)

        return card_rect

    def clean_pdf_ligatures(self) -> None:
        """Safely decompose ligatures (fi, fl, ff, ffi, ffl) into standard ASCII letters in ToUnicode CMaps."""
        for xref in range(1, self.doc.xref_length()):
            if not self.doc.xref_is_stream(xref):
                continue
            try:
                stream_data = self.doc.xref_stream(xref)
                if b"/CIDInit" not in stream_data:
                    continue
                txt = stream_data.decode("latin1")
                
                txt_new = re.sub(r"<([0-9a-fA-F]+)>\s+<([0-9a-fA-F]+)>\s+<fb01>",
                                 r"<\1> <\2> [ <00660069> <0066006c> ]", txt)
                txt_new = re.sub(r"<([0-9a-fA-F]+)>\s+<([0-9a-fA-F]+)>\s+<fb03>",
                                 r"<\1> <\2> [ <006600660069> <00660066006c> ]", txt_new)
                txt_new = re.sub(r"<([0-9a-fA-F]+)>\s+<fb00>", r"<\1> <00660066>", txt_new)
                txt_new = re.sub(r"<([0-9a-fA-F]+)>\s+<fb01>", r"<\1> <00660069>", txt_new)
                txt_new = re.sub(r"<([0-9a-fA-F]+)>\s+<fb02>", r"<\1> <0066006c>", txt_new)
                txt_new = re.sub(r"<([0-9a-fA-F]+)>\s+<fb03>", r"<\1> <006600660069>", txt_new)
                txt_new = re.sub(r"<([0-9a-fA-F]+)>\s+<fb04>", r"<\1> <00660066006c>", txt_new)
                
                if txt_new != txt:
                    self.doc.update_stream(xref, txt_new.encode("latin1"))
            except Exception:
                pass


def build_operational_guidebook(lang: str = "id") -> Path:
    """Compile dedicated 18-page operational technical manual with proportional screenshots."""
    lang = lang.lower()
    is_id = (lang == "id")
    edition_label = "Bahasa Indonesia" if is_id else "English Edition"
    print(f"Building BSMA Dedicated Operational Guidebook [{edition_label}]...")

    doc = pymupdf.open()
    builder = OperationalGuidebookBuilder(doc, lang=lang)
    font_bold_obj = pymupdf.Font(FONT_BOLD)

    TOTAL_BODY_PAGES = 15

    # =========================================================================
    # HALAMAN i (Sampul / Cover Page)
    # =========================================================================
    p1 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p1)

    p1.draw_rect(pymupdf.Rect(0, 0, PAGE_W, 330), color=COLOR_NAVY, fill=COLOR_NAVY)
    p1.draw_rect(pymupdf.Rect(0, 325, PAGE_W, 330), color=COLOR_SKY, fill=COLOR_SKY)

    bmkg_icon = LOGO_BMKG_ICON_PATH if LOGO_BMKG_ICON_PATH.is_file() else LOGO_JUDUL_PATH
    if bmkg_icon.is_file():
        p1.insert_image(pymupdf.Rect(50, 42, 114, 106), filename=str(bmkg_icon))
    bmkg_tw = font_bold_obj.text_length("BMKG", fontsize=11.0)
    p1.insert_text((82 - bmkg_tw / 2, 122), "BMKG", fontsize=11.0, fontname=FONT_BOLD, color=COLOR_WHITE)

    p1.draw_line(pymupdf.Point(130, 42), pymupdf.Point(130, 124), color=(0.35, 0.60, 0.85), width=1.2)

    itera_icon = LOGO_ITERA_ICON_PATH if LOGO_ITERA_ICON_PATH.is_file() else LOGO_ITERA_PATH
    if itera_icon.is_file():
        p1.insert_image(pymupdf.Rect(146, 42, 210, 106), filename=str(itera_icon))
    itera_tw = font_bold_obj.text_length("ITERA", fontsize=11.0)
    p1.insert_text((178 - itera_tw / 2, 122), "ITERA", fontsize=11.0, fontname=FONT_BOLD, color=COLOR_WHITE)

    if is_id:
        text_header_1 = "BUKU PANDUAN PENGOPERASIAN PERANGKAT LUNAK (USER MANUAL)"
        text_header_2 = "Program Studi Teknik Geofisika, Fakultas Teknologi Industri"
        text_header_3 = "Institut Teknologi Sumatera (ITERA)"
        tag_sup = "PANDUAN TEKNIS OPERASIONAL PERANGKAT LUNAK"
        tag_dev = "Dikembangkan di BMKG Stasiun Geofisika Sleman"
        sub_title = "Platform Komputasi Sinyal Akselerograf Gerakan Tanah Kuat & Kinematika Seismik"
        meta_hdr = "INFORMASI DOKUMEN & IDENTITAS PENGEMBANGAN"
        inst_disclaimer = "Karya Akademik Mandiri Mahasiswa - Afiliasi kelembagaan tidak menyatakan bahwa BSMA merupakan perangkat lunak resmi BMKG."
    else:
        text_header_1 = "SOFTWARE OPERATIONAL GUIDEBOOK & USER MANUAL"
        text_header_2 = "Department of Geophysical Engineering, Faculty of Industrial Technology"
        text_header_3 = "Institut Teknologi Sumatera (ITERA)"
        tag_sup = "SOFTWARE OPERATIONAL TECHNICAL MANUAL"
        tag_dev = "Developed at BMKG Sleman Geophysical Station"
        sub_title = "Computational Platform for Strong Ground Motion Accelerograph Signals & Seismic Kinematics"
        meta_hdr = "DOCUMENT METADATA & DEVELOPMENT INFORMATION"
        inst_disclaimer = "Independent Student Academic Work - Institutional affiliation does not imply that BSMA is official BMKG software."

    p1.insert_text((230, 62), text_header_1, fontsize=9.0, fontname=FONT_BOLD, color=COLOR_WHITE)
    p1.insert_text((230, 78), text_header_2, fontsize=8.0, fontname=FONT_REG, color=(0.85, 0.92, 1.0))
    p1.insert_text((230, 93), text_header_3, fontsize=7.4, fontname=FONT_IT, color=(0.75, 0.85, 0.95))

    p1.insert_text((50, 175), tag_sup, fontsize=9.5, fontname=FONT_BOLD, color=(0.75, 0.90, 1.0))
    p1.insert_text((50, 212), "BMKG Strong Motion Analyzer", fontsize=24.0, fontname=FONT_BOLD, color=COLOR_WHITE)
    p1.insert_text((50, 242), "(BSMA v2.0.0)", fontsize=18.0, fontname=FONT_BOLD, color=COLOR_SKY)

    p1.insert_text((50, 272), tag_dev, fontsize=12.0, fontname=FONT_BOLD, color=COLOR_GOLD)
    p1.insert_text((50, 292), sub_title, fontsize=9.0, fontname=FONT_REG, color=(0.90, 0.95, 1.0))

    builder.draw_card(p1, pymupdf.Rect(50, 360, 545, 750), bg_col=COLOR_WHITE, border_col=COLOR_CARD_BORDER)

    p1.insert_text((75, 395), meta_hdr, fontsize=11.0, fontname=FONT_BOLD, color=COLOR_NAVY)
    p1.draw_line(pymupdf.Point(75, 405), pymupdf.Point(520, 405), color=COLOR_SKY, width=1.2)

    if is_id:
        fields = [
            ("Nama Perangkat Lunak", "BMKG Strong Motion Analyzer (BSMA)"),
            ("Versi Rilis / Build", "v2.0.0 (Build 2026.08)"),
            ("Lokasi Pengembangan", "Stasiun Geofisika Kelas I Sleman, BMKG D.I. Yogyakarta"),
            ("Penyusun / Pengembang", "Ahmad Didane Setyawan Putra"),
            ("Nomor Induk Mahasiswa", "123120094"),
            ("Program Studi", "Teknik Geofisika"),
            ("Fakultas", "Fakultas Teknologi Industri"),
            ("Perguruan Tinggi", "Institut Teknologi Sumatera (ITERA)"),
            ("Instansi Mitra Kerja Praktik", "Stasiun Geofisika Kelas I Sleman, BMKG D.I. Yogyakarta"),
            ("Periode Kerja Praktik", "20 Juli 2026 s.d. 20 Agustus 2026"),
            ("Klasifikasi Proyek", "Karya Akademik Mandiri Mahasiswa (Non-Komersial)"),
            ("Alamat Aplikasi Web", "https://strong-motion.streamlit.app/"),
            ("Repositori Kode Sumber", "https://github.com/ahmaddidan/BSMA-v.2"),
        ]
    else:
        fields = [
            ("Software System Name", "BMKG Strong Motion Analyzer (BSMA)"),
            ("Release Version / Build", "v2.0.0 (Build 2026.08)"),
            ("Host Institution", "Sleman Geophysical Station Class I, BMKG D.I. Yogyakarta"),
            ("Author / Developer", "Ahmad Didane Setyawan Putra"),
            ("Student ID (NIM)", "123120094"),
            ("Study Program", "Geophysical Engineering"),
            ("Faculty", "Faculty of Industrial Technology"),
            ("University", "Institut Teknologi Sumatera (ITERA)"),
            ("Host Partner Institution", "Sleman Geophysical Station Class I, BMKG D.I. Yogyakarta"),
            ("Internship Period", "July 20, 2026 - August 20, 2026"),
            ("Project Classification", "Independent Student Academic Work (Non-Commercial)"),
            ("Web Application URL", "https://strong-motion.streamlit.app/"),
            ("Source Code Repository", "https://github.com/ahmaddidan/BSMA-v.2"),
        ]

    cur_y = 435
    for label, val in fields:
        p1.insert_text((75, cur_y), f"{label}", fontsize=8.4, fontname=FONT_BOLD, color=COLOR_SLATE)
        if "http" in val:
            p1.insert_text((230, cur_y), f":  {val}", fontsize=8.4, fontname=FONT_REG, color=COLOR_SKY)
            link_w = font_bold_obj.text_length(f":  {val}", fontsize=8.4)
            p1.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(230, cur_y - 8, 230 + link_w, cur_y + 2), "uri": val})
        else:
            p1.insert_text((230, cur_y), f":  {val}", fontsize=8.4, fontname=FONT_REG, color=COLOR_DARK)
        cur_y += 24

    p1.insert_text((50, 790), inst_disclaimer, fontsize=7.2, fontname=FONT_IT, color=COLOR_SLATE)

    # =========================================================================
    # HALAMAN ii (Daftar Isi / Table of Contents)
    # =========================================================================
    p2 = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p2)
    p2_badge = "DAFTAR ISI" if is_id else "TABLE OF CONTENTS"
    p2_pstr = "Halaman ii dari iii" if is_id else "Page ii of iii"
    builder.add_page_header_footer(p2, p2_badge, p2_pstr)

    toc_banner = "DAFTAR ISI PANDUAN PENGOPERASIAN" if is_id else "OPERATIONAL MANUAL TABLE OF CONTENTS"
    y = builder.draw_chapter_banner(p2, toc_banner)

    builder.draw_card(p2, pymupdf.Rect(LEFT_X, y + 10, RIGHT_X, y + 710), bg_col=COLOR_WHITE, border_col=COLOR_CARD_BORDER)
    toc_sub = "SISTEMATIKA BAB & STRUKTUR HALAMAN DOKUMEN" if is_id else "DOCUMENT CHAPTERS & PAGE STRUCTURE"
    p2.insert_text((LEFT_X + 24, y + 38), toc_sub, fontsize=10.0, fontname=FONT_BOLD, color=COLOR_NAVY)
    p2.draw_line(pymupdf.Point(LEFT_X + 24, y + 46), pymupdf.Point(RIGHT_X - 24, y + 46), color=COLOR_SKY, width=1.2)

    if is_id:
        toc_items = [
            ("BAGIAN AWAL (FRONT MATTER)", "", True),
            ("Halaman Sampul & Identitas Pengembangan", "i", False),
            ("Daftar Isi Panduan Pengoperasian", "ii", False),
            ("Panduan Cepat 5 Langkah Operasional (Quick Start Workflow)", "iii", False),
            ("BAGIAN I: PERSIAPAN & KONFIGURASI MENU SIDEBAR", "", True),
            ("BAB I: AKSES SISTEM & PENGUNGGAHAN DATA (DATA INPUT)", "1", False),
            ("BAB II: PEMILIHAN ASAL-USUL REKAMAN (DATA PROVENANCE)", "2", False),
            ("BAB III: KONFIGURASI PARAMETER DSP (PROCESSING & UNIT)", "3", False),
            ("BAB IV: PARAMETER DINAMIKA STRUKTUR, EVENT GEMPA, & BENCHMARK", "4", False),
            ("BAGIAN II: NAVIGASI 7 TAB ANALISIS INTERAKTIF", "", True),
            ("BAB V: EKSEKUSI PIPELINE (RUN ANALYSIS) & TAB 1 SUMMARY", "5", False),
            ("BAB VI: TAB 2 - RIWAYAT WAKTU KINEMATIKA LENGKAP (WAVEFORMS)", "6", False),
            ("BAB VII: TAB 3 - KENDALI MUTU SINYAL (QUALITY CONTROL GATES)", "7", False),
            ("BAB VIII: TAB 4 - GERAKAN TANAH KUAT & ENERGI SEISMIK (STRONG MOTION)", "8", False),
            ("BAB IX: TAB 5 - ESTIMASI INTENSITAS INSTRUMENTAL (INTENSITY MMI)", "9", False),
            ("BAB X: TAB 6 - SPEKTRUM RESPONS STRUKTUR SDOF & STANDAR SNI 1726", "10", False),
            ("BAB XI: ANALISIS SPEKTRAL LANJUTAN (SPEKTRUM FAS & KURVA HUSID)", "11", False),
            ("BAB XII: TAB 7 - EKSPOR LAPORAN TEKNIK PDF & DELIVERABLES", "12", False),
            ("BAGIAN III: PEMROSESAN MASSAL & DUKUNGAN OPERASIONAL", "", True),
            ("BAB XIII: PROTOKOL PEMROSESAN BATCH MULTI-STASIUN REGIONAL", "13", False),
            ("BAB XIV: PANDUAN PEMECAHAN MASALAH OPERASIONAL (TROUBLESHOOTING)", "14", False),
            ("PROFIL PENGEMBANG & PENGESAHAN KERJA PRAKTIK", "15", False),
        ]
    else:
        toc_items = [
            ("FRONT MATTER", "", True),
            ("Cover Page & Development Metadata", "i", False),
            ("Operational Manual Table of Contents", "ii", False),
            ("5-Step Quick Start Operational Workflow", "iii", False),
            ("PART I: PREPARATION & SIDEBAR CONFIGURATION", "", True),
            ("CHAPTER I: SYSTEM ACCESS & SEISMIC DATA INGESTION (DATA INPUT)", "1", False),
            ("CHAPTER II: RECORDING ORIGIN DECLARATION (DATA PROVENANCE)", "2", False),
            ("CHAPTER III: DSP PARAMETERS CONFIGURATION (PROCESSING & UNIT)", "3", False),
            ("CHAPTER IV: STRUCTURAL PARAMETERS, EVENT METADATA, & BENCHMARK", "4", False),
            ("PART II: INTERACTIVE 7-TAB WORKSTATION NAVIGATION", "", True),
            ("CHAPTER V: PIPELINE EXECUTION (RUN ANALYSIS) & TAB 1 SUMMARY", "5", False),
            ("CHAPTER VI: TAB 2 - FULL KINEMATIC TIME HISTORIES (WAVEFORMS)", "6", False),
            ("CHAPTER VII: TAB 3 - SIGNAL QUALITY CONTROL GATES (QC GATES)", "7", False),
            ("CHAPTER VIII: TAB 4 - STRONG GROUND MOTION & SEISMIC ENERGY", "8", False),
            ("CHAPTER IX: TAB 5 - INSTRUMENTAL INTENSITY ESTIMATION (INTENSITY MMI)", "9", False),
            ("CHAPTER X: TAB 6 - SDOF RESPONSE SPECTRA & SNI 1726 DESIGN OVERLAYS", "10", False),
            ("CHAPTER XI: ADVANCED SPECTRAL ANALYSIS (FAS SPECTRUM & HUSID PLOT)", "11", False),
            ("CHAPTER XII: TAB 7 - PDF ENGINEERING REPORT EXPORT & DELIVERABLES", "12", False),
            ("PART III: MASS PROCESSING & OPERATIONAL SUPPORT", "", True),
            ("CHAPTER XIII: REGIONAL MULTI-STATION BATCH PROCESSING PROTOCOL", "13", False),
            ("CHAPTER XIV: OPERATIONAL TROUBLESHOOTING GUIDE & FIELD PROCEDURES", "14", False),
            ("DEVELOPER PROFILE & INTERNSHIP ENDORSEMENT", "15", False),
        ]

    cur_toc_y = y + 72
    for title, page_str, is_header in toc_items:
        if is_header:
            cur_toc_y += 6
            p2.draw_rect(pymupdf.Rect(LEFT_X + 20, cur_toc_y - 2, RIGHT_X - 20, cur_toc_y + 13), color=COLOR_CARD_BG, fill=COLOR_CARD_BG)
            p2.insert_text((LEFT_X + 26, cur_toc_y + 9), title, fontsize=7.8, fontname=FONT_BOLD, color=COLOR_SKY)
            cur_toc_y += 20
            continue

        font_name = FONT_BOLD if page_str in ["i", "ii", "iii"] else FONT_REG
        font_col = COLOR_NAVY if page_str in ["i", "ii", "iii"] else COLOR_DARK
        p2.insert_text((LEFT_X + 26, cur_toc_y), title, fontsize=8.0, fontname=font_name, color=font_col)

        text_len = font_bold_obj.text_length(title, fontsize=8.0)
        leader_start_x = LEFT_X + 32 + text_len
        leader_end_x = RIGHT_X - 48
        if leader_end_x > leader_start_x:
            dots_count = int((leader_end_x - leader_start_x) / 4.4)
            dots = ". " * (dots_count // 2)
            p2.insert_text((leader_start_x, cur_toc_y), dots, fontsize=7.2, fontname=FONT_REG, color=COLOR_MUTED)

        p2.insert_text((RIGHT_X - 42, cur_toc_y), page_str, fontsize=8.2, fontname=FONT_BOLD, color=COLOR_NAVY)
        cur_toc_y += 22.5

    # =========================================================================
    # HALAMAN iii (Panduan Cepat / Quick Start 5 Langkah)
    # =========================================================================
    p3_front = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p3_front)
    p3_badge = "PANDUAN CEPAT OPERASIONAL" if is_id else "QUICK START WORKFLOW"
    p3_pstr = "Halaman iii dari iii" if is_id else "Page iii of iii"
    builder.add_page_header_footer(p3_front, p3_badge, p3_pstr)

    qs_banner = "PANDUAN CEPAT 5 LANGKAH OPERASIONAL (QUICK START)" if is_id else "5-STEP QUICK START OPERATIONAL WORKFLOW"
    y = builder.draw_chapter_banner(p3_front, qs_banner)

    if is_id:
        qs_intro = """
        <p>Bagan alur kerja di bawah ini merangkum 5 langkah operasional utama dalam menjalankan perangkat lunak 
        <b>BMKG Strong Motion Analyzer (BSMA v2.0.0)</b> untuk pemrosesan rekaman gerakan tanah kuat secara cepat, 
        mulai dari pemuatan berkas hingga pengunduhan deliverables rekayasa:</p>
        """
        steps = [
            ("Langkah 1: Ingesti Data Seismik (Data Ingestion)",
             "Akses aplikasi web/lokal, lalu unggah berkas akselerograf (.mseed atau .sac) pada sidebar DATA INPUT. "
             "Unggah berkas StationXML (.xml) jika rekaman masih berdimensi hitungan digital mentah (counts). Tentukan provenance data.",
             COLOR_SKY),
            ("Langkah 2: Konfigurasi Parameter DSP (Parameter Settings)",
             "Pilih satuan fisik luaran (Gal atau m/s²). Klik tombol pintas 'Preset Baku Strong-Motion (0.05 - 40.0 Hz)' "
             "untuk menerapkan filter lolos pita (Bandpass) standar BMKG secara instan.",
             COLOR_SKY),
            ("Langkah 3: Eksekusi Analisis Single-Station (Run Analysis)",
             "Pilih stasiun pada 'Select Recording Window', lalu klik tombol utama '[Run Analysis]'. "
             "Pipeline otomatis mengeksekusi kendali mutu QC, filter SOS orde-4 zero-phase, koreksi baseline, serta integrasi kinematika ganda.",
             COLOR_GREEN),
            ("Langkah 4: Eksplorasi 7 Tab Analisis Interaktif (Dashboard Exploration)",
             "Navigasi tab analisis untuk meninjau ringkasan kinematika (Tab 1), grafik 3-komponen (Tab 2), audit QC (Tab 3), "
             "energi getaran & CAV (Tab 4), intensitas MMI (Tab 5), dan spektrum respons desain gempa SNI 1726 (Tab 6).",
             COLOR_AMBER),
            ("Langkah 5: Ekspor & Distribusi Deliverables (Export Deliverables)",
             "Pada Tab 7, tinjau pratinjau laporan teknik resmi 2-halaman, lalu unduh laporan PDF siap cetak, tabel deret waktu "
             "CSV percepatan/kecepatan/perpindahan, CSV spektrum respons, serta paket arsip ZIP terpadu.",
             COLOR_NAVY),
        ]
        callout_qs_title = "Praktek Terbaik Operasional Cepat"
        callout_qs = (
            "Tips Operasional Harian: Untuk data akselerograf operasional BMKG yang telah dikonversi langsung oleh stasiun lapangan ke "
            "satuan fisik percepatan (Gal atau m/s²), cukup unggah berkas .mseed, biarkan filter pada preset baku 0.05 - 40.0 Hz, "
            "dan seluruh tab analisis beserta laporan PDF langsung terkomputasi secara otomatis dalam hitungan detik!"
        )
    else:
        qs_intro = """
        <p>The workflow diagram below summarizes the 5 primary operational steps to execute 
        <b>BMKG Strong Motion Analyzer (BSMA v2.0.0)</b> for rapid strong ground motion processing, 
        spanning file loading to engineering deliverables export:</p>
        """
        steps = [
            ("Step 1: Seismic Data Ingestion (Data Ingestion)",
             "Access the cloud web platform or local instance, then upload accelerogram files (.mseed or .sac) in the DATA INPUT sidebar. "
             "Upload StationXML (.xml) if records are raw digital counts. Specify recording data provenance.",
             COLOR_SKY),
            ("Step 2: DSP Parameter Configuration (Parameter Settings)",
             "Select output physical units (Gal or m/s²). Click the 'Standard Strong-Motion Preset (0.05 - 40.0 Hz)' button "
             "to instantly apply the standard BMKG-recommended bandpass filter settings.",
             COLOR_SKY),
            ("Step 3: Single-Station Analysis Execution (Run Analysis)",
             "Select the target station from 'Select Recording Window', then click the main '[Run Analysis]' button. "
             "The pipeline automatically runs QC screening, 4th-order zero-phase SOS filtering, baseline detrending, and dual kinematic integration.",
             COLOR_GREEN),
            ("Step 4: Interactive 7-Tab Dashboard Exploration (Dashboard Exploration)",
             "Navigate interactive tabs to examine kinematic summaries (Tab 1), 3-component time series (Tab 2), QC diagnostics (Tab 3), "
             "vibration energy & CAV (Tab 4), MMI intensity (Tab 5), and SNI 1726 structural design response spectra (Tab 6).",
             COLOR_AMBER),
            ("Step 5: Deliverables Export & Distribution (Export Deliverables)",
             "On Tab 7, review the print-ready 2-page engineering report preview, then download the official PDF report, "
             "time-history CSVs (acc/vel/disp), response spectra CSVs, and unified ZIP archive packages.",
             COLOR_NAVY),
        ]
        callout_qs_title = "Fast Operational Best Practice"
        callout_qs = (
            "Daily Operational Best Practice: For BMKG operational accelerograph files pre-calibrated by station dataloggers into "
            "physical acceleration units (Gal or m/s²), simply upload the .mseed file, retain default 0.05 - 40.0 Hz bandpass settings, "
            "and all analytical tabs and publication-grade PDF reports will be computed automatically in seconds!"
        )

    builder.insert_html_safe(p3_front, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 45), qs_intro, font_size="8.0pt")

    card_h = 74.0
    cur_step_y = y + 55
    for idx, (step_title, step_desc, step_color) in enumerate(steps):
        s_rect = pymupdf.Rect(LEFT_X, cur_step_y, RIGHT_X, cur_step_y + card_h)
        builder.draw_card(p3_front, s_rect, bg_col=COLOR_CARD_BG, border_col=COLOR_CARD_BORDER)
        p3_front.draw_rect(pymupdf.Rect(LEFT_X, cur_step_y, LEFT_X + 5, cur_step_y + card_h), color=step_color, fill=step_color)
        p3_front.insert_text((LEFT_X + 16, cur_step_y + 18), step_title, fontsize=8.6, fontname=FONT_BOLD, color=COLOR_NAVY)

        desc_html = f"<p style='font-size: 7.6pt; line-height: 1.34; margin: 0; color: #475569;'>{step_desc}</p>"
        builder.insert_html_safe(p3_front, pymupdf.Rect(LEFT_X + 16, cur_step_y + 26, RIGHT_X - 16, cur_step_y + card_h - 6), desc_html)
        cur_step_y += card_h + 12.0

    builder.draw_callout(p3_front, pymupdf.Rect(LEFT_X, cur_step_y + 10, RIGHT_X, cur_step_y + 85),
                         callout_qs_title, callout_qs, callout_type="success")

    # =========================================================================
    # HALAMAN 1 (BAB I: Akses Sistem & Data Input)
    # =========================================================================
    p1_body = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p1_body)
    b1_badge = "BAB I - DATA INPUT" if is_id else "CH. I - DATA INPUT"
    b1_pstr = f"Halaman 1 dari {TOTAL_BODY_PAGES}" if is_id else f"Page 1 of {TOTAL_BODY_PAGES}"
    builder.add_page_header_footer(p1_body, b1_badge, b1_pstr)

    b1_banner = "BAB I: AKSES SISTEM & PENGUNGGAHAN DATA (DATA INPUT)" if is_id else "CHAPTER I: SYSTEM ACCESS & SEISMIC DATA INGESTION (DATA INPUT)"
    y = builder.draw_chapter_banner(p1_body, b1_banner)

    if is_id:
        c1_text = """
        <p><b>1.1 Tata Cara Akses Antarmuka Perangkat Lunak</b></p>
        <p>Perangkat lunak <b>BMKG Strong Motion Analyzer (BSMA v2.0.0)</b> dapat diakses secara langsung melalui peramban web modern 
        (Google Chrome, Microsoft Edge, Mozilla Firefox) tanpa proses instalasi yang rumit pada portal resmi: 
        <a href="https://strong-motion.streamlit.app/">https://strong-motion.streamlit.app/</a>. Untuk pengoperasian lokal pada komputer operasional 
        stasiun geofisika yang tidak terhubung jaringan luar, buka terminal PowerShell pada direktori instalasi dan jalankan perintah: 
        <code>streamlit run app.py</code>. Pastikan akselerasi perangkat keras WebGL pada peramban aktif guna mendukung rendering grafik interaktif berkecepatan tinggi.</p>
        
        <p><b>1.2 Bagian DATA INPUT pada Sidebar Kontrol</b></p>
        <p>&bull; <b>Uploader Berkas Akselerogram:</b> Area interaktif untuk memilih atau menyeret (<i>drag-and-drop</i>) berkas rekaman gempa bumi. 
        Format yang didukung mencakup <b>MiniSEED (.mseed, .miniseed, .msd)</b> dan <b>SAC (.sac)</b> standar FDSN. Sistem secara otomatis membaca 
        dan memvalidasi struktur header berkas, mengekstrak identitas stasiun (Network, Station), komponen kanal (Z, N, E), waktu awal kejadian, 
        serta laju cuplik (<i>sampling rate</i>). Perangkat lunak dioptimalkan untuk rekaman dengan laju cuplik 20 Hz hingga 250 Hz.</p>
        <p>&bull; <b>Uploader StationXML (Opsional):</b> Kolom unggah berkas metadata respon sensor (*.xml). Kolom ini hanya dibutuhkan apabila 
        rekaman masih berupa hitungan digital mentah (<i>raw counts</i>) guna mengonversi nilai digital ke satuan fisik percepatan melalui kutub dan nol (<i>poles and zeros</i>).</p>
        <p>&bull; <b>Tombol 'Reset / Clear Data':</b> Terletak di bawah uploader untuk membersihkan memori penampung sesi (<i>session cache</i>), 
        mengosongkan data rekaman sebelumnya, dan mengembalikan seluruh parameter kontrol ke pengaturan baku sebelum memproses sinyal baru.</p>
        """
        c1_cap = "Gambar 1.1: Panel Pengunggahan Berkas Akselerograf, StationXML, dan Tombol Reset Data pada Sidebar"
        c1_sub_html = """
        <p><b>1.3 Rangkuman Format Berkas Input yang Didukung BSMA v2.0.0</b></p>
        <table>
            <tr><th width="22%">Tipe Format</th><th width="28%">Ekstensi Berkas</th><th width="50%">Deskripsi & Ketentuan Teknis</th></tr>
            <tr><td><b>MiniSEED</b></td><td><code>.mseed</code>, <code>.miniseed</code>, <code>.msd</code></td><td>Format baku FDSN/IRIS. Mendukung rekaman triaksial maupun kanal tunggal terkompresi STEIM-1/STEIM-2.</td></tr>
            <tr><td><b>SAC Binary</b></td><td><code>.sac</code></td><td>Format Seismic Analysis Code. Memuat header seismologi lengkap (koordinat stasiun, delta waktu, komponen).</td></tr>
            <tr><td><b>StationXML</b></td><td><code>.xml</code></td><td>Berkas metadata respons instrumen berisi tahapan kalibrasi (stages), sensitivitas, dan transfer function sensor.</td></tr>
        </table>
        """
        c1_callout_title = "Panduan Penyiapan Rekaman Sinyal Lapangan"
        callout_c1 = (
            "Ketentuan Rekaman Seismik: Pastikan rekaman akselerograf memiliki jendela hening (pre-event quiet window) minimal 5 detik "
            "sebelum gelombang P tiba. Jendela ini dibutuhkan mesin Quality Control (QC) otomatis BSMA untuk mengevaluasi lantai derau latar (noise floor) "
            "dan rasio sinyal terhadap derau (SNR). Rekaman tanpa jendela hening yang memadai dapat menurunkan skor kelayakan audit sinyal."
        )
    else:
        c1_text = """
        <p><b>1.1 Software Interface Access Procedures</b></p>
        <p><b>BMKG Strong Motion Analyzer (BSMA v2.0.0)</b> can be accessed directly via modern web browsers 
        (Google Chrome, Microsoft Edge, Mozilla Firefox) without complex installation procedures at the public portal: 
        <a href="https://strong-motion.streamlit.app/">https://strong-motion.streamlit.app/</a>. For offline operation on local operational computers 
        within isolated station environments, launch a terminal in the project directory and execute: 
        <code>streamlit run app.py</code>. Ensure browser WebGL hardware acceleration is enabled to guarantee high-performance interactive chart rendering.</p>
        
        <p><b>1.2 DATA INPUT Controls on the Sidebar Panel</b></p>
        <p>&bull; <b>Accelerogram File Uploader:</b> Interactive widget to select or drag-and-drop seismic record files. 
        Supported formats include standard FDSN <b>MiniSEED (.mseed, .miniseed, .msd)</b> and <b>SAC (.sac)</b> binaries. The system automatically parses 
        and validates file header structures, extracting network/station codes, component channels (Z, N, E), start times, 
        and sampling rates. BSMA is optimized for records sampled between 20 Hz and 250 Hz.</p>
        <p>&bull; <b>StationXML Uploader (Optional):</b> Upload zone for sensor response metadata (*.xml). Required only when raw records 
        are uncalibrated digital counts, enabling precise instrument response deconvolution via poles, zeros, and stage gains.</p>
        <p>&bull; <b>'Reset / Clear Data' Button:</b> Clears the browser session cache, wipes previously loaded data streams, 
        and resets all parameters back to default configurations prior to evaluating new events.</p>
        """
        c1_cap = "Figure 1.1: Accelerograph File Upload Widget, StationXML Uploader, and Reset Data Button on Sidebar"
        c1_sub_html = """
        <p><b>1.3 Supported Input File Formats in BSMA v2.0.0</b></p>
        <table>
            <tr><th width="22%">Format Type</th><th width="28%">File Extension</th><th width="50%">Description & Technical Specifications</th></tr>
            <tr><td><b>MiniSEED</b></td><td><code>.mseed</code>, <code>.miniseed</code>, <code>.msd</code></td><td>Standard FDSN/IRIS format. Supports triaxial or single-component STEIM-1/STEIM-2 data streams.</td></tr>
            <tr><td><b>SAC Binary</b></td><td><code>.sac</code></td><td>Seismic Analysis Code format. Stores full seismological headers (station coordinates, sampling delta).</td></tr>
            <tr><td><b>StationXML</b></td><td><code>.xml</code></td><td>FDSN response metadata containing calibration stages, sensitivity gains, and sensor transfer functions.</td></tr>
        </table>
        """
        c1_callout_title = "Field Seismic Recording Guidelines"
        callout_c1 = (
            "Field Recording Prerequisite: Ensure accelerograph files contain at least 5.0 seconds of pre-event quiet window "
            "prior to initial P-wave onset. This window is mandatory for BSMA's automated Quality Control (QC) engine to establish "
            "the baseline noise floor and compute dynamic Signal-to-Noise Ratios (SNR). Files lacking sufficient quiet windows will trigger QC score penalties."
        )

    builder.insert_html_safe(p1_body, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 215), c1_text, font_size="7.5pt")

    img1_w = 405.0
    img1_x = LEFT_X + (CONTENT_W - img1_w) / 2
    r_img1 = builder.insert_screenshot_card(p1_body, img1_x, y + 225, img1_w,
                                           "Data Input.png", c1_cap)

    builder.insert_html_safe(p1_body, pymupdf.Rect(LEFT_X, r_img1.y1 + 10, RIGHT_X, r_img1.y1 + 105), c1_sub_html)

    builder.draw_callout(p1_body, pymupdf.Rect(LEFT_X, r_img1.y1 + 112, RIGHT_X, r_img1.y1 + 180),
                         c1_callout_title, callout_c1, callout_type="info")

    # =========================================================================
    # HALAMAN 2 (BAB II: Pemilihan Asal Rekaman / Data Provenance)
    # =========================================================================
    p2_body = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p2_body)
    b2_badge = "BAB II - DATA PROVENANCE" if is_id else "CH. II - DATA PROVENANCE"
    b2_pstr = f"Halaman 2 dari {TOTAL_BODY_PAGES}" if is_id else f"Page 2 of {TOTAL_BODY_PAGES}"
    builder.add_page_header_footer(p2_body, b2_badge, b2_pstr)

    b2_banner = "BAB II: PEMILIHAN ASAL-USUL REKAMAN (DATA PROVENANCE)" if is_id else "CHAPTER II: RECORDING ORIGIN DECLARATION (DATA PROVENANCE)"
    y = builder.draw_chapter_banner(p2_body, b2_banner)

    if is_id:
        c2_text = """
        <p><b>2.1 Urgensi Ilmiah Deklarasi Asal-Usul Data (Data Provenance)</b></p>
        <p>Dalam seismologi teknik, data rekaman akselerograf yang didistribusikan dari stasiun lapangan memiliki dua kondisi asal yang sangat berbeda: 
        data yang telah dikalibrasi langsung menjadi besaran percepatan fisik oleh datalogger stasiun (seperti instrumen Nanometrics Titan, Guralp, 
        atau Episensor), dan data mentah yang masih berdimensi hitungan digital (<i>counts</i>). Deklarasi asal data (<i>Data Provenance</i>) pada BSMA v2.0.0 
        merupakan langkah krusial untuk mencegah terjadinya <b>dekonvolusi instrumen ganda</b>. Menerapkan dekonvolusi respon sensor pada data yang telah 
        berdimensi fisik akan memicu distorsi frekuensi tinggi yang sangat parah (<i>high-frequency over-correction</i>), melipatgandakan derau instrumen, 
        dan merusak bentuk gelombang asli serta menghasilkan nilai estimasi PGA palsu yang membahayakan analisis keselamatan rekayasa.</p>

        <p><b>2.2 Rincian Pilihan Status Rekaman pada Menu PROJECT</b></p>
        <p>&bull; <b>'Already processed physical acceleration':</b> Pilih opsi ini jika berkas rekaman telah dikalibrasi ke satuan percepatan fisik 
        (m/s&sup2;, Gal, atau cm/s&sup2;). Sistem secara otomatis mengaktifkan <i>calibration bypass protocol</i>, melewati tahap dekonvolusi, dan 
        langsung mengalirkan sinyal ke filter lolos pita serta koreksi garis dasar (<i>baseline correction</i>).</p>
        <p>&bull; <b>'Raw instrument counts with StationXML':</b> Pilih opsi ini jika data masih dalam hitungan digital mentah. Sistem akan membaca 
        fungsi transfer instrumen dari berkas StationXML yang diunggah dan menerapkan dekonvolusi respons di domain frekuensi dengan <i>water-level stabilizer</i>.</p>
        <p>&bull; <b>'Unknown (Require Scientific Review)':</b> Digunakan saat operator tidak yakin mengenai status kalibrasi rekaman. Sistem akan 
        melakukan pengujian statistik heuristik terhadap amplitudo puncak guna mendeteksi dimensi sinyal secara otomatis.</p>
        """
        c2_cap_a = "Gambar 2.1: Panel Menu Pilihan Data Provenance"
        c2_cap_b = "Gambar 2.2: Dropdown Pilihan Status Asal Rekaman"
        c2_sub_html = """
        <p><b>2.3 Matriks Perbandingan Status Rekaman & Tindakan Sistem BSMA</b></p>
        <table>
            <tr><th width="25%">Status Pilihan</th><th width="25%">Ciri Amplitudo Data</th><th width="25%">Perlakuan Sistem</th><th width="25%">Resiko Kesalahan</th></tr>
            <tr><td><b>Physical Acceleration</b><br>(Rekomendasi Baku)</td><td>Amplitudo bernilai riil wajar (0.001 - 500 Gal / m/s²)</td><td>Bypass dekonvolusi; langsung ke filter lolos pita & baseline</td><td>Jika salah pilih, data mentah counts akan menghasilkan PGA 10⁶ Gal</td></tr>
            <tr><td><b>Raw Counts + XML</b></td><td>Amplitudo berupa angka bulat besar (10⁴ - 10⁷ counts)</td><td>Membaca StationXML; membagi respon sensor di domain frekuensi</td><td>Jika diterapkan pada data fisik, memicu over-correction parah</td></tr>
            <tr><td><b>Unknown Review</b></td><td>Tidak teridentifikasi pasti</td><td>Pengujian statistik puncak; memberi rekomendasi otomatis</td><td>Membutuhkan konfirmasi visual operator terhadap bentuk gelombang</td></tr>
        </table>
        """
        c2_callout_title = "Prosedur Baku Kalibrasi Sinyal Seismik BMKG"
        callout_c2 = (
            "Pedoman Operasional BMKG: Mayoritas berkas akselerograf BMKG yang didistribusikan pascakejadian gempa bumi (format MiniSEED dari repository "
            "operasional) telah dikonversi langsung oleh datalogger stasiun menjadi percepatan fisik. Oleh karena itu, operator sangat disarankan memilih "
            "opsi 'Already processed physical acceleration' guna menjamin integritas komputasi kinematika tanah dan spektrum respons elastis."
        )
    else:
        c2_text = """
        <p><b>2.1 Scientific Importance of Data Provenance Declarations</b></p>
        <p>In earthquake engineering seismology, accelerograph data distributed from field stations typically exhibit two fundamentally different states: 
        records pre-calibrated into physical acceleration by field dataloggers (e.g., Nanometrics Titan, Guralp, or Kinemetrics Episensor), 
        and raw records still scaled in digital telemetry units (<i>counts</i>). Declaring <i>Data Provenance</i> in BSMA v2.0.0 is critical 
        to prevent <b>accidental double deconvolution</b>. Applying instrument deconvolution to pre-calibrated physical acceleration signals causes catastrophic 
        high-frequency over-correction, amplifying ambient electronic noise, distorting phase arrivals, and producing false PGA values that invalidate safety analyses.</p>

        <p><b>2.2 Recording Status Options in the PROJECT Sidebar</b></p>
        <p>&bull; <b>'Already processed physical acceleration':</b> Select this option when files are pre-calibrated to physical units (m/s&sup2;, Gal, cm/s&sup2;). 
        The system engages the <i>calibration bypass protocol</i>, skipping instrument deconvolution and routing signals directly to zero-phase bandpass filtering and detrending.</p>
        <p>&bull; <b>'Raw instrument counts with StationXML':</b> Select this mode when recordings consist of raw integer counts. 
        BSMA ingests the StationXML transfer function and applies frequency-domain response deconvolution stabilized with a water-level regularizer.</p>
        <p>&bull; <b>'Unknown (Require Scientific Review)':</b> Selected when the operator is uncertain of the recording's provenance. 
        Heuristic statistical checks are executed against peak amplitudes to recommend appropriate calibration pathways.</p>
        """
        c2_cap_a = "Figure 2.1: Data Provenance Configuration Menu"
        c2_cap_b = "Figure 2.2: Recording Provenance Dropdown Selector"
        c2_sub_html = """
        <p><b>2.3 Recording Provenance Matrix & BSMA Processing Pipelines</b></p>
        <table>
            <tr><th width="25%">Provenance Mode</th><th width="25%">Typical Amplitude Range</th><th width="25%">Engine Pipeline Action</th><th width="25%">Risk of Misconfiguration</th></tr>
            <tr><td><b>Physical Acceleration</b><br>(Default Recommendation)</td><td>Physical floating-point values (0.001 - 500 Gal or m/s²)</td><td>Bypasses deconvolution; direct bandpass & baseline detrending</td><td>Selecting for raw counts results in severe PGA underestimation</td></tr>
            <tr><td><b>Raw Counts + XML</b></td><td>Large integer digital values (10⁴ - 10⁷ digital counts)</td><td>Parses StationXML; applies frequency-domain deconvolution</td><td>Applying to pre-calibrated signals causes catastrophic over-correction</td></tr>
            <tr><td><b>Unknown Review</b></td><td>Indeterminate provenance</td><td>Executes peak amplitude heuristics; flags review warning</td><td>Requires visual inspection before confirming baseline integration</td></tr>
        </table>
        """
        c2_callout_title = "Standard BMKG Signal Calibration Protocol"
        callout_c2 = (
            "BMKG Operational Guideline: The vast majority of accelerograph MiniSEED records archived and distributed by BMKG post-earthquake "
            "are converted to physical acceleration directly by station field dataloggers. Operators should select 'Already processed physical acceleration' "
            "to guarantee kinematic integration stability and response spectra fidelity."
        )

    builder.insert_html_safe(p2_body, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 215), c2_text, font_size="7.5pt")

    col_w = (CONTENT_W - 14.0) / 2
    r_img2a = builder.insert_screenshot_card(p2_body, LEFT_X, y + 225, col_w,
                                            "Data Provenance.png", c2_cap_a, forced_h=98.0)
    r_img2b = builder.insert_screenshot_card(p2_body, LEFT_X + col_w + 14.0, y + 225, col_w,
                                            "Data Provenance Pilihan.png", c2_cap_b, forced_h=98.0)

    builder.insert_html_safe(p2_body, pymupdf.Rect(LEFT_X, r_img2a.y1 + 10, RIGHT_X, r_img2a.y1 + 115), c2_sub_html)

    builder.draw_callout(p2_body, pymupdf.Rect(LEFT_X, r_img2a.y1 + 122, RIGHT_X, r_img2a.y1 + 192),
                         c2_callout_title, callout_c2, callout_type="success")

    # =========================================================================
    # HALAMAN 3 (BAB III: Parameter DSP / Processing & Unit)
    # =========================================================================
    p3_body = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p3_body)
    b3_badge = "BAB III - PARAMETER DSP & SATUAN" if is_id else "CH. III - DSP & UNITS"
    b3_pstr = f"Halaman 3 dari {TOTAL_BODY_PAGES}" if is_id else f"Page 3 of {TOTAL_BODY_PAGES}"
    builder.add_page_header_footer(p3_body, b3_badge, b3_pstr)

    b3_banner = "BAB III: KONFIGURASI PARAMETER DSP (PROCESSING & UNIT)" if is_id else "CHAPTER III: DSP PARAMETERS CONFIGURATION (PROCESSING & UNIT)"
    y = builder.draw_chapter_banner(p3_body, b3_banner)

    if is_id:
        c3_text = """
        <p><b>3.1 Penapisan Sinyal Digital (Digital Filtering) & Pengolahan Sinyal</b></p>
        <p>Penapisan digital bertujuan mengisolasi pita frekuensi getaran tanah kuat yang relevan dengan respons struktur bangunan serta 
        mengeliminasi derau instrumen dan fluktuasi garis dasar non-seismik. BSMA v2.0.0 menyediakan konfigurasi DSP terstandarisasi:</p>
        <p>&bull; <b>Pilihan Tipe Filter:</b> Tersedia filter <b>Bandpass</b> (lolos pita, standar baku gerakan kuat), <b>Lowpass</b> (lolos rendah), 
        dan <b>Highpass</b> (lolos tinggi). Filter diterapkan secara dua-arah (<i>zero-phase forward-backward filtering</i>) berbasis IIR Butterworth 
        orde-4 dalam format <i>Second-Order Sections</i> (SOS). Metode dua-arah ini menjamin bahwa fase gelombang tidak mengalami pergeseran waktu (&Delta;&phi; = 0), 
        sehingga waktu tiba gelombang P dan fase geser S tetap presisi sesuai rekaman aslinya.</p>
        <p>&bull; <b>Frekuensi Sudut (Corner Frequencies):</b> Slider interaktif untuk mengatur frekuensi potong batas bawah <b>f<sub>min</sub></b> 
        (default: 0.05 Hz) guna menekan derau frekuensi rendah pemicu hanyutan integrasi (<i>baseline drift</i>), dan batas atas <b>f<sub>max</sub></b> 
        (default: 40.0 Hz) guna menyingkirkan derau listrik dan derau mekanis frekuensi tinggi.</p>
        <p>&bull; <b>Tombol Pintas 'Preset Baku Strong-Motion (0.05 - 40.0 Hz)':</b> Tombol satu-klik untuk langsung memulihkan parameter filter ke 
        standar rekomendasi operasional BMKG dan USGS untuk pemrosesan gempa kuat.</p>
        <p>&bull; <b>Detrending & Tapering:</b> Opsi <i>Polynomial Detrending</i> untuk menghilangkan offset garis dasar dan tren termal linier/kuadratik, 
        serta <i>Cosine Tapering</i> 5% jendela Tukey pada ujung sinyal guna mencegah kebocoran spektral (<i>spectral leakage</i>).</p>
        <p><b>3.2 Pemilihan Satuan Fisik Luaran (Physical Unit Selector)</b></p>
        <p>Pengguna dapat memilih satuan percepatan sesuai kebutuhan analisis: <b>m/s&sup2;</b> (satuan internasional SI), <b>Gal</b> (1 Gal = 1 cm/s&sup2; = 0.01 m/s&sup2;, 
        satuan baku operasional BMKG), atau <b>cm/s&sup2;</b>.</p>
        """
        c3_cap_a = "Gambar 3.1: Panel Pengaturan Processing Sinyal"
        c3_cap_b = "Gambar 3.2: Dropdown Pilihan Jenis Filter"
        c3_cap_c = "Gambar 3.3: Dropdown Pilihan Satuan Fisik"
        c3_callout_title = "Perlindungan Batas Frekuensi Shannon-Nyquist"
        callout_c3 = (
            "Proteksi Batas Shannon-Nyquist: Sistem secara cerdas membatasi nilai fmax maksimal sebesar 0.40 x laju cuplik (fs). "
            "Batas proteksi ini menjamin bahwa seluruh proses penapisan digital beroperasi aman di bawah ambang teoritis Nyquist (0.50 fs), "
            "sehingga mencegah timbulnya distorsi aliasing frekuensi yang dapat merusak keabsahan data rekayasa."
        )
    else:
        c3_text = """
        <p><b>3.1 Digital Signal Filtering & Conditioning</b></p>
        <p>Digital filtering isolates the engineering-relevant frequency bandwidth of strong ground motions while eliminating 
        instrument noise, thermal drifts, and non-seismic baseline perturbations. BSMA v2.0.0 incorporates standardized DSP workflows:</p>
        <p>&bull; <b>Filter Type Options:</b> Offers <b>Bandpass</b> (standard for strong motion), <b>Lowpass</b>, and <b>Highpass</b> configurations. 
        Filters are applied using a two-pass <i>zero-phase forward-backward scheme</i> based on a 4th-order IIR Butterworth topology in 
        <i>Second-Order Sections</i> (SOS) formulation. This ensures zero phase distortion (&Delta;&phi; = 0), preserving exact P-wave arrivals and S-wave kinematics.</p>
        <p>&bull; <b>Corner Frequencies:</b> Interactive sliders configure the lower corner <b>f<sub>min</sub></b> (default: 0.05 Hz) 
        to suppress low-frequency drift during double integration, and the upper corner <b>f<sub>max</sub></b> (default: 40.0 Hz) 
        to eliminate high-frequency cultural and electrical sensor noise.</p>
        <p>&bull; <b>'Standard Strong-Motion Preset (0.05 - 40.0 Hz)' Shortcut:</b> A single click restores optimal BMKG/USGS 
        operational corner frequency recommendations.</p>
        <p>&bull; <b>Detrending & Tapering:</b> <i>Polynomial Detrending</i> removes static offsets and thermal polynomial drift, while a 
        5% <i>Tukey Cosine Taper</i> suppresses end-window spectral leakage.</p>
        <p><b>3.2 Physical Engineering Unit Selector</b></p>
        <p>Operators can toggle output units: <b>m/s&sup2;</b> (SI standard), <b>Gal</b> (1 Gal = 1 cm/s&sup2; = 0.01 m/s&sup2;, BMKG operational standard), 
        or <b>cm/s&sup2;</b>.</p>
        """
        c3_cap_a = "Figure 3.1: Signal Processing Configuration Sidebar"
        c3_cap_b = "Figure 3.2: Digital Filter Type Dropdown"
        c3_cap_c = "Figure 3.3: Engineering Unit Dropdown"
        c3_callout_title = "Shannon-Nyquist Frequency Safeguard"
        callout_c3 = (
            "Shannon-Nyquist Safeguard: BSMA automatically restricts fmax to a maximum of 0.40 x sampling frequency (fs). "
            "This conservative ceiling ensures digital filtering operates safely below the theoretical Nyquist limit (0.50 fs), "
            "preventing high-frequency aliasing and filter instability from corrupting engineering data."
        )

    builder.insert_html_safe(p3_body, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 235), c3_text, font_size="7.5pt")

    col_left_w = 230.0
    col_right_w = CONTENT_W - col_left_w - 14.0
    right_x = LEFT_X + col_left_w + 14.0

    r_img3a = builder.insert_screenshot_card(p3_body, LEFT_X, y + 245, col_left_w,
                                            "Processing.png", c3_cap_a, forced_h=230.0)
    r_img3b = builder.insert_screenshot_card(p3_body, right_x, y + 245, col_right_w,
                                            "Filter Pilihan.png", c3_cap_b, forced_h=98.0)
    r_img3c = builder.insert_screenshot_card(p3_body, right_x, r_img3b.y1 + 10, col_right_w,
                                            "Unit Pilihan.png", c3_cap_c, forced_h=100.0)

    max_img_y = max(r_img3a.y1, r_img3c.y1)
    builder.draw_callout(p3_body, pymupdf.Rect(LEFT_X, max_img_y + 12, RIGHT_X, max_img_y + 82),
                         c3_callout_title, callout_c3, callout_type="info")

    # =========================================================================
    # HALAMAN 4 (BAB IV: Parameter Struktur, Event Gempa, & Benchmark)
    # =========================================================================
    p4_body = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p4_body)
    b4_badge = "BAB IV - STRUKTUR & METADATA EVENT" if is_id else "CH. IV - SDOF & EVENT DATA"
    b4_pstr = f"Halaman 4 dari {TOTAL_BODY_PAGES}" if is_id else f"Page 4 of {TOTAL_BODY_PAGES}"
    builder.add_page_header_footer(p4_body, b4_badge, b4_pstr)

    b4_banner = "BAB IV: PARAMETER STRUKTUR, METADATA EVENT, & BENCHMARK" if is_id else "CHAPTER IV: STRUCTURAL PARAMETERS, EVENT METADATA, & BENCHMARK"
    y = builder.draw_chapter_banner(p4_body, b4_banner)

    if is_id:
        c4_text = """
        <p><b>4.1 Bagian ANALYSIS (Dinamika Struktur SDOF & Pilihan Solver Numerik)</b></p>
        <p>Pemodelan osilator berderajat kebebasan tunggal (<i>Single-Degree-of-Freedom</i> / SDOF) merupakan fondasi utama dalam perancangan struktur 
        tahan gempa. BSMA v2.0.0 menyediakan kontrol parameter dinamis yang fleksibel:</p>
        <p>&bull; <b>Rasio Redaman Kritis (&xi;):</b> Slider rasio redaman viskos struktur dengan rentang 0.01 hingga 0.20. Nilai standar baku adalah 
        <b>0.05 (5%)</b>, yang merupakan acuan normatif perancangan struktur beton bertulang dan baja sesuai standar SNI 1726:2019 dan ASCE 7-22.</p>
        <p>&bull; <b>Algoritma Solver SDOF:</b> Pengguna dapat memilih solver komputasi: <b>Nigam-Jennings (1969)</b> (metode analitik eksak matriks transisi 
        dengan eksitasi sepotong-sepotong, bebas dispersi numerik dan sangat akurat pada periode pendek T &lt; 0.1 s) atau <b>Newmark-Beta (1959)</b> 
        (metode integrasi waktu implisit percepatan rata-rata konstan &gamma;=0.5, &beta;=0.25 tanpa redaman numerik artifisial).</p>
        <p><b>4.2 Bagian OUTPUT & BENCHMARK (Metadata Kejadian Gempa & Validasi Silang)</b></p>
        <p>&bull; <b>Formulir Event Metadata:</b> Menyediakan kolom isian parameter kegempaan: Origin Time (UTC), Magnitudo (Mw), Kedalaman Hiposenter (km), 
        Koordinat Episenter (Lintang/Bujur), dan Jarak Episentral (km). Parameter ini otomatis disuntikkan ke dalam kop resmi laporan teknik PDF.</p>
        <p>&bull; <b>Ground Truth Benchmark:</b> Fitur pengunggahan berkas CSV spektrum acuan eksternal untuk validasi silang akurasi solver numerik.</p>
        """
        c4_cap_a = "Gambar 4.1: Panel Kontrol Dynamic Analysis"
        c4_cap_b = "Gambar 4.2: Pilihan Solver SDOF Dinamika Struktur"
        c4_cap_c = "Gambar 4.3: Formulir Metadata Event Gempa"
        c4_cap_d = "Gambar 4.4: Panel Unggah Kurva Benchmark"
        c4_callout_title = "Kelengkapan Metadata Laporan"
        callout_c4 = (
            "Kelengkapan Laporan Resmi: Selalu isi parameter kejadian gempa pada 'Output Event Data' sebelum mengunduh laporan PDF di Tab 7. "
            "Informasi magnitudo, kedalaman, dan jarak episentral sangat vital bagi rekayasawan struktur dalam menilai dampak atenuasi getaran."
        )
    else:
        c4_text = """
        <p><b>4.1 ANALYSIS Section (SDOF Structural Dynamics & Numerical Solvers)</b></p>
        <p>Single-Degree-of-Freedom (SDOF) elastic oscillator modeling forms the foundation of modern structural earthquake design. 
        BSMA v2.0.0 delivers flexible and rigorous dynamic controls:</p>
        <p>&bull; <b>Critical Damping Ratio (&xi;):</b> Configures structural viscous damping from 0.01 to 0.20. The default value is 
        <b>0.05 (5%)</b>, matching normative code provisions for reinforced concrete and structural steel under SNI 1726:2019 and ASCE 7-22.</p>
        <p>&bull; <b>SDOF Solver Algorithms:</b> Operators can choose between: <b>Nigam-Jennings (1969)</b> (exact piece-wise analytical state transition 
        matrix, immune to numerical dispersion and exceptionally accurate at short periods T &lt; 0.1 s) or <b>Newmark-Beta (1959)</b> 
        (constant average acceleration implicit step-by-step integration with &gamma;=0.5, &beta;=0.25, introducing zero artificial numerical damping).</p>
        <p><b>4.2 OUTPUT & BENCHMARK Sections (Event Metadata & Cross-Validation)</b></p>
        <p>&bull; <b>Event Metadata Form:</b> Interactive fields for Origin Time (UTC), Magnitude (Mw), Hypocentral Depth (km), Epicentral Coordinates, 
        and Epicentral Distance (km). These parameters populate the official header of the exportable 2-page PDF engineering report.</p>
        <p>&bull; <b>Ground Truth Benchmark:</b> Upload portal for external reference spectral CSVs, facilitating instant numerical cross-validation.</p>
        """
        c4_cap_a = "Figure 4.1: Dynamic Analysis Control Sidebar"
        c4_cap_b = "Figure 4.2: SDOF Structural Dynamics Solver Dropdown"
        c4_cap_c = "Figure 4.3: Earthquake Event Metadata Form"
        c4_cap_d = "Figure 4.4: Ground Truth Benchmark Upload Widget"
        c4_callout_title = "Report Metadata Completeness"
        callout_c4 = (
            "Engineering Report Completeness: Always complete the earthquake event parameters in 'Output Event Data' prior to downloading "
            "the official PDF report in Tab 7. Magnitude, depth, and epicentral distance provide essential seismological context "
            "for structural engineers evaluating ground motion attenuation and seismic demand."
        )

    builder.insert_html_safe(p4_body, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 210), c4_text, font_size="7.5pt")

    col_w = (CONTENT_W - 14.0) / 2
    r_img4a = builder.insert_screenshot_card(p4_body, LEFT_X, y + 220, col_w,
                                            "Analysis.png", c4_cap_a, forced_h=95.0)
    r_img4b = builder.insert_screenshot_card(p4_body, LEFT_X + col_w + 14.0, y + 220, col_w,
                                            "SDOF Solver Pilihan.png", c4_cap_b, forced_h=95.0)

    y_row2 = r_img4a.y1 + 10.0
    r_img4c = builder.insert_screenshot_card(p4_body, LEFT_X, y_row2, col_w,
                                            "Output Event Data.png", c4_cap_c, forced_h=190.0)
    r_img4d = builder.insert_screenshot_card(p4_body, LEFT_X + col_w + 14.0, y_row2, col_w,
                                            "Benchmark.png", c4_cap_d, forced_h=95.0)

    builder.draw_callout(p4_body, pymupdf.Rect(LEFT_X + col_w + 14.0, r_img4d.y1 + 10.0, RIGHT_X, r_img4c.y1),
                         c4_callout_title, callout_c4, callout_type="info")

    # =========================================================================
    # HALAMAN 5 (BAB V: Eksekusi Pipeline Run Analysis & Tab 1 Summary)
    # =========================================================================
    p5_body = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p5_body)
    b5_badge = "BAB V - EKSEKUSI & SUMMARY" if is_id else "CH. V - PIPELINE & SUMMARY"
    b5_pstr = f"Halaman 5 dari {TOTAL_BODY_PAGES}" if is_id else f"Page 5 of {TOTAL_BODY_PAGES}"
    builder.add_page_header_footer(p5_body, b5_badge, b5_pstr)

    b5_banner = "BAB V: EKSEKUSI PIPELINE (RUN ANALYSIS) & TAB 1 SUMMARY" if is_id else "CHAPTER V: PIPELINE EXECUTION (RUN ANALYSIS) & TAB 1 SUMMARY"
    y = builder.draw_chapter_banner(p5_body, b5_banner)

    if is_id:
        c5_text1 = """
        <p><b>5.1 Pemilihan Mode Alur Kerja & Eksekusi Tombol 'Run Analysis'</b></p>
        <p>Setelah seluruh parameter pada bilah sisi (<i>sidebar</i>) selesai dikonfigurasi, operator beralih ke kanvas kerja utama. 
        BSMA v2.0.0 menyediakan tiga mode kerja utama pada bilah <b>Workflow Mode</b>: <i>Single-station review</i>, <i>Multi-station processing</i>, dan <i>Export results</i>:</p>
        <p>&bull; <b>Aktivasi Mode Single-Station:</b> Pastikan opsi <b>'Single-station review'</b> terpilih untuk melakukan analisis mendalam satu stasiun.</p>
        <p>&bull; <b>Menu Tarik-Turun 'Select Recording Window':</b> Pilih berkas atau jendela waktu stasiun yang hendak dianalisis (misal: <code>IA.PPJR | 2026-02-05 18:04:27 UTC</code>).</p>
        <p>&bull; <b>Teks Status Kalibrasi:</b> Sistem menampilkan konfirmasi respons instrumen, misalnya <i>Physical Acceleration Mode (StationXML correction bypassed)</i>.</p>
        <p>&bull; <b>Tombol Eksekusi '[Run Analysis]':</b> Klik tombol biru utama <b>'Run Analysis'</b> untuk mengeksekusi pipeline komputasi terpadu. Sistem akan menampilkan animasi indikator putar <code>Executing pipeline for station IA.PPJR...</code>. Tombol diproteksi (berstatus <i>disabled</i>) apabila asal data masih 'Unknown' atau berkas hitungan mentah (counts) belum disertai StationXML.</p>
        """
        c5_cap_a = "Gambar 5.1: Panel Workflow Mode, Dropdown Jendela Rekaman Stasiun, dan Tombol Eksekusi Utama 'Run Analysis'"
        c5_text2 = """
        <p><b>5.2 Lencana 'ANALYSIS COMPLETE' & Dasbor Eksekutif Tab SUMMARY</b></p>
        <p>Segera setelah pipeline komputasi selesai, lencana hijau <b>ANALYSIS COMPLETE</b> akan menyala di pojok kanan atas, dan seluruh 7 tab analisis interaktif terbuka secara simultan. 
        Tab <b>SUMMARY</b> merangkum identitas rekaman stasiun (laju cuplik, durasi, komponen), triad parameter kinematika puncak (<b>PGA</b> [Gal], <b>PGV</b> [cm/s], <b>PGD</b> [cm]), skor mutu sinyal QC, serta estimasi intensitas instrumental MMI.</p>
        """
        c5_cap_b = "Gambar 5.2: Tampilan Baris Navigasi 7 Tab Analisis Interaktif dan Header Stasiun Berstatus ANALYSIS COMPLETE"
        c5_cap_c = "Gambar 5.3: Kartu Parameter Kinematika Puncak (PGA, PGV, PGD), Riwayat Provenance, dan Audit Mutu QC"
        c5_callout_title = "Prosedur Eksekusi & Tanggap Darurat Cepat"
        callout_c5 = (
            "Prosedur Verifikasi Cepat Pascaguncangan: Setelah menekan tombol 'Run Analysis' dan lencana ANALYSIS COMPLETE muncul, "
            "segera periksa triad nilai puncak PGA, PGV, dan badge MMI pada Tab Summary. Parameter ini menjadi indikator cepat bagi operator BMKG "
            "dalam menentukan tingkat keparahan guncangan gempa bumi di lokasi stasiun sebelum meninjau detail bentuk gelombang pada tab berikutnya."
        )
    else:
        c5_text1 = """
        <p><b>5.1 Workflow Mode Selection & 'Run Analysis' Button Execution</b></p>
        <p>Once sidebar parameters are configured, operators engage the primary workstation area. 
        BSMA v2.0.0 features three main operational modes on the <b>Workflow Mode</b> selector: <i>Single-station review</i>, <i>Multi-station processing</i>, and <i>Export results</i>:</p>
        <p>&bull; <b>Single-Station Mode Activation:</b> Ensure <b>'Single-station review'</b> is active for comprehensive station-level signal processing.</p>
        <p>&bull; <b>'Select Recording Window' Dropdown:</b> Choose the target station stream (e.g., <code>IA.PPJR | 2026-02-05 18:04:27 UTC</code>).</p>
        <p>&bull; <b>Calibration Status Feedback:</b> Displays explicit confirmation, e.g., <i>Physical Acceleration Mode (StationXML correction bypassed)</i>.</p>
        <p>&bull; <b>'[Run Analysis]' Execution Button:</b> Click the primary blue <b>'Run Analysis'</b> button to trigger the end-to-end processing pipeline. 
        A dynamic spinner confirms <code>Executing pipeline for station IA.PPJR...</code>. The button is automatically disabled if provenance remains 'Unknown' or raw counts lack an accompanying StationXML file.</p>
        """
        c5_cap_a = "Figure 5.1: Workflow Mode Selector, Station Stream Dropdown, and Primary 'Run Analysis' Execution Button"
        c5_text2 = """
        <p><b>5.2 'ANALYSIS COMPLETE' Status Badge & Tab 1 SUMMARY Executive Dashboard</b></p>
        <p>Upon pipeline completion, a green <b>ANALYSIS COMPLETE</b> badge illuminates in the header, unlocking all 7 interactive tabs simultaneously. 
        The <b>SUMMARY</b> tab consolidates station stream metadata (sampling rate, duration, components), the peak kinematic triad (<b>PGA</b> [Gal], <b>PGV</b> [cm/s], <b>PGD</b> [cm]), QC health audit scores, and the instrumental MMI rating.</p>
        """
        c5_cap_b = "Figure 5.2: 7 Interactive Analysis Tabs Navigation Bar and Station Header with ANALYSIS COMPLETE Badge"
        c5_cap_c = "Figure 5.3: Peak Kinematic Parameter Cards (PGA, PGV, PGD), Provenance History, and QC Health Audit"
        c5_callout_title = "Rapid Post-Earthquake Verification Protocol"
        callout_c5 = (
            "Rapid Post-Earthquake Verification: After triggering 'Run Analysis' and confirming the green badge, "
            "immediately review the PGA, PGV, and MMI cards on Tab 1. These metrics provide BMKG duty officers with instant, "
            "objective situational awareness of ground shaking severity at the station site prior to detailed waveform inspection."
        )

    builder.insert_html_safe(p5_body, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 106), c5_text1, font_size="7.4pt", line_height="1.28")

    r_img5a = builder.insert_screenshot_card(p5_body, LEFT_X, y + 110, CONTENT_W,
                                            "Single Station Review and Run Analysis.png", c5_cap_a, forced_h=105.0)

    builder.insert_html_safe(p5_body, pymupdf.Rect(LEFT_X, r_img5a.y1 + 8, RIGHT_X, r_img5a.y1 + 55), c5_text2, font_size="7.4pt", line_height="1.28")

    r_img5b = builder.insert_screenshot_card(p5_body, LEFT_X, r_img5a.y1 + 58, CONTENT_W,
                                            "Tab Summary .png", c5_cap_b, forced_h=75.0)
    r_img5c = builder.insert_screenshot_card(p5_body, LEFT_X, r_img5b.y1 + 8, CONTENT_W,
                                            "Isi Tab Summary.png", c5_cap_c, forced_h=130.0)

    builder.draw_callout(p5_body, pymupdf.Rect(LEFT_X, r_img5c.y1 + 8, RIGHT_X, r_img5c.y1 + 65),
                         c5_callout_title, callout_c5, callout_type="success")

    # =========================================================================
    # HALAMAN 6 (BAB VI: Tab 2 Waveforms)
    # =========================================================================
    p6_body = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p6_body)
    b6_badge = "BAB VI - TAB WAVEFORMS" if is_id else "CH. VI - TAB WAVEFORMS"
    b6_pstr = f"Halaman 6 dari {TOTAL_BODY_PAGES}" if is_id else f"Page 6 of {TOTAL_BODY_PAGES}"
    builder.add_page_header_footer(p6_body, b6_badge, b6_pstr)

    b6_banner = "BAB VI: TAB 2 - RIWAYAT WAKTU KINEMATIKA LENGKAP (WAVEFORMS)" if is_id else "CHAPTER VI: TAB 2 - FULL KINEMATIC TIME HISTORIES (WAVEFORMS)"
    y = builder.draw_chapter_banner(p6_body, b6_banner)

    if is_id:
        c6_text = """
        <p><b>6.1 Visualisasi Riwayat Waktu Kinematika Triaksial Sinkron</b></p>
        <p>Tab Waveforms menyajikan deret waktu lengkap getaran tanah hasil integrasi ganda kinematika secara interaktif beresolusi tinggi:</p>
        <p>&bull; <b>Tiga Baris Kinematika Sinkron:</b> Menampilkan grafik deret waktu <b>Percepatan a(t)</b> [Gal], <b>Kecepatan v(t)</b> [cm/s], 
        dan <b>Perpindahan d(t)</b> [cm] yang disusun sejajar pada satu sumbu waktu horizontal yang sama (<i>shared time axis</i>). Susunan ini 
        memungkinkan pengamat membandingkan fase getaran secara langsung antara percepatan, kecepatan, dan perpindahan tanah.</p>
        <p>&bull; <b>Selektor Komponen Interaktif:</b> Pengguna dapat memilih untuk memeriksa komponen tertentu (Timur-Barat HNE, Utara-Selatan HNN, 
        atau Vertikal HNZ) atau menampilkan ketiga komponen secara serentak untuk evaluasi gerakan tanah tiga dimensi.</p>
        <p><b>6.2 Fitur Interaktivitas WebGL Plotly & Anotasi Puncak</b></p>
        <p>&bull; <b>Eksplorasi Fase Gelombang:</b> Pengguna dapat memperbesar tampilan (<i>box zoom</i>) pada fase kedatangan gelombang primer (P-wave) 
        dan gelombang sekunder (S-wave), menggeser rekaman (<i>pan</i>), serta membaca nilai amplitudo seketika menggunakan kursor penunjuk (<i>hover</i>).</p>
        <p>&bull; <b>Anotasi Nilai Puncak Otomatis:</b> Penanda visual vertikal putus-putus dan teks anotasi menunjukkan waktu terjadinya PGA, PGV, dan PGD.</p>
        """
        c6_cap = "Gambar 6.1: Visualisasi Riwayat Waktu Kinematika Percepatan, Kecepatan, dan Perpindahan Sinkron"
        c6_sub_html = """
        <p><b>6.3 Makna Fisik Kinematika Tanah dalam Rekayasa Struktur & Prosedur Baseline Drift</b></p>
        <p>&bull; <b>Percepatan Tanah a(t):</b> Berbanding lurus dengan gaya inersia sesaat (F = m&middot;a) yang bekerja pada massa struktur gedung kaku.</p>
        <p>&bull; <b>Kecepatan Tanah v(t):</b> Merefleksikan energi kinetik seismik dan deformasi inersia pada struktur fleksibel dan fasilitas tertanam (pipa/terowongan).</p>
        <p>&bull; <b>Perpindahan Tanah d(t):</b> Menggambarkan pergeseran relatif antarlantai (<i>drift ratio</i>) dan regangan plastis tanah fondasi.</p>
        <p>&bull; <b>Investigasi Drift Integrasi:</b> Jika kurva perpindahan di ekor sinyal melayang menjauhi nol, naikkan f<sub>min</sub> pada sidebar (misal dari 0.05 Hz ke 0.10 Hz).</p>
        """
        c6_callout_title = "Pemeriksaan Kestabilan Riwayat Perpindahan Tanah"
        callout_c6 = (
            "Diagnostik Kestabilan Garis Dasar (Baseline): Perhatikan kurva riwayat perpindahan (Displacement) pada bagian akhir rekaman. "
            "Bila kurva perpindahan menunjukkan hanyutan linier atau kuadratik menjauhi nol, naikkan batas fmin pada sidebar (misal dari 0.05 Hz ke 0.10 Hz) "
            "dan pastikan Polynomial Detrending aktif untuk menekan sisa derau frekuensi rendah pemicu drift integrasi."
        )
    else:
        c6_text = """
        <p><b>6.1 Synchronous Triaxial Kinematic Time Histories</b></p>
        <p>The Waveforms tab provides high-resolution interactive time series resulting from dual kinematic numerical integration:</p>
        <p>&bull; <b>Three-Tier Synchronized Kinematic Rows:</b> Plots <b>Acceleration a(t)</b> [Gal], <b>Velocity v(t)</b> [cm/s], 
        and <b>Displacement d(t)</b> [cm] stacked along a shared horizontal time axis. This layout facilitates direct physical correlation 
        between high-frequency acceleration pulses, intermediate velocity energy bursts, and long-period displacement offsets.</p>
        <p>&bull; <b>Interactive Component Channel Selector:</b> Allows isolating individual orthogonal channels (East-West HNE, North-South HNN, 
        or Vertical HNZ) or displaying all three components simultaneously for comprehensive 3D motion characterization.</p>
        <p><b>6.2 Plotly WebGL Interactivity & Automated Peak Annotations</b></p>
        <p>&bull; <b>Wave Phase Inspection:</b> Supports seamless box zoom on initial P-wave and high-energy S-wave arrivals, horizontal panning, 
        and precision cursor hover tooltips displaying instantaneous amplitudes and timestamps.</p>
        <p>&bull; <b>Automated Peak Annotations:</b> Prominent vertical dashed lines and dynamic text tags mark the exact occurrence of PGA, PGV, and PGD.</p>
        """
        c6_cap = "Figure 6.1: Synchronized Kinematic Acceleration, Velocity, and Displacement Waveform Time Histories"
        c6_sub_html = """
        <p><b>6.3 Physical Significance of Ground Kinematics in Earthquake Engineering</b></p>
        <p>&bull; <b>Ground Acceleration a(t):</b> Directly governs instantaneous inertial shear forces (F = m&middot;a) transmitted to stiff low-rise structures.</p>
        <p>&bull; <b>Ground Velocity v(t):</b> Reflects kinetic energy flux and governs demand in mid-rise structures and buried lifeline networks (pipelines).</p>
        <p>&bull; <b>Ground Displacement d(t):</b> Controls transient inter-story drift demands in flexible high-rises and permanent ground strain.</p>
        <p>&bull; <b>Integration Drift Check:</b> If displacement tails drift non-linearly away from zero, raise f<sub>min</sub> (e.g., 0.05 Hz &rarr; 0.10 Hz) in the sidebar.</p>
        """
        c6_callout_title = "Displacement Baseline Stability Verification"
        callout_c6 = (
            "Baseline Stability Diagnostics: Closely examine the displacement trace toward the end of the recording. "
            "If the displacement waveform drifts upwards or downwards away from zero, increase fmin in the sidebar (e.g., from 0.05 Hz to 0.10 Hz) "
            "and ensure Polynomial Detrending is enabled to suppress residual low-frequency integration noise."
        )

    builder.insert_html_safe(p6_body, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 175), c6_text, font_size="7.5pt")

    r_img6 = builder.insert_screenshot_card(p6_body, LEFT_X, y + 182, CONTENT_W,
                                           "Tab Waveforms.png", c6_cap)

    builder.insert_html_safe(p6_body, pymupdf.Rect(LEFT_X, r_img6.y1 + 10, RIGHT_X, r_img6.y1 + 110), c6_sub_html, font_size="7.2pt")

    builder.draw_callout(p6_body, pymupdf.Rect(LEFT_X, r_img6.y1 + 116, RIGHT_X, r_img6.y1 + 186),
                         c6_callout_title, callout_c6, callout_type="warning")

    # =========================================================================
    # HALAMAN 7 (BAB VII: Tab 3 Quality Control)
    # =========================================================================
    p7_body = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p7_body)
    b7_badge = "BAB VII - KENDALI MUTU (QC)" if is_id else "CH. VII - QUALITY CONTROL"
    b7_pstr = f"Halaman 7 dari {TOTAL_BODY_PAGES}" if is_id else f"Page 7 of {TOTAL_BODY_PAGES}"
    builder.add_page_header_footer(p7_body, b7_badge, b7_pstr)

    b7_banner = "BAB VII: TAB 3 - KENDALI MUTU SINYAL (QUALITY CONTROL GATES)" if is_id else "CHAPTER VII: TAB 3 - SIGNAL QUALITY CONTROL GATES (QC GATES)"
    y = builder.draw_chapter_banner(p7_body, b7_banner)

    if is_id:
        c7_text = """
        <p><b>7.1 Audit Kesehatan Sinyal Otomatis (QC Diagnostic Engine)</b></p>
        <p>Tab Quality Control menyajikan evaluasi objektif mengenai kelayakan sinyal rekaman mentah sebelum diproses oleh pipeline rekayasa. 
        Skor komposit dihitung dari basis 100 poin: <code>Skor QC = max(0, min(100, 100 - &sum; Penalti))</code> melalui 6 gerbang diagnostik fisik:</p>
        """
        c7_cap = "Gambar 7.1: Panel Kendali Mutu Sinyal (QC Health Score, Pre-Event SNR, dan 6 QC Gates)"
        qc_table_html = """
        <p><b>7.2 Matriks 6 Gerbang Evaluasi Fisik (QC Gates) & Skema Penalti</b></p>
        <table>
            <tr><th width="18%">Tingkat Keparahan</th><th width="44%">Kriteria Anomali Fisik & Parameter Gate</th><th width="16%">Bobot Penalti</th><th width="22%">Status Kelayakan</th></tr>
            <tr><td><b>Nominal / Normal</b></td><td>Sinyal bersih, noise floor rendah, pre-event SNR &ge; 10 dB, baseline stabil</td><td>0 (Basis 100)</td><td><b>PASS (&ge; 70 Poin)</b><br>Layak analisis penuh</td></tr>
            <tr><td><b>Peringatan Ringan</b><br>(Warning Tier)</td><td>• Baseline Offset (&gt; 2% PGA)<br>• Baseline Drift termal (&gt; 5%)<br>• Pre-event window &lt; 5.0 detik<br>• Pre-event SNR marginal (3 s.d. 10 dB)</td><td><b>-15 Poin</b><br>(per anomali)</td><td><b>WARNING (50-69 Poin)</b><br>Dapat digunakan dengan catatan rekayasa</td></tr>
            <tr><td><b>Kesalahan Berat</b><br>(Major Error)</td><td>• Sinyal derau tinggi parah (SNR &lt; 3 dB)<br>• Multiple impulsive spikes (MAD &ge; 6.0)</td><td><b>-40 Poin</b><br>(per anomali)</td><td>Zona degradasi kualitas</td></tr>
            <tr><td><b>Anomali Kritis</b><br>(Fatal Override)</td><td>• Sensor terpotong (<i>clipping</i> &ge; 0.02 s / 98% FS)<br>• Sinyal datar hilang (<i>flatline</i> &ge; 1.0 s) / berkas korup</td><td><b>Fatal Override<br>(Skor = 0)</b></td><td><b>FAIL / REJECT (&lt; 50 Poin)</b><br>Ditolak untuk integrasi</td></tr>
        </table>
        """
        c7_callout_title = "Interpretasi Skor 70 Data Sinyal Lapangan"
        callout_c7 = (
            "Interpretasi Skor 70 pada Rekaman Alami Lapangan: Skor 70 bukan berarti data rusak, melainkan rekaman mentah alami stasiun yang membawa "
            "(1) Baseline Offset (-15 Poin) dan (2) Drift Termal (-15 Poin). Evaluasi: 100 - 30 = 70 Poin (Status PASS / Lolos). "
            "Kedua anomali non-fatal ini otomatis dibersihkan secara tuntas oleh filter lolos pita dan polynomial detrending BSMA!"
        )
    else:
        c7_text = """
        <p><b>7.1 Automated Signal Quality Auditing (QC Diagnostic Engine)</b></p>
        <p>The Quality Control tab provides an objective multi-gate audit of raw accelerograms prior to downstream kinematic integration. 
        A composite score is calculated from a 100-point base: <code>QC Score = max(0, min(100, 100 - &sum; Penalties))</code> across 6 physical gates:</p>
        """
        c7_cap = "Figure 7.1: Signal Quality Control Dashboard (QC Health Score, Pre-Event SNR, and 6 Physical QC Gates)"
        qc_table_html = """
        <p><b>7.2 Six Physical Diagnostic Gates (QC Gates) & Penalty Structure</b></p>
        <table>
            <tr><th width="18%">Severity Tier</th><th width="44%">Physical Anomaly Criteria & Gate Parameters</th><th width="16%">Penalty Deduction</th><th width="22%">Engineering Eligibility</th></tr>
            <tr><td><b>Nominal / Pristine</b></td><td>Clean waveform, low noise floor, pre-event SNR &ge; 10 dB, stable baseline</td><td>0 (Base 100)</td><td><b>PASS (&ge; 70 Points)</b><br>Fully approved for analysis</td></tr>
            <tr><td><b>Warning Tier</b><br>(Advisory)</td><td>• Static baseline offset (&gt; 2% PGA)<br>• Thermal baseline drift (&gt; 5%)<br>• Pre-event window &lt; 5.0 s<br>• Marginal pre-event SNR (3 to 10 dB)</td><td><b>-15 Points</b><br>(per anomaly)</td><td><b>WARNING (50-69 Points)</b><br>Acceptable with engineering caveats</td></tr>
            <tr><td><b>Major Degradation</b><br>(Severe Noise)</td><td>• Severe ambient noise (SNR &lt; 3 dB)<br>• Multiple high-amplitude spikes (MAD &ge; 6.0)</td><td><b>-40 Points</b><br>(per anomaly)</td><td>Quality degradation zone</td></tr>
            <tr><td><b>Critical Anomaly</b><br>(Fatal Override)</td><td>• ADC saturation (<i>clipping</i> &ge; 0.02 s / 98% FS)<br>• Channel dropout (<i>flatline</i> &ge; 1.0 s) / corrupted file</td><td><b>Fatal Override<br>(Score = 0)</b></td><td><b>FAIL / REJECT (&lt; 50 Points)</b><br>Rejected from integration</td></tr>
        </table>
        """
        c7_callout_title = "Interpretation of Field Record Score 70"
        callout_c7 = (
            "Interpreting Score 70 on Natural Field Records: A score of 70 denotes a fully valid raw recording that simply carries "
            "(1) Baseline Offset (-15 Points) and (2) Natural Thermal Drift (-15 Points). Score: 100 - 30 = 70 (PASS status). "
            "Both non-fatal anomalies are completely eliminated by BSMA's polynomial detrending and zero-phase Butterworth filter!"
        )

    builder.insert_html_safe(p7_body, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 42), c7_text, font_size="7.6pt")

    r_img7 = builder.insert_screenshot_card(p7_body, LEFT_X, y + 48, CONTENT_W,
                                           "Tab QC.png", c7_cap)

    builder.insert_html_safe(p7_body, pymupdf.Rect(LEFT_X, r_img7.y1 + 10, RIGHT_X, r_img7.y1 + 165), qc_table_html)

    builder.draw_callout(p7_body, pymupdf.Rect(LEFT_X, r_img7.y1 + 172, RIGHT_X, r_img7.y1 + 242),
                         c7_callout_title, callout_c7, callout_type="info")

    # =========================================================================
    # HALAMAN 8 (BAB VIII: Tab 4 Strong Motion)
    # =========================================================================
    p8_body = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p8_body)
    b8_badge = "BAB VIII - STRONG MOTION" if is_id else "CH. VIII - STRONG MOTION"
    b8_pstr = f"Halaman 8 dari {TOTAL_BODY_PAGES}" if is_id else f"Page 8 of {TOTAL_BODY_PAGES}"
    builder.add_page_header_footer(p8_body, b8_badge, b8_pstr)

    b8_banner = "BAB VIII: TAB 4 - GERAKAN TANAH KUAT & ENERGI SEISMIK (STRONG MOTION)" if is_id else "CHAPTER VIII: TAB 4 - STRONG GROUND MOTION & SEISMIC ENERGY"
    y = builder.draw_chapter_banner(p8_body, b8_banner)

    if is_id:
        c8_text = """
        <p><b>8.1 Karakterisasi Parameter Energi Seismik Rekayasa Kegempaan</b></p>
        <p>Tab Strong Motion menyajikan parameter fungsional yang merefleksikan daya rusak, kandungan energi getaran, dan potensi beban dinamis 
        getaran gempa pada struktur bangunan serta fondasi tanah geoteknik:</p>
        <p>&bull; <b>Intensitas Arias (Ia):</b> Ukuran energi getaran tanah total yang diintegrasikan dari kuadrat percepatan terhadap durasi gempa: 
        <code>Ia = (&pi; / 2g) &int; a&sup2;(t) dt</code> (satuan: m/s). Parameter ini sangat penting dalam analisis kestabilan lereng, bahaya likuifaksi tanah, 
        dan evaluasi potensi deformasi permanen bendungan maupun timbunan tanah pascagempa.</p>
        <p>&bull; <b>Cumulative Absolute Velocity (CAV):</b> Integrasi nilai absolut percepatan terhadap waktu: <code>CAV = &int; |a(t)| dt</code> (satuan: g&middot;s). 
        Berdasarkan standar EPRI (1988), ambang batas <b>CAV &ge; 0.16 g&middot;s</b> dan <b>PGA &ge; 0.10 g</b> digunakan secara luas sebagai kriteria 
        penentu perlunya inspeksi keselamatan rekayasa pada fasilitas infrastruktur strategis dan reaktor nuklir.</p>
        <p>&bull; <b>Durasi Signifikan Trifunac-Brady (D<sub>5-95</sub>):</b> Interval waktu yang diperlukan untuk mengakumulasikan energi dari 5% 
        hingga 95% total energi Arias Ia. Interval ini merepresentasikan durasi efektif pelepasan energi guncangan kuat yang paling bertanggung jawab 
        terhadap akumulasi kerusakan siklik dan keruntuhan struktur bangunan gedung.</p>
        <p>&bull; <b>Rasio Kinematika V<sub>max</sub> / A<sub>max</sub>:</b> Memberikan indikasi periode getaran dominan tanah (<i>predominant period</i>) 
        dan respon tanah lokal. Rasio yang tinggi mencirikan keberadaan lapisan endapan tanah lunak yang mampu memperkuat getaran.</p>
        """
        c8_cap = "Gambar 8.1: Panel Parameter Gerakan Kuat (Arias Intensity, CAV, Durasi D5-95, dan Rasio Kinematika)"
        c8_sub_html = """
        <p><b>8.2 Matriks Hubungan Parameter Gerakan Kuat dengan Karakteristik Kerusakan Bangunan</b></p>
        <table>
            <tr><th width="24%">Parameter Seismik</th><th width="16%">Satuan</th><th width="60%">Aplikasi Rekayasa Sipil & Dampak Struktural</th></tr>
            <tr><td><b>Intensitas Arias (Ia)</b></td><td>m/s</td><td>Evaluasi potensi likuifaksi pasir jenuh dan pergeseran blok lereng seismik (Newmark sliding block).</td></tr>
            <tr><td><b>Cumulative Abs. Vel. (CAV)</b></td><td>g&middot;s</td><td>Kriteria skrining kerusakan kumulatif struktur kaku (EPRI 1988 threshold: CAV &ge; 0.16 g&middot;s).</td></tr>
            <tr><td><b>Durasi Signifikan (D₅₋₉₅)</b></td><td>detik</td><td>Menentukan jumlah siklus pembebanan inersia bolak-balik pemicu degradasi kekakuan struktur.</td></tr>
            <tr><td><b>Rasio Vmax / Amax</b></td><td>detik</td><td>Estimasi periode dominan tanah (Ts &approx; 2&pi; Vmax/Amax); indikasi efek perangkap lembah sedimen.</td></tr>
        </table>
        """
        c8_callout_title = "Standar Evaluasi Ambang Batas CAV (EPRI 1988)"
        callout_c8 = (
            "Kriteria Skrining Kerusakan EPRI: Apabila nilai CAV tercatat melampaui 0.16 g·s dan PGA melebihi 0.10 g, fasilitas infrastruktur strategis "
            "seperti jembatan bentang panjang, bendungan, instalasi industri, dan bangunan bertingkat tinggi disarankan untuk segera menjalani inspeksi "
            "keselamatan struktural visual di lapangan guna mendeteksi potensi retak fatik mikro atau penurunan daya dukung tanah fondasi."
        )
    else:
        c8_text = """
        <p><b>8.1 Earthquake Engineering Energy Parameter Characterization</b></p>
        <p>The Strong Motion tab quantifies engineering parameters reflecting destructive energy potential, ground motion duration, 
        and cumulative dynamic demands imposed on superstructure and geotechnical foundations:</p>
        <p>&bull; <b>Arias Intensity (Ia):</b> Total seismic energy flux derived by integrating squared acceleration across record duration: 
        <code>Ia = (&pi; / 2g) &int; a&sup2;(t) dt</code> (units: m/s). This parameter is vital for slope stability assessments (Newmark sliding block), 
        liquefaction triggering evaluations, and residual deformation modeling in dams and embankments.</p>
        <p>&bull; <b>Cumulative Absolute Velocity (CAV):</b> Time integral of absolute acceleration: <code>CAV = &int; |a(t)| dt</code> (units: g&middot;s). 
        Under EPRI (1988) criteria, thresholds of <b>CAV &ge; 0.16 g&middot;s</b> combined with <b>PGA &ge; 0.10 g</b> serve as the benchmark 
        standard mandating structural safety inspections in nuclear power stations and strategic infrastructure.</p>
        <p>&bull; <b>Trifunac-Brady Significant Duration (D<sub>5-95</sub>):</b> The time interval spanning 5% to 95% total Arias Intensity accumulation. 
        It represents the strong shaking duration responsible for cyclic stiffness degradation and structural collapse mechanisms.</p>
        <p>&bull; <b>Kinematic Ratio V<sub>max</sub> / A<sub>max</sub>:</b> Provides insight into dominant site ground periods (Ts &approx; 2&pi; Vmax/Amax) 
        and local amplification due to soft sedimentary cover.</p>
        """
        c8_cap = "Figure 8.1: Strong Motion Engineering Parameters (Arias Intensity, CAV, Duration D5-95, and Kinematic Ratios)"
        c8_sub_html = """
        <p><b>8.2 Strong Motion Parameters & Structural Damage Correlations</b></p>
        <table>
            <tr><th width="24%">Seismic Parameter</th><th width="16%">Unit</th><th width="60%">Civil Engineering Application & Structural Impact</th></tr>
            <tr><td><b>Arias Intensity (Ia)</b></td><td>m/s</td><td>Liquefaction triggering in saturated sands and seismic slope displacement evaluations.</td></tr>
            <tr><td><b>Cumulative Abs. Vel. (CAV)</b></td><td>g&middot;s</td><td>Damage screening for stiff structures (EPRI 1988 threshold: CAV &ge; 0.16 g&middot;s).</td></tr>
            <tr><td><b>Significant Duration (D₅₋₉₅)</b></td><td>seconds</td><td>Governs cyclic loading demand and progressive structural fatigue degradation.</td></tr>
            <tr><td><b>Kinematic Ratio Vmax / Amax</b></td><td>seconds</td><td>Estimates dominant site period (Ts &approx; 2&pi; Vmax/Amax); flags sedimentary basin resonance.</td></tr>
        </table>
        """
        c8_callout_title = "EPRI 1988 Damage Screening Standard"
        callout_c8 = (
            "EPRI Damage Screening Criteria: When recorded CAV exceeds 0.16 g·s and PGA surpasses 0.10 g, critical civil infrastructure "
            "including bridges, dams, industrial plants, and multi-story structures should undergo visual structural safety inspections "
            "to detect micro-fatigue cracking or differential foundation settlements."
        )

    builder.insert_html_safe(p8_body, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 215), c8_text, font_size="7.5pt")

    r_img8 = builder.insert_screenshot_card(p8_body, LEFT_X, y + 225, CONTENT_W,
                                           "Tab Strong Motion.png", c8_cap)

    builder.insert_html_safe(p8_body, pymupdf.Rect(LEFT_X, r_img8.y1 + 10, RIGHT_X, r_img8.y1 + 115), c8_sub_html)

    builder.draw_callout(p8_body, pymupdf.Rect(LEFT_X, r_img8.y1 + 122, RIGHT_X, r_img8.y1 + 192),
                         c8_callout_title, callout_c8, callout_type="success")

    # =========================================================================
    # HALAMAN 9 (BAB IX: Tab 5 Intensity MMI)
    # =========================================================================
    p9_body = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p9_body)
    b9_badge = "BAB IX - ESTIMASI INTENSITAS MMI" if is_id else "CH. IX - INTENSITY MMI"
    b9_pstr = f"Halaman 9 dari {TOTAL_BODY_PAGES}" if is_id else f"Page 9 of {TOTAL_BODY_PAGES}"
    builder.add_page_header_footer(p9_body, b9_badge, b9_pstr)

    b9_banner = "BAB IX: TAB 5 - ESTIMASI INTENSITAS INSTRUMENTAL (INTENSITY MMI)" if is_id else "CHAPTER IX: TAB 5 - INSTRUMENTAL INTENSITY ESTIMATION (INTENSITY MMI)"
    y = builder.draw_chapter_banner(p9_body, b9_banner)

    if is_id:
        c9_text = """
        <p><b>9.1 Estimasi Intensitas Instrumental Modified Mercalli Intensity (MMI)</b></p>
        <p>Tab Intensity memetakan parameter kinematika tanah hasil rekaman menjadi skala intensitas instrumental objektif 
        menggunakan formulasi baku Ground-Motion to Intensity Conversion Equations (GMICE):</p>
        <p>&bull; <b>Formulasi GMICE Worden et al. (2012):</b> Merupakan acuan standar operasional USGS ShakeMap. Estimasi MMI dihitung berdasarkan 
        komponen horizontal maksimum (<b>Max-H</b> antara komponen Timur-Barat HNE dan Utara-Selatan HNN, tanpa melibatkan komponen vertikal HNZ).</p>
        <p>&bull; <b>Logika Max-H Horizontal:</b> Komponen vertikal sengaja dikecualikan dalam penentuan intensitas guncangan gempa karena 
        struktur bangunan gedung umumnya dirancang memiliki faktor keamanan tinggi terhadap gravitasi, sedangkan kerusakan struktural 
        hampir selalu dipicu oleh gaya geser inersia lateral horizontal.</p>
        <p>&bull; <b>Transisi Dominansi Kinematika (Kinematic Crossover):</b> Nilai PGA mengontrol estimasi pada getaran ringan (MMI &lt; 5.0). 
        Sebaliknya, nilai PGV mengontrol estimasi pada getaran sedang hingga destruktif (MMI &ge; 5.0) yang berkaitan langsung dengan energi kinetik dan deformasi inersia bangunan.</p>
        """
        c9_cap_a = "Gambar 9.1: Tampilan Skala Intensitas MMI Instrumental Max-H dan Rangkuman Parameter Tab Intensity"
        c9_sub_html = """
        <p><b>9.2 Klasifikasi Skala MMI Baku USGS ShakeMap (Worden et al., 2011/2012)</b></p>
        <p>Matriks baku di bawah ini merupakan acuan resmi hubungan kuantitatif antara persepsi guncangan (<i>Perceived Shaking</i>), 
        potensi dampak kerusakan (<i>Potential Damage</i>), percepatan puncak <b>PGA (%g)</b>, kecepatan puncak <b>PGV (cm/s)</b>, 
        serta tingkatan <b>Instrumental Intensity (MMI I - X+)</b> berbasis GMICE Worden et al. (2011):</p>
        """
        c9_cap_b = "Gambar 9.2: Matriks Hubungan Kuantitatif Skala Intensitas Instrumental MMI USGS ShakeMap (Worden et al., 2011)"
        c9_callout_title = "Batasan Penggunaan Skala MMI Instrumental & Konversi Satuan"
        callout_c9 = (
            "Batasan Penafsiran MMI Instrumental: Nilai MMI pada Tab 5 merupakan estimasi fisik instrumental yang diturunkan secara objektif dari data rekaman "
            "akselerometer pada satu titik stasiun. Nilai ini sangat penting untuk diseminasi dan asesmen cepat pascagempa, namun tidak menggantikan survei "
            "observasi visual kerusakan langsung di lapangan yang mempertimbangkan variasi kualitas konstruksi bangunan dan kondisi geologi mikro sekitar. "
            "Catatan konversi satuan percepatan: 1 %g setara dengan kurang lebih 9.81 Gal."
        )
    else:
        c9_text = """
        <p><b>9.1 Instrumental Modified Mercalli Intensity (MMI) Estimation</b></p>
        <p>The Intensity tab translates recorded ground kinematics into objective instrumental intensity levels 
        using peer-reviewed Ground-Motion to Intensity Conversion Equations (GMICE):</p>
        <p>&bull; <b>GMICE Formulations (Worden et al., 2012):</b> Standard operational relationship utilized by USGS ShakeMap. 
        MMI is derived from the maximum horizontal peak amplitude (<b>Max-H</b> across HNE and HNN channels, excluding the vertical HNZ component).</p>
        <p>&bull; <b>Horizontal Max-H Rationale:</b> Vertical motions are deliberately excluded because civil structures are designed with substantial 
        static gravity margins; severe damage and collapse are predominantly triggered by lateral horizontal inertial shear demands.</p>
        <p>&bull; <b>Kinematic Crossover Logic:</b> PGA governs intensity estimates during weak shaking (MMI &lt; 5.0). 
        Conversely, PGV governs moderate to destructive intensity ranges (MMI &ge; 5.0), where structural damage correlates with kinetic energy and velocity pulse deformations.</p>
        """
        c9_cap_a = "Figure 9.1: Max-H Instrumental MMI Intensity Scale Display and Parameter Summary on Tab 5"
        c9_sub_html = """
        <p><b>9.2 USGS ShakeMap Standard MMI Classification Matrix (Worden et al., 2011/2012)</b></p>
        <p>The standard reference matrix below details quantitative correlations across Perceived Shaking, Potential Damage, 
        Peak Acceleration <b>PGA (%g)</b>, Peak Velocity <b>PGV (cm/s)</b>, and <b>Instrumental Intensity (MMI I - X+)</b>:</p>
        """
        c9_cap_b = "Figure 9.2: USGS ShakeMap Instrumental Intensity (MMI) Quantitative Relationship Matrix (Worden et al., 2011)"
        c9_callout_title = "Instrumental MMI Operational Boundaries & Unit Equivalencies"
        callout_c9 = (
            "Instrumental MMI Operational Boundaries: Tab 5 intensity ratings represent objective point measurements derived from single station instruments. "
            "While critical for rapid automated situational awareness and immediate post-earthquake briefing, instrumental MMI does not replace "
            "comprehensive post-event field macroseismic surveys accounting for building typology variations and micro-geological site effects. "
            "Unit equivalence note: 1 %g equals approximately 9.81 Gal."
        )

    builder.insert_html_safe(p9_body, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 155), c9_text, font_size="7.4pt", line_height="1.28")

    r_img9 = builder.insert_screenshot_card(p9_body, LEFT_X, y + 160, CONTENT_W,
                                           "Tab Intensity.png", c9_cap_a, forced_h=130.0)

    builder.insert_html_safe(p9_body, pymupdf.Rect(LEFT_X, r_img9.y1 + 8, RIGHT_X, r_img9.y1 + 55), c9_sub_html, font_size="7.4pt", line_height="1.28")

    r_img9b = builder.insert_screenshot_card(p9_body, LEFT_X, r_img9.y1 + 58, CONTENT_W,
                                            "Klasifikasi Skala MMI Worden.png", c9_cap_b, forced_h=104.0)

    builder.draw_callout(p9_body, pymupdf.Rect(LEFT_X, r_img9b.y1 + 8, RIGHT_X, r_img9b.y1 + 75),
                         c9_callout_title, callout_c9, callout_type="info")

    # =========================================================================
    # HALAMAN 10 (BAB X: Tab 6 Spektrum Respons & SNI 1726)
    # =========================================================================
    p10_body = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p10_body)
    b10_badge = "BAB X - SPEKTRUM RESPONS & SNI" if is_id else "CH. X - SDOF & SNI 1726"
    b10_pstr = f"Halaman 10 dari {TOTAL_BODY_PAGES}" if is_id else f"Page 10 of {TOTAL_BODY_PAGES}"
    builder.add_page_header_footer(p10_body, b10_badge, b10_pstr)

    b10_banner = "BAB X: TAB 6 - SPEKTRUM RESPONS STRUKTUR SDOF & STANDAR SNI 1726" if is_id else "CHAPTER X: TAB 6 - SDOF RESPONSE SPECTRA & SNI 1726 CODE OVERLAYS"
    y = builder.draw_chapter_banner(p10_body, b10_banner)

    if is_id:
        c10_text = """
        <p><b>10.1 Spektrum Respons Elastis SDOF & Perbandingan Desain SNI 1726:2019</b></p>
        <p>Tab Spectrum menyajikan analisis respons getaran osilator Single-Degree-of-Freedom elastis teredam 5% (&xi; = 0.05) pada rentang periode alami 
        T = 0.01 s hingga 10.0 s untuk mengevaluasi potensi resonansi struktur bangunan:</p>
        <p>&bull; <b>Kurva Spektrum Respons Pseudo-Acceleration (PSA):</b> Menunjukkan percepatan puncak yang dialami struktur bangunan berdasarkan 
        periode alami getarannya. Skala sumbu X dapat dialihkan antara skala Linier (untuk eksplorasi detail periode pendek) dan skala Logaritmik.</p>
        <p>&bull; <b>Overlay Spektrum Desain Gempa SNI 1726:2019:</b> Pengguna dapat membandingkan spektrum respons aktual terhadap kurva batas desain 
        resmi SNI 1726:2019 untuk kelas situs <b>Tanah Lunak (SE)</b>, <b>Tanah Sedang (SD)</b>, dan <b>Tanah Keras (SC)</b> guna mengevaluasi apakah beban 
        gempa nyata melampaui kapasitas desain gedung di wilayah tersebut.</p>
        <p>&bull; <b>Panel Validasi Silang Solver SDOF:</b> Membandingkan kurva spektrum Nigam-Jennings (1969) dan Newmark-Beta (1959) untuk membuktikan 
        kehandalan dan akurasi numerik komputasi perangkat lunak.</p>
        """
        c10_cap_a = "Gambar 10.1: Spektrum Respons Elastis PSA 5% Dibandingkan terhadap Kurva Desain Gempa SNI 1726:2019"
        c10_cap_b = "Gambar 10.2: Panel Validasi Silang Numerik Solver Nigam-Jennings vs Newmark-Beta"
        c10_callout_title = "Deteksi Potensi Bahaya Resonansi Gedung"
        callout_c10 = (
            "Deteksi Bahaya Resonansi Bangunan: Perhatikan periode saat kurva PSA mencapai puncak tertinggi. Jika puncak PSA berada pada rentang "
            "T = 0.1 - 0.4 detik, bangunan rendah hingga menengah (1-4 lantai) berada pada kondisi paling rentan mengalami resonansi struktural. "
            "Sebaliknya, jika puncak berada pada T > 1.0 detik, bangunan tinggi fleksibel atau jembatan yang paling terancam."
        )
    else:
        c10_text = """
        <p><b>10.1 SDOF Elastic Response Spectra & SNI 1726:2019 Design Code Overlays</b></p>
        <p>The Spectrum tab models the response of 5% critically damped (&xi; = 0.05) linear Single-Degree-of-Freedom systems across 
        natural periods T = 0.01 s to 10.0 s, identifying resonance vulnerabilities:</p>
        <p>&bull; <b>Pseudo-Spectral Acceleration (PSA) Curves:</b> Quantifies peak spectral acceleration acting on structures as a function 
        of their fundamental vibration periods. Horizontal axes toggle between Linear (for short-period resolution) and Logarithmic scales.</p>
        <p>&bull; <b>SNI 1726:2019 Design Spectrum Overlays:</b> Compares actual recorded response spectra against official Indonesian Building Code 
        design envelopes for <b>Soft Soil (SE)</b>, <b>Medium Soil (SD)</b>, and <b>Hard Soil (SC)</b>, revealing whether seismic ground demands exceeded design capacity.</p>
        <p>&bull; <b>SDOF Solver Cross-Validation Panel:</b> Direct comparative plotting of Nigam-Jennings (1969) analytical vs Newmark-Beta (1959) 
        numerical spectra, demonstrating near-zero numerical divergence across all evaluated periods.</p>
        """
        c10_cap_a = "Figure 10.1: 5% Damped Elastic Response Spectra PSA Compared with SNI 1726:2019 Code Design Envelopes"
        c10_cap_b = "Figure 10.2: SDOF Solver Numerical Cross-Validation Panel (Nigam-Jennings vs Newmark-Beta)"
        c10_callout_title = "Structural Resonance Vulnerability Identification"
        callout_c10 = (
            "Structural Resonance Detection: Identify the dominant period where the PSA spectrum peaks. If peak amplification concentrates "
            "between T = 0.1 - 0.4 s, low-to-medium rise buildings (1-4 stories) face the highest dynamic resonance risk. "
            "Conversely, peak spectral demands at T > 1.0 s threaten flexible high-rises and long-span bridges."
        )

    builder.insert_html_safe(p10_body, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 175), c10_text, font_size="7.6pt")

    r_img10a = builder.insert_screenshot_card(p10_body, LEFT_X, y + 185, CONTENT_W,
                                             "Tab Spectrum.png", c10_cap_a)
    r_img10b = builder.insert_screenshot_card(p10_body, LEFT_X, r_img10a.y1 + 10.0, CONTENT_W,
                                             "SDOF Solver Cross Validation & Numerical Benchmark dalam tab spektrum.png", c10_cap_b)

    builder.draw_callout(p10_body, pymupdf.Rect(LEFT_X, r_img10b.y1 + 10.0, RIGHT_X, r_img10b.y1 + 76.0),
                         c10_callout_title, callout_c10, callout_type="info")

    # =========================================================================
    # HALAMAN 11 (BAB XI: Spektrum FAS & Kurva Husid)
    # =========================================================================
    p11_body = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p11_body)
    b11_badge = "BAB XI - FAS & KURVA HUSID" if is_id else "CH. XI - FAS & HUSID PLOT"
    b11_pstr = f"Halaman 11 dari {TOTAL_BODY_PAGES}" if is_id else f"Page 11 of {TOTAL_BODY_PAGES}"
    builder.add_page_header_footer(p11_body, b11_badge, b11_pstr)

    b11_banner = "BAB XI: ANALISIS SPEKTRAL LANJUTAN (SPEKTRUM FAS & KURVA HUSID)" if is_id else "CHAPTER XI: ADVANCED SPECTRAL ANALYSIS (FAS SPECTRUM & HUSID PLOT)"
    y = builder.draw_chapter_banner(p11_body, b11_banner)

    if is_id:
        c11_text = """
        <p><b>11.1 Fourier Amplitude Spectrum (FAS) & Kandungan Frekuensi Dominan</b></p>
        <p>Spektrum Amplitudo Fourier (FAS) mentransformasikan sinyal deret waktu percepatan tanah ke domain frekuensi (Hz) menggunakan Fast Fourier Transform 
        (FFT). FAS membedah komposisi harmonik getaran gempa tanpa dipengaruhi oleh faktor redaman struktur:</p>
        <p>&bull; <b>Frekuensi Dominan Tapak (f<sub>0</sub>):</b> Puncak amplitudo tertinggi pada kurva FAS menunjukkan frekuensi alami resonansi tanah di lokasi 
        stasiun, yang mencerminkan ketebalan lapisan sedimen permukaan dan kontras kecepatan gelombang geser (Vs) terhadap batuan dasar.</p>
        <p>&bull; <b>Interaksi Tanah-Struktur:</b> Apabila frekuensi dominan tanah berhimpit dengan frekuensi alami bangunan di atasnya, fenomena 
        resonansi ganda (<i>double resonance</i>) dapat terjadi, memperbesar gaya inersia dan memicu kerusakan struktural yang luas.</p>
        
        <p><b>11.2 Kurva Akumulasi Energi Seismik (Husid Energy Growth Plot)</b></p>
        <p>Kurva Husid menggambarkan laju pertumbuhan integral kuadrat percepatan terhadap waktu yang telah dinormalisasi dari 0% hingga 100%:</p>
        <p>&bull; <b>Analisis Bentuk Kurva:</b> Kemiringan kurva yang sangat curam mengindikasikan pelepasan energi guncangan kuat yang terkonsentrasi secara 
        tiba-tiba (<i>impulsive ground motion</i> / efek directivity sesar dekat). Sebaliknya, kurva dengan kenaikan bertahap mencirikan guncangan berdurasi 
        panjang yang umum terjadi pada zona subduksi megathrust.</p>
        <p>&bull; <b>Penentuan Durasi Efektif:</b> Garis batas horizontal 5% dan 95% pada plot Husid menjadi dasar penentuan durasi signifikan Trifunac-Brady (D<sub>5-95</sub>).</p>
        """
        c11_cap_a = "Gambar 11.1: Fourier Amplitude Spectrum (FAS)"
        c11_cap_b = "Gambar 11.2: Kurva Akumulasi Energi Seismik Husid"
        c11_sub_html = """
        <p><b>11.3 Karakterisasi Tipe Sumber Gempa Bumi Berdasarkan Bentuk Kurva Husid</b></p>
        <table>
            <tr><th width="25%">Mekanisme Gempa</th><th width="25%">Bentuk Profil Husid</th><th width="20%">Durasi D₅₋₉₅</th><th width="30%">Implikasi Rekayasa Struktur</th></tr>
            <tr><td><b>Sesar Dekat (Near-Fault)</b></td><td>Lonjakan tegak curam dalam beberapa detik awal</td><td>Sangat pendek (&lt; 10 detik)</td><td>Pukulan impulsif satu arah (fling-step); deformasi plastis besar seketika</td></tr>
            <tr><td><b>Kerak Dangkal (Crustal)</b></td><td>Kenaikan lereng konstan dengan kemiringan sedang</td><td>Sedang (10 - 25 detik)</td><td>Beban bolak-balik berimbang; keretakan siklik elemen beton bertulang</td></tr>
            <tr><td><b>Subduksi Megathrust</b></td><td>Kurva mendatar panjang, akumulasi energi bertahap</td><td>Panjang (&gt; 35 detik)</td><td>Kelelahan material (fatigue damage) tinggi; risiko likuifaksi parah</td></tr>
        </table>
        """
        c11_callout_title = "Aplikasi Analisis Sinyal Lanjutan untuk Geoteknik"
        callout_c11 = (
            "Aplikasi Rekayasa Geoteknik: Kombinasi antara analisis kurva FAS dan spektrum respons PSA memberikan wawasan komprehensif bagi praktisi "
            "mengenai kecocokan frekuensi getaran tanah dengan frekuensi alami struktur guna merumuskan rekomendasi perkuatan bangunan (retrofitting) "
            "dan mitigasi risiko bencana gempa bumi secara presisi."
        )
    else:
        c11_text = """
        <p><b>11.1 Fourier Amplitude Spectrum (FAS) & Dominant Site Frequencies</b></p>
        <p>The Fourier Amplitude Spectrum transforms acceleration time series into the continuous frequency domain (Hz) via Fast Fourier Transform (FFT). 
        Unlike SDOF response spectra, FAS deconstructs pure harmonic energy distributions uncoupled from structural damping:</p>
        <p>&bull; <b>Dominant Site Frequency (f<sub>0</sub>):</b> Peak spectral amplitudes identify fundamental soil resonance frequencies, 
        governed by sedimentary overburden thickness and shear wave velocity (Vs) contrasts with underlying engineering bedrock.</p>
        <p>&bull; <b>Soil-Structure Resonance:</b> When site dominant frequencies coincide with the fundamental frequency of the overlying building, 
        <i>double resonance</i> phenomena can occur, amplifying structural base shear and catastrophic damage.</p>
        
        <p><b>11.2 Seismic Energy Accumulation (Husid Energy Growth Plot)</b></p>
        <p>The Husid plot portrays the cumulative time-rate of squared acceleration energy normalized from 0% to 100%:</p>
        <p>&bull; <b>Profile Morphology:</b> Steep vertical slopes indicate impulsive, concentrated energy release (near-fault forward directivity / fling-step). 
        Gradual, extended rises characterize prolonged energy dissipation common in large megathrust subduction earthquakes.</p>
        <p>&bull; <b>Significant Duration Calibration:</b> Horizontal markers at 5% and 95% threshold lines calibrate Trifunac-Brady duration (D<sub>5-95</sub>).</p>
        """
        c11_cap_a = "Figure 11.1: Fourier Amplitude Spectrum (FAS)"
        c11_cap_b = "Figure 11.2: Cumulative Husid Seismic Energy Growth Curve"
        c11_sub_html = """
        <p><b>11.3 Earthquake Source Rupture Mechanisms vs Husid Curve Morphology</b></p>
        <table>
            <tr><th width="25%">Rupture Mechanism</th><th width="25%">Husid Profile Morphology</th><th width="20%">Duration D₅₋₉₅</th><th width="30%">Structural Engineering Implications</th></tr>
            <tr><td><b>Near-Fault Directivity</b></td><td>Abrupt vertical step within initial seconds</td><td>Short (&lt; 10 seconds)</td><td>High-velocity pulse loading; severe plastic deformation demands</td></tr>
            <tr><td><b>Shallow Crustal</b></td><td>Uniform moderate-slope monotonic rise</td><td>Moderate (10 - 25 s)</td><td>Symmetric cyclic reversals; progressive structural degradation</td></tr>
            <tr><td><b>Megathrust Subduction</b></td><td>Extended multi-phase gradual rise</td><td>Long (&gt; 35 seconds)</td><td>High-cycle fatigue damage; severe soil liquefaction triggering</td></tr>
        </table>
        """
        c11_callout_title = "Geotechnical Engineering Applications of Spectral Analysis"
        callout_c11 = (
            "Geotechnical & Structural Synergy: Combining FAS frequency peaks with PSA design spectra yields comprehensive insight into "
            "ground motion site effects. Engineers can pinpoint potential resonance risks, guide seismic retrofitting prioritization, "
            "and calibrate foundation damping requirements."
        )

    builder.insert_html_safe(p11_body, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 215), c11_text, font_size="7.5pt")

    half_w = (CONTENT_W - 14.0) / 2
    r_img11a = builder.insert_screenshot_card(p11_body, LEFT_X, y + 225, half_w,
                                             "Fourier Spectrum (FAS) dalam Tab Spectrum.png", c11_cap_a, forced_h=105.0)
    r_img11b = builder.insert_screenshot_card(p11_body, LEFT_X + half_w + 14.0, y + 225, half_w,
                                             "Husid Energy Growth dalam Tab Spectrum.png", c11_cap_b, forced_h=105.0)

    builder.insert_html_safe(p11_body, pymupdf.Rect(LEFT_X, r_img11a.y1 + 10, RIGHT_X, r_img11a.y1 + 115), c11_sub_html)

    builder.draw_callout(p11_body, pymupdf.Rect(LEFT_X, r_img11a.y1 + 122, RIGHT_X, r_img11a.y1 + 192),
                         c11_callout_title, callout_c11, callout_type="success")

    # =========================================================================
    # HALAMAN 12 (BAB XII: Tab 7 Ekspor Deliverables)
    # =========================================================================
    p12_body = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p12_body)
    b12_badge = "BAB XII - EKSPOR DELIVERABLES" if is_id else "CH. XII - DELIVERABLES"
    b12_pstr = f"Halaman 12 dari {TOTAL_BODY_PAGES}" if is_id else f"Page 12 of {TOTAL_BODY_PAGES}"
    builder.add_page_header_footer(p12_body, b12_badge, b12_pstr)

    b12_banner = "BAB XII: TAB 7 - EKSPOR LAPORAN TEKNIK PDF & DELIVERABLES" if is_id else "CHAPTER XII: TAB 7 - PDF ENGINEERING REPORT EXPORT & DELIVERABLES"
    y = builder.draw_chapter_banner(p12_body, b12_banner)

    if is_id:
        c12_text = """
        <p><b>12.1 Panel Ekspor Produk Rekayasa Seismologi</b></p>
        <p>Tab Report menyediakan sarana pengunduhan produk rekayasa kegempaan yang lengkap, terstandarisasi, dan siap pakai:</p>
        <p>&bull; <b>Laporan Teknik PDF 2-Halaman:</b> Menghasilkan berkas PDF siap cetak beresolusi tinggi (vektor PyMuPDF) yang memuat kop resmi BMKG-ITERA, 
        metadata gempa, kartu parameter PGA/PGV/PGD, visualisasi riwayat waktu 3-komponen, kurva Husid, spektrum respons PSA vs SNI 1726, serta tabel audit QC.</p>
        <p>&bull; <b>Tabel CSV Deret Waktu Kinematika:</b> Menyimpan data akselerasi, kecepatan, dan perpindahan terkoreksi untuk analisis riwayat waktu (<i>time-history</i>) 
        pada perangkat lunak analisis struktur sipil (ETABS, SAP2000, OpenSees).</p>
        <p>&bull; <b>Tabel CSV Spektrum Respons:</b> Tabel diskret Periode T versus ordinat spektral PSA untuk analisis beban gempa rencana.</p>
        <p>&bull; <b>Paket Arsip ZIP Terpadu:</b> Menyatukan seluruh produk luaran (PDF, CSV, JSON log) dalam satu berkas terkompresi untuk arsip stasiun.</p>
        """
        c12_cap_a = "Gambar 12.1: Panel Pratinjau Interaktif Laporan Teknik Rekayasa Seismologi 2-Halaman Berformat PDF"
        c12_cap_b = "Gambar 12.2: Tombol Pengunduhan Produk Deliverables (PDF, CSV Kinematika, CSV Spektral, JSON, dan ZIP Paket)"
        table_deliv_html = """
        <p><b>12.2 Rangkuman Format Berkas Deliverables BSMA v2.0.0</b></p>
        <table>
            <tr><th width="24%">Nama Berkas Luaran</th><th width="20%">Format / Tipe</th><th width="56%">Kegunaan & Integrasi Sistem</th></tr>
            <tr><td><b>Laporan Teknik Rekayasa</b></td><td>Berkas PDF (*.pdf)</td><td>Diseminasi pimpinan, arsip stasiun BMKG, dan laporan teknis.</td></tr>
            <tr><td><b>Tabel Kinematika Terkoreksi</b></td><td>Spreadsheet CSV (*.csv)</td><td>Analisis time-history pada software perancangan gempa (ETABS, SAP2000).</td></tr>
            <tr><td><b>Tabel Spektrum Respons</b></td><td>Spreadsheet CSV (*.csv)</td><td>Evaluasi kurva beban spektral perancangan gedung dan struktur jembatan.</td></tr>
            <tr><td><b>Paket Arsip Terpadu</b></td><td>Arsip Kompresi (*.zip)</td><td>Dokumentasi lengkap satu rekaman stasiun untuk arsip permanen institusi.</td></tr>
        </table>
        """
    else:
        c12_text = """
        <p><b>12.1 Engineering Seismological Deliverables Export Panel</b></p>
        <p>The Report tab provides a unified export suite generating standardized, publication-ready engineering outputs:</p>
        <p>&bull; <b>2-Page Official Engineering PDF Report:</b> Generates a high-resolution vector PDF report (via PyMuPDF) containing BMKG-ITERA logos, 
        event metadata, PGA/PGV/PGD KPI cards, synchronized 3-component kinematic traces, Husid curves, PSA vs SNI 1726 overlays, and QC audit tables.</p>
        <p>&bull; <b>Corrected Kinematic Time History CSVs:</b> Exports discrete acceleration, velocity, and displacement series for dynamic time-history 
        analyses in structural engineering software (ETABS, SAP2000, OpenSees).</p>
        <p>&bull; <b>Response Spectra CSVs:</b> Tabulates natural period T against PSA, PSV, and SD spectral ordinates for design code verification.</p>
        <p>&bull; <b>Unified ZIP Archive Package:</b> Bundles all output files (PDF reports, CSV tables, JSON provenance logs) into a single archive.</p>
        """
        c12_cap_a = "Figure 12.1: Interactive Preview Panel of the Official 2-Page Engineering Seismology PDF Report"
        c12_cap_b = "Figure 12.2: Deliverables Download Widgets (PDF Report, Kinematic CSV, Spectral CSV, JSON, and ZIP Archive)"
        table_deliv_html = """
        <p><b>12.2 BSMA v2.0.0 Engineering Deliverables Format Summary</b></p>
        <table>
            <tr><th width="24%">Deliverable File Name</th><th width="20%">File Format</th><th width="56%">Primary Engineering Use & System Integration</th></tr>
            <tr><td><b>Engineering PDF Report</b></td><td>Vector PDF (*.pdf)</td><td>Official post-event stakeholder briefings, BMKG archives, and technical reports.</td></tr>
            <tr><td><b>Corrected Kinematic CSV</b></td><td>Spreadsheet CSV (*.csv)</td><td>Non-linear dynamic time-history analysis in structural software (ETABS, OpenSees).</td></tr>
            <tr><td><b>Response Spectra CSV</b></td><td>Spreadsheet CSV (*.csv)</td><td>Equivalent lateral force and spectral modal analysis for building design.</td></tr>
            <tr><td><b>Unified Archive Package</b></td><td>Compressed ZIP (*.zip)</td><td>Permanent institutional record archiving complete analytical data packages.</td></tr>
        </table>
        """

    builder.insert_html_safe(p12_body, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 130), c12_text, font_size="7.6pt")

    r_img12a = builder.insert_screenshot_card(p12_body, LEFT_X, y + 138, CONTENT_W,
                                             "Tab Report.png", c12_cap_a)
    r_img12b = builder.insert_screenshot_card(p12_body, LEFT_X, r_img12a.y1 + 10.0, CONTENT_W,
                                             "Export Results.png", c12_cap_b)

    builder.insert_html_safe(p12_body, pymupdf.Rect(LEFT_X, r_img12b.y1 + 10, RIGHT_X, r_img12b.y1 + 120), table_deliv_html)

    # =========================================================================
    # HALAMAN 13 (BAB XIII: Mode Batch Multi-Stasiun)
    # =========================================================================
    p13_body = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p13_body)
    b13_badge = "BAB XIII - MODE BATCH MULTI-STASIUN" if is_id else "CH. XIII - BATCH MODE"
    b13_pstr = f"Halaman 13 dari {TOTAL_BODY_PAGES}" if is_id else f"Page 13 of {TOTAL_BODY_PAGES}"
    builder.add_page_header_footer(p13_body, b13_badge, b13_pstr)

    b13_banner = "BAB XIII: PROTOKOL PEMROSESAN BATCH MULTI-STASIUN REGIONAL" if is_id else "CHAPTER XIII: REGIONAL MULTI-STATION BATCH PROCESSING PROTOCOL"
    y = builder.draw_chapter_banner(p13_body, b13_banner)

    if is_id:
        c13_text = """
        <p><b>13.1 Urgensi Pemrosesan Jaringan Gempa Regional Massal</b></p>
        <p>Ketika gempa bumi signifikan terjadi, guncangan terekam secara serentak di puluhan stasiun akselerograf BMKG yang tersebar di seluruh wilayah. 
        Mengolah data stasiun satu per satu secara manual sangat tidak efisien untuk tanggap darurat bencana. <b>Mode Pemrosesan Batch Multi-Stasiun</b> 
        pada BSMA v2.0.0 mengotomatisasi pemrosesan kumpulan berkas stasiun secara massal, sekuensial, dan menyusun matriks sebaran guncangan regional.</p>
        <p><b>13.2 Alur Pengoperasian Mode Batch</b></p>
        <p>&bull; <b>Aktivasi Mode Batch:</b> Beralih ke <b>'Batch Processing Mode'</b> pada bilah navigasi mode di bagian atas antarmuka.</p>
        <p>&bull; <b>Unggah Massal Berkas:</b> Seret seluruh berkas MiniSEED (.mseed) atau SAC (.sac) dari berbagai stasiun ke area uploader massal.</p>
        <p>&bull; <b>Proteksi Isolasi Kegagalan (Fault Isolation):</b> Setiap stasiun dieksekusi dalam wadah mandiri (<i>isolated boundary</i>). 
        Berkas yang korup atau cacat akan ditandai <b>FAILED</b> tanpa menghentikan atau membatalkan proses stasiun-stasiun lain.</p>
        <p>&bull; <b>Tabel Matriks Komparasi Regional:</b> Menampilkan rangkuman stasiun: PGA, PGV, PGD, Arias Ia, D5-95, MMI, dan Skor Mutu QC.</p>
        <p>&bull; <b>Tombol 'Export Regional Summary CSV':</b> Mengunduh seluruh rekapitulasi ke format CSV untuk diimpor ke software GIS (QGIS/ArcGIS).</p>
        """
        c13_cap = "Gambar 13.1: Dasbor Pemrosesan Batch Multi-Stasiun Regional dan Tabel Matriks Komparasi Jaringan Seismik"
        c13_sub_html = """
        <p><b>13.3 Struktur Kolom Rekapitulasi CSV & Integrasi Perangkat Lunak GIS</b></p>
        <table>
            <tr><th width="24%">Kolom Data CSV</th><th width="24%">Satuan Pengukuran</th><th width="52%">Pemanfaatan dalam Analisis Spasial GIS</th></tr>
            <tr><td><b>Station, Net, Chan</b></td><td>Teks Identitas</td><td>Atribut pengenal unik titik stasiun pengamatan gempa bumi.</td></tr>
            <tr><td><b>PGA_MaxH, PGV_MaxH</b></td><td>Gal, cm/s</td><td>Variabel utama interpolasi spasial kontur guncangan (IDW / Kriging).</td></tr>
            <tr><td><b>Arias_Ia, CAV</b></td><td>m/s, g&middot;s</td><td>Layer analisis risiko geoteknik; pemetaan potensi zona likuifaksi tanah.</td></tr>
            <tr><td><b>MMI_MaxH, QC_Score</b></td><td>Skala Angka, Poin</td><td>Simbolisasi warna peta intensitas instrumental (ShakeMap) dan filter mutu.</td></tr>
        </table>
        """
        c13_callout_title = "Pemanfaatan Mode Batch untuk Mitigasi Bencana Gempa"
        callout_c13 = (
            "Efisiensi Tanggap Darurat Bencana: Dalam situasi darurat bencana gempa bumi, Mode Batch memungkinkan operator BMKG memetakan sebaran "
            "nilai PGA, PGV, dan estimasi intensitas MMI seluruh stasiun jaringan regional hanya dalam hitungan detik, sangat mendukung percepatan "
            "pembuatan peta guncangan gempa bumi (ShakeMap BMKG) dan pelaporan resmi kepada pimpinan serta Badan Penanggulangan Bencana Daerah (BPBD)."
        )
    else:
        c13_text = """
        <p><b>13.1 Regional Seismic Network Processing Rationale</b></p>
        <p>Significant earthquake events trigger ground motion recordings across dozens of regional BMKG accelerograph stations simultaneously. 
        Processing each station stream sequentially by hand is inefficient during emergency crisis operations. <b>Multi-Station Batch Processing Mode</b> 
        in BSMA v2.0.0 automates bulk folder ingestion, parallel screening, and regional summary matrix synthesis.</p>
        <p><b>13.2 Batch Workflow Execution Steps</b></p>
        <p>&bull; <b>Batch Mode Selection:</b> Switch to <b>'Batch Processing Mode'</b> on the top navigation bar.</p>
        <p>&bull; <b>Bulk File Upload:</b> Drag-and-drop a collection of MiniSEED (.mseed) or SAC (.sac) files from regional stations into the bulk uploader.</p>
        <p>&bull; <b>Fault Isolation Architecture:</b> Each station stream executes within an isolated boundary. Corrupted traces are flagged 
        as <b>FAILED</b> without aborting or stalling batch processing for remaining stations.</p>
        <p>&bull; <b>Regional Comparison Matrix:</b> Compiles all processed station metrics: PGA, PGV, PGD, Arias Ia, D5-95, MMI, and QC Health Scores.</p>
        <p>&bull; <b>'Export Regional Summary CSV' Button:</b> Downloads the synthesized regional matrix for immediate GIS mapping (QGIS/ArcGIS).</p>
        """
        c13_cap = "Figure 13.1: Regional Multi-Station Batch Processing Dashboard and Seismic Network Comparison Matrix"
        c13_sub_html = """
        <p><b>13.3 Regional Summary CSV Column Structure & GIS Software Integration</b></p>
        <table>
            <tr><th width="24%">CSV Data Column</th><th width="24%">Measurement Unit</th><th width="52%">Spatial Mapping & GIS Analysis Application</th></tr>
            <tr><td><b>Station, Net, Chan</b></td><td>Identity String</td><td>Unique primary key for joining station geospatial coordinate layers.</td></tr>
            <tr><td><b>PGA_MaxH, PGV_MaxH</b></td><td>Gal, cm/s</td><td>Core variables for spatial shaking contour interpolation (Kriging / IDW).</td></tr>
            <tr><td><b>Arias_Ia, CAV</b></td><td>m/s, g&middot;s</td><td>Geotechnical hazard overlays for liquefaction and slope instability mapping.</td></tr>
            <tr><td><b>MMI_MaxH, QC_Score</b></td><td>Numerical Scale, Points</td><td>Color symbology for instrumental ShakeMaps and quality assurance filtering.</td></tr>
        </table>
        """
        c13_callout_title = "Emergency Disaster Response Utilization"
        callout_c13 = (
            "Disaster Response Acceleration: During emergency post-earthquake operations, Batch Mode allows duty seismologists to compile "
            "comprehensive PGA, PGV, and MMI ShakeMap distributions across entire regional networks within seconds, dramatically accelerating "
            "critical situation briefings for civil protection agencies (BPBD/BNPB)."
        )

    builder.insert_html_safe(p13_body, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 215), c13_text, font_size="7.6pt")

    r_img13 = builder.insert_screenshot_card(p13_body, LEFT_X, y + 225, CONTENT_W,
                                            "Multi Station Processing .png", c13_cap)

    builder.insert_html_safe(p13_body, pymupdf.Rect(LEFT_X, r_img13.y1 + 10, RIGHT_X, r_img13.y1 + 115), c13_sub_html)

    builder.draw_callout(p13_body, pymupdf.Rect(LEFT_X, r_img13.y1 + 122, RIGHT_X, r_img13.y1 + 192),
                         c13_callout_title, callout_c13, callout_type="success")

    # =========================================================================
    # HALAMAN 14 (BAB XIV: Troubleshooting Operasional)
    # =========================================================================
    p14_body = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p14_body)
    b14_badge = "BAB XIV - TROUBLESHOOTING" if is_id else "CH. XIV - TROUBLESHOOTING"
    b14_pstr = f"Halaman 14 dari {TOTAL_BODY_PAGES}" if is_id else f"Page 14 of {TOTAL_BODY_PAGES}"
    builder.add_page_header_footer(p14_body, b14_badge, b14_pstr)

    b14_banner = "BAB XIV: PANDUAN PEMECAHAN MASALAH OPERASIONAL (TROUBLESHOOTING)" if is_id else "CHAPTER XIV: OPERATIONAL TROUBLESHOOTING GUIDE & FIELD PROCEDURES"
    y = builder.draw_chapter_banner(p14_body, b14_banner)

    if is_id:
        trouble_intro = """
        <p>Tabel berikut merangkum panduan pemecahan masalah teknis yang sering ditemui selama pengoperasian perangkat lunak 
        beserta langkah-langkah praktis penanganannya di lapangan:</p>
        """
        trouble_table_html = """
        <table>
            <tr><th width="24%">Gejala Kendala</th><th width="32%">Kemungkinan Penyebab</th><th width="44%">Solusi Praktis & Tindakan Operasional</th></tr>
            <tr><td><b>Berkas gagal dimuat / format ditolak</b></td><td>Berkas korup, ekstensi salah, atau terdapat diskontinuitas waktu (gap).</td><td>Pastikan berkas berformat MiniSEED (*.mseed) atau SAC (*.sac) standar FDSN tanpa jeda waktu cuplikan.</td></tr>
            <tr><td><b>Riwayat perpindahan (Displacement) melayang</b></td><td>Derau frekuensi rendah residual belum tereliminasi sempurna.</td><td>Naikkan batas f<sub>min</sub> pada filter (misal: 0.05 &rarr; 0.10 Hz) dan pastikan opsi Polynomial Detrending aktif.</td></tr>
            <tr><td><b>Amplitudo getaran melonjak tidak wajar</b></td><td>Dekonvolusi instrumen diterapkan ganda pada data yang sudah fisik.</td><td>Ubah opsi Data Provenance menjadi <i>'Already processed physical acceleration'</i> untuk bypass kalibrasi.</td></tr>
            <tr><td><b>Status QC menunjukkan WARNING (Skor 70)</b></td><td>Baseline offset atau drift termal ringan alami pada sensor mentah.</td><td>Status normal untuk data lapangan. Filter bandpass BSMA otomatis membersihkan anomali ini saat komputasi.</td></tr>
            <tr><td><b>Status QC menunjukkan FAIL / REJECT (Skor 0)</b></td><td>Sensor terpotong (clipping ADC) atau kanal mati (flatline).</td><td>Data cacat secara fisik; rekaman tidak boleh diintegrasikan ke kecepatan/perpindahan karena tidak valid.</td></tr>
            <tr><td><b>Laporan PDF gagal diunduh</b></td><td>Data rekaman belum dimuat atau memori peramban penuh.</td><td>Klik tombol 'Reset / Clear Data' pada sidebar, muat ulang berkas, dan pastikan proses komputasi selesai sempurna.</td></tr>
            <tr><td><b>Batch mode berhenti di tengah jalan</b></td><td>Terdapat berkas korup tanpa header standar.</td><td>Arsitektur fault isolation BSMA otomatis melewati berkas tersebut dan melanjutkan stasiun berikutnya.</td></tr>
        </table>
        """
        trouble_tips_html = """
        <p><b>14.2 Batasan Operasional Sistem (System Limitations)</b></p>
        <p>&bull; <b>Integritas Kanal:</b> Sinyal idealnya memuat 3 komponen ortogonal (Z, N, E). Bila berkas hanya memuat 1 komponen, 
        estimasi intensitas MMI Max-H akan dihitung dari komponen horizontal yang tersedia.</p>
        <p>&bull; <b>Laju Cuplik Data:</b> Perangkat lunak mendukung laju cuplik seragam mulai dari 20 Hz hingga 250 Hz. Rekaman dengan laju cuplik 
        berubah-ubah di tengah rekaman harus di-resampling terlebih dahulu sebelum dimuat.</p>
        <p>&bull; <b>Kebutuhan Memori:</b> Untuk direktori batch berukuran besar (&gt; 50 stasiun), disarankan menggunakan perangkat komputer 
        dengan RAM minimal 8 GB guna menjaga kelancaran buffering data.</p>
        """
        c14_callout_title = "Dukungan Pengguna & Pelaporan Kendala"
        callout_c14 = (
            "Bantuan Teknis & Pelaporan Kendala: Apabila kendala operasional tetap berlanjut setelah mengikuti panduan pemecahan masalah di atas, "
            "silakan hubungi tim pengembang melalui surel resmi akademik atau buat laporan kendala (Issue) pada repositori GitHub resmi BSMA."
        )
    else:
        trouble_intro = """
        <p>The table below summarizes common technical anomalies encountered during software operation 
        along with standardized operational resolution workflows:</p>
        """
        trouble_table_html = """
        <table>
            <tr><th width="24%">Observed Symptom</th><th width="32%">Probable Root Cause</th><th width="44%">Operational Solution & Corrective Action</th></tr>
            <tr><td><b>File ingestion failure / rejected format</b></td><td>Corrupted file header, non-standard extension, or temporal telemetry gaps.</td><td>Verify file is standard FDSN MiniSEED (*.mseed) or SAC (*.sac) without sample discontinuities.</td></tr>
            <tr><td><b>Displacement time history drifts upward/downward</b></td><td>Residual long-period noise amplified during numerical double integration.</td><td>Increase lower corner f<sub>min</sub> (e.g., 0.05 &rarr; 0.10 Hz) and ensure Polynomial Detrending is enabled.</td></tr>
            <tr><td><b>Unreasonable amplitude surge (10⁶ Gal)</b></td><td>Instrument deconvolution erroneously applied to pre-calibrated physical data.</td><td>Set Data Provenance to <i>'Already processed physical acceleration'</i> to engage calibration bypass.</td></tr>
            <tr><td><b>QC status flags WARNING (Score 70)</b></td><td>Natural static baseline offset or moderate thermal drift in raw field sensor.</td><td>Normal status for field records. BSMA bandpass filtering and detrending cleanse anomalies automatically.</td></tr>
            <tr><td><b>QC status flags FAIL / REJECT (Score 0)</b></td><td>ADC saturation (sensor clipping) or channel dead flatline.</td><td>Physically corrupted record; data cannot be integrated into velocity/displacement validly.</td></tr>
            <tr><td><b>PDF report download fails</b></td><td>Session cache exhaustion or incomplete pipeline execution.</td><td>Click 'Reset / Clear Data' on sidebar, reload record, and ensure pipeline execution completes fully.</td></tr>
            <tr><td><b>Batch mode stalls on single station</b></td><td>Corrupted file lacking mandatory FDSN headers.</td><td>BSMA fault isolation automatically skips the invalid record and continues processing remaining stations.</td></tr>
        </table>
        """
        trouble_tips_html = """
        <p><b>14.2 System Operational Boundaries & Computational Constraints</b></p>
        <p>&bull; <b>Channel Integrity:</b> Requires standard triaxial orthogonal components (Z, N, E). If single-channel records are loaded, 
        Max-H MMI is calculated from the available horizontal component.</p>
        <p>&bull; <b>Sampling Rate Uniformity:</b> Supports uniform rates between 20 Hz and 250 Hz. Streams with variable sampling intervals 
        must be resampled before loading.</p>
        <p>&bull; <b>Memory Recommendations:</b> For extensive batch directories (&gt; 50 stations), a machine with &ge; 8 GB RAM is recommended 
        to ensure fluid buffer streaming and chart rendering.</p>
        """
        c14_callout_title = "User Technical Support & Issue Reporting"
        callout_c14 = (
            "Technical Support & Issue Tracking: If operational difficulties persist after applying the troubleshooting steps above, "
            "contact the developer via official university email or submit a detailed issue report on the official BSMA GitHub repository."
        )

    builder.insert_html_safe(p14_body, pymupdf.Rect(LEFT_X, y, RIGHT_X, y + 28), trouble_intro, font_size="7.6pt")
    builder.insert_html_safe(p14_body, pymupdf.Rect(LEFT_X, y + 34, RIGHT_X, y + 310), trouble_table_html)
    builder.insert_html_safe(p14_body, pymupdf.Rect(LEFT_X, y + 322, RIGHT_X, y + 455), trouble_tips_html, font_size="7.5pt")
    builder.draw_callout(p14_body, pymupdf.Rect(LEFT_X, y + 465, RIGHT_X, y + 540),
                         c14_callout_title, callout_c14, callout_type="info")

    # =========================================================================
    # HALAMAN 15 (PROFIL PENGEMBANG & PENGESAHAN)
    # =========================================================================
    p15_body = doc.new_page(width=PAGE_W, height=PAGE_H)
    builder.init_page_fonts(p15_body)
    b15_badge = "PROFIL PENGEMBANG" if is_id else "DEVELOPER PROFILE"
    b15_pstr = f"Halaman 15 dari {TOTAL_BODY_PAGES}" if is_id else f"Page 15 of {TOTAL_BODY_PAGES}"
    builder.add_page_header_footer(p15_body, b15_badge, b15_pstr)

    b15_banner = "PROFIL PENGEMBANG & PENGESAHAN KERJA PRAKTIK" if is_id else "DEVELOPER PROFILE & INTERNSHIP ENDORSEMENT"
    y = builder.draw_chapter_banner(p15_body, b15_banner)

    prof_box_y = y + 10
    builder.draw_card(p15_body, pymupdf.Rect(LEFT_X, prof_box_y, RIGHT_X, prof_box_y + 360), bg_col=COLOR_WHITE, border_col=COLOR_CARD_BORDER)

    prof_title = "IDENTITAS PENYUSUN & AKADEMIK" if is_id else "AUTHOR & ACADEMIC IDENTITY"
    p15_body.insert_text((LEFT_X + 24, prof_box_y + 32), prof_title, fontsize=11.0, fontname=FONT_BOLD, color=COLOR_NAVY)
    p15_body.draw_line(pymupdf.Point(LEFT_X + 24, prof_box_y + 40), pymupdf.Point(RIGHT_X - 24, prof_box_y + 40), color=COLOR_SKY, width=1.2)

    if is_id:
        prof_fields = [
            ("Nama Lengkap", "Ahmad Didane Setyawan Putra"),
            ("Nomor Induk Mahasiswa (NIM)", "123120094"),
            ("Program Studi", "Teknik Geofisika"),
            ("Fakultas", "Fakultas Teknologi Industri"),
            ("Perguruan Tinggi", "Institut Teknologi Sumatera (ITERA)"),
            ("Instansi Penyelenggara KP", "Stasiun Geofisika Kelas I Sleman, BMKG D.I. Yogyakarta"),
            ("Periode Pelaksanaan", "20 Juli 2026 s.d. 20 Agustus 2026"),
            ("Peran dalam Proyek", "Lead Developer & Geophysical Computing Researcher"),
            ("Kontak Surel Resmi", "ahmad.123120094@student.itera.ac.id"),
            ("Tautan Repositori GitHub", "https://github.com/ahmaddidan/BSMA-v.2"),
            ("Tautan Aplikasi Web Online", "https://strong-motion.streamlit.app/"),
        ]
        c15_callout_title = "Pernyataan Akademik & Komitmen Terbuka"
        callout_closing = (
            "Pernyataan Hak Cipta & Penggunaan Akademik: Perangkat lunak BMKG Strong Motion Analyzer (BSMA v2.0.0) dan buku panduan operasional ini "
            "merupakan hasil karya ilmiah mahasiswa program sarjana Teknik Geofisika ITERA dalam program kerja praktik di BMKG Stasiun Geofisika Sleman. "
            "Seluruh kode sumber bersifat terbuka (open-source) untuk kemajuan riset seismologi teknik dan mitigasi bencana gempa bumi di Indonesia."
        )
    else:
        prof_fields = [
            ("Full Name", "Ahmad Didane Setyawan Putra"),
            ("Student ID (NIM)", "123120094"),
            ("Study Program", "Geophysical Engineering"),
            ("Faculty", "Faculty of Industrial Technology"),
            ("University", "Institut Teknologi Sumatera (ITERA)"),
            ("Host Institution", "Sleman Geophysical Station Class I, BMKG D.I. Yogyakarta"),
            ("Internship Period", "July 20, 2026 - August 20, 2026"),
            ("Project Role", "Lead Developer & Geophysical Computing Researcher"),
            ("Official Academic Email", "ahmad.123120094@student.itera.ac.id"),
            ("GitHub Source Repository", "https://github.com/ahmaddidan/BSMA-v.2"),
            ("Online Web Application", "https://strong-motion.streamlit.app/"),
        ]
        c15_callout_title = "Academic Attribution & Open-Source Statement"
        callout_closing = (
            "Academic Attribution & Open-Source Statement: BMKG Strong Motion Analyzer (BSMA v2.0.0) and this operational technical manual "
            "constitute original academic works developed by an undergraduate Geophysical Engineering student of ITERA during an internship at BMKG Sleman. "
            "All source code is open-source to support advancements in engineering seismology and earthquake disaster mitigation in Indonesia."
        )

    cur_p_y = prof_box_y + 70
    for label, val in prof_fields:
        p15_body.insert_text((LEFT_X + 24, cur_p_y), f"{label}", fontsize=8.4, fontname=FONT_BOLD, color=COLOR_SLATE)
        if "http" in val:
            p15_body.insert_text((LEFT_X + 175, cur_p_y), f":  {val}", fontsize=8.4, fontname=FONT_REG, color=COLOR_SKY)
            link_w = font_bold_obj.text_length(f":  {val}", fontsize=8.4)
            p15_body.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(LEFT_X + 175, cur_p_y - 8, LEFT_X + 175 + link_w, cur_p_y + 2), "uri": val})
        else:
            p15_body.insert_text((LEFT_X + 175, cur_p_y), f":  {val}", fontsize=8.4, fontname=FONT_REG, color=COLOR_DARK)
        cur_p_y += 24.5

    builder.draw_callout(p15_body, pymupdf.Rect(LEFT_X, prof_box_y + 380, RIGHT_X, prof_box_y + 475),
                         c15_callout_title, callout_closing, callout_type="info")

    # Native PDF Bookmarks Outline (TOC)
    toc_id = [
        [1, "Halaman Sampul & Informasi Dokumen", 1],
        [1, "Daftar Isi Panduan Pengoperasian", 2],
        [1, "Panduan Cepat 5 Langkah Operasional (Quick Start)", 3],
        [1, "Bab I: Akses Sistem & Pengunggahan Data (Data Input)", 4],
        [1, "Bab II: Seleksi Stasiun & Parameter DSP (Sidebar)", 5],
        [1, "Bab III: Tab 1 - Ringkasan Eksekutif & Tinjauan Sinyal", 6],
        [1, "Bab IV: Tab 2 - Bentuk Gelombang Kinematika Lengkap", 7],
        [1, "Bab V: Tab 3 - Diagnostik Kendali Mutu Sinyal (QC)", 8],
        [1, "Bab VI: Tab 4 - Karakteristik Gerakan Kuat & Energi", 9],
        [1, "Bab VII: Tab 5 - Estimasi Intensitas Instrumental MMI", 10],
        [1, "Bab VIII: Tab 6 - Spektrum Respons SDOF Elastis 5%", 11],
        [1, "Bab IX: Evaluasi Spektrum Desain SNI 1726:2019", 12],
        [1, "Bab X: Tab 7 - Pelaporan Teknik & Ekspor Data", 13],
        [1, "Bab XI: Mode Pemrosesan Batch Multi-Stasiun", 14],
        [1, "Bab XII: Alur Kerja Ekspor Batch & Arsip ZIP", 15],
        [1, "Bab XIII: Panduan Penanganan Galat & Validasi Data", 16],
        [1, "Bab XIV: Spesifikasi Runtime & Dependensi Sistem", 17],
        [1, "Bab XV: Profil Pengembang & Penafian Tanggung Jawab", 18],
    ]
    toc_en = [
        [1, "Title Page & Document Information", 1],
        [1, "Table of Contents & Guidebook Architecture", 2],
        [1, "Quick Start Guide (5-Step Operational Flow)", 3],
        [1, "Chapter I: System Access & Ingestion (Data Input)", 4],
        [1, "Chapter II: Station Selection & DSP Settings (Sidebar)", 5],
        [1, "Chapter III: Tab 1 - Executive Summary & Signal Overview", 6],
        [1, "Chapter IV: Tab 2 - Complete Kinematic Waveforms", 7],
        [1, "Chapter V: Tab 3 - Signal Quality Control (QC) Diagnostics", 8],
        [1, "Chapter VI: Tab 4 - Strong Ground Motion & Energy Metrics", 9],
        [1, "Chapter VII: Tab 5 - Instrumental MMI Intensity Estimation", 10],
        [1, "Chapter VIII: Tab 6 - Elastic 5% Damped SDOF Spectra", 11],
        [1, "Chapter IX: SNI 1726:2019 Seismic Code & Solver Benchmark", 12],
        [1, "Chapter X: Tab 7 - Technical Reporting & Data Export", 13],
        [1, "Chapter XI: Regional Multi-Station Batch Processing", 14],
        [1, "Chapter XII: Batch Export Workflows & Full Archive ZIP", 15],
        [1, "Chapter XIII: Error Handling & Diagnostic Validation", 16],
        [1, "Chapter XIV: Computational Runtime & Dependencies", 17],
        [1, "Chapter XV: Developer Profile & Academic Disclaimer", 18],
    ]
    doc.set_toc(toc_id if is_id else toc_en)

    builder.clean_pdf_ligatures()

    suffix = "ID" if is_id else "EN"
    out_path = OUTPUT_DIR / f"BSMA_Panduan_Teknis_Operasional_{suffix}.pdf"
    doc.save(str(out_path), deflate=True, garbage=4)
    total_pages = len(doc)
    doc.close()

    print(f"[SUCCESS] Operational Guidebook [{suffix}] successfully created at: {out_path}")
    print(f"Total Pages: {total_pages} (3 Front Matter + {TOTAL_BODY_PAGES} Body Pages)")
    return out_path


def main():
    print("=" * 72)
    print("BMKG Strong Motion Analyzer (BSMA v2.0.0) - Operational Manual Generator")
    print("=" * 72)

    pdf_id = build_operational_guidebook(lang="id")
    pdf_en = build_operational_guidebook(lang="en")

    print("\nOperational Manual generation completed successfully!")
    print(f"  - Indonesian Edition: {pdf_id} ({pdf_id.stat().st_size / 1024:.1f} KB)")
    print(f"  - English Edition:    {pdf_en} ({pdf_en.stat().st_size / 1024:.1f} KB)")
    print("=" * 72)


if __name__ == "__main__":
    main()
