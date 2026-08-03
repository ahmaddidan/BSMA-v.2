"""
BMKG Strong Motion Analyzer (BSMA)
Module: utils/pdf_exporter.py
Description: Generates a highly professional multi-page PDF report (BMKG standard).
"""
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF

from core.types.context import ProcessingContext

class BSMAPDFReport(FPDF):
    def header(self):
        # Kop Surat Laporan
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(0, 51, 102) # Dark Blue
        self.cell(0, 8, 'LAPORAN ANALISIS STRONG MOTION (BSMA BMKG)', border=0, ln=1, align='C')
        self.set_line_width(0.5)
        self.line(10, 20, 200, 20)
        self.ln(8)

    def footer(self):
        # Footer Halaman
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Dicetak pada: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Halaman {self.page_no()}', 0, 0, 'C')

def get_sig_bmkg(pga_gal: float) -> tuple:
    """
    Menghitung SIG BMKG berdasarkan PGA (dalam unit Gal).
    Berdasarkan referensi gambar tabel BMKG resmi.
    Returns: (Skala SIG, Warna Nama, RGB Tuple, Deskripsi Singkat, Skala MMI, Deskripsi Rinci)
    """
    if pga_gal < 2.9:
        return ("I", "Putih", (255, 255, 255), "TIDAK DIRASAKAN", "I-II", "Tidak dirasakan atau dirasakan hanya oleh beberapa orang tetapi terekam oleh alat.")
    elif pga_gal <= 88.0:
        return ("II", "Hijau", (146, 208, 80), "DIRASAKAN", "III-V", "Dirasakan oleh orang banyak tetapi tidak menimbulkan kerusakan. Benda-benda ringan yang digantung bergoyang dan jendela kaca bergetar.")
    elif pga_gal <= 167.0:
        return ("III", "Kuning", (255, 255, 0), "KERUSAKAN RINGAN", "VI", "Bagian non struktur bangunan mengalami kerusakan ringan, seperti retak rambut pada dinding, atap bergeser ke bawah dan sebagian berjatuhan.")
    elif pga_gal <= 564.0:
        return ("IV", "Jingga", (255, 192, 0), "KERUSAKAN SEDANG", "VII-VIII", "Banyak Retakan terjadi pada dinding bangunan sederhana, sebagian roboh, kaca pecah. Sebagian plester dinding lepas. Hampir sebagian besar atap bergeser ke bawah atau jatuh. Struktur bangunan mengalami kerusakan ringan sampai sedang.")
    else:
        return ("V", "Merah", (255, 0, 0), "KERUSAKAN BERAT", "IX-XII", "Sebagian besar dinding bangunan permanen roboh. Struktur bangunan mengalami kerusakan berat. Rel kereta api melengkung.")

def generate_spectrum_overlay(contexts: dict[str, ProcessingContext], output_path: str, station_code: str):
    """Membuat grafik overlay Response Spectrum (PSA) dengan Legenda."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = {"HNE": "purple", "HNN": "goldenrod", "HNZ": "darkgreen"}
    has_data = False
    
    for comp, ctx in contexts.items():
        spec = ctx.spectral_data
        
        # Ekstraksi aman: mendukung "periods" maupun "Periods"
        psa_data = spec.get("PSA", spec.get("psa"))
        periods_data = spec.get("periods", spec.get("Periods"))
        
        if psa_data is not None and periods_data is not None:
            max_sa = np.max(psa_data) / 9.80665  # Convert m/s2 to g
            psa_g = psa_data / 9.80665
            color = colors.get(comp[-3:], "blue")
            ax.plot(periods_data, psa_g, label=f"{comp} (Max Sa: {max_sa:.6f} g)", color=color, linewidth=1.5)
            has_data = True
            
    if has_data:
        ax.set_title(f"Response Spectrum Overlay (N, E, Z) - Stasiun {station_code}", fontweight='bold')
        ax.set_xlabel("Building Period (seconds)")
        ax.set_ylabel("Spectral Acc (g)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def generate_4panel_waveform(ctx: ProcessingContext, comp: str, output_path: str, station_code: str):
    """Membuat plot 4 panel (Raw, Acc, Vel, Disp)."""
    fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    
    raw = ctx.raw_waveform.data if ctx.raw_waveform else np.array([])
    acc = ctx.acceleration.data if ctx.acceleration else np.array([])
    vel = ctx.velocity.data if ctx.velocity else np.array([])
    disp = ctx.displacement.data if ctx.displacement else np.array([])
    
    sr = ctx.sampling_rate
    time = np.arange(len(acc)) / sr if len(acc) > 0 else np.array([])

    m = ctx.metrics
    pga = m.get("PGA", 0.0)
    pgv = m.get("PGV", 0.0) * 100  # cm/s
    pgd = m.get("PGD", 0.0) * 100  # cm

    if len(raw) > 0: axs[0].plot(np.arange(len(raw))/sr, raw * 100, color='gray', linewidth=0.5)
    axs[0].set_title(f"1. Raw Waveform ({comp})", fontweight='bold', fontsize=10)
    axs[0].set_ylabel("Amp (gal)")
    axs[0].grid(True, linestyle="--", alpha=0.5)

    if len(acc) > 0: axs[1].plot(time, acc, color='black', linewidth=0.5)
    axs[1].set_title(f"2. Filtered Acceleration ({comp} | PGA: {pga:.6f} m/s²)", fontweight='bold', fontsize=10)
    axs[1].set_ylabel("Acc (m/s²)")
    axs[1].grid(True, linestyle="--", alpha=0.5)

    if len(vel) > 0: axs[2].plot(time, vel * 100, color='mediumblue', linewidth=0.5)
    axs[2].set_title(f"3. Velocity ({comp} | PGV: {pgv:.6f} cm/s)", fontweight='bold', fontsize=10)
    axs[2].set_ylabel("Vel (cm/s)")
    axs[2].grid(True, linestyle="--", alpha=0.5)

    if len(disp) > 0: axs[3].plot(time, disp * 100, color='crimson', linewidth=0.5)
    axs[3].set_title(f"4. Displacement ({comp} | PGD: {pgd:.6f} cm)", fontweight='bold', fontsize=10)
    axs[3].set_ylabel("Disp (cm)")
    axs[3].set_xlabel("Time (seconds)", fontweight='bold')
    axs[3].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(f"Analisis Kinematik Gelombang Waktu\nStasiun: {station_code} | Channel: {comp}", y=0.98, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def export_station_report(station_code: str, contexts: dict[str, ProcessingContext], output_dir: str = "outputs/reports") -> Path:
    """Menggabungkan hasil 3 komponen stasiun ke dalam PDF profesional multi-halaman."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = out_dir / f"BSMA_Report_{station_code}_{timestamp}.pdf"
    
    # 1. Cari Max PGA
    max_pga_ms2 = 0.0
    strongest_comp = ""
    for comp, ctx in contexts.items():
        pga = ctx.metrics.get("PGA", 0.0)
        if pga > max_pga_ms2:
            max_pga_ms2 = pga
            strongest_comp = comp
            
    pga_gal_max = max_pga_ms2 * 100.0
    sig_id, sig_color, rgb, sig_desc, mmi, sig_detail = get_sig_bmkg(pga_gal_max)

    # INISIALISASI PDF
    pdf = BSMAPDFReport()
    pdf.add_page()
    
    # KESIMPULAN UTAMA
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "1. RINGKASAN GUNCANGAN TERKUAT", ln=1)
    
    pdf.set_font("helvetica", "", 11)
    summary = (f"Stasiun {station_code} mencatat guncangan terkuat pada komponen {strongest_comp} "
               f"dengan nilai PGA maksimum sebesar {max_pga_ms2:.6f} m/s² ({pga_gal_max:.2f} Gal). "
               f"Berdasarkan standar BMKG, guncangan ini masuk dalam kategori:")
    pdf.multi_cell(0, 6, summary)
    pdf.ln(3)

    # KOTAK WARNA SIG
    pdf.set_fill_color(*rgb)
    # Jika warnanya gelap (Merah/Jingga), gunakan teks putih
    text_c = 255 if sig_id in ["IV", "V"] else 0
    pdf.set_text_color(text_c, text_c, text_c)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 12, f"SKALA {sig_id} SIG-BMKG (Setara {mmi}) - {sig_desc}", border=1, ln=1, align='C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("helvetica", "I", 10)
    pdf.multi_cell(0, 6, f"Dampak: {sig_detail}", border=1)
    pdf.ln(8)

    # TABEL PARAMETER KOMPONEN
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "2. TABEL PARAMETER STRONG MOTION KESELURUHAN", ln=1)
    
    col_w = [25, 25, 25, 25, 30, 25, 35] # Total 190
    pdf.set_fill_color(220, 220, 220)
    pdf.set_font("helvetica", "B", 9)
    headers = ["Channel", "PGA (Gal)", "PGV (cm/s)", "PGD (cm)", "Arias (m/s)", "Durasi (s)", "Intensitas"]
    for i, head in enumerate(headers):
        pdf.cell(col_w[i], 8, head, border=1, align="C", fill=True)
    pdf.ln()
    
    pdf.set_font("helvetica", "", 9)
    for comp in sorted(contexts.keys()):
        ctx = contexts[comp]
        m = ctx.metrics
        c_pga_gal = m.get("PGA", 0.0) * 100.0
        c_sig, _, _, _, c_mmi, _ = get_sig_bmkg(c_pga_gal)
        
        pdf.cell(col_w[0], 8, comp, border=1, align="C")
        pdf.cell(col_w[1], 8, f"{c_pga_gal:.3f}", border=1, align="C")
        pdf.cell(col_w[2], 8, f"{m.get('PGV', 0.0) * 100:.4f}", border=1, align="C")
        pdf.cell(col_w[3], 8, f"{m.get('PGD', 0.0) * 100:.4f}", border=1, align="C")
        pdf.cell(col_w[4], 8, f"{m.get('Arias_Intensity', 0.0):.6f}", border=1, align="C")
        pdf.cell(col_w[5], 8, f"{m.get('Significant_Duration_D5_95', 0.0):.2f}", border=1, align="C")
        pdf.cell(col_w[6], 8, f"{c_sig} ({c_mmi})", border=1, align="C")
        pdf.ln()
    pdf.ln(8)

    # GRAFIK OVERLAY SPEKTRUM
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "3. RESPONSE SPECTRUM (REDAMAN 5%)", ln=1)
    spec_img = out_dir / f"temp_spec_{timestamp}.png"
    generate_spectrum_overlay(contexts, str(spec_img), station_code)
    if spec_img.exists():
        pdf.image(str(spec_img), x=10, w=190)
        os.remove(spec_img)

    # HALAMAN WAVEFORM PER KOMPONEN
    for comp in sorted(contexts.keys()):
        pdf.add_page()
        ctx = contexts[comp]
        wave_img = out_dir / f"temp_wave_{comp}_{timestamp}.png"
        generate_4panel_waveform(ctx, comp, str(wave_img), station_code)
        
        if wave_img.exists():
            pdf.image(str(wave_img), x=10, w=190)
            os.remove(wave_img)

    pdf.output(str(pdf_path))
    return pdf_path