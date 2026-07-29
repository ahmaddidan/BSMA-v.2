"""
BMKG Strong Motion Analyzer (BSMA)
core/io/pdf_exporter.py
"""
from pathlib import Path
from fpdf import FPDF

class PDFReportGenerator(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 14)
        self.cell(0, 10, 'BMKG Strong Motion Analyzer (BSMA) - Laporan Analisis', 0, 1, 'C')
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

def generate_station_pdf(metadata, params, sig_status, plot_paths: dict, output_pdf_path: str):
    pdf = PDFReportGenerator()
    pdf.add_page()
    
    # Metadata Stasiun
    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 7, f"Informasi Stasiun: {metadata.station} ({metadata.channel})", 0, 1)
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 6, f"Network: {metadata.network} | Sampling Rate: {metadata.sampling_rate} Hz", 0, 1)
    pdf.ln(3)

    # Parameter Hasil Analisis
    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 7, "Parameter Rekayasa Gempa & Klasifikasi", 0, 1)
    pdf.set_font('helvetica', '', 10)
    if params:
        pdf.cell(0, 6, f"- Peak Acceleration (PGA): {params.pga:.2f} cm/s²", 0, 1)
        pdf.cell(0, 6, f"- Peak Velocity (PGV): {params.pgv:.4f} cm/s", 0, 1)
        pdf.cell(0, 6, f"- Peak Displacement (PGD): {params.pgd:.4f} cm", 0, 1)
        pdf.cell(0, 6, f"- Arias Intensity: {params.arias_intensity:.4f} m/s", 0, 1)
        pdf.cell(0, 6, f"- Klasifikasi SIG BMKG: {sig_status}", 0, 1)
    pdf.ln(5)

    # Sisipkan Gambar Plot
    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 7, "Visualisasi Grafik Analisis", 0, 1)
    
    for title, p_path in plot_paths.items():
        if p_path and Path(p_path).exists():
            pdf.set_font('helvetica', 'I', 9)
            pdf.cell(0, 5, title, 0, 1)
            pdf.image(p_path, w=170)
            pdf.ln(3)

    pdf.output(output_pdf_path)
    return output_pdf_path