"""
BMKG Strong Motion Analyzer (BSMA)
core/io/pdf_exporter.py

Generator laporan PDF hasil analisis strong motion.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from fpdf import FPDF


class PDFReportGenerator(FPDF):
    """
    Template PDF laporan analisis BSMA.
    """

    def __init__(self, logo_path: str | Path | None = None) -> None:
        super().__init__()
        self.logo_path = Path(logo_path) if logo_path else None
        # Operational reports must identify the system, never a personal
        # author/developer. These values are PDF metadata, not visible text.
        self.set_title("BSMA Strong Motion Report")
        self.set_author("BMKG Strong Motion Analyzer")
        self.set_creator("BMKG Strong Motion Analyzer")

    def header(self) -> None:
        """Render header pada setiap halaman."""
        if self.logo_path and self.logo_path.is_file():
            self.image(str(self.logo_path), x=12, y=8, w=20)
        self.set_font("helvetica", "B", 14)
        self.cell(
            0,
            10,
            "BMKG Strong Motion Analyzer (BSMA)",
            border=0,
            ln=1,
            align="C",
        )

        self.set_font("helvetica", "", 9)
        self.cell(
            0,
            6,
            "Laporan Analisis Strong Motion",
            border=0,
            ln=1,
            align="C",
        )

        self.ln(3)

    def footer(self) -> None:
        """Render nomor halaman."""
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(
            0,
            10,
            f"Halaman {self.page_no()}",
            border=0,
            align="C",
        )


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Mengubah nilai menjadi float secara aman.

    Parameters
    ----------
    value
        Nilai yang akan dikonversi.
    default
        Nilai fallback apabila konversi gagal.

    Returns
    -------
    float
        Nilai float hasil konversi.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric(
    metrics: Mapping[str, Any],
    key: str,
    default: float = 0.0,
) -> float:
    """
    Mengambil metric dari mapping secara aman.
    """
    return _safe_float(
        metrics.get(key, default),
        default,
    )


def _value(source: Any, key: str, default: Any = "-") -> Any:
    """Read a value from either a mapping or a metadata object."""
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def generate_station_pdf(
    metadata: Any,
    params: Mapping[str, Any],
    sig_status: str,
    plot_paths: Mapping[str, str | Path] | None,
    output_pdf_path: str | Path,
    *,
    event_info: Mapping[str, Any] | None = None,
    logo_path: str | Path | None = None,
) -> Path:
    """
    Membuat laporan PDF analisis strong motion.

    Parameters
    ----------
    metadata
        Metadata trace/stasiun.

    params
        Dictionary parameter hasil analisis BSMA.

        Metric internal BSMA yang digunakan:
        - PGA                  : m/s²
        - PGV                  : m/s
        - PGD                  : m
        - Arias_Intensity      : sesuai hasil pipeline
        - Significant_Duration_D5_95 : s

    sig_status
        Klasifikasi SIG BMKG yang telah dihitung oleh
        layer pemanggil.

    plot_paths
        Mapping:
            judul grafik -> path file gambar.

    output_pdf_path
        Lokasi file PDF keluaran.

    Returns
    -------
    pathlib.Path
        Path file PDF yang berhasil dibuat.

    Raises
    ------
    OSError
        Jika file PDF gagal ditulis.
    """

    output_path = Path(output_pdf_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf = PDFReportGenerator(logo_path=logo_path)

    pdf.set_auto_page_break(
        auto=True,
        margin=20,
    )

    pdf.add_page()

    # ==============================================================
    # INFORMASI STASIUN
    # ==============================================================

    station = _value(metadata, "station")
    channel = _value(metadata, "channel")
    network = _value(metadata, "network")
    location = _value(metadata, "location")
    sampling_rate = _value(metadata, "sampling_rate")

    pdf.set_font(
        "helvetica",
        "B",
        11,
    )

    pdf.cell(
        0,
        7,
        f"Informasi Stasiun: {station} ({channel})",
        border=0,
        ln=1,
    )

    pdf.set_font(
        "helvetica",
        "",
        10,
    )

    pdf.cell(
        0,
        6,
        f"Network: {network}",
        border=0,
        ln=1,
    )

    pdf.cell(
        0,
        6,
        f"Location: {location}",
        border=0,
        ln=1,
    )

    pdf.cell(
        0,
        6,
        f"Sampling Rate: {sampling_rate} Hz",
        border=0,
        ln=1,
    )

    pdf.ln(4)

    # ==============================================================
    # INFORMASI EVENT
    # ==============================================================
    if event_info:
        pdf.set_font("helvetica", "B", 11)
        pdf.cell(0, 7, "Informasi Event", border=0, ln=1)
        pdf.set_font("helvetica", "", 10)
        event_rows = (
            ("Waktu", _value(event_info, "time")),
            ("Koordinat", f"{_value(event_info, 'latitude')}, {_value(event_info, 'longitude')}"),
            ("Magnitudo", _value(event_info, "magnitude")),
            ("Kedalaman", f"{_value(event_info, 'depth_km')} km"),
            ("Jarak episentral", f"{_value(event_info, 'epicentral_distance_km')} km"),
        )
        for label, value in event_rows:
            pdf.cell(0, 6, f"{label}: {value}", border=0, ln=1)
        pdf.ln(4)

    # ==============================================================
    # PARAMETER STRONG MOTION
    # ==============================================================

    pdf.set_font(
        "helvetica",
        "B",
        11,
    )

    pdf.cell(
        0,
        7,
        "Parameter Strong Motion",
        border=0,
        ln=1,
    )

    pdf.set_font(
        "helvetica",
        "",
        10,
    )

    pga_ms2 = _metric(
        params,
        "PGA",
    )

    pgv_ms = _metric(
        params,
        "PGV",
    )

    pgd_m = _metric(
        params,
        "PGD",
    )

    arias = _metric(
        params,
        "Arias_Intensity",
    )

    duration = _metric(
        params,
        "Significant_Duration_D5_95",
    )

    # ==============================================================
    # KONVERSI SATUAN
    # ==============================================================

    # m/s² -> Gal
    pga_gal = pga_ms2 * 100.0

    # m/s -> cm/s
    pgv_cm_s = pgv_ms * 100.0

    # m -> cm
    pgd_cm = pgd_m * 100.0

    # ==============================================================
    # OUTPUT PARAMETER
    # ==============================================================

    pdf.cell(
        0,
        6,
        f"PGA: {pga_gal:.4f} Gal",
        border=0,
        ln=1,
    )

    pdf.cell(
        0,
        6,
        f"PGV: {pgv_cm_s:.4f} cm/s",
        border=0,
        ln=1,
    )

    pdf.cell(
        0,
        6,
        f"PGD: {pgd_cm:.4f} cm",
        border=0,
        ln=1,
    )

    pdf.cell(
        0,
        6,
        f"Arias Intensity: {arias:.6f}",
        border=0,
        ln=1,
    )

    pdf.cell(
        0,
        6,
        f"Significant Duration D5-95: {duration:.2f} s",
        border=0,
        ln=1,
    )

    pdf.cell(
        0,
        6,
        f"SIG BMKG: {sig_status}",
        border=0,
        ln=1,
    )

    pdf.ln(5)

    # ==============================================================
    # VISUALISASI
    # ==============================================================

    # Charts are intentionally started on a fresh page.  Letting FPDF's
    # automatic break split a chart title from the chart itself produces an
    # unprofessional orphan heading at the bottom of the summary page.
    if plot_paths:
        pdf.add_page()
        # The content on this page begins below the header logo rather than
        # sharing its visual row with the section title.
        pdf.ln(8)

    pdf.set_font(
        "helvetica",
        "B",
        11,
    )

    pdf.cell(
        0,
        7,
        "Visualisasi Grafik Analisis",
        border=0,
        ln=1,
    )

    if plot_paths:
        for title, plot_path in plot_paths.items():

            if not plot_path:
                continue

            image_path = Path(plot_path)

            if not image_path.is_file():
                continue

            pdf.set_font(
                "helvetica",
                "I",
                9,
            )

            pdf.cell(
                0,
                5,
                str(title),
                border=0,
                ln=1,
            )

            pdf.image(
                str(image_path),
                w=170,
            )

            pdf.ln(4)

    # ==============================================================
    # WRITE PDF
    # ==============================================================

    try:
        pdf.output(str(output_path))
    except OSError as exc:
        raise OSError(
            f"Gagal menulis laporan PDF: {output_path}"
        ) from exc

    return output_path
