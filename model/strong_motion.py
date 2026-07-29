"""
BMKG Strong Motion Analyzer (BSMA)
Model: Strong Motion Parameter Extractor

Mengekstrak parameter fisik gempa dari data akselerasi:
- PGA (Peak Ground Acceleration)
- PGV (Peak Ground Velocity)
- PGD (Peak Ground Displacement)
- Arias Intensity (Ia)
- Significant Duration (D_5-95)
- Skala MMI & SIG BMKG (Skala Intensitas Gempabumi)
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from dataclasses import dataclass, field
from typing import Any

__all__ = ["StrongMotionConfig", "StrongMotionData", "StrongMotionCalculator"]

@dataclass(slots=True)
class StrongMotionConfig:
    """Konfigurasi ekstraksi parameter strong motion."""
    gravity: float = 9.80665  # Percepatan gravitasi standar (m/s^2)
    calculate_pgd: bool = True


@dataclass(slots=True)
class StrongMotionData:
    """Kontainer hasil kalkulasi parameter strong motion dan intensitas."""
    trace_id: str
    pga: float = 0.0          # m/s^2
    pgv: float = 0.0          # m/s
    pgd: float = 0.0          # m
    arias_intensity: float = 0.0  # m/s
    duration_5_95: float = 0.0    # s
    mmi: float = 1.0
    sig_bmkg_scale: str = "I"
    sig_bmkg_color: str = "Putih"
    sig_bmkg_desc: str = "Terekam alat, umumnya tidak dirasakan."

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "pga_m_s2": round(self.pga, 6),
            "pgv_m_s": round(self.pgv, 6),
            "pgd_m": round(self.pgd, 6),
            "arias_intensity_m_s": round(self.arias_intensity, 6),
            "duration_5_95_s": round(self.duration_5_95, 3),
            "mmi": round(self.mmi, 1),
            "sig_bmkg": {
                "skala": self.sig_bmkg_scale,
                "warna": self.sig_bmkg_color,
                "deskripsi": self.sig_bmkg_desc
            }
        }


class StrongMotionCalculator:
    """Mesin komputasi utama untuk mengekstrak parameter intensitas dan energi gempa."""

    def __init__(self, config: StrongMotionConfig | None = None):
        self.config = config or StrongMotionConfig()

    def _convert_pga_to_mmi(self, pga_m_s2: float) -> float:
        """
        Mengonversi PGA menjadi MMI menggunakan relasi empiris standar (Worden / Wald).
        Untuk penyederhanaan, menggunakan relasi logaritmik Wald et al. (1999)
        MMI = 3.66 * log10(PGA_gal) - 1.66
        """
        pga_gal = abs(pga_m_s2) * 100.0  # Konversi m/s^2 ke cm/s^2 (Gal)
        if pga_gal < 0.1:
            return 1.0
        
        mmi = 3.66 * np.log10(pga_gal) - 1.66
        return float(np.clip(mmi, 1.0, 12.0))

    def _get_sig_bmkg(self, mmi: float) -> tuple[str, str, str]:
        """
        Memetakan nilai MMI ke Skala Intensitas Gempabumi (SIG) BMKG.
        SIG I   : MMI I - II
        SIG II  : MMI III - V
        SIG III : MMI VI
        SIG IV  : MMI VII - VIII
        SIG V   : MMI IX - XII
        """
        if mmi < 3.0:
            return "I", "Putih", "Terekam alat, umumnya tidak dirasakan."
        elif mmi < 6.0:
            return "II", "Hijau", "Dirasakan banyak orang, benda bergoyang, kaca bergetar, tanpa kerusakan."
        elif mmi < 7.0:
            return "III", "Kuning", "Kerusakan ringan, dinding retak rambut, genteng bergeser."
        elif mmi < 9.0:
            return "IV", "Jingga", "Kerusakan sedang, dinding rumah banyak retak, sebagian roboh."
        else:
            return "V", "Merah", "Kerusakan berat, bangunan roboh, jembatan putus, tanah terbelah."

    def process_trace(self, acc_ground: np.ndarray, dt: float, trace_id: str = "unknown") -> StrongMotionData:
        """Mengekstrak seluruh parameter dari satu trace akselerasi tanah (m/s^2)."""
        if acc_ground.size == 0 or not np.isfinite(acc_ground).all():
            raise ValueError(f"Data akselerasi pada trace {trace_id} tidak valid (mengandung NaN/Inf).")

        npts = len(acc_ground)
        time_array = np.arange(npts) * dt

        # 1. Hitung PGA
        pga = float(np.max(np.abs(acc_ground)))

        # 2. Integrasi untuk PGV dan PGD
        # cumulative_trapezoid mengembalikan array berukuran N-1 jika initial tidak diatur.
        velocity = cumulative_trapezoid(acc_ground, dx=dt, initial=0.0)
        pgv = float(np.max(np.abs(velocity)))

        pgd = 0.0
        if self.config.calculate_pgd:
            displacement = cumulative_trapezoid(velocity, dx=dt, initial=0.0)
            pgd = float(np.max(np.abs(displacement)))

        # 3. Hitung Arias Intensity (Ia)
        # Rumus: Ia = (pi / 2g) * integral(a(t)^2) dt
        acc_squared = acc_ground ** 2
        integral_a2 = cumulative_trapezoid(acc_squared, dx=dt, initial=0.0)
        ia_series = (np.pi / (2.0 * self.config.gravity)) * integral_a2
        total_ia = float(ia_series[-1])

        # 4. Hitung Significant Duration (D_5-95)
        d_5_95 = 0.0
        if total_ia > 0.0:
            ia_normalized = ia_series / total_ia
            # Cari indeks di mana kumulatif energi mencapai 5% dan 95%
            idx_5 = np.searchsorted(ia_normalized, 0.05)
            idx_95 = np.searchsorted(ia_normalized, 0.95)
            d_5_95 = float(time_array[idx_95] - time_array[idx_5])

        # 5. Konversi ke MMI dan SIG BMKG
        mmi = self._convert_pga_to_mmi(pga)
        sig_scale, sig_color, sig_desc = self._get_sig_bmkg(mmi)

        return StrongMotionData(
            trace_id=trace_id,
            pga=pga,
            pgv=pgv,
            pgd=pgd,
            arias_intensity=total_ia,
            duration_5_95=d_5_95,
            mmi=mmi,
            sig_bmkg_scale=sig_scale,
            sig_bmkg_color=sig_color,
            sig_bmkg_desc=sig_desc
        )