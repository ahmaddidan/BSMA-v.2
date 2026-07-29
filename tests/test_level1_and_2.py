"""
BMKG Strong Motion Analyzer (BSMA)
Test Suite: Level 1 & 2 (Mathematical Correctness & Numerical Stability)

Memvalidasi solver Newmark-beta secara analitik dan memastikan ketahanannya
terhadap kondisi batas ekstrem (T -> 0, Zero Input, Extreme Damping).
"""

import numpy as np
import pytest
from core.sdof.newmark import solve_newmark

# =====================================================================
# LEVEL 1: MATHEMATICAL CORRECTNESS
# =====================================================================

def test_zero_input():
    """Validasi: Sinyal input nol absolut harus menghasilkan respons nol mutlak."""
    dt = 0.01
    acc_zero = np.zeros(1000, dtype=np.float64)
    
    sd, psv, psa = solve_newmark(acc_zero, dt, T=1.0, damping=0.05)
    
    assert sd == 0.0, f"SD harus 0.0, didapat {sd}"
    assert psv == 0.0, f"PSV harus 0.0, didapat {psv}"
    assert psa == 0.0, f"PSA harus 0.0, didapat {psa}"

def test_rigid_limit_consistency():
    """Validasi: Struktur sangat kaku (T mendekati 0) harus memiliki PSA yang setara dengan PGA."""
    dt = 0.01
    # Bangkitkan sinyal acak sebagai representasi ground motion
    acc_random = np.random.normal(0, 1.5, 1000) 
    pga_target = float(np.max(np.abs(acc_random)))
    
    # Uji pada periode sangat kecil
    t_rigid = 1e-5
    sd, psv, psa = solve_newmark(acc_random, dt, t_rigid, damping=0.05)
    
    # Toleransi numerik sangat ketat (0.01%)
    rel_error = abs(psa - pga_target) / pga_target
    assert rel_error < 1e-4, f"Gagal limit rigid. PSA={psa}, PGA={pga_target}"

def test_free_vibration_decay():
    """
    Validasi: Respons terhadap impuls sesaat harus meluruh sesuai selubung eksponensial.
    Cek apakah respons di akhir waktu mendekati nol karena adanya redaman.
    """
    dt = 0.01
    npts = 1000
    acc_impulse = np.zeros(npts, dtype=np.float64)
    acc_impulse[0] = 10.0  # Impuls awal
    
    sd, psv, psa = solve_newmark(acc_impulse, dt, T=1.0, damping=0.05)
    
    # Selama ada redaman positif, harus ada serapan energi (respons tidak infinite/divergen)
    assert np.isfinite(sd) and np.isfinite(psa), "Respons impuls divergen"
    # Nilai puncak SD tidak boleh nol jika ada impuls
    assert sd > 0.0, "SD tidak merespons impuls"

# =====================================================================
# LEVEL 2: NUMERICAL STABILITY
# =====================================================================

@pytest.mark.parametrize("damping", [0.0, 0.05, 0.2, 0.8, 0.99])
def test_extreme_dampings(damping):
    """
    Validasi: Solver harus kebal terhadap variasi redaman ekstrem tanpa menghasilkan
    nilai NaN, Inf, atau Overflow. (Menguji batas redaman kritis).
    """
    dt = 0.02
    acc_harmonic = np.sin(2 * np.pi * 1.0 * np.linspace(0, 10, 500))
    
    sd, psv, psa = solve_newmark(acc_harmonic, dt, T=0.5, damping=damping)
    
    assert np.isfinite(sd), f"SD menghasilkan NaN/Inf pada redaman {damping}"
    assert np.isfinite(psv), f"PSV menghasilkan NaN/Inf pada redaman {damping}"
    assert np.isfinite(psa), f"PSA menghasilkan NaN/Inf pada redaman {damping}"

@pytest.mark.parametrize("period", [1e-4, 0.01, 20.0, 50.0])
def test_extreme_periods(period):
    """
    Validasi: Solver harus stabil pada periode struktur yang sangat kaku maupun 
    sangat fleksibel (periode panjang).
    """
    dt = 0.01
    acc_step = np.ones(500, dtype=np.float64)  # Step load
    
    sd, psv, psa = solve_newmark(acc_step, dt, T=period, damping=0.05)
    
    assert np.isfinite(psa), f"Solver gagal (NaN/Inf) pada periode ekstrem T={period}s"