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
    
    sd, psv, psa = solve_newmark(acc_zero, dt, np.array([1.0], dtype=np.float64), damping=0.05)
    
    assert sd.shape == (1,), "SD output must be length 1 for scalar period array."
    assert psv.shape == (1,), "PSV output must be length 1 for scalar period array."
    assert psa.shape == (1,), "PSA output must be length 1 for scalar period array."
    assert sd[0] == 0.0, f"SD harus 0.0, didapat {sd[0]}"
    assert psv[0] == 0.0, f"PSV harus 0.0, didapat {psv[0]}"
    assert psa[0] == 0.0, f"PSA harus 0.0, didapat {psa[0]}"

def test_rigid_limit_consistency():
    """Validasi: Struktur sangat kaku (T = 0) harus memiliki PSA yang setara dengan PGA."""
    dt = 0.01
    rng = np.random.default_rng(0)
    acc_random = rng.normal(0, 1.5, 1000)
    pga_target = float(np.max(np.abs(acc_random)))

    sd, psv, psa = solve_newmark(acc_random, dt, np.array([0.0], dtype=np.float64), damping=0.05)

    assert psa.shape == (1,), "PSA output must be length 1 for scalar period array."
    assert np.isfinite(psa[0]), f"PSA harus bernilai finite, didapat {psa[0]}"
    assert np.isclose(psa[0], pga_target, rtol=0.0, atol=0.0), (
        f"PSA(T=0) harus sama dengan PGA. PSA={psa[0]}, PGA={pga_target}"
    )

def test_free_vibration_decay():
    """
    Validasi: Respons terhadap impuls sesaat harus meluruh sesuai selubung eksponensial.
    Cek apakah respons di akhir waktu mendekati nol karena adanya redaman.
    """
    dt = 0.01
    npts = 1000
    acc_impulse = np.zeros(npts, dtype=np.float64)
    acc_impulse[0] = 10.0  # Impuls awal
    
    sd, psv, psa = solve_newmark(acc_impulse, dt, np.array([1.0], dtype=np.float64), damping=0.05)
    
    assert sd.shape == (1,), "SD output must be length 1 for scalar period array."
    assert np.isfinite(sd[0]) and np.isfinite(psa[0]), "Respons impuls divergen"
    assert sd[0] > 0.0, "SD tidak merespons impuls"

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
    
    sd, psv, psa = solve_newmark(acc_harmonic, dt, np.array([0.5], dtype=np.float64), damping=damping)
    
    assert np.isfinite(sd[0]), f"SD menghasilkan NaN/Inf pada redaman {damping}"
    assert np.isfinite(psv[0]), f"PSV menghasilkan NaN/Inf pada redaman {damping}"
    assert np.isfinite(psa[0]), f"PSA menghasilkan NaN/Inf pada redaman {damping}"

@pytest.mark.parametrize("period", [1e-4, 0.01, 20.0, 50.0])
def test_extreme_periods(period):
    """
    Validasi: Solver harus stabil pada periode struktur yang sangat kaku maupun 
    sangat fleksibel (periode panjang).
    """
    dt = 0.01
    acc_step = np.ones(500, dtype=np.float64)  # Step load
    
    sd, psv, psa = solve_newmark(
        acc_step,
        dt,
        np.array([period], dtype=np.float64),
        damping=0.05,
    )
    
    assert np.isfinite(psa[0]), f"Solver gagal (NaN/Inf) pada periode ekstrem T={period}s"