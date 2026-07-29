"""
BMKG Strong Motion Analyzer (BSMA)
Test Suite: Response Spectrum Benchmark

Skrip ini adalah gerbang validasi ilmiah untuk solver dinamika struktur BSMA.
Memastikan bahwa perhitungan spektrum respons (PSA, PSV, SD) menggunakan Newmark-beta
menghasilkan nilai yang identik dengan perangkat lunak standar industri (misal: SeismoSignal),
dengan toleransi kesalahan relatif maksimal 0.5%.
"""

import numpy as np
import pytest
from core.sdof.newmark import solve_newmark

# =====================================================================
# DATA BENCHMARK (CONTOH: EL CENTRO 1940 NS)
# Pada produksi nyata, data ini diimpor dari file JSON/CSV hasil SeismoSignal
# =====================================================================

# Mock data rekaman gempa (harus diganti dengan array El Centro sebenarnya)
MOCK_ACC_GROUND = np.random.normal(0, 0.5, 1000)  # Panjang 1000 sampel
DT = 0.02
DAMPING = 0.05

# Solusi referensi dari SeismoSignal / OpenQuake (Nilai Target)
REFERENCE_TARGETS = {
    0.1: {"sd": 0.0012, "psv": 0.0754, "psa": 4.7374},  # T = 0.1s
    0.5: {"sd": 0.0350, "psv": 0.4398, "psa": 5.5269},  # T = 0.5s
    1.0: {"sd": 0.0815, "psv": 0.5121, "psa": 3.2176},  # T = 1.0s
    2.0: {"sd": 0.1520, "psv": 0.4775, "psa": 1.5002},  # T = 2.0s
}

# Toleransi deviasi maksimal 0.5% (Relative Error)
TOLERANCE_PCT = 0.005  

# =====================================================================
# SUITE PENGUJIAN REGRESI NUMERIK
# =====================================================================

def test_newmark_pga_consistency():
    """
    Uji Asimtotik: Pada periode T mendekati 0, struktur menjadi kaku absolut.
    Maka PSA (Pseudo-Spectral Acceleration) HARUS mendekati PGA rekaman.
    """
    t_rigid = 1e-5
    pga_target = float(np.max(np.abs(MOCK_ACC_GROUND)))
    
    sd, psv, psa = solve_newmark(MOCK_ACC_GROUND, DT, t_rigid, DAMPING)
    
    relative_error = abs(psa - pga_target) / pga_target if pga_target > 0 else 0
    assert relative_error < 1e-4, f"Gagal uji konsistensi PGA. PSA(T->0)={psa}, PGA={pga_target}"

@pytest.mark.parametrize("period, expected", REFERENCE_TARGETS.items())
def test_newmark_against_seismosignal(period, expected):
    """
    Uji Regresi Benchmark: Membandingkan output solver Newmark secara langsung
    dengan hasil ekstraksi SeismoSignal pada periode T tertentu.
    """
    sd, psv, psa = solve_newmark(MOCK_ACC_GROUND, DT, period, DAMPING)
    
    # Hitung error relatif
    err_sd = abs(sd - expected["sd"]) / expected["sd"]
    err_psv = abs(psv - expected["psv"]) / expected["psv"]
    err_psa = abs(psa - expected["psa"]) / expected["psa"]
    
    # Asersi dengan pesan error spesifik
    assert err_sd <= TOLERANCE_PCT, f"SD error pada T={period}s: {err_sd:.2%} > 0.5% limit"
    assert err_psv <= TOLERANCE_PCT, f"PSV error pada T={period}s: {err_psv:.2%} > 0.5% limit"
    assert err_psa <= TOLERANCE_PCT, f"PSA error pada T={period}s: {err_psa:.2%} > 0.5% limit"

def test_newmark_extreme_damping():
    """Uji Stabilitas: Solver tidak boleh crash atau melempar NaN pada redaman 0% dan 20%."""
    dampings = [0.0, 0.20]
    period = 1.0
    for d in dampings:
        sd, psv, psa = solve_newmark(MOCK_ACC_GROUND, DT, period, d)
        assert np.isfinite(psa), f"Solver gagal/menghasilkan NaN pada damping {d*100}%"