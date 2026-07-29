"""
BMKG Strong Motion Analyzer (BSMA)
Test Suite: Level 4 (Cross-Solver Validation on Real Earthquake Data)

Menguji solver numerik menggunakan rekaman data seismik riil (.mseed).
Karena solusi analitik tidak ada untuk sinyal acak gempa, kita menjadikan
solusi eksak Nigam-Jennings sebagai "Ground Truth" internal, dan mengukur
seberapa besar deviasi/error yang dihasilkan oleh metode aproksimasi Newmark.
"""

import numpy as np
import pytest
from pathlib import Path
import obspy

from core.sdof.newmark import solve_newmark
from core.sdof.nigam_jennings import solve_nigam_jennings

# =====================================================================
# KONFIGURASI FILE DATA (.MSEED)
# =====================================================================
# Sesuaikan nama file ini dengan file mseed yang ada di folder Data/mseed Anda
FILE_NAME = "20260127061533_IA_BBJM_00HNE_BP4_0.1_40.mseed"

# Mencari jalur dinamis ke folder Data di root proyek
PROJECT_ROOT = Path(__file__).parent.parent
FILE_PATH = PROJECT_ROOT / "Data" / "mseed" / FILE_NAME

@pytest.fixture
def real_earthquake_data():
    """Fixture untuk memuat data mseed secara otomatis sebelum pengujian."""
    if not FILE_PATH.exists():
        pytest.skip(f"File data riil tidak ditemukan di: {FILE_PATH}")
    
    # Baca data menggunakan ObsPy
    st = obspy.read(str(FILE_PATH))
    trace = st[0]  # Ambil komponen/trace pertama
    
    # Asumsikan data sudah berupa akselerasi. Pastikan tipe datanya float64
    acc = trace.data.astype(np.float64)
    dt = trace.stats.delta
    
    return acc, dt

# =====================================================================
# UJI SILANG (CROSS-VALIDATION) SOLVER
# =====================================================================
@pytest.mark.parametrize("period", [0.2, 1.0, 3.0])
def test_cross_solver_on_real_data(real_earthquake_data, period):
    """
    Mengadu Newmark vs Nigam-Jennings pada rekaman gempa riil.
    Kita menetapkan hasil Nigam-Jennings sebagai target kebenaran.
    """
    acc, dt = real_earthquake_data
    damping = 0.05  # Damping standar 5%
    
    # Eksekusi kedua solver
    sd_nj, psv_nj, psa_nj = solve_nigam_jennings(acc, dt, period, damping)
    sd_nm, psv_nm, psa_nm = solve_newmark(acc, dt, period, damping)
    
    # Hitung error relatif Newmark terhadap Nigam-Jennings
    err_psa = abs(psa_nm - psa_nj) / psa_nj if psa_nj > 0 else 0.0
    
    print("\n" + "="*60)
    print(f"BENCHMARK GEMPA RIIL (Periode T = {period}s, Damping 5%)")
    print("="*60)
    print(f"PSA (Nigam-Jennings) : {psa_nj:.8f} m/s^2 (Exact Target)")
    print(f"PSA (Newmark-beta)   : {psa_nm:.8f} m/s^2")
    print(f"Relative Error       : {err_psa:.4%}")
    print("="*60)
    
    # Peringatan dini jika Newmark mulai menyimpang terlalu jauh (toleransi 1%)
    assert err_psa < 0.01, f"Deviasi Newmark melebihi 1% pada data riil! Error: {err_psa:.2%}"