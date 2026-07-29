"""
BMKG Strong Motion Analyzer (BSMA)
Test Suite: Level 3 (Analytical Benchmark)

Membenturkan solver numerik Newmark dan Nigam-Jennings secara langsung dengan 
Solusi Analitik Eksak (Kebenaran Mutlak Matematika). 
Ini akan membuktikan keunggulan algoritma rekursif Nigam-Jennings.
"""

import numpy as np
import pytest
from core.sdof.newmark import solve_newmark
from core.sdof.nigam_jennings import solve_nigam_jennings

def test_step_load_undamped_accuracy():
    """
    Analytical Benchmark 1: Undamped Step Load (Beban Konstan).
    Secara teori, respons maksimum perpindahan untuk beban konstan A0 
    pada struktur tanpa redaman adalah tepat 2 * A0 / w^2.
    """
    dt = 0.02           # Sampling rate standar gempa (50 Hz)
    T = 1.0             # Periode struktur 1 detik
    damping = 0.0       # Tanpa redaman
    A0 = 1.0            # Akselerasi tanah konstan 1 m/s^2
    
    # Bangkitkan Sinyal Step Load selama 10 detik
    time_array = np.arange(0, 10, dt)
    acc_step = np.full_like(time_array, A0)
    
    # 1. Hitung Solusi Eksak (Buku Teks)
    w = 2.0 * np.pi / T
    exact_sd = 2.0 * A0 / (w**2)
    
    # 2. Hitung Solusi Numerik Menggunakan Kedua Solver
    sd_newmark, _, _ = solve_newmark(acc_step, dt, T, damping)
    sd_nigam, _, _ = solve_nigam_jennings(acc_step, dt, T, damping)
    
    # 3. Hitung Error Relatif
    err_newmark = abs(sd_newmark - exact_sd) / exact_sd
    err_nigam = abs(sd_nigam - exact_sd) / exact_sd

    # Cetak hasil ke terminal agar kita bisa melihat perbedaannya secara langsung
    print("\n" + "="*50)
    print("HASIL UJI ANALITIK (STEP LOAD - T=1.0s, dt=0.02s)")
    print("="*50)
    print(f"Nilai Matematika Eksak : {exact_sd:.8f} m")
    print(f"Solver Nigam-Jennings  : {sd_nigam:.8f} m (Error: {err_nigam:.6%})")
    print(f"Solver Newmark-beta    : {sd_newmark:.8f} m (Error: {err_newmark:.6%})")
    print("="*50)

    # Asersi Ilmiah: Nigam-Jennings harus memiliki tingkat error mendekati nol absolut
    assert err_nigam < 1e-5, f"Nigam-Jennings gagal mencapai presisi eksak. Error: {err_nigam:.4%}"
    
    # Kita tidak melakukan assert (penggagalan) pada Newmark di sini, 
    # karena kita tahu Newmark pasti memiliki error lebih besar.
    # Kita hanya ingin membandingkannya.