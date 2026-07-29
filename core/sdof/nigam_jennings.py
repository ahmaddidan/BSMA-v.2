"""
BMKG Strong Motion Analyzer (BSMA)
Core SDOF Solver: Nigam & Jennings (1969) Exact Recursive Method

Implementasi eksak berbasis algoritma rekursif Navin C. Nigam & Paul C. Jennings (1969).
Makalah Referensi: "Calculation of Response Spectra from Strong-Motion Earthquake Records",
Bulletin of the Seismological Society of America, Vol. 59, No. 2, pp. 909-922, April 1969.

Modul ini menggunakan formulasi Matriks A dan B yang didefinisikan secara eksplisit
pada Lampiran (Appendix) makalah tersebut. Formulasi ini memberikan solusi eksak 
tanpa truncation error, dengan asumsi akselerasi tanah bervariasi linear di antara 
titik-titik sampel (Piecewise Linear Exact Method).
"""

import numpy as np

def solve_nigam_jennings(
    acc_ground: np.ndarray, 
    dt: float, 
    T: float, 
    damping: float
) -> tuple[float, float, float]:
    """
    Menyelesaikan respons SDOF menggunakan algoritma eksak Nigam & Jennings (1969).
    
    Parameters:
    - acc_ground : np.ndarray (Akselerasi tanah dalam m/s^2)
    - dt         : float (Interval waktu / delta t)
    - T          : float (Periode natural struktur dalam detik)
    - damping    : float (Rasio redaman, zeta)
    
    Returns:
    - max_sd  : Spectral Displacement (meter)
    - max_psv : Pseudo-Spectral Velocity (m/s)
    - max_psa : Pseudo-Spectral Acceleration (m/s^2)
    """
    # Limit struktur kaku mutlak (Rigid Structure, T -> 0)
    if T <= 1e-5:
        pga = float(np.max(np.abs(acc_ground)))
        return 0.0, 0.0, pga

    # Algoritma ini dirumuskan khusus untuk struktur Underdamped (zeta < 1.0).
    # Untuk menghindari Division by Zero (pada akar kuadrat 1 - zeta^2), kita batasi nilainya.
    if damping >= 1.0:
        damping = 0.9999

    # Parameter Dasar
    w = 2.0 * np.pi / T
    w2 = w * w
    w3 = w2 * w
    z = damping

    wd = w * np.sqrt(1.0 - z * z)
    z_sq = z / np.sqrt(1.0 - z * z)

    E = np.exp(-z * w * dt)
    S = np.sin(wd * dt)
    C = np.cos(wd * dt)

    # ==============================================================================
    # Matriks Transisi Keadaan (A Matrix) - Nigam & Jennings (1969)
    # ==============================================================================
    a11 = E * (C + z_sq * S)
    a12 = (E / wd) * S
    a21 = -(w / np.sqrt(1.0 - z * z)) * E * S
    a22 = E * (C - z_sq * S)

    # ==============================================================================
    # Matriks Koefisien Eksitasi (B Matrix) - Nigam & Jennings (1969) Appendix
    # ==============================================================================
    # Suku bantu untuk mencegah pengulangan komputasi berlebihan
    term1 = (2.0 * z * z - 1.0) / (w2 * dt)
    term2 = 2.0 * z / (w3 * dt)

    # b11 dan b12 (Koefisien untuk update Displacement)
    b11 = E * ((term1 + z / w) * (S / wd) + (term2 + 1.0 / w2) * C) - term2
    b12 = -E * (term1 * (S / wd) + term2 * C) - 1.0 / w2 + term2

    # b21 dan b22 (Koefisien untuk update Velocity)
    b21 = E * ((term1 + z / w) * (C - z_sq * S) - (term2 + 1.0 / w2) * (wd * S + z * w * C)) + 1.0 / (w2 * dt)
    b22 = -E * (term1 * (C - z_sq * S) - term2 * (wd * S + z * w * C)) - 1.0 / (w2 * dt)

    # ==============================================================================
    # Iterasi Rekursif (Step-by-Step Exact State Update)
    # ==============================================================================
    npts = len(acc_ground)
    u = np.zeros(npts, dtype=np.float64)  # Perpindahan relatif
    v = np.zeros(npts, dtype=np.float64)  # Kecepatan relatif

    # Kondisi Awal: u[0] = 0, v[0] = 0
    u[0] = 0.0
    v[0] = 0.0

    # Iterasi eksak O(N) tanpa error akumulasi integrasi (truncation error)
    for i in range(npts - 1):
        ag_i = acc_ground[i]
        ag_ip1 = acc_ground[i+1]

        u[i+1] = a11 * u[i] + a12 * v[i] + b11 * ag_i + b12 * ag_ip1
        v[i+1] = a21 * u[i] + a22 * v[i] + b21 * ag_i + b22 * ag_ip1

    # ==============================================================================
    # Ekstraksi Parameter Output (Peak Values)
    # ==============================================================================
    max_sd = float(np.max(np.abs(u)))
    max_psv = w * max_sd
    max_psa = w2 * max_sd

    return max_sd, max_psv, max_psa