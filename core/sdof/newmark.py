"""
BMKG Strong Motion Analyzer (BSMA)
Core SDOF Solver: Newmark-beta Method (Average Acceleration, beta=0.25, gamma=0.5)

Implementasi mutlak berbasis formulasi baku Anil K. Chopra (Dynamics of Structures, Chapter 5).
Menggunakan pendekatan Absolute State Update (Effective Load at t+dt -> Displacement -> Acceleration -> Velocity).
"""

import numpy as np

def solve_newmark(
    acc_ground: np.ndarray, 
    dt: float, 
    T: float, 
    damping: float
) -> tuple[float, float, float]:
    """
    Menyelesaikan persamaan gerak SDOF menggunakan Metode Newmark-beta (Average Acceleration).
    
    Parameters:
    - acc_ground : np.ndarray (Akselerasi tanah dalam m/s^2)
    - dt         : float (Interval waktu / delta t)
    - T          : float (Periode natural struktur dalam detik)
    - damping    : float (Rasio redaman, misal 0.05 untuk 5%)
    
    Returns:
    - max_sd  : Spectral Displacement (meter)
    - max_psv : Pseudo-Spectral Velocity (m/s)
    - max_psa : Pseudo-Spectral Acceleration (m/s^2)
    """
    # Kasus khusus: Struktur sangat kaku (Rigid Structure, T -> 0, PSA -> PGA)
    if T <= 1e-4:
        peak_acc = float(np.max(np.abs(acc_ground)))
        return 0.0, 0.0, peak_acc

    omega = 2.0 * np.pi / T
    omega2 = omega * omega
    
    m = 1.0
    c = 2.0 * damping * omega
    k = omega2

    beta = 0.25
    gamma = 0.5
    dt2 = dt * dt

    # 1. Konstanta Integrasi Newmark (Chopra Chapter 5)
    a0 = 1.0 / (beta * dt2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    # 2. Kekakuan Efektif (K_hat)
    k_hat = k + a1 * c + a0 * m

    npts = len(acc_ground)
    u = np.zeros(npts, dtype=np.float64)  # Perpindahan relatif
    v = np.zeros(npts, dtype=np.float64)  # Kecepatan relatif
    a = np.zeros(npts, dtype=np.float64)  # Akselerasi relatif

    # 3. Kondisi Awal (t = 0)
    # P_eff(0) = -m * ag(0)
    # m*a0 + c*v0 + k*u0 = P_eff(0) -> a0 = (-m*ag[0] - c*v0 - k*u0) / m (karena m=1)
    u[0] = 0.0
    v[0] = 0.0
    a[0] = -acc_ground[0] - c * v[0] - k * u[0]

    # 4. Step-by-Step Numerical Integration (Chopra Algorithm)
    for i in range(npts - 1):
        # Beban efektif pada t + dt
        p_eff_next = -m * acc_ground[i+1]
        
        # Beban efektif gabungan / terkoreksi (P_hat at i+1)
        p_hat_next = p_eff_next + m * (a0 * u[i] + a2 * v[i] + a3 * a[i]) + c * (a1 * u[i] + a4 * v[i] + a5 * a[i])
        
        # Solusi perpindahan absolut pada langkah berikutnya (u at i+1)
        u[i+1] = p_hat_next / k_hat
        
        # Pembaruan state akselerasi dan kecepatan absolut pada langkah berikutnya
        a[i+1] = a0 * (u[i+1] - u[i]) - a2 * v[i] - a3 * a[i]
        v[i+1] = v[i] + dt * ((1.0 - gamma) * a[i] + gamma * a[i+1])

    # 5. Ekstraksi Peak Response
    max_sd = float(np.max(np.abs(u)))
    max_psv = omega * max_sd
    max_psa = omega2 * max_sd

    return max_sd, max_psv, max_psa