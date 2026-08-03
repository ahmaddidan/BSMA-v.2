"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/sdof/nigam_jennings.py
Description: Exact Analytical SDOF Oscillator Solver (Nigam & Jennings, 1968).
Uses the piecewise exact method for linear interpolation of ground excitation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["solve_nigam_jennings"]

FloatArray = NDArray[np.float64]


def solve_nigam_jennings(
    acceleration: FloatArray,
    dt: float,
    periods: FloatArray,
    damping: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Menyelesaikan persamaan gerak SDOF menggunakan metode analitik eksak
    Nigam-Jennings (Piecewise Linear Exact Method).
    """
    n_samples = acceleration.size
    n_periods = periods.size

    u = np.zeros((n_periods, n_samples), dtype=np.float64)
    v = np.zeros((n_periods, n_samples), dtype=np.float64)
    a_abs = np.zeros((n_periods, n_samples), dtype=np.float64)

    valid_idx = periods > 0.0
    p_valid = periods[valid_idx]
    n_valid = p_valid.size

    if n_valid > 0:
        z = damping
        w = 2.0 * np.pi / p_valid
        w2 = w ** 2
        wd = w * np.sqrt(1.0 - z ** 2)
        
        # Eksponensial dan Trigonometri
        E = np.exp(-z * w * dt)
        S = np.sin(wd * dt)
        C = np.cos(wd * dt)

        z_s_term = z / np.sqrt(1.0 - z ** 2)
        w_s_term = w / np.sqrt(1.0 - z ** 2)

        # Matriks Transisi Keadaan (State Transition Matrices)
        A11 = E * (z_s_term * S + C)
        A12 = E * (S / wd)
        A21 = -E * (w_s_term * S)
        A22 = E * (C - z_s_term * S)

        # Konstanta Solusi Partikular (Particular Solution Matrices)
        # Menghindari pembagian berulang di dalam loop
        term1 = (2.0 * z) / (w * dt)
        term2 = (1.0 - 2.0 * z ** 2) / (wd * dt) - z_s_term
        term3 = (2.0 * z ** 2 - 1.0) / (wd * dt)
        
        B11 = (term1 + E * (term2 * S - (1.0 + term1) * C)) / w2
        B12 = (1.0 - term1 + E * (term3 * S + term1 * C)) / w2

        term4 = w_s_term + z / (dt * np.sqrt(1.0 - z ** 2))
        term5 = z / (dt * np.sqrt(1.0 - z ** 2))

        B21 = (-1.0/dt + E * (term4 * S + C/dt)) / w2
        B22 = (1.0/dt - E * (term5 * S + C/dt)) / w2

        # Workspace pre-allocation
        u_v = np.zeros((n_valid, n_samples), dtype=np.float64)
        v_v = np.zeros((n_valid, n_samples), dtype=np.float64)

        # Loop Time-Stepping O(1) Memory
        for i in range(n_samples - 1):
            u_i = u_v[:, i]
            v_i = v_v[:, i]
            
            # Percepatan tanah diubah menjadi gaya efektif (p = -ag)
            p_i = -acceleration[i]
            p_i1 = -acceleration[i+1]

            # Update State secara Rekursif
            u_v[:, i+1] = A11 * u_i + A12 * v_i + B11 * p_i + B12 * p_i1
            v_v[:, i+1] = A21 * u_i + A22 * v_i + B21 * p_i + B22 * p_i1

        u[valid_idx, :] = u_v
        v[valid_idx, :] = v_v
        
        # Secara fisis: a_abs = a_rel + ag = -2*zeta*w*v_rel - w^2*u_rel
        # Menggunakan broadcasting otomatis dari numpy untuk perkalian elemen matriks [M,N] x [M,1]
        v_term = 2.0 * z * w[:, np.newaxis] * v_v
        u_term = w2[:, np.newaxis] * u_v
        a_abs[valid_idx, :] = -v_term - u_term

    zero_idx = ~valid_idx
    if np.any(zero_idx):
        a_abs[zero_idx, :] = acceleration

    return u, v, a_abs