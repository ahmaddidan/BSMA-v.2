"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/sdof/newmark.py
Description: Linear SDOF Oscillator Solver using Newmark-Beta (Average Acceleration).
Optimized with memory-view array slicing to eliminate inside-loop allocations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["solve_newmark"]

FloatArray = NDArray[np.float64]


def solve_newmark(
    acceleration: FloatArray,
    dt: float,
    periods: FloatArray,
    damping: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Menyelesaikan persamaan gerak SDOF menggunakan Newmark-Beta.
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
        omega = 2.0 * np.pi / p_valid
        c = 2.0 * damping * omega
        k = omega ** 2

        a0 = 4.0 / (dt ** 2)
        a1 = 2.0 / dt
        a2 = 4.0 / dt
        a3 = 1.0
        a4 = 1.0

        k_hat = k + a0 + a1 * c

        # Workspace pre-allocation
        u_v = np.zeros((n_valid, n_samples), dtype=np.float64)
        v_v = np.zeros((n_valid, n_samples), dtype=np.float64)
        a_v = np.zeros((n_valid, n_samples), dtype=np.float64)

        a_v[:, 0] = -acceleration[0]

        # Vektorisasi eksekusi loop tanpa alokasi memori internal
        for i in range(n_samples - 1):
            u_i = u_v[:, i]
            v_i = v_v[:, i]
            a_i = a_v[:, i]

            p_i1 = -acceleration[i+1]

            p_hat = p_i1 + (a0 * u_i + a2 * v_i + a3 * a_i) + c * (a1 * u_i + a4 * v_i)

            u_i1 = p_hat / k_hat
            a_i1 = a0 * (u_i1 - u_i) - a2 * v_i - a3 * a_i
            v_i1 = v_i + 0.5 * dt * (a_i + a_i1)

            u_v[:, i+1] = u_i1
            v_v[:, i+1] = v_i1
            a_v[:, i+1] = a_i1

        u[valid_idx, :] = u_v
        v[valid_idx, :] = v_v
        a_abs[valid_idx, :] = a_v + acceleration

    zero_idx = ~valid_idx
    if np.any(zero_idx):
        a_abs[zero_idx, :] = acceleration

    return u, v, a_abs