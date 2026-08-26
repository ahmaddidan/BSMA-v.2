"""
BMKG Strong Motion Analyzer (BSMA)
Test Suite: Level 3 (Analytical Benchmark)
"""

import numpy as np
import pytest
from core.sdof.newmark import solve_newmark
from core.sdof.nigam_jennings import solve_nigam_jennings


def test_step_load_undamped_accuracy():
    """Analytical Benchmark 1: Undamped Step Load (Constant Acceleration)."""
    dt = 0.02
    T = 1.0
    damping = 0.0
    A0 = 1.0

    time_array = np.arange(0, 10, dt)
    acc_step = np.full_like(time_array, A0)

    w = 2.0 * np.pi / T
    exact_sd = 2.0 * A0 / (w ** 2)

    u_nm, _, _ = solve_newmark(acc_step, dt, np.array([T], dtype=np.float64), damping)
    u_nj, _, _ = solve_nigam_jennings(acc_step, dt, np.array([T], dtype=np.float64), damping)

    sd_newmark = float(np.max(np.abs(u_nm[0])))
    sd_nigam = float(np.max(np.abs(u_nj[0])))

    err_newmark = abs(sd_newmark - exact_sd) / exact_sd
    err_nigam = abs(sd_nigam - exact_sd) / exact_sd

    assert err_nigam < 0.05, f"Nigam-Jennings error {err_nigam:.6%} exceeds 5%"
    assert err_newmark < 0.05, f"Newmark error {err_newmark:.6%} exceeds 5%"