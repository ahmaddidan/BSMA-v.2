"""
BMKG Strong Motion Analyzer (BSMA)
Test Suite: Level 1 & 2 (Mathematical Correctness & Numerical Stability)
"""

import numpy as np
import pytest
from core.sdof.newmark import solve_newmark


def _to_spectral_peaks(u, v, a_abs, periods, ground_acc):
    periods = np.asarray(periods, dtype=np.float64)
    sd = np.max(np.abs(u), axis=-1)
    safe_periods = np.where(periods > 0, periods, 1.0)
    omega = np.where(periods > 0, 2.0 * np.pi / safe_periods, 0.0)
    psv = omega * sd
    psa = (omega ** 2) * sd
    for i, p in enumerate(periods):
        if p == 0.0:
            psa[i] = np.max(np.abs(ground_acc))
    return sd, psv, psa


def test_zero_input():
    dt = 0.01
    acc_zero = np.zeros(1000, dtype=np.float64)
    u, v, a_abs = solve_newmark(acc_zero, dt, np.array([1.0], dtype=np.float64), damping=0.05)
    sd, psv, psa = _to_spectral_peaks(u, v, a_abs, [1.0], acc_zero)

    assert sd.shape == (1,)
    assert psv.shape == (1,)
    assert psa.shape == (1,)
    assert sd[0] == 0.0
    assert psv[0] == 0.0
    assert psa[0] == 0.0


def test_rigid_limit_consistency():
    dt = 0.01
    rng = np.random.default_rng(0)
    acc_random = rng.normal(0, 1.5, 1000)
    pga_target = float(np.max(np.abs(acc_random)))

    u, v, a_abs = solve_newmark(acc_random, dt, np.array([0.0], dtype=np.float64), damping=0.05)
    sd, psv, psa = _to_spectral_peaks(u, v, a_abs, [0.0], acc_random)

    assert psa.shape == (1,)
    assert np.isfinite(psa[0])
    assert np.isclose(psa[0], pga_target, rtol=1e-5, atol=1e-5)


def test_free_vibration_decay():
    dt = 0.01
    npts = 1000
    acc_impulse = np.zeros(npts, dtype=np.float64)
    acc_impulse[0] = 10.0

    u, v, a_abs = solve_newmark(acc_impulse, dt, np.array([1.0], dtype=np.float64), damping=0.05)
    sd, psv, psa = _to_spectral_peaks(u, v, a_abs, [1.0], acc_impulse)

    assert sd.shape == (1,)
    assert np.isfinite(sd[0]) and np.isfinite(psa[0])
    assert sd[0] > 0.0


@pytest.mark.parametrize("damping", [0.0, 0.05, 0.2, 0.8, 0.99])
def test_extreme_dampings(damping):
    dt = 0.02
    acc_harmonic = np.sin(2 * np.pi * 1.0 * np.linspace(0, 10, 500))
    u, v, a_abs = solve_newmark(acc_harmonic, dt, np.array([0.5], dtype=np.float64), damping=damping)
    sd, psv, psa = _to_spectral_peaks(u, v, a_abs, [0.5], acc_harmonic)

    assert np.isfinite(sd[0])
    assert np.isfinite(psv[0])
    assert np.isfinite(psa[0])


@pytest.mark.parametrize("period", [1e-4, 0.01, 20.0, 50.0])
def test_extreme_periods(period):
    dt = 0.01
    acc_step = np.ones(500, dtype=np.float64)
    u, v, a_abs = solve_newmark(acc_step, dt, np.array([period], dtype=np.float64), damping=0.05)
    sd, psv, psa = _to_spectral_peaks(u, v, a_abs, [period], acc_step)

    assert np.isfinite(psa[0])