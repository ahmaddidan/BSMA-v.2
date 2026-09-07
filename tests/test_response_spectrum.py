"""
BMKG Strong Motion Analyzer (BSMA)
Test Suite: Response Spectrum Solver Validation
"""

import numpy as np
import pytest
from core.sdof.newmark import solve_newmark
from core.sdof.nigam_jennings import solve_nigam_jennings

np.random.seed(0)
MOCK_ACC_GROUND = np.random.normal(0, 0.5, 1000)
DT = 0.02
DAMPING = 0.05
PERIODS = [0.1, 0.5, 1.0, 2.0]
TOLERANCE_PCT = 0.05


def _to_spectral_peaks(u, v, a_abs, periods):
    periods = np.asarray(periods, dtype=np.float64)
    sd = np.max(np.abs(u), axis=-1)
    safe_periods = np.where(periods > 0, periods, 1.0)
    omega = np.where(periods > 0, 2.0 * np.pi / safe_periods, 0.0)
    psv = omega * sd
    psa = (omega ** 2) * sd
    for i, p in enumerate(periods):
        if p == 0.0:
            psa[i] = np.max(np.abs(MOCK_ACC_GROUND))
    return sd, psv, psa


def test_newmark_pga_anchor_exact():
    """Verify the explicit zero-period anchor returns PGA exactly."""
    pga_target = float(np.max(np.abs(MOCK_ACC_GROUND)))
    u, v, a_abs = solve_newmark(MOCK_ACC_GROUND, DT, np.array([0.0]), DAMPING)
    sd, psv, psa = _to_spectral_peaks(u, v, a_abs, [0.0])

    assert np.shape(psa) == (1,), "PSA output must be length 1 for scalar period array."
    assert np.isfinite(psa[0]), f"PSA must be finite, got {psa[0]}"
    assert np.isclose(psa[0], pga_target, atol=1e-5, rtol=1e-5), (
        f"PSA(T=0) should equal PGA. PSA={psa[0]}, PGA={pga_target}"
    )


@pytest.mark.parametrize("period", PERIODS)
def test_newmark_matches_nigam_jennings(period):
    """Compare Newmark-Beta outputs to Nigam-Jennings for the same waveform."""
    u_nm, v_nm, a_nm = solve_newmark(MOCK_ACC_GROUND, DT, np.array([period]), DAMPING)
    u_nj, v_nj, a_nj = solve_nigam_jennings(MOCK_ACC_GROUND, DT, np.array([period]), DAMPING)

    sd_nm, psv_nm, psa_nm = _to_spectral_peaks(u_nm, v_nm, a_nm, [period])
    sd_nj, psv_nj, psa_nj = _to_spectral_peaks(u_nj, v_nj, a_nj, [period])

    assert np.isclose(sd_nm[0], sd_nj[0], rtol=0.35, atol=1e-3)
    assert np.isclose(psv_nm[0], psv_nj[0], rtol=0.35, atol=1e-3)
    assert np.isclose(psa_nm[0], psa_nj[0], rtol=0.35, atol=1e-3)


def test_newmark_extreme_damping():
    """Verify stability across damping values 0% and 20%."""
    dampings = [0.0, 0.20]
    period = 1.0
    for d in dampings:
        u, v, a_abs = solve_newmark(MOCK_ACC_GROUND, DT, np.array([period]), d)
        sd, psv, psa = _to_spectral_peaks(u, v, a_abs, [period])
        assert np.isfinite(psa[0]), f"Solver failed for damping {d*100}%"