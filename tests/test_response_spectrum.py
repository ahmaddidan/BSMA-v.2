"""
BMKG Strong Motion Analyzer (BSMA)
Test Suite: Response Spectrum Solver Validation

This suite validates the Newmark-Beta SDOF solver against a seeded synthetic
waveform and cross-checks its spectral outputs against the analytical
Nigam-Jennings solver.
"""

import numpy as np
import pytest
from core.sdof.newmark import solve_newmark
from core.sdof.nigam_jennings import solve_nigam_jennings

# =====================================================================
# DATA SET
# =====================================================================
np.random.seed(0)
MOCK_ACC_GROUND = np.random.normal(0, 0.5, 1000)
DT = 0.02
DAMPING = 0.05
PERIODS = [0.1, 0.5, 1.0, 2.0]
TOLERANCE_PCT = 0.01

# =====================================================================
# TEST SUITE
# =====================================================================

def test_newmark_pga_anchor_exact():
    """Verify the explicit zero-period anchor returns PGA exactly."""
    pga_target = float(np.max(np.abs(MOCK_ACC_GROUND)))
    sd, psv, psa = solve_newmark(MOCK_ACC_GROUND, DT, np.array([0.0]), DAMPING)

    assert np.shape(psa) == (1,), "PSA output must be length 1 for scalar period array."
    assert np.isfinite(psa[0]), f"PSA must be finite, got {psa[0]}"
    assert np.isclose(psa[0], pga_target, atol=0.0, rtol=0.0), (
        f"PSA(T=0) should equal PGA. PSA={psa[0]}, PGA={pga_target}"
    )

@pytest.mark.parametrize("period", PERIODS)
def test_newmark_matches_nigam_jennings(period):
    """Compare Newmark-Beta outputs to Nigam-Jennings for the same waveform."""
    sd_nm, psv_nm, psa_nm = solve_newmark(
        MOCK_ACC_GROUND,
        DT,
        np.asarray([period], dtype=np.float64),
        DAMPING,
    )
    sd_nj, psv_nj, psa_nj = solve_nigam_jennings(
        MOCK_ACC_GROUND,
        DT,
        np.asarray([period], dtype=np.float64),
        DAMPING,
    )

    assert np.isclose(sd_nm[0], sd_nj[0], rtol=TOLERANCE_PCT, atol=0.0), (
        f"SD mismatch at T={period}s: Newmark={sd_nm[0]}, Nigam-Jennings={sd_nj[0]}"
    )
    assert np.isclose(psv_nm[0], psv_nj[0], rtol=TOLERANCE_PCT, atol=0.0), (
        f"PSV mismatch at T={period}s: Newmark={psv_nm[0]}, Nigam-Jennings={psv_nj[0]}"
    )
    assert np.isclose(psa_nm[0], psa_nj[0], rtol=TOLERANCE_PCT, atol=0.0), (
        f"PSA mismatch at T={period}s: Newmark={psa_nm[0]}, Nigam-Jennings={psa_nj[0]}"
    )

def test_newmark_extreme_damping():
    """Uji Stabilitas: Solver tidak boleh crash atau melempar NaN pada redaman 0% dan 20%."""
    dampings = [0.0, 0.20]
    period = 1.0
    for d in dampings:
        sd, psv, psa = solve_newmark(
            MOCK_ACC_GROUND,
            DT,
            np.array([period], dtype=np.float64),
            d,
        )
        assert np.isfinite(psa[0]), f"Solver gagal/menghasilkan NaN pada damping {d*100}%"