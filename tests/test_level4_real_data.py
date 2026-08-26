"""
BMKG Strong Motion Analyzer (BSMA)
Test Suite: Level 4 (Cross-Solver Validation on Real Earthquake Data)
"""

from pathlib import Path
import numpy as np
import obspy
import pytest

from core.sdof.newmark import solve_newmark
from core.sdof.nigam_jennings import solve_nigam_jennings

FILE_NAME = "20260205180614_IA_BBJM_00HNE_BP4_0.05_40.mseed"
PROJECT_ROOT = Path(__file__).parent.parent
FILE_PATH = PROJECT_ROOT / "Data" / "mseed" / FILE_NAME


@pytest.fixture
def real_earthquake_data():
    if not FILE_PATH.exists():
        pytest.skip(f"Real data file not found: {FILE_PATH}")
    st = obspy.read(str(FILE_PATH))
    trace = st[0]
    acc = trace.data.astype(np.float64)
    dt = float(trace.stats.delta)
    return acc, dt


@pytest.mark.parametrize("period", [0.2, 1.0, 3.0])
def test_cross_solver_on_real_data(real_earthquake_data, period):
    acc, dt = real_earthquake_data
    damping = 0.05

    u_nj, v_nj, a_nj = solve_nigam_jennings(acc, dt, np.array([period]), damping)
    u_nm, v_nm, a_nm = solve_newmark(acc, dt, np.array([period]), damping)

    sd_nj = float(np.max(np.abs(u_nj[0])))
    sd_nm = float(np.max(np.abs(u_nm[0])))

    error_relatif = abs(sd_nm - sd_nj) / sd_nj
    assert error_relatif < 0.05, f"Cross-solver mismatch at T={period}s: {error_relatif:.4%}"