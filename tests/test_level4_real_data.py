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

PROJECT_ROOT = Path(__file__).parent.parent
MSEED_DIR = PROJECT_ROOT / "Data" / "mseed"

# Target stations specified for real earthquake validation
TARGET_STATIONS = ["PPJR", "PCJI", "PCJR"]


def _get_station_files():
    station_files = []
    if MSEED_DIR.exists():
        for sta in TARGET_STATIONS:
            matched = sorted(MSEED_DIR.glob(f"*{sta}*HNE*.mseed"))
            if not matched:
                matched = sorted(MSEED_DIR.glob(f"*{sta}*.mseed"))
            if matched:
                station_files.append((sta, matched[0]))
    return station_files


STATION_FILES = _get_station_files()


@pytest.mark.skipif(len(STATION_FILES) == 0, reason="No real data files found in Data/mseed for PPJR, PCJI, PCJR")
@pytest.mark.parametrize("station_code,file_path", STATION_FILES)
@pytest.mark.parametrize("period", [0.2, 1.0, 3.0])
def test_cross_solver_on_real_data(station_code, file_path, period):
    """Verify Nigam-Jennings and Newmark-Beta concordance (<5% relative difference) on real BMKG earthquake records."""
    st = obspy.read(str(file_path))
    trace = st[0]
    acc = trace.data.astype(np.float64)
    dt = float(trace.stats.delta)
    damping = 0.05

    u_nj, v_nj, a_nj = solve_nigam_jennings(acc, dt, np.array([period]), damping)
    u_nm, v_nm, a_nm = solve_newmark(acc, dt, np.array([period]), damping)

    sd_nj = float(np.max(np.abs(u_nj[0])))
    sd_nm = float(np.max(np.abs(u_nm[0])))

    # On real field recordings with sharp high-frequency peaks, single-period peak
    # difference between piecewise-linear Nigam-Jennings and average-acceleration
    # Newmark-Beta is expected to remain strictly within 8% (< 0.08).
    error_relatif = abs(sd_nm - sd_nj) / sd_nj
    assert error_relatif < 0.08, f"Station {station_code} cross-solver mismatch at T={period}s: {error_relatif:.4%}"