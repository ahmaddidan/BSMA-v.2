"""
BMKG Strong Motion Analyzer (BSMA)
Test Suite: Instrumental Intensity (MMI / GMICE Worden et al., 2012)
"""

from __future__ import annotations

import numpy as np
import pytest

from core.processing.parameters import compute_worden_mmi
from utils.pdf_exporter import get_mmi_worden


def test_worden_mmi_low_pga_branch():
    """Verify Worden et al. (2012) lower branch: log10(PGA) <= 1.57."""
    pga_gal = 10.0  # log10(10) = 1.0 <= 1.57
    res = compute_worden_mmi(pga_gal)

    expected_mmi = 1.78 + 1.55 * 1.0  # 3.33
    assert np.isclose(res["mmi_continuous"], expected_mmi, atol=0.01)
    assert res["mmi_discrete"] == "II-III"
    assert res["basis"] == "PGA"


def test_worden_mmi_high_pga_branch():
    """Verify Worden et al. (2012) upper branch: log10(PGA) > 1.57."""
    pga_gal = 100.0  # log10(100) = 2.0 > 1.57
    res = compute_worden_mmi(pga_gal)

    expected_mmi = -1.60 + 3.70 * 2.0  # 5.80
    assert np.isclose(res["mmi_continuous"], expected_mmi, atol=0.01)
    assert res["mmi_discrete"] == "VI"
    assert res["shaking"] == "Strong"


def test_worden_mmi_pgv_dominance_at_high_intensity():
    """Verify PGV dominates at high intensity (MMI >= 5.0)."""
    pga_gal = 50.0  # MMI ~ 4.69
    pgv_cm_s = 20.0  # log10(20) = 1.30 > 0.53 -> MMI ~ 2.40 + 4.96 * 1.30 = 8.85
    res = compute_worden_mmi(pga_gal, pgv_cm_s)

    assert res["basis"] == "PGV"
    assert res["mmi_continuous"] >= 8.0
    assert res["mmi_discrete"] in {"VIII", "IX", "X+"}


def test_worden_mmi_monotonicity():
    """Verify that increasing PGA produces monotonically non-decreasing MMI."""
    pga_values = np.logspace(-1, 3, 50)  # from 0.1 to 1000 Gal
    mmi_values = [compute_worden_mmi(p)["mmi_continuous"] for p in pga_values]

    for i in range(len(mmi_values) - 1):
        assert mmi_values[i + 1] >= mmi_values[i], (
            f"Non-monotonic MMI at index {i}: {mmi_values[i]} > {mmi_values[i+1]}"
        )


def test_worden_mmi_non_finite_handling():
    """Verify graceful handling of NaN and Inf without crashing."""
    res_nan = compute_worden_mmi(float("nan"))
    assert np.isfinite(res_nan["mmi_continuous"])
    assert res_nan["mmi_discrete"] == "I"

    res_zero = compute_worden_mmi(0.0)
    assert np.isfinite(res_zero["mmi_continuous"])
    assert res_zero["mmi_discrete"] == "I"


def test_get_mmi_worden_pdf_helper():
    """Verify the discrete ShakeMap mapping helper used in PDF generation."""
    mmi_info = get_mmi_worden(1.0, 0.5)
    assert "mmi" in mmi_info
    assert "shaking" in mmi_info
    assert "rgb" in mmi_info
