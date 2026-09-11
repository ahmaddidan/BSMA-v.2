"""
BMKG Strong Motion Analyzer (BSMA)
Test Suite: Golden End-to-End Analysis Pipeline Test
====================================================
Executes a full station 3-component analysis stream (HNE, HNN, HNZ) through
AnalysisService from ingestion, QC gate, deconvolution/scaling, baseline detrending,
tapering, Butterworth zero-phase filtering, kinematic integration, parameter extraction,
Worden et al. (2012) GMICE MMI, to SDOF response spectrum.

Asserts exact, deterministic scientific regression targets to guarantee reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from core.processing.parameters import compute_worden_mmi
from services.analysis_service import (
    AnalysisConfiguration,
    AnalysisService,
    extract_summary_data,
)


@pytest.fixture
def canonical_station_stream() -> Stream:
    """Create a calibrated 3-component ground-motion station stream."""
    fs = 100.0
    duration = 20.0
    n_pts = int(duration * fs) + 1
    t = np.linspace(0.0, duration, n_pts)

    # Component HNE (dominant 5 Hz + 2.5 Hz carrier)
    a0_e, beta_e, f1_e, f2_e = 5.0, 0.5, 2.5, 5.0
    acc_e = a0_e * t * np.exp(-beta_e * t) * (
        np.sin(2.0 * np.pi * f1_e * t) + 0.5 * np.sin(2.0 * np.pi * f2_e * t)
    )
    acc_e -= np.mean(acc_e)

    # Component HNN (dominant 3.0 Hz carrier)
    a0_n, beta_n, f1_n = 4.0, 0.6, 3.0
    acc_n = a0_n * t * np.exp(-beta_n * t) * np.sin(2.0 * np.pi * f1_n * t)
    acc_n -= np.mean(acc_n)

    # Component HNZ (dominant 8.0 Hz carrier)
    a0_z, beta_z, f1_z = 2.5, 0.8, 8.0
    acc_z = a0_z * t * np.exp(-beta_z * t) * np.sin(2.0 * np.pi * f1_z * t)
    acc_z -= np.mean(acc_z)

    start = UTCDateTime("2026-01-01T00:00:00")
    return Stream([
        Trace(data=acc_e, header={"network": "IA", "station": "GLD", "location": "00", "channel": "HNE", "sampling_rate": fs, "starttime": start}),
        Trace(data=acc_n, header={"network": "IA", "station": "GLD", "location": "00", "channel": "HNN", "sampling_rate": fs, "starttime": start}),
        Trace(data=acc_z, header={"network": "IA", "station": "GLD", "location": "00", "channel": "HNZ", "sampling_rate": fs, "starttime": start}),
    ])


def test_golden_pipeline_end_to_end(canonical_station_stream):
    """Verify complete multi-channel analysis pipeline against deterministic reference targets."""
    cfg = AnalysisConfiguration(
        input_unit="m/s^2",
        freq_min_hz=0.1,
        freq_max_hz=25.0,
        response_periods=(0.1, 0.2, 0.5, 1.0, 2.0),
        adaptive_filter=False,
    )
    service = AnalysisService(cfg)
    contexts = service.process_station_stream(canonical_station_stream)

    # All 3 channels must be successfully analyzed
    assert set(contexts.keys()) == {"HNE", "HNN", "HNZ"}

    # 1. Component HNE (Strongest horizontal)
    ctx_e = contexts["HNE"]
    assert ctx_e.processing_state.is_qc_complete
    assert ctx_e.processing_state.is_integrated
    assert ctx_e.processing_state.has_strong_motion_parameters
    assert ctx_e.processing_state.has_response_spectrum

    pga_e = ctx_e.metrics["PGA"] * 100.0
    pgv_e = ctx_e.metrics["PGV"] * 100.0
    pgd_e = ctx_e.metrics["PGD"] * 100.0
    ia_e = ctx_e.metrics["Arias_Intensity"]
    d595_e = ctx_e.metrics["Significant_Duration_D5_95"]

    assert pga_e == pytest.approx(479.725, rel=1e-3)
    assert pgv_e == pytest.approx(46.086, rel=1e-3)
    assert pgd_e == pytest.approx(258.327, rel=1e-3)
    assert ia_e == pytest.approx(4.986, rel=1e-3)
    assert d595_e == pytest.approx(5.489, rel=1e-3)

    # HNE PSA resonance at T=0.2s (5 Hz carrier)
    psa_e = dict(zip(ctx_e.spectral_data["periods"], ctx_e.spectral_data["PSA"]))
    assert psa_e[0.2] == pytest.approx(21.702, rel=1e-3)
    assert psa_e[0.1] == pytest.approx(5.513, rel=1e-3)
    assert psa_e[0.5] == pytest.approx(8.075, rel=1e-3)

    # 2. Component HNN
    ctx_n = contexts["HNN"]
    pga_n = ctx_n.metrics["PGA"] * 100.0
    pgv_n = ctx_n.metrics["PGV"] * 100.0
    assert pga_n == pytest.approx(252.252, rel=1e-3)
    assert pgv_n == pytest.approx(38.631, rel=1e-3)

    # 3. Component HNZ (Vertical)
    ctx_z = contexts["HNZ"]
    pga_z = ctx_z.metrics["PGA"] * 100.0
    pgv_z = ctx_z.metrics["PGV"] * 100.0
    assert pga_z == pytest.approx(119.881, rel=1e-3)
    assert pgv_z == pytest.approx(15.622, rel=1e-3)

    # 4. Worden et al. (2012) MMI based strictly on Horizontal Max (Max-H)
    pga_max_h = max(pga_e, pga_n)
    pgv_max_h = max(pgv_e, pgv_n)
    mmi_res = compute_worden_mmi(pga_max_h, pgv_max_h)
    assert mmi_res["mmi_discrete"] == "X+"
    assert mmi_res["basis"] == "PGV"
    assert mmi_res["shaking"] == "Extreme"

    # Vertical channel must NOT contaminate Max-H MMI
    assert pga_max_h > pga_z

    # 5. Summary tabular export consistency
    rows = extract_summary_data("GLD", contexts)
    assert len(rows) == 3
    channels_in_summary = [r["channel"] for r in rows]
    assert channels_in_summary == ["HNE", "HNN", "HNZ"]
    for r in rows:
        assert r["station"] == "GLD"
        assert r["qc_valid"] is True
        assert r["qc_score"] >= 70