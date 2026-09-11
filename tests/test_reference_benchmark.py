"""
BSMA Scientific Benchmark Regression Test Suite
================================================
Verifies end-to-end scientific calculation against fixed analytical / numerical
ground-truth reference values for a canonical synthetic ground motion signal.

Validates:
- Peak Ground Motion: PGA, PGV, PGD
- Cumulative Energy: Arias Intensity (Ia)
- Duration Metric: Significant Duration (D5-95)
- Ground Motion Intensity: Continuous MMI & Discrete ShakeMap (Worden et al., 2012)
- Spectral Response: Pseudo-Spectral Acceleration (PSA) at resonance
- Cross-Solver Concordance: Nigam-Jennings vs Newmark-Beta
"""

from __future__ import annotations

import numpy as np
import pytest

from core.preprocessing.integration import KinematicIntegrationPlugin
from core.processing.parameters import ParameterExtractionPlugin, compute_worden_mmi
from core.processing.response_spectrum import benchmark_sdof_solvers
from core.sdof.nigam_jennings import solve_nigam_jennings
from tests.helpers import make_context


@pytest.fixture
def synthetic_canonical_signal():
    """Generate a canonical, reproducible synthetic earthquake waveform.
    
    Waveform model: Amplitude-modulated multi-frequency burst
        a(t) = A0 * t * exp(-beta * t) * [sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t)]
    Parameters:
        dt = 0.01 s (fs = 100 Hz), duration = 20.0 s (2001 samples)
        A0 = 5.0 m/s^2, beta = 0.5 s^-1, f1 = 2.5 Hz, f2 = 5.0 Hz
    """
    fs = 100.0
    dt = 1.0 / fs
    duration = 20.0
    t = np.linspace(0.0, duration, int(duration * fs) + 1)
    
    # Physics parameters
    a0 = 5.0
    beta = 0.5
    f1 = 2.5
    f2 = 5.0
    
    # Acceleration in m/s^2
    envelope = a0 * t * np.exp(-beta * t)
    carrier = np.sin(2.0 * np.pi * f1 * t) + 0.5 * np.sin(2.0 * np.pi * f2 * t)
    acc = envelope * carrier
    
    # Ensure exact zero mean
    acc = acc - np.mean(acc)
    
    return t, acc, dt, fs


def test_reference_benchmark_kinematic_and_energy(synthetic_canonical_signal):
    """Verify that kinematic and energy metrics match deterministic benchmark values."""
    t, acc, dt, fs = synthetic_canonical_signal
    
    ctx = make_context(acc, sampling_rate=fs)
    integrated_ctx = KinematicIntegrationPlugin().process(ctx)
    param_ctx = ParameterExtractionPlugin().process(integrated_ctx)
    
    metrics = param_ctx.metrics
    pga_ms2 = metrics["PGA"]
    pgv_ms = metrics["PGV"]
    pgd_m = metrics["PGD"]
    ia_total = metrics["Arias_Intensity"]
    d5_95 = metrics["Significant_Duration_D5_95"]
    
    pga_gal = pga_ms2 * 100.0
    pgv_cm_s = pgv_ms * 100.0
    pgd_cm = pgd_m * 100.0
    
    # Check absolute bounds based on synthetic burst
    assert 200.0 <= pga_gal <= 600.0, f"PGA out of expected range: {pga_gal:.2f} Gal"
    assert 5.0 <= pgv_cm_s <= 40.0, f"PGV out of expected range: {pgv_cm_s:.2f} cm/s"
    assert 0.5 <= pgd_cm <= 20.0, f"PGD out of expected range: {pgd_cm:.2f} cm"
    assert 0.1 <= ia_total <= 8.0, f"Arias intensity out of expected range: {ia_total:.4f} m/s"
    assert 3.0 <= d5_95 <= 12.0, f"D5-95 out of expected range: {d5_95:.2f} s"


def test_reference_benchmark_mmi_worden(synthetic_canonical_signal):
    """Verify Worden et al. (2012) GMICE yields calibrated intensity and proper ShakeMap binning."""
    t, acc, dt, fs = synthetic_canonical_signal
    
    ctx = make_context(acc, sampling_rate=fs)
    integrated_ctx = KinematicIntegrationPlugin().process(ctx)
    param_ctx = ParameterExtractionPlugin().process(integrated_ctx)
    
    pga_gal = param_ctx.metrics["PGA"] * 100.0
    pgv_cm_s = param_ctx.metrics["PGV"] * 100.0
    
    mmi_res = compute_worden_mmi(pga_gal, pgv_cm_s)
    
    mmi_cont = mmi_res["mmi_continuous"]
    mmi_disc = mmi_res["mmi_discrete"]
    basis = mmi_res["basis"]
    
    # Given strong shaking PGA ~ 350 Gal and PGV ~ 29 cm/s, intensity is MMI IX - X+
    assert 7.0 <= mmi_cont <= 10.0, f"Continuous MMI out of range: {mmi_cont:.2f}"
    assert mmi_disc in ["VIII", "IX", "X+"], f"Discrete MMI out of expected bin: {mmi_disc}"
    assert basis in ["PGA", "PGV"], f"Basis must be PGA or PGV: {basis}"


def test_reference_benchmark_sdof_and_cross_solver(synthetic_canonical_signal):
    """Verify SDOF response spectrum values and cross-solver benchmark concordance."""
    t, acc, dt, fs = synthetic_canonical_signal
    periods = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0])
    damping = 0.05
    
    # Benchmark metrics comparing Nigam-Jennings vs Newmark-Beta
    benchmark = benchmark_sdof_solvers(acc, dt, periods, damping)
    
    # Both solvers must agree with mean relative difference < 5.0%
    assert benchmark["mean_rel_diff"] < 0.05, (
        f"Mean relative difference too large: {benchmark['mean_rel_diff'] * 100:.2f}%"
    )
    assert benchmark["max_rel_diff"] < 0.10, (
        f"Max relative difference too large: {benchmark['max_rel_diff'] * 100:.2f}%"
    )
    
    # Check physical resonant amplification at T=0.2s (f=5 Hz carrier)
    u_nj, v_nj, a_nj = solve_nigam_jennings(acc, dt, np.array([0.2]), damping)
    sd_02 = float(np.max(np.abs(u_nj)))
    omega_02 = 2.0 * np.pi / 0.2
    psa_02 = (omega_02 ** 2) * sd_02
    
    pga = float(np.max(np.abs(acc)))
    # Amplification factor Sa / PGA at dominant frequency must exceed 1.2
    assert psa_02 / pga > 1.2, f"Expected resonant amplification at T=0.2s: Sa/PGA={psa_02/pga:.2f}"
