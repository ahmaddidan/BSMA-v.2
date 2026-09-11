from __future__ import annotations

import numpy as np
import pytest

from core.preprocessing.integration import (
    IntegrationConfig,
    KinematicIntegrationPlugin,
)
from core.types.processing_state import StageStatus
from tests.helpers import make_context
from utils.exceptions import ProcessingError


def test_zero_acceleration_integrates_to_zero():
    result = KinematicIntegrationPlugin().process(make_context(np.zeros(100)))

    assert np.allclose(result.velocity.data, 0.0)
    assert np.allclose(result.displacement.data, 0.0)
    assert result.processing_state.integration is StageStatus.SUCCESS


def test_constant_acceleration_uses_cumulative_trapezoid():
    sampling_rate = 100.0
    acceleration = np.ones(1000)
    result = KinematicIntegrationPlugin(
        IntegrationConfig(condition_acceleration=False)
    ).process(make_context(acceleration, sampling_rate=sampling_rate))

    expected_velocity = np.arange(acceleration.size) / sampling_rate
    assert np.allclose(result.velocity.data, expected_velocity)


def test_integration_preserves_original_context():
    context = make_context(np.sin(np.linspace(0.0, 10.0, 1000)))
    result = KinematicIntegrationPlugin().process(context)

    assert result is not context
    assert context.velocity is None
    assert context.displacement is None
    assert result.velocity is not None
    assert result.displacement is not None


def test_short_waveform_is_rejected():
    context = make_context(np.array([1.0]))

    with pytest.raises(ProcessingError, match="at least two samples"):
        KinematicIntegrationPlugin().process(context)


def test_synthetic_sinusoidal_integration_accuracy():
    """
    Scientific Validation Test:
    Verify kinematic integration against known analytical solutions for:
        a(t) = A * sin(omega * t)
        v(t) = (A / omega) * (1 - cos(omega * t))
        d(t) = (A / omega) * t - (A / omega^2) * sin(omega * t)
    """
    fs = 500.0  # High sampling rate for fine discretization
    dt = 1.0 / fs
    duration = 5.0
    time = np.arange(0, duration, dt, dtype=np.float64)

    A = 2.0  # m/s^2
    f_sig = 1.0  # Hz
    omega = 2.0 * np.pi * f_sig

    a_analytical = A * np.sin(omega * time)
    v_analytical = (A / omega) * (1.0 - np.cos(omega * time))
    d_analytical = (A / omega) * time - (A / (omega ** 2)) * np.sin(omega * time)

    context = make_context(a_analytical, sampling_rate=fs)
    plugin = KinematicIntegrationPlugin(IntegrationConfig(condition_acceleration=False))
    result = plugin.process(context)

    # Validate velocity: relative error should be < 0.1% of peak velocity
    v_num = result.velocity.data
    v_peak = float(np.max(np.abs(v_analytical)))
    v_err = np.max(np.abs(v_num - v_analytical)) / v_peak
    assert v_err < 1e-3, f"Velocity integration relative error {v_err:.6%} exceeds 0.1%"

    # Validate displacement: relative error should be < 0.1% of peak displacement
    d_num = result.displacement.data
    d_peak = float(np.max(np.abs(d_analytical)))
    d_err = np.max(np.abs(d_num - d_analytical)) / d_peak
    assert d_err < 1e-3, f"Displacement integration relative error {d_err:.6%} exceeds 0.1%"
