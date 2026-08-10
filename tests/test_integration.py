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
