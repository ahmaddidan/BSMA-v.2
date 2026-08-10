from __future__ import annotations

import numpy as np
import pytest

from core.preprocessing.integration import KinematicIntegrationPlugin
from core.processing.parameters import ParameterExtractionPlugin
from core.types.processing_state import StageStatus
from tests.helpers import make_context
from utils.exceptions import ProcessingError


def test_parameter_extraction_uses_absolute_peaks_and_arias_definition():
    acceleration = np.array([-2.0, -1.0, 0.0, 3.0, -5.0])
    integrated = KinematicIntegrationPlugin().process(make_context(acceleration))

    result = ParameterExtractionPlugin().process(integrated)

    assert result.metrics["PGA"] == pytest.approx(5.0)
    assert result.metrics["PGV"] >= 0.0
    assert result.metrics["PGD"] >= 0.0
    assert result.metrics["Arias_Intensity"] > 0.0
    assert result.cache.acceleration_energy > 0.0
    assert result.processing_state.strong_motion_parameters is StageStatus.SUCCESS


def test_parameter_extraction_requires_kinematic_products():
    with pytest.raises(ProcessingError, match="Velocity data is unavailable"):
        ParameterExtractionPlugin().process(make_context(np.ones(100)))


def test_zero_signal_has_zero_energy_and_duration():
    integrated = KinematicIntegrationPlugin().process(make_context(np.zeros(100)))
    result = ParameterExtractionPlugin().process(integrated)

    assert result.metrics["PGA"] == 0.0
    assert result.metrics["Arias_Intensity"] == 0.0
    assert result.metrics["Significant_Duration_D5_95"] == 0.0
