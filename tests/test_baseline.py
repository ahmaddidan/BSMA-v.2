import numpy as np
import pytest
from unittest.mock import MagicMock

from core.preprocessing.baseline import BaselineCorrectionPlugin
from core.types.context import ProcessingContext, WaveformData
from core.types.processing_state import ProcessingState


def create_mock_context(data_array: np.ndarray) -> ProcessingContext:
    waveform = WaveformData(data=data_array, sampling_rate=100.0, unit="m/s^2")
    state = ProcessingState()
    return ProcessingContext(
        trace_id="TEST.MOCK.00.HNN",
        raw_waveform=waveform,
        acceleration=waveform,
        metadata=MagicMock(),
        processing_state=state,
        metrics={},
        spectral_data={},
        history=(),
    )


def test_baseline_constant_removal():
    t = np.linspace(0, 10, 1000)
    pure_sine = np.sin(2 * np.pi * 1.0 * t)
    offset_sine = pure_sine + 5.0

    context = create_mock_context(offset_sine)
    plugin = BaselineCorrectionPlugin(method="constant")
    new_context = plugin.process(context)

    corrected_data = new_context.acceleration.data
    assert np.abs(np.mean(corrected_data)) < 1e-10
    assert new_context.processing_state.baseline.name == "SUCCESS"


def test_baseline_empty_array_error():
    with pytest.raises(Exception):
        context = create_mock_context(np.array([]))
        plugin = BaselineCorrectionPlugin(method="linear")
        plugin.process(context)