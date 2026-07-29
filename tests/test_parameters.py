"""
Unit Tests
==========

Parameter Extraction Plugin
"""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta

import numpy as np
import pytest

from core.preprocessing.integration import (
    IntegrationPlugin,
)
from core.processing.parameters import (
    ParameterExtractionPlugin,
)
from core.types.cache import (
    ProcessingCache,
)
from core.types.context import (
    ProcessingContext,
)
from core.types.metadata import (
    TraceMetadata,
)
from core.types.processing_state import (
    ProcessingState,
)


# ==========================================================
# Helpers
# ==========================================================


def create_metadata(
    npts: int,
    sampling_rate: float = 100.0,
) -> TraceMetadata:
    """
    Create valid metadata for testing.
    """

    start = datetime(
        2025,
        1,
        1,
    )

    duration = (
        npts - 1
    ) / sampling_rate

    end = (
        start
        + timedelta(
            seconds=duration
        )
    )

    return TraceMetadata(
        network="IA",
        station="TEST",
        location="00",
        channel="HNZ",
        sampling_rate=sampling_rate,
        starttime=start,
        endtime=end,
        npts=npts,
        units="m/s²",
    )


def create_context(
    waveform: np.ndarray,
    sampling_rate: float = 100.0,
) -> ProcessingContext:
    """
    Create ProcessingContext and run integration
    so velocity and displacement are available.
    """

    waveform = np.asarray(
        waveform,
        dtype=np.float64,
    )

    metadata = create_metadata(
        len(waveform),
        sampling_rate,
    )

    context = ProcessingContext(
        waveform=waveform,
        metadata=metadata,
        processing_state=ProcessingState(),
        cache=ProcessingCache(),
    )

    integration = IntegrationPlugin()

    return integration.process(
        context
    )


# ==========================================================
# Plugin Information
# ==========================================================


def test_plugin_name():

    plugin = ParameterExtractionPlugin()

    assert (
        plugin.plugin_name
        == "ParameterExtraction"
    )


def test_plugin_description():

    plugin = ParameterExtractionPlugin()

    assert isinstance(
        plugin.plugin_description,
        str,
    )

    assert (
        len(
            plugin.plugin_description
        )
        > 0
    )


def test_repr():

    plugin = ParameterExtractionPlugin()

    text = repr(plugin)

    assert (
        "ParameterExtractionPlugin"
        in text
    )

    assert (
        "gravity"
        in text
    )

    assert (
        "D5-75"
        in text
    )

    assert (
        "D5-95"
        in text
    )
    # ==========================================================
# Validation
# ==========================================================


def test_waveform_none():

    metadata = create_metadata(
        10,
    )

    context = ProcessingContext(
        waveform=None,
        metadata=metadata,
        processing_state=ProcessingState(),
        cache=ProcessingCache(),
    )

    plugin = ParameterExtractionPlugin()

    with pytest.raises(
        ValueError,
    ):
        plugin.process(
            context,
        )


def test_short_waveform():

    waveform = np.array(
        [1.0],
        dtype=np.float64,
    )

    metadata = create_metadata(
        1,
    )

    context = ProcessingContext(
        waveform=waveform,
        metadata=metadata,
        processing_state=ProcessingState(),
        cache=ProcessingCache(
            velocity=np.array(
                [0.0],
                dtype=np.float64,
            ),
            displacement=np.array(
                [0.0],
                dtype=np.float64,
            ),
        ),
    )

    plugin = ParameterExtractionPlugin()

    with pytest.raises(
        ValueError,
    ):
        plugin.process(
            context,
        )


def test_nan_waveform():

    waveform = np.zeros(
        100,
        dtype=np.float64,
    )

    waveform[10] = np.nan

    metadata = create_metadata(
        len(waveform),
    )

    context = ProcessingContext(
        waveform=waveform,
        metadata=metadata,
        processing_state=ProcessingState(),
        cache=ProcessingCache(
            velocity=np.zeros_like(
                waveform,
            ),
            displacement=np.zeros_like(
                waveform,
            ),
        ),
    )

    plugin = ParameterExtractionPlugin()

    with pytest.raises(
        ValueError,
    ):
        plugin.process(
            context,
        )


def test_inf_waveform():

    waveform = np.zeros(
        100,
        dtype=np.float64,
    )

    waveform[20] = np.inf

    metadata = create_metadata(
        len(waveform),
    )

    context = ProcessingContext(
        waveform=waveform,
        metadata=metadata,
        processing_state=ProcessingState(),
        cache=ProcessingCache(
            velocity=np.zeros_like(
                waveform,
            ),
            displacement=np.zeros_like(
                waveform,
            ),
        ),
    )

    plugin = ParameterExtractionPlugin()

    with pytest.raises(
        ValueError,
    ):
        plugin.process(
            context,
        )


def test_missing_velocity():

    waveform = np.random.randn(
        500,
    )

    context = create_context(
        waveform,
    )

    context.cache.velocity = None

    plugin = ParameterExtractionPlugin()

    with pytest.raises(
        ValueError,
    ):
        plugin.process(
            context,
        )


def test_missing_displacement():

    waveform = np.random.randn(
        500,
    )

    context = create_context(
        waveform,
    )

    context.cache.displacement = None

    plugin = ParameterExtractionPlugin()

    with pytest.raises(
        ValueError,
    ):
        plugin.process(
            context,
        )


def test_velocity_length_mismatch():

    waveform = np.random.randn(
        500,
    )

    context = create_context(
        waveform,
    )

    context.cache.velocity = np.zeros(
        100,
        dtype=np.float64,
    )

    plugin = ParameterExtractionPlugin()

    with pytest.raises(
        ValueError,
    ):
        plugin.process(
            context,
        )


def test_displacement_length_mismatch():

    waveform = np.random.randn(
        500,
    )

    context = create_context(
        waveform,
    )

    context.cache.displacement = np.zeros(
        100,
        dtype=np.float64,
    )

    plugin = ParameterExtractionPlugin()

    with pytest.raises(
        ValueError,
    ):
        plugin.process(
            context,
        )


def test_invalid_sampling_rate():

    waveform = np.random.randn(
        100,
    )

    metadata = create_metadata(
        len(waveform),
    )

    object.__setattr__(
        metadata,
        "sampling_rate",
        -100.0,
    )

    context = ProcessingContext(
        waveform=waveform,
        metadata=metadata,
        processing_state=ProcessingState(),
        cache=ProcessingCache(
            velocity=np.zeros_like(
                waveform,
            ),
            displacement=np.zeros_like(
                waveform,
            ),
        ),
    )

    plugin = ParameterExtractionPlugin()

    with pytest.raises(
        ValueError,
    ):
        plugin.process(
            context,
        )
        # ==========================================================
# Processing Results
# ==========================================================


def test_zero_signal():

    waveform = np.zeros(
        2000,
        dtype=np.float64,
    )

    context = create_context(
        waveform,
    )

    plugin = ParameterExtractionPlugin()

    result = plugin.process(
        context,
    )

    assert result.cache.pga == pytest.approx(
        0.0,
    )

    assert result.cache.pgv == pytest.approx(
        0.0,
    )

    assert result.cache.pgd == pytest.approx(
        0.0,
    )

    assert result.cache.arias_intensity == pytest.approx(
        0.0,
    )

    assert result.cache.cumulative_absolute_velocity == pytest.approx(
        0.0,
    )

    assert np.allclose(
        result.cache.husid_curve,
        0.0,
    )


def test_constant_signal():

    waveform = np.ones(
        3000,
        dtype=np.float64,
    )

    context = create_context(
        waveform,
    )

    plugin = ParameterExtractionPlugin()

    result = plugin.process(
        context,
    )

    assert result.cache.pga == pytest.approx(
        1.0,
    )

    assert result.cache.pgv >= 0.0

    assert result.cache.pgd >= 0.0

    assert result.cache.energy > 0.0


def test_sinusoidal_signal():

    fs = 100.0

    t = np.arange(
        0.0,
        20.0,
        1.0 / fs,
    )

    waveform = np.sin(
        2.0
        * np.pi
        * 2.0
        * t,
    )

    context = create_context(
        waveform,
        sampling_rate=fs,
    )

    plugin = ParameterExtractionPlugin()

    result = plugin.process(
        context,
    )

    assert result.cache.pga == pytest.approx(
        1.0,
        rel=1e-2,
    )

    assert result.cache.arias_intensity > 0.0

    assert (
        result.cache.cumulative_absolute_velocity
        > 0.0
    )


def test_waveform_statistics():

    waveform = np.array(
        [
            -1.0,
            0.0,
            1.0,
            2.0,
            3.0,
        ],
        dtype=np.float64,
    )

    context = create_context(
        waveform,
    )

    plugin = ParameterExtractionPlugin()

    result = plugin.process(
        context,
    )

    dt = 1.0 / context.sampling_rate

    assert result.cache.mean == pytest.approx(
        np.mean(
            waveform,
        )
    )

    assert result.cache.standard_deviation == pytest.approx(
        np.std(
            waveform,
        )
    )

    assert result.cache.root_mean_square == pytest.approx(
        np.sqrt(
            np.mean(
                waveform ** 2,
            )
        )
    )

    assert result.cache.energy == pytest.approx(
        np.sum(
            waveform ** 2,
        )
        * dt
    )


def test_peak_ground_motion():

    waveform = np.array(
        [
            -2.0,
            -1.0,
            0.0,
            3.0,
            -5.0,
        ],
        dtype=np.float64,
    )

    context = create_context(
        waveform,
    )

    plugin = ParameterExtractionPlugin()

    result = plugin.process(
        context,
    )

    assert result.cache.pga == pytest.approx(
        5.0,
    )

    assert result.cache.pgv >= 0.0

    assert result.cache.pgd >= 0.0