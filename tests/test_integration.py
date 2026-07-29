from __future__ import annotations

from datetime import datetime
from datetime import timedelta

import numpy as np
import pytest

from core.preprocessing.integration import (
    IntegrationConfig,
    IntegrationPlugin,
)
from core.types.cache import ProcessingCache
from core.types.context import ProcessingContext
from core.types.metadata import TraceMetadata
from core.types.processing_state import (
    ProcessingState,
    StageStatus,
)
from core.types.qc import QCReport


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def create_metadata(
    sampling_rate: float,
    npts: int,
) -> TraceMetadata:

    start = datetime(2025, 1, 1)

    end = start + timedelta(
        seconds=(npts - 1) / sampling_rate
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
    )


def create_context(
    waveform: np.ndarray,
) -> ProcessingContext:

    waveform = waveform.astype(np.float64)

    metadata = create_metadata(
        sampling_rate=100.0,
        npts=waveform.size,
    )

    return ProcessingContext(
        waveform=waveform,
        metadata=metadata,
        processing_state=ProcessingState(),
        cache=ProcessingCache(),
        qc=QCReport(),
    )


# ---------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------


def test_plugin_name():

    plugin = IntegrationPlugin()

    assert plugin.plugin_name == "Integration"


def test_plugin_description():

    plugin = IntegrationPlugin()

    assert "velocity" in plugin.plugin_description.lower()


# ---------------------------------------------------------------------
# Zero signal
# ---------------------------------------------------------------------


def test_zero_signal():

    waveform = np.zeros(1000)

    ctx = create_context(waveform)

    plugin = IntegrationPlugin()

    result = plugin.process(ctx)

    assert np.allclose(
        result.cache.velocity,
        0.0,
    )

    assert np.allclose(
        result.cache.displacement,
        0.0,
    )


# ---------------------------------------------------------------------
# Constant acceleration
# ---------------------------------------------------------------------


def test_constant_acceleration():

    waveform = np.ones(1000)

    ctx = create_context(waveform)

    plugin = IntegrationPlugin(
        IntegrationConfig(
            remove_mean_before_integrating=False
        )
    )

    result = plugin.process(ctx)

    dt = 1.0 / ctx.sampling_rate

    expected_velocity = (
        np.arange(waveform.size)
        * dt
    )

    assert np.allclose(
        result.cache.velocity,
        expected_velocity,
        atol=1e-12,
    )


# ---------------------------------------------------------------------
# Sinusoid
# ---------------------------------------------------------------------


def test_sinusoid():

    fs = 100.0

    t = np.arange(0, 10, 1 / fs)

    waveform = np.sin(
        2 * np.pi * t
    )

    ctx = create_context(waveform)

    plugin = IntegrationPlugin()

    result = plugin.process(ctx)

    assert result.cache.velocity is not None

    assert result.cache.displacement is not None

    assert result.cache.velocity.size == waveform.size

    assert result.cache.displacement.size == waveform.size


# ---------------------------------------------------------------------
# NaN
# ---------------------------------------------------------------------


def test_nan():

    waveform = np.ones(100)

    waveform[10] = np.nan

    ctx = create_context(waveform)

    plugin = IntegrationPlugin()

    with pytest.raises(ValueError):

        plugin.process(ctx)


# ---------------------------------------------------------------------
# Inf
# ---------------------------------------------------------------------


def test_inf():

    waveform = np.ones(100)

    waveform[20] = np.inf

    ctx = create_context(waveform)

    plugin = IntegrationPlugin()

    with pytest.raises(ValueError):

        plugin.process(ctx)


# ---------------------------------------------------------------------
# Short waveform
# ---------------------------------------------------------------------

def test_short_waveform():
    """
    Integration requires at least two samples.

    This test intentionally constructs a valid ProcessingContext whose
    waveform contains only a single sample. The metadata remain valid,
    while the IntegrationPlugin is expected to reject the waveform.
    """

    sampling_rate = 100.0

    start = datetime(2025, 1, 1)

    # endtime harus > starttime agar TraceMetadata valid
    end = start + timedelta(seconds=1.0 / sampling_rate)

    metadata = TraceMetadata(
        network="IA",
        station="TEST",
        location="00",
        channel="HNZ",
        sampling_rate=sampling_rate,
        starttime=start,
        endtime=end,
        npts=1,
    )

    context = ProcessingContext(
        waveform=np.array(
            [1.0],
            dtype=np.float64,
        ),
        metadata=metadata,
        processing_state=ProcessingState(),
        cache=ProcessingCache(),
        qc=QCReport(),
    )

    plugin = IntegrationPlugin()

    with pytest.raises(
        ValueError,
        match="Waveform must contain at least two samples.",
    ):
        plugin.process(context)


# ---------------------------------------------------------------------
# Processing state
# ---------------------------------------------------------------------


def test_processing_state_updated():

    waveform = np.random.randn(1000)

    ctx = create_context(waveform)

    plugin = IntegrationPlugin()

    result = plugin.process(ctx)

    assert (
        result.processing_state.integration
        is StageStatus.SUCCESS
    )


# ---------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------


def test_cache_created():

    waveform = np.random.randn(500)

    ctx = create_context(waveform)

    plugin = IntegrationPlugin()

    result = plugin.process(ctx)

    assert result.cache.velocity is not None

    assert result.cache.displacement is not None


# ---------------------------------------------------------------------
# Immutable
# ---------------------------------------------------------------------


def test_context_immutable():

    waveform = np.random.randn(500)

    ctx = create_context(waveform)

    plugin = IntegrationPlugin()

    result = plugin.process(ctx)

    assert result is not ctx

    assert ctx.cache.velocity is None

    assert ctx.cache.displacement is None


# ---------------------------------------------------------------------
# History
# ---------------------------------------------------------------------


def test_history():

    waveform = np.random.randn(500)

    ctx = create_context(waveform)

    plugin = IntegrationPlugin()

    result = plugin.process(ctx)

    assert len(result.history) == 1

    assert "Integration" in result.history[0]


# ---------------------------------------------------------------------
# Dtype
# ---------------------------------------------------------------------


def test_dtype_preserved():

    waveform = np.random.randn(
        500
    ).astype(np.float64)

    ctx = create_context(waveform)

    plugin = IntegrationPlugin()

    result = plugin.process(ctx)

    assert (
        result.cache.velocity.dtype
        == np.float64
    )

    assert (
        result.cache.displacement.dtype
        == np.float64
    )


# ---------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------


def test_repr():

    plugin = IntegrationPlugin()

    text = repr(plugin)

    assert "IntegrationPlugin" in text

    assert "method" in text