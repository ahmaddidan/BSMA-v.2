"""
Unit tests for ProcessingContext root aggregate.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from core.types.context import ProcessingContext
from core.types.metadata import TraceMetadata
from core.types.processing_state import ProcessingState, StageStatus
from core.types.cache import ProcessingCache
from core.types.qc import QCReport


def create_metadata() -> TraceMetadata:
    """Helper to create a standard TraceMetadata object for testing."""
    start = datetime(2025, 1, 1, 0, 0, 0)
    end = start + timedelta(seconds=10)
    
    return TraceMetadata(
        network="IA",
        station="BBJI",
        location="00",
        channel="HNE",
        sampling_rate=100.0,
        starttime=start,
        endtime=end,
        npts=1000,
        units="m/s²",
        instrument="Etna2",
    )


def create_context() -> ProcessingContext:
    """Helper to create a standard ProcessingContext object."""
    waveform = np.zeros(1000, dtype=np.float64)
    return ProcessingContext(
        waveform=waveform,
        metadata=create_metadata()
    )


def test_context_creation():
    ctx = create_context()
    assert ctx.waveform is not None
    assert len(ctx.waveform) == 1000
    assert ctx.npts == 1000
    assert ctx.sampling_rate == 100.0


def test_defaults():
    ctx = create_context()
    assert isinstance(ctx.processing_state, ProcessingState)
    assert isinstance(ctx.cache, ProcessingCache)
    assert isinstance(ctx.qc, QCReport)
    assert ctx.history == ()
    assert dict(ctx.config) == {}


def test_copy_returns_new_instance():
    ctx = create_context()
    ctx2 = ctx.copy()
    assert ctx is not ctx2


def test_copy_keeps_metadata():
    ctx = create_context()
    ctx2 = ctx.copy()
    assert ctx.metadata is ctx2.metadata


def test_copy_keeps_cache_reference():
    ctx = create_context()
    ctx2 = ctx.copy()
    assert ctx.cache is ctx2.cache


def test_copy_keeps_qc():
    ctx = create_context()
    ctx2 = ctx.copy()
    assert ctx.qc is ctx2.qc


def test_add_history():
    ctx = create_context()
    ctx2 = ctx.add_history("step 1")
    assert ctx.history == ()
    assert ctx2.history == ("step 1",)


def test_add_multiple_history():
    ctx = create_context()
    ctx2 = ctx.add_history("step 1").add_history("step 2")
    assert ctx2.history == ("step 1", "step 2")


def test_config_is_available():
    ctx = create_context()
    ctx2 = ctx.copy(config={"test": True})
    assert ctx2.config["test"] is True
    
    with pytest.raises(TypeError):
        ctx2.config["test"] = False  # MappingProxyType prevents mutation


def test_to_dict():
    ctx = create_context()
    d = ctx.to_dict()
    assert d["sampling_rate"] == 100.0
    assert d["npts"] == 1000
    assert d["metadata"]["network"] == "IA"
    assert "waveform" not in d


def test_waveform_optional():
    ctx = ProcessingContext(
        waveform=None,
        metadata=create_metadata(),
    )
    assert bool(ctx) is False
    assert ctx.npts == 1000


def test_processing_state_exists():
    ctx = create_context()
    assert hasattr(ctx.processing_state, "filter")
    assert ctx.processing_state.filter == StageStatus.PENDING