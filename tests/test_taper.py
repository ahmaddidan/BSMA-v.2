from datetime import datetime
from datetime import timedelta

import numpy as np

from core.preprocessing.taper import (
    TaperPlugin,
)

from core.types.context import ProcessingContext
from core.types.metadata import TraceMetadata


def create_context():

    fs = 100.0

    waveform = np.ones(1000)

    metadata = TraceMetadata(
        network="IA",
        station="TEST",
        location="00",
        channel="HNZ",
        sampling_rate=fs,
        starttime=datetime.now(),
        endtime=datetime.now() + timedelta(seconds=10),
        npts=1000,
    )

    return ProcessingContext(
        waveform=waveform,
        metadata=metadata,
    )


def test_taper():

    ctx = create_context()

    plugin = TaperPlugin()

    result = plugin.process(ctx)

    assert result.waveform[0] < 0.05

    assert result.waveform[-1] < 0.05

    assert result.waveform[500] > 0.95

    assert len(result.history) == 1


def test_original_context_not_modified():

    ctx = create_context()

    original = ctx.waveform.copy()

    plugin = TaperPlugin()

    result = plugin.process(ctx)

    assert np.array_equal(
        ctx.waveform,
        original,
    )

    assert not np.array_equal(
        ctx.waveform,
        result.waveform,
    )