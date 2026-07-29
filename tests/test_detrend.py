from datetime import datetime, timedelta

import numpy as np

from core.preprocessing.detrend import (
    DetrendPlugin,
)

from core.types.context import ProcessingContext
from core.types.metadata import TraceMetadata


def create_context():

    fs = 100.0

    waveform = np.linspace(
        0,
        10,
        1000,
    )

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


def test_linear_detrend():

    ctx = create_context()

    plugin = DetrendPlugin()

    new_ctx = plugin.process(ctx)

    assert np.abs(np.mean(new_ctx.waveform)) < 1e-8

    assert len(new_ctx.history) == 1