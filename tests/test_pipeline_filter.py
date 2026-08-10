from __future__ import annotations

import numpy as np

from core.preprocessing.filter import (
    ButterworthFilterPlugin,
    FilterConfig,
    FilterType,
)
from tests.helpers import make_context


def test_bandpass_filter_attenuates_out_of_band_noise():
    sampling_rate = 100.0
    time = np.arange(0.0, 20.0, 1.0 / sampling_rate)
    target = np.sin(2 * np.pi * 5.0 * time)
    noise = 0.5 * np.sin(2 * np.pi * 40.0 * time)
    context = make_context(target + noise, sampling_rate=sampling_rate)

    result = ButterworthFilterPlugin(
        FilterConfig(
            type=FilterType.BANDPASS,
            freq_min=0.5,
            freq_max=20.0,
            zerophase=True,
        )
    ).process(context)

    error = result.acceleration.data[200:-200] - target[200:-200]
    assert np.sqrt(np.mean(error**2)) < 0.05
    assert np.array_equal(context.raw_waveform.data, context.acceleration.data)
