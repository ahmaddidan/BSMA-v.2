from __future__ import annotations

import numpy as np

from core.preprocessing.baseline import BaselineCorrectionPlugin
from tests.helpers import make_context


def test_linear_detrend() -> None:
    data = np.linspace(0, 10, 1000)
    ctx = make_context(data, sampling_rate=100.0)
    plugin = BaselineCorrectionPlugin(method="linear")
    result = plugin.process(ctx)

    assert result.acceleration is not None
    assert np.abs(np.mean(result.acceleration.data)) < 1e-8
    assert len(result.history) == 1