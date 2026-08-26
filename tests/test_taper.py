from __future__ import annotations

import numpy as np

from core.preprocessing.taper import TaperPlugin
from tests.helpers import make_context


def test_taper() -> None:
    ctx = make_context(np.ones(1000), sampling_rate=100.0)
    plugin = TaperPlugin()
    result = plugin.process(ctx)

    assert result.acceleration is not None
    assert result.acceleration.data[0] < 0.05
    assert result.acceleration.data[-1] < 0.05
    assert result.acceleration.data[500] > 0.95
    assert len(result.history) == 1


def test_original_context_not_modified() -> None:
    ctx = make_context(np.ones(1000), sampling_rate=100.0)
    original = ctx.acceleration.data.copy()

    plugin = TaperPlugin()
    plugin.process(ctx)

    np.testing.assert_array_equal(ctx.acceleration.data, original)