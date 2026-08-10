from __future__ import annotations

import numpy as np

from core.pipeline import PipelineBuilder
from core.preprocessing.baseline import BaselineCorrectionPlugin
from tests.helpers import make_context


def test_pipeline_linear_baseline_removes_offset_and_trend():
    samples = np.arange(1000, dtype=np.float64)
    signal = 4.0 + 0.01 * samples + np.sin(samples / 10.0)
    context = make_context(signal)

    result = (
        PipelineBuilder()
        .add(BaselineCorrectionPlugin(method="linear"))
        .build()
        .run(context)
    )

    slope, _ = np.polyfit(samples, result.acceleration.data, 1)
    assert abs(slope) < 1.0e-12
    assert np.mean(context.acceleration.data) > 4.0
    assert result is not context
