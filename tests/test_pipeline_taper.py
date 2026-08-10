from __future__ import annotations

import numpy as np

from core.preprocessing.taper import TaperConfig, TaperPlugin
from tests.helpers import make_context


def test_tukey_taper_reduces_both_record_edges_without_mutating_raw_data():
    context = make_context(np.ones(100))
    result = TaperPlugin(TaperConfig(alpha=0.10)).process(context)

    assert result.acceleration.data[0] == 0.0
    assert result.acceleration.data[-1] == 0.0
    assert result.acceleration.data[50] == 1.0
    assert context.raw_waveform.data[0] == 1.0
