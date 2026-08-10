from __future__ import annotations

import numpy as np

from core.types.context import ProcessingContext, WaveformData
from core.types.processing_state import ProcessingState


def make_context(
    data: np.ndarray,
    *,
    sampling_rate: float = 100.0,
    trace_id: str = "XX.TEST.00.HNZ",
) -> ProcessingContext:
    """Create a valid immutable context for signal-processing tests."""
    waveform = WaveformData(
        data=np.asarray(data, dtype=np.float64),
        sampling_rate=sampling_rate,
        unit="m/s^2",
    )
    return ProcessingContext(
        trace_id=trace_id,
        metadata={},
        raw_waveform=waveform,
        acceleration=waveform,
        processing_state=ProcessingState(),
    )
