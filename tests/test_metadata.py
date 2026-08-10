from datetime import datetime, timedelta

import pytest

from core.types.metadata import TraceMetadata


def _metadata(**updates):
    start = datetime(2025, 1, 1)
    values = {
        "network": "IA", "station": "BBJI", "location": "00", "channel": "HNE",
        "sampling_rate": 100.0, "starttime": start,
        "endtime": start + timedelta(seconds=9.99), "npts": 1000, "unit": "m/s^2",
    }
    values.update(updates)
    return TraceMetadata(**values)


def test_metadata_exposes_consistent_time_and_identifier():
    metadata = _metadata()
    assert metadata.trace_id == "IA.BBJI.00.HNE"
    assert metadata.delta == 0.01
    assert metadata.duration == 9.99
    assert metadata.component == "E"


def test_metadata_rejects_inconsistent_timing():
    with pytest.raises(ValueError, match="Inconsistent trace timing"):
        _metadata(endtime=datetime(2025, 1, 1) + timedelta(seconds=10))
