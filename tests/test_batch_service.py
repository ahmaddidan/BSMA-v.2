from __future__ import annotations

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from services import AnalysisConfiguration, AnalysisService, BatchService


def _valid_stream() -> Stream:
    sampling_rate = 100.0
    time = np.arange(0.0, 10.0, 1.0 / sampling_rate)
    trace = Trace(
        data=(0.3 * np.sin(2.0 * np.pi * time)).astype(np.float64),
        header={
            "network": "XX",
            "station": "TEST",
            "location": "00",
            "channel": "HNE",
            "sampling_rate": sampling_rate,
            "starttime": UTCDateTime(2024, 1, 1),
        },
    )
    return Stream([trace])


def test_batch_isolates_failures_and_reports_progress():
    service = AnalysisService(
        AnalysisConfiguration(
            input_unit="m/s^2",
            freq_max_hz=20.0,
            response_periods=(0.1, 0.5, 1.0),
        )
    )
    progress: list[tuple[int, int, str]] = []

    result = BatchService(service).process_stations(
        {"TEST": _valid_stream(), "EMPTY": Stream()},
        progress_callback=lambda index, total, station: progress.append(
            (index, total, station)
        ),
    )

    assert set(result.contexts_by_station) == {"TEST"}
    assert set(result.failures) == {"EMPTY"}
    assert len(result.summary_rows()) == 1
    assert progress == [(1, 2, "TEST"), (2, 2, "EMPTY")]
