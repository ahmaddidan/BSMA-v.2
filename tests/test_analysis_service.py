from __future__ import annotations

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from services.analysis_service import (
    AnalysisConfiguration,
    AnalysisService,
    AnalysisServiceError,
    extract_summary_data,
)


def _physical_acceleration_stream() -> Stream:
    sampling_rate = 100.0
    time = np.arange(0.0, 20.0, 1.0 / sampling_rate)
    data = 0.5 * np.sin(2.0 * np.pi * 1.0 * time)
    trace = Trace(
        data=data.astype(np.float64),
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


def _service() -> AnalysisService:
    return AnalysisService(
        AnalysisConfiguration(
            input_unit="m/s^2",
            freq_min_hz=0.1,
            freq_max_hz=20.0,
            response_periods=(0.1, 0.2, 0.5, 1.0),
        )
    )


def test_process_station_stream_preserves_input_and_produces_products():
    stream = _physical_acceleration_stream()
    source = stream[0].data.copy()

    contexts = _service().process_station_stream(stream)

    assert np.array_equal(stream[0].data, source)
    context = contexts["HNE"]
    assert context.raw_waveform.unit == "m/s^2"
    assert context.acceleration is not None
    assert context.velocity is not None
    assert context.displacement is not None
    assert context.processing_state.is_qc_complete
    assert context.processing_state.is_integrated
    assert context.processing_state.has_strong_motion_parameters
    assert context.processing_state.has_response_spectrum
    assert context.metrics["PGA"] > 0.0


def test_physical_unit_must_be_declared_without_inventory():
    stream = _physical_acceleration_stream()
    service = AnalysisService(
        AnalysisConfiguration(
            freq_min_hz=0.1,
            freq_max_hz=20.0,
            response_periods=(0.1, 0.5),
        )
    )

    with pytest.raises(AnalysisServiceError, match="Declare input_unit"):
        service.process_station_stream(stream)


def test_duplicate_channel_is_not_silently_overwritten():
    stream = _physical_acceleration_stream()
    duplicate = stream[0].copy()
    duplicate.stats.starttime += 20.0
    stream.append(duplicate)

    with pytest.raises(AnalysisServiceError, match="Duplicate channel"):
        _service().process_station_stream(stream)


def test_extract_summary_data_exposes_physical_values_and_qc():
    contexts = _service().process_station_stream(_physical_acceleration_stream())

    rows = extract_summary_data("TEST", contexts)

    assert rows[0]["station"] == "TEST"
    assert rows[0]["channel"] == "HNE"
    assert rows[0]["pga_gal"] > 0.0
    assert rows[0]["qc_valid"] is True