from datetime import datetime, timedelta

import pytest

from core.types.metadata import TraceMetadata


def create_metadata():

    return TraceMetadata(
        network="IA",
        station="BMKG",
        location="00",
        channel="HNE",
        sampling_rate=100.0,
        starttime=datetime(2024, 1, 1, 0, 0, 0),
        endtime=datetime(2024, 1, 1, 0, 0, 10),
        npts=1000,
        units="m/s²",
        instrument="Etna2",
    )


def test_trace_id():

    md = create_metadata()

    assert md.trace_id == "IA.BMKG.00.HNE"


def test_delta():

    md = create_metadata()

    assert md.delta == pytest.approx(0.01)


def test_duration():

    md = create_metadata()

    assert md.duration == pytest.approx(10.0)


def test_to_dict():

    md = create_metadata()

    d = md.to_dict()

    assert d["network"] == "IA"
    assert d["channel"] == "HNE"
    assert d["sampling_rate"] == 100.0


def test_invalid_sampling_rate():

    with pytest.raises(ValueError):

        TraceMetadata(
            network="IA",
            station="AAA",
            location="00",
            channel="HNE",
            sampling_rate=0,
            starttime=datetime.now(),
            endtime=datetime.now() + timedelta(seconds=10),
            npts=100,
        )


def test_invalid_unit():

    with pytest.raises(ValueError):

        TraceMetadata(
            network="IA",
            station="AAA",
            location="00",
            channel="HNE",
            sampling_rate=100,
            starttime=datetime.now(),
            endtime=datetime.now() + timedelta(seconds=10),
            npts=100,
            units="volt",
        )


def test_invalid_channel():

    with pytest.raises(ValueError):

        TraceMetadata(
            network="IA",
            station="AAA",
            location="00",
            channel="HN",
            sampling_rate=100,
            starttime=datetime.now(),
            endtime=datetime.now() + timedelta(seconds=10),
            npts=100,
        )


def test_invalid_npts():

    with pytest.raises(ValueError):

        TraceMetadata(
            network="IA",
            station="AAA",
            location="00",
            channel="HNE",
            sampling_rate=100,
            starttime=datetime.now(),
            endtime=datetime.now() + timedelta(seconds=10),
            npts=0,
        )


def test_invalid_time_order():

    with pytest.raises(ValueError):

        now = datetime.now()

        TraceMetadata(
            network="IA",
            station="AAA",
            location="00",
            channel="HNE",
            sampling_rate=100,
            starttime=now,
            endtime=now,
            npts=100,
        )