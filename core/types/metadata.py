"""
BMKG Strong Motion Analyzer (BSMA)

Domain Types
============

Trace metadata shared across the processing pipeline.

This module intentionally contains no dependency on ObsPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


_ALLOWED_UNITS = {
    "m/s²",
    "m/s^2",
    "gal",
    "cm/s²",
    "cm/s^2",
    "g",
}


@dataclass(slots=True, frozen=True)
class TraceMetadata:
    """
    Immutable metadata describing one waveform.

    Parameters
    ----------
    network
        Seismic network code.

    station
        Station code.

    location
        SEED location code.

    channel
        Channel code (HNE, HNN, HNZ, etc.).

    sampling_rate
        Sampling frequency (Hz).

    starttime
        Record start time.

    endtime
        Record end time.

    npts
        Number of samples.

    units
        Physical unit.

    instrument
        Sensor or recorder model.
    """

    network: str
    station: str
    location: str
    channel: str

    sampling_rate: float

    starttime: datetime
    endtime: datetime

    npts: int

    units: str = "m/s²"
    instrument: str = ""

    def __post_init__(self) -> None:

        if self.sampling_rate <= 0:
            raise ValueError(
                "sampling_rate must be > 0 Hz."
            )

        if self.npts <= 0:
            raise ValueError(
                "npts must be positive."
            )

        if self.endtime <= self.starttime:
            raise ValueError(
                "endtime must be later than starttime."
            )

        if self.units not in _ALLOWED_UNITS:
            raise ValueError(
                f"Unsupported unit '{self.units}'."
            )

        if len(self.channel) < 3:
            raise ValueError(
                "Invalid SEED channel code."
            )

    @property
    def duration(self) -> float:
        """
        Record duration (seconds).
        """
        return (
            self.endtime - self.starttime
        ).total_seconds()

    @property
    def delta(self) -> float:
        """
        Sampling interval (seconds).
        """
        return 1.0 / self.sampling_rate

    @property
    def trace_id(self) -> str:
        """
        Canonical trace identifier.

        NET.STA.LOC.CHA
        """
        return (
            f"{self.network}."
            f"{self.station}."
            f"{self.location}."
            f"{self.channel}"
        )

    def to_dict(self) -> dict:

        return {
            "network": self.network,
            "station": self.station,
            "location": self.location,
            "channel": self.channel,
            "sampling_rate": self.sampling_rate,
            "delta": self.delta,
            "npts": self.npts,
            "duration": self.duration,
            "units": self.units,
            "instrument": self.instrument,
            "starttime": self.starttime.isoformat(),
            "endtime": self.endtime.isoformat(),
            "trace_id": self.trace_id,
        }