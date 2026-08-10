"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/types/metadata.py

Domain Types
============

Trace metadata shared across the BSMA processing pipeline.

This module intentionally contains no dependency on ObsPy.

The metadata model describes the physical and acquisition properties
of one waveform trace and is used as a lightweight domain object
throughout the processing pipeline.

Design Principles
-----------------
- Immutable domain model
- No ObsPy dependency
- Explicit physical units
- Strong validation of acquisition parameters
- SEED-compatible trace identification
- Numerically consistent timing information
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math


__all__ = [
    "TraceMetadata",
]


# ============================================================================
# Supported Physical Units
# ============================================================================

_ALLOWED_UNITS = frozenset(
    {
        "counts",
        "m/s²",
        "m/s^2",
        "cm/s²",
        "cm/s^2",
        "m/s",
        "cm/s",
        "m",
        "cm",
        "gal",
        "g",
    }
)


# ============================================================================
# Trace Metadata
# ============================================================================


@dataclass(frozen=True, slots=True)
class TraceMetadata:
    """
    Immutable metadata describing one waveform trace.

    Parameters
    ----------
    network
        Seismic network code.

    station
        Station code.

    location
        SEED location code. An empty string is valid when the source
        waveform does not define a location code.

    channel
        SEED channel code, for example ``HNE``, ``HNN`` or ``HNZ``.

    sampling_rate
        Sampling frequency in Hz.

    starttime
        Record start time.

    endtime
        Record end time corresponding to the timestamp of the final
        waveform sample.

    npts
        Number of waveform samples.

    unit
        Physical unit of the waveform.

    instrument
        Sensor or recorder model, if available.

    Notes
    -----
    For a uniformly sampled waveform:

        duration = (npts - 1) / sampling_rate

    when ``npts > 1``.

    Therefore ``endtime`` is expected to satisfy:

        endtime = starttime + (npts - 1) * delta

    within the precision of the source metadata.

    The model does not require the unit to be ``m/s²`` because ingestion
    and calibration may temporarily operate in counts or other physical
    units. Unit normalization belongs to the waveform/calibration stage.
    """

    # ======================================================================
    # SEED Identity
    # ======================================================================

    network: str
    station: str
    location: str
    channel: str

    # ======================================================================
    # Acquisition
    # ======================================================================

    sampling_rate: float

    starttime: datetime
    endtime: datetime

    npts: int

    # ======================================================================
    # Physical Metadata
    # ======================================================================

    unit: str = "m/s²"
    instrument: str = ""

    # ======================================================================
    # Validation
    # ======================================================================

    def __post_init__(self) -> None:
        """Validate metadata consistency and normalize textual fields."""

        # ------------------------------------------------------------------
        # Text fields
        # ------------------------------------------------------------------

        if not isinstance(self.network, str):
            raise TypeError("network must be a string.")

        if not isinstance(self.station, str):
            raise TypeError("station must be a string.")

        if not isinstance(self.location, str):
            raise TypeError("location must be a string.")

        if not isinstance(self.channel, str):
            raise TypeError("channel must be a string.")

        if not isinstance(self.unit, str):
            raise TypeError("unit must be a string.")

        if not isinstance(self.instrument, str):
            raise TypeError("instrument must be a string.")

        network = self.network.strip()
        station = self.station.strip()
        location = self.location.strip()
        channel = self.channel.strip().upper()
        unit = self.unit.strip()

        if not network:
            raise ValueError("network cannot be empty.")

        if not station:
            raise ValueError("station cannot be empty.")

        # Empty location code is valid in SEED.
        if len(location) > 2:
            raise ValueError(
                "SEED location code must contain at most 2 characters."
            )

        # SEED channel code consists of three characters:
        # band + instrument + orientation.
        if len(channel) != 3:
            raise ValueError(
                f"Invalid SEED channel code {channel!r}. "
                "Expected exactly 3 characters."
            )

        if not all(
            character.isalnum()
            for character in channel
        ):
            raise ValueError(
                f"Invalid SEED channel code {channel!r}."
            )

        # ------------------------------------------------------------------
        # Sampling rate
        # ------------------------------------------------------------------

        sampling_rate = float(self.sampling_rate)

        if not math.isfinite(sampling_rate):
            raise ValueError(
                "sampling_rate must be finite."
            )

        if sampling_rate <= 0.0:
            raise ValueError(
                "sampling_rate must be > 0 Hz."
            )

        # ------------------------------------------------------------------
        # Number of samples
        # ------------------------------------------------------------------

        if isinstance(self.npts, bool):
            raise TypeError("npts must be an integer, not bool.")

        npts = int(self.npts)

        if npts <= 0:
            raise ValueError(
                "npts must be positive."
            )

        # ------------------------------------------------------------------
        # Datetime validation
        # ------------------------------------------------------------------

        if not isinstance(self.starttime, datetime):
            raise TypeError(
                "starttime must be a datetime."
            )

        if not isinstance(self.endtime, datetime):
            raise TypeError(
                "endtime must be a datetime."
            )

        # ------------------------------------------------------------------
        # Normalize timezone information.
        #
        # Naive datetime is retained because some existing BSMA ingestion
        # paths may provide naive UTC datetimes. We do NOT silently alter
        # the instant here.
        # ------------------------------------------------------------------

        if (
            self.starttime.tzinfo is not None
            and self.endtime.tzinfo is not None
        ):
            start_utc = self.starttime.astimezone(timezone.utc)
            end_utc = self.endtime.astimezone(timezone.utc)

            if end_utc < start_utc:
                raise ValueError(
                    "endtime must not precede starttime."
                )

        else:
            if self.endtime < self.starttime:
                raise ValueError(
                    "endtime must not precede starttime."
                )

        # ------------------------------------------------------------------
        # Physical unit
        # ------------------------------------------------------------------

        if unit not in _ALLOWED_UNITS:
            raise ValueError(
                f"Unsupported waveform unit {unit!r}. "
                f"Supported units: {sorted(_ALLOWED_UNITS)}"
            )

        # ------------------------------------------------------------------
        # Instrument
        # ------------------------------------------------------------------

        instrument = self.instrument.strip()

        # ------------------------------------------------------------------
        # Store normalized values
        # ------------------------------------------------------------------

        object.__setattr__(self, "network", network)
        object.__setattr__(self, "station", station)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "channel", channel)

        object.__setattr__(
            self,
            "sampling_rate",
            sampling_rate,
        )

        object.__setattr__(
            self,
            "npts",
            npts,
        )

        object.__setattr__(
            self,
            "unit",
            unit,
        )

        object.__setattr__(
            self,
            "instrument",
            instrument,
        )

        # ------------------------------------------------------------------
        # Timing consistency
        # ------------------------------------------------------------------

        self._validate_timing_consistency()

    # ======================================================================
    # Timing
    # ======================================================================

    @property
    def delta(self) -> float:
        """
        Sampling interval in seconds.

        Defined as:

            Δt = 1 / fs
        """
        return 1.0 / self.sampling_rate

    @property
    def duration(self) -> float:
        """
        Record duration in seconds.

        For N samples:

            T = (N - 1) / fs

        This represents elapsed time between the first and final sample.
        """
        if self.npts <= 1:
            return 0.0

        return float(
            (self.npts - 1) / self.sampling_rate
        )

    # ======================================================================
    # SEED Identity
    # ======================================================================

    @property
    def trace_id(self) -> str:
        """
        Canonical SEED trace identifier.

        Format:

            NET.STA.LOC.CHA

        An empty location code is intentionally preserved:

            IA.SBSSI..HNN
        """
        return (
            f"{self.network}."
            f"{self.station}."
            f"{self.location}."
            f"{self.channel}"
        )

    # ======================================================================
    # Convenience Properties
    # ======================================================================

    @property
    def component(self) -> str:
        """
        Return the channel orientation/component code.

        Examples
        --------
        ``HNE`` -> ``E``

        ``HNN`` -> ``N``

        ``HNZ`` -> ``Z``
        """
        return self.channel[-1]

    @property
    def band_code(self) -> str:
        """
        Return the SEED channel band code.
        """
        return self.channel[0]

    @property
    def instrument_code(self) -> str:
        """
        Return the SEED channel instrument code.
        """
        return self.channel[1]

    # ======================================================================
    # Serialization
    # ======================================================================

    def to_dict(self) -> dict[str, object]:
        """
        Serialize metadata into a JSON-compatible dictionary.
        """
        return {
            "network": self.network,
            "station": self.station,
            "location": self.location,
            "channel": self.channel,
            "trace_id": self.trace_id,
            "component": self.component,
            "band_code": self.band_code,
            "instrument_code": self.instrument_code,
            "sampling_rate": self.sampling_rate,
            "delta": self.delta,
            "npts": self.npts,
            "duration": self.duration,
            "unit": self.unit,
            "instrument": self.instrument,
            "starttime": self.starttime.isoformat(),
            "endtime": self.endtime.isoformat(),
        }

    # ======================================================================
    # Internal Validation
    # ======================================================================

    def _validate_timing_consistency(self) -> None:
        """
        Validate consistency between timestamps, sample count and sampling
        rate.

        The expected elapsed duration is:

            T_expected = (N - 1) / fs

        A small tolerance is allowed because timestamps originating from
        different file formats may have finite precision.
        """

        if self.npts == 1:
            if self.endtime != self.starttime:
                raise ValueError(
                    "For a single-sample trace, endtime must equal starttime."
                )
            return

        if self.starttime.tzinfo is not None and self.endtime.tzinfo is not None:
            elapsed = (
                self.endtime.astimezone(timezone.utc)
                - self.starttime.astimezone(timezone.utc)
            ).total_seconds()
        else:
            elapsed = (
                self.endtime - self.starttime
            ).total_seconds()

        expected = self.duration

        tolerance = max(
            1e-6,
            0.5 * self.delta,
        )

        if not math.isclose(
            elapsed,
            expected,
            rel_tol=1e-6,
            abs_tol=tolerance,
        ):
            raise ValueError(
                "Inconsistent trace timing: "
                f"observed duration={elapsed:.9f} s, "
                f"expected duration={expected:.9f} s "
                f"from npts={self.npts} and "
                f"sampling_rate={self.sampling_rate} Hz."
            )