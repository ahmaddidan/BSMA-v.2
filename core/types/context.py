"""
BMKG Strong Motion Analyzer (BSMA)

Domain Types
============

Root aggregate for the BSMA processing pipeline.

The ProcessingContext is the single object flowing through every stage
of the preprocessing and analysis pipeline. It groups together waveform
data, metadata, QC information, processing state, cache, provenance,
and configuration.

Design goals
------------
- Single Source of Truth (SSOT)
- Lightweight root aggregate
- Compatible with immutable workflow
- No ObsPy dependency
- Production-grade typing
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .cache import ProcessingCache
from .metadata import TraceMetadata
from .processing_state import ProcessingState
from .qc import QCReport

__all__ = [
    "ProcessingContext",
]

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class ProcessingContext:
    """
    Root aggregate of the BSMA processing pipeline.
    """

    waveform: FloatArray | None

    metadata: TraceMetadata

    processing_state: ProcessingState = field(
        default_factory=ProcessingState
    )

    cache: ProcessingCache = field(
        default_factory=ProcessingCache
    )

    qc: QCReport = field(
        default_factory=QCReport
    )

    history: tuple[str, ...] = field(
        default_factory=tuple
    )

    config: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def sampling_rate(self) -> float:
        """Sampling rate in Hz."""
        return self.metadata.sampling_rate

    @property
    def npts(self) -> int:
        """Number of waveform samples (SSOT from metadata)."""
        return self.metadata.npts

    @property
    def duration(self) -> float:
        """Waveform duration in seconds."""
        return self.metadata.duration

    # ------------------------------------------------------------------
    # Functional update
    # ------------------------------------------------------------------

    def copy(self, **changes: Any) -> "ProcessingContext":
        """
        Return a new ProcessingContext with selected fields updated.
        """
        if "config" in changes:
            changes["config"] = MappingProxyType(
                dict(changes["config"])
            )

        return replace(self, **changes)

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def add_history(self, message: str) -> "ProcessingContext":
        """
        Return a new context with an additional history record.
        """
        return self.copy(
            history=self.history + (message,)
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize context into lightweight dictionary.
        Numerical arrays themselves are intentionally omitted.
        """
        return {
            "sampling_rate": self.sampling_rate,
            "npts": self.npts,
            "duration": self.duration,
            "history_length": len(self.history),
            "processing_state": self.processing_state.to_dict(),
            "cache": self.cache.to_dict(),
            "qc": self.qc.to_dict(),
            "config": dict(self.config),
            "metadata": {
                "network": self.metadata.network,
                "station": self.metadata.station,
                "location": self.metadata.location,
                "channel": self.metadata.channel,
                "instrument": self.metadata.instrument,
                "units": self.metadata.units,
            },
        }

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return waveform length based on metadata."""
        return self.npts

    def __bool__(self) -> bool:
        """True if waveform is available."""
        return self.waveform is not None