"""
BMKG Strong Motion Analyzer (BSMA)

Domain Types: Processing State
"""

from enum import Enum
from dataclasses import dataclass

__all__ = [
    "StageStatus",
    "ProcessingState",
]


class StageStatus(str, Enum):
    """Enumeration of possible processing stage statuses."""
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass(slots=True)
class ProcessingState:
    """
    Tracks the status of each preprocessing and analysis stage.
    """
    baseline: StageStatus = StageStatus.PENDING
    detrend: StageStatus = StageStatus.PENDING
    taper: StageStatus = StageStatus.PENDING
    filter: StageStatus = StageStatus.PENDING
    integration: StageStatus = StageStatus.PENDING

    def to_dict(self) -> dict[str, str]:
        """Serialize state into a lightweight dictionary."""
        return {
            "baseline": self.baseline.value,
            "detrend": self.detrend.value,
            "taper": self.taper.value,
            "filter": self.filter.value,
            "integration": self.integration.value,
        }