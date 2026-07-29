from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class SeverityLevel(str, Enum):
    """Severity level for BSMA exceptions."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ErrorCode(str, Enum):
    """Standardized BSMA error codes."""

    # ------------------------------------------------------------------
    # Generic (GEN)
    # ------------------------------------------------------------------
    GEN001 = "GEN001"  # Unknown error
    GEN002 = "GEN002"  # Invalid argument
    GEN003 = "GEN003"  # Internal application error

    # ------------------------------------------------------------------
    # Configuration (CFG)
    # ------------------------------------------------------------------
    CFG001 = "CFG001"  # Configuration file not found
    CFG002 = "CFG002"  # Invalid configuration
    CFG003 = "CFG003"  # Missing required parameter

    # ------------------------------------------------------------------
    # Waveform (WF)
    # ------------------------------------------------------------------
    WF001 = "WF001"  # Sampling rate mismatch
    WF002 = "WF002"  # Unsupported waveform format
    WF003 = "WF003"  # Empty waveform stream
    WF004 = "WF004"  # Waveform read failure

    # ------------------------------------------------------------------
    # Quality Control (QC)
    # ------------------------------------------------------------------
    QC001 = "QC001"  # Spike detected
    QC002 = "QC002"  # Clipping detected
    QC003 = "QC003"  # Gap detected
    QC004 = "QC004"  # Offset detected
    QC005 = "QC005"  # QC validation failed
    QC006 = "QC006"  # Noise level too high
    QC007 = "QC007"  # Signal saturation
    QC008 = "QC008"  # Dead channel

    # ------------------------------------------------------------------
    # Preprocessing (PR)
    # ------------------------------------------------------------------
    PR001 = "PR001"  # Preprocessing failure
    PR002 = "PR002"  # Numerical integration failure
    PR003 = "PR003"  # Filtering failure

    # ------------------------------------------------------------------
    # Strong Motion (SM)
    # ------------------------------------------------------------------
    SM001 = "SM001"  # PGA computation failure
    SM002 = "SM002"  # PGV integration drift
    SM003 = "SM003"  # Arias Intensity computation failure

    # ------------------------------------------------------------------
    # Response Spectrum (RS)
    # ------------------------------------------------------------------
    RS001 = "RS001"  # Response spectrum failure

    # ------------------------------------------------------------------
    # Reporting (RP)
    # ------------------------------------------------------------------
    RP001 = "RP001"  # Report generation failure
    RP002 = "RP002"  # Figure export failure
    RP003 = "RP003"  # Template missing
    RP004 = "RP004"  # PDF write failure

    # ------------------------------------------------------------------
    # Batch Processing (BT)
    # ------------------------------------------------------------------
    BT001 = "BT001"  # Batch processing failure
    BT002 = "BT002"  # Worker timeout
    BT003 = "BT003"  # Queue failed
    BT004 = "BT004"  # Task cancelled


@dataclass(slots=True, repr=False)
class BSMAError(Exception):
    """Base exception for all BSMA modules."""

    message: str
    error_code: ErrorCode = ErrorCode.GEN001
    severity: SeverityLevel = SeverityLevel.ERROR

    context: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=UTC),
        init=False,
    )

    cause: Exception | None = field(
        default=None,
    )

    def __post_init__(self) -> None:
        super().__init__(self.message)
        # Mendukung exception chaining bawaan Python secara native
        if self.cause is not None:
            self.__cause__ = self.cause

    # --- Properties Akses Cepat ---
    @property
    def module(self) -> str | None:
        return self.context.get("module")

    @property
    def station(self) -> str | None:
        return self.context.get("station")

    @property
    def network(self) -> str | None:
        return self.context.get("network")
        
    @property
    def network_station(self) -> str | None:
        if self.network and self.station:
            return f"{self.network}.{self.station}"
        return None

    @property
    def channel(self) -> str | None:
        return self.context.get("channel")
        
    @property
    def location(self) -> str | None:
        return self.context.get("location")

    @property
    def event_id(self) -> str | None:
        return self.context.get("event_id")

    @property
    def filename(self) -> str | None:
        return self.context.get("file")

    @property
    def event_time(self) -> str | None:
        return self.context.get("event_time")

    @property
    def magnitude(self) -> float | None:
        return self.context.get("magnitude")

    def __str__(self) -> str:
        """Return a human-readable representation of the exception."""
        return (
            f"[{self.error_code}] "
            f"{self.severity.value}: "
            f"{self.message}"
        )

    def __repr__(self) -> str:
        """Return a detailed representation for debugging purposes."""
        cause_repr = repr(self.cause) if self.cause else None
        return (
            f"{self.__class__.__name__}("
            f"error_code={self.error_code.value!r}, "
            f"severity={self.severity.value!r}, "
            f"message={self.message!r}, "
            f"context={self.context!r}, "
            f"details={self.details!r}, "
            f"timestamp={self.timestamp.isoformat()!r}, "
            f"cause={cause_repr!r}"
            f")"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the exception as a JSON-serializable dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "exception": self.__class__.__name__,
            "error_code": self.error_code.value,
            "severity": self.severity.value,
            "message": self.message,
            "module": self.module,
            "context": self.context,
            "details": self.details,
            "cause": repr(self.cause) if self.cause is not None else None,
        }

    def to_json(self, indent: int = 4) -> str:
        """Return the exception as a formatted JSON string (Unicode safe)."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def log(self, logger: logging.Logger, stack_info: bool = False) -> None:
        """Log this exception using the configured BSMA logger."""
        level = {
            SeverityLevel.INFO: logging.INFO,
            SeverityLevel.WARNING: logging.WARNING,
            SeverityLevel.ERROR: logging.ERROR,
            SeverityLevel.CRITICAL: logging.CRITICAL,
        }[self.severity]

        # Logging bersih tanpa memaksakan exc_info yang rentan jika di luar blok except.
        # Traceback tetap aman karena self.__cause__ sudah menyimpan error aslinya.
        logger.log(
            level,
            str(self),
            extra={"bsma_exception": self.to_dict()},
            stack_info=stack_info,
        )


@dataclass(slots=True, repr=False)
class ConfigurationError(BSMAError):
    error_code: ErrorCode = ErrorCode.CFG001
    severity: SeverityLevel = SeverityLevel.ERROR


@dataclass(slots=True, repr=False)
class WaveformError(BSMAError):
    error_code: ErrorCode = ErrorCode.WF004
    severity: SeverityLevel = SeverityLevel.ERROR


@dataclass(slots=True, repr=False)
class QCError(BSMAError):
    error_code: ErrorCode = ErrorCode.QC005
    severity: SeverityLevel = SeverityLevel.WARNING


@dataclass(slots=True, repr=False)
class ProcessingError(BSMAError):
    error_code: ErrorCode = ErrorCode.PR001
    severity: SeverityLevel = SeverityLevel.ERROR


@dataclass(slots=True, repr=False)
class StrongMotionError(BSMAError):
    error_code: ErrorCode = ErrorCode.SM001
    severity: SeverityLevel = SeverityLevel.ERROR


@dataclass(slots=True, repr=False)
class ResponseSpectrumError(BSMAError):
    error_code: ErrorCode = ErrorCode.RS001
    severity: SeverityLevel = SeverityLevel.ERROR


@dataclass(slots=True, repr=False)
class ReportError(BSMAError):
    error_code: ErrorCode = ErrorCode.RP001
    severity: SeverityLevel = SeverityLevel.ERROR


@dataclass(slots=True, repr=False)
class BatchProcessingError(BSMAError):
    error_code: ErrorCode = ErrorCode.BT001
    severity: SeverityLevel = SeverityLevel.CRITICAL