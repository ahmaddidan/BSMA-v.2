"""
BMKG Strong Motion Analyzer (BSMA)
Module: utils/exceptions.py

Description
-----------
Centralized exception hierarchy for BSMA.

This module provides:

- Standardized severity levels.
- Standardized error codes.
- Structured contextual information.
- UTC-aware timestamps.
- Native Python exception chaining.
- JSON serialization.
- Structured logging support.
- Specialized exception classes for each BSMA subsystem.

Architectural principles
------------------------
- Single Source of Truth for application errors.
- Strong typing.
- Production-grade diagnostics.
- JSON-friendly serialization.
- Compatible with standard Python exception handling.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


__all__ = [
    "SeverityLevel",
    "ErrorCode",
    "BSMAError",
    "ConfigurationError",
    "WaveformError",
    "QCError",
    "ProcessingError",
    "StrongMotionError",
    "ResponseSpectrumError",
    "ReportError",
    "BatchProcessingError",
]


# ============================================================================
# SEVERITY
# ============================================================================


class SeverityLevel(str, Enum):
    """
    Severity level assigned to a BSMA exception.

    Levels
    ------
    INFO
        Informational condition.

    WARNING
        Recoverable or non-fatal condition.

    ERROR
        Processing or operational failure.

    CRITICAL
        Severe failure requiring immediate attention.
    """

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ============================================================================
# ERROR CODES
# ============================================================================


class ErrorCode(str, Enum):
    """
    Standardized BSMA error codes.

    Error codes are grouped by subsystem:

    GEN : Generic application errors
    CFG : Configuration
    WF  : Waveform ingestion
    QC  : Quality control
    PR  : Preprocessing
    SM  : Strong-motion parameters
    RS  : Response spectrum
    RP  : Reporting
    BT  : Batch processing
    """

    # ------------------------------------------------------------------
    # Generic (GEN)
    # ------------------------------------------------------------------

    GEN001 = "GEN001"
    GEN002 = "GEN002"
    GEN003 = "GEN003"

    # ------------------------------------------------------------------
    # Configuration (CFG)
    # ------------------------------------------------------------------

    CFG001 = "CFG001"
    CFG002 = "CFG002"
    CFG003 = "CFG003"

    # ------------------------------------------------------------------
    # Waveform (WF)
    # ------------------------------------------------------------------

    WF001 = "WF001"
    WF002 = "WF002"
    WF003 = "WF003"
    WF004 = "WF004"

    # ------------------------------------------------------------------
    # Quality Control (QC)
    # ------------------------------------------------------------------

    QC001 = "QC001"
    QC002 = "QC002"
    QC003 = "QC003"
    QC004 = "QC004"
    QC005 = "QC005"
    QC006 = "QC006"
    QC007 = "QC007"
    QC008 = "QC008"

    # ------------------------------------------------------------------
    # Preprocessing (PR)
    # ------------------------------------------------------------------

    PR001 = "PR001"
    PR002 = "PR002"
    PR003 = "PR003"

    # ------------------------------------------------------------------
    # Strong Motion (SM)
    # ------------------------------------------------------------------

    SM001 = "SM001"
    SM002 = "SM002"
    SM003 = "SM003"

    # ------------------------------------------------------------------
    # Response Spectrum (RS)
    # ------------------------------------------------------------------

    RS001 = "RS001"

    # ------------------------------------------------------------------
    # Reporting (RP)
    # ------------------------------------------------------------------

    RP001 = "RP001"
    RP002 = "RP002"
    RP003 = "RP003"
    RP004 = "RP004"

    # ------------------------------------------------------------------
    # Batch Processing (BT)
    # ------------------------------------------------------------------

    BT001 = "BT001"
    BT002 = "BT002"
    BT003 = "BT003"
    BT004 = "BT004"


# ============================================================================
# BASE BSMA EXCEPTION
# ============================================================================


@dataclass(slots=True, repr=False)
class BSMAError(Exception):
    """
    Base exception for all BSMA application errors.

    Parameters
    ----------
    message:
        Human-readable description of the failure.

    error_code:
        Standardized BSMA error code.

    severity:
        Severity classification.

    context:
        Operational context such as module, trace ID, station,
        network, channel, event ID, or filename.

    details:
        Additional structured diagnostic information.

    timestamp:
        UTC timestamp generated automatically.

    cause:
        Optional underlying exception.

    Notes
    -----
    The class inherits from :class:`Exception`, therefore it remains
    compatible with normal Python exception handling:

    >>> raise ProcessingError(
    ...     message="Integration failed",
    ... )

    Native chaining is also supported:

    >>> try:
    ...     ...
    ... except ValueError as exc:
    ...     raise ProcessingError(
    ...         message="Numerical processing failed",
    ...         cause=exc,
    ...     ) from exc
    """

    message: str

    error_code: ErrorCode = ErrorCode.GEN001

    severity: SeverityLevel = SeverityLevel.ERROR

    context: dict[str, Any] = field(
        default_factory=dict,
    )

    details: dict[str, Any] = field(
        default_factory=dict,
    )

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        init=False,
    )

    cause: Exception | None = field(
        default=None,
    )

    def __post_init__(self) -> None:
        """
        Initialize the underlying Python Exception object and establish
        explicit exception chaining when a cause is supplied.
        """
        Exception.__init__(self, self.message)

        if self.cause is not None:
            self.__cause__ = self.cause

    # ========================================================================
    # QUICK ACCESS PROPERTIES
    # ========================================================================

    @property
    def module(self) -> str | None:
        """Return originating BSMA module."""
        return self.context.get("module")

    @property
    def station(self) -> str | None:
        """Return seismic station code."""
        return self.context.get("station")

    @property
    def network(self) -> str | None:
        """Return seismic network code."""
        return self.context.get("network")

    @property
    def network_station(self) -> str | None:
        """Return canonical NETWORK.STATION identifier."""
        if self.network and self.station:
            return f"{self.network}.{self.station}"

        return None

    @property
    def channel(self) -> str | None:
        """Return seismic channel code."""
        return self.context.get("channel")

    @property
    def location(self) -> str | None:
        """Return SEED location code."""
        return self.context.get("location")

    @property
    def event_id(self) -> str | None:
        """Return event identifier."""
        return self.context.get("event_id")

    @property
    def filename(self) -> str | None:
        """Return associated waveform filename."""
        return self.context.get("file")

    @property
    def event_time(self) -> str | None:
        """Return event origin time if available."""
        return self.context.get("event_time")

    @property
    def magnitude(self) -> float | None:
        """Return event magnitude if available."""
        value = self.context.get("magnitude")

        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ========================================================================
    # STRING REPRESENTATION
    # ========================================================================

    def __str__(self) -> str:
        """
        Return concise human-readable exception representation.
        """
        return (
            f"[{self.error_code.value}] "
            f"{self.severity.value}: "
            f"{self.message}"
        )

    def __repr__(self) -> str:
        """
        Return detailed debugging representation.
        """
        cause_repr = (
            repr(self.cause)
            if self.cause is not None
            else None
        )

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

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the exception to a JSON-compatible dictionary.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "exception": self.__class__.__name__,
            "error_code": self.error_code.value,
            "severity": self.severity.value,
            "message": self.message,
            "module": self.module,
            "context": dict(self.context),
            "details": dict(self.details),
            "cause": (
                repr(self.cause)
                if self.cause is not None
                else None
            ),
        }

    def to_json(self, indent: int = 4) -> str:
        """
        Serialize the exception to formatted JSON.

        Parameters
        ----------
        indent:
            JSON indentation level.

        Returns
        -------
        str
            Unicode-safe JSON representation.
        """
        if indent < 0:
            raise ValueError("indent must be >= 0.")

        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            default=str,
        )

    # ========================================================================
    # LOGGING
    # ========================================================================

    def log(
        self,
        logger: logging.Logger,
        *,
        stack_info: bool = False,
    ) -> None:
        """
        Log this BSMA exception using a supplied logger.

        The complete structured exception payload is passed through
        ``extra["bsma_exception"]``.
        """
        if not isinstance(logger, logging.Logger):
            raise TypeError(
                "logger must be an instance of logging.Logger."
            )

        log_level = {
            SeverityLevel.INFO: logging.INFO,
            SeverityLevel.WARNING: logging.WARNING,
            SeverityLevel.ERROR: logging.ERROR,
            SeverityLevel.CRITICAL: logging.CRITICAL,
        }[self.severity]

        logger.log(
            log_level,
            str(self),
            extra={
                "bsma_exception": self.to_dict(),
            },
            stack_info=stack_info,
        )


# ============================================================================
# SPECIALIZED EXCEPTIONS
# ============================================================================


@dataclass(slots=True, repr=False)
class ConfigurationError(BSMAError):
    """
    Exception raised for BSMA configuration failures.
    """

    error_code: ErrorCode = ErrorCode.CFG001
    severity: SeverityLevel = SeverityLevel.ERROR


@dataclass(slots=True, repr=False)
class WaveformError(BSMAError):
    """
    Exception raised for waveform ingestion or validation failures.
    """

    error_code: ErrorCode = ErrorCode.WF004
    severity: SeverityLevel = SeverityLevel.ERROR


@dataclass(slots=True, repr=False)
class QCError(BSMAError):
    """
    Exception raised for quality-control failures.

    Default severity is WARNING because individual QC findings may be
    recoverable. Critical QC conditions can still explicitly override
    ``severity``.
    """

    error_code: ErrorCode = ErrorCode.QC005
    severity: SeverityLevel = SeverityLevel.WARNING


@dataclass(slots=True, repr=False)
class ProcessingError(BSMAError):
    """
    Exception raised during preprocessing or numerical processing.
    """

    error_code: ErrorCode = ErrorCode.PR001
    severity: SeverityLevel = SeverityLevel.ERROR


@dataclass(slots=True, repr=False)
class StrongMotionError(BSMAError):
    """
    Exception raised during strong-motion parameter extraction.
    """

    error_code: ErrorCode = ErrorCode.SM001
    severity: SeverityLevel = SeverityLevel.ERROR


@dataclass(slots=True, repr=False)
class ResponseSpectrumError(BSMAError):
    """
    Exception raised during response-spectrum computation.
    """

    error_code: ErrorCode = ErrorCode.RS001
    severity: SeverityLevel = SeverityLevel.ERROR


@dataclass(slots=True, repr=False)
class ReportError(BSMAError):
    """
    Exception raised during report generation or export.
    """

    error_code: ErrorCode = ErrorCode.RP001
    severity: SeverityLevel = SeverityLevel.ERROR


@dataclass(slots=True, repr=False)
class BatchProcessingError(BSMAError):
    """
    Exception raised during batch-processing execution.
    """

    error_code: ErrorCode = ErrorCode.BT001
    severity: SeverityLevel = SeverityLevel.CRITICAL