"""
BMKG Strong Motion Analyzer (BSMA)

Module: core/types/processing_state.py

Description
-----------
Domain state model for the BSMA processing pipeline.

The ProcessingState tracks the lifecycle of waveform processing from
raw waveform ingestion through instrument-response correction,
baseline conditioning, filtering, tapering, kinematic integration,
and downstream engineering analysis.

Design Principles
-----------------
- Pure domain layer
- No ObsPy dependency
- Explicit processing-state transitions
- Immutable-friendly usage
- Suitable for MiniSEED + StationXML workflows
- No station-specific assumptions
- Safe for batch processing
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "StageStatus",
    "ProcessingState",
]


class StageStatus(str, Enum):
    """
    Status of an individual BSMA processing stage.

    PENDING
        Stage has not been executed.

    SUCCESS
        Stage completed successfully.

    FAILED
        Stage was attempted but failed.

    SKIPPED
        Stage was intentionally not executed.
    """

    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass(slots=True, frozen=True)
class ProcessingState:
    """
    Immutable processing-state model for BSMA.

    The state describes what has already been performed on the
    waveform. It does not contain numerical waveform data.

    Processing order
    ----------------
    A typical strong-motion workflow is:

        raw
          ↓
        response correction
          ↓
        baseline correction / detrending
          ↓
        taper
          ↓
        filtering
          ↓
        integration
          ↓
        strong-motion parameters
          ↓
        response spectrum
          ↓
        reporting

    Notes
    -----
    The pipeline must not assume that every dataset requires exactly
    the same processing sequence. Station metadata, instrument response,
    sampling rate, record duration, and scientific objective determine
    which stages are valid.

    This class therefore records state only; it does not enforce a
    station-specific processing recipe.
    """

    # ==========================================================
    # Ingestion / Instrument Response
    # ==========================================================

    raw: StageStatus = StageStatus.SUCCESS

    response_correction: StageStatus = StageStatus.PENDING

    # ==========================================================
    # Signal Conditioning
    # ==========================================================

    baseline: StageStatus = StageStatus.PENDING

    detrend: StageStatus = StageStatus.PENDING

    taper: StageStatus = StageStatus.PENDING

    filter: StageStatus = StageStatus.PENDING

    # ==========================================================
    # Kinematic Processing
    # ==========================================================

    integration: StageStatus = StageStatus.PENDING

    # ==========================================================
    # Engineering / Strong-Motion Analysis
    # ==========================================================

    strong_motion_parameters: StageStatus = StageStatus.PENDING

    response_spectrum: StageStatus = StageStatus.PENDING

    # ==========================================================
    # Quality Control
    # ==========================================================

    qc: StageStatus = StageStatus.PENDING

    # ==========================================================
    # Reporting
    # ==========================================================

    report: StageStatus = StageStatus.PENDING

    # ==========================================================
    # General State Helpers
    # ==========================================================

    @property
    def is_raw(self) -> bool:
        """
        Return True when the waveform is still in its raw state.

        A waveform is considered raw when instrument-response
        correction and subsequent signal-conditioning stages have
        not successfully modified the physical waveform.
        """
        return (
            self.raw == StageStatus.SUCCESS
            and self.response_correction != StageStatus.SUCCESS
            and self.baseline != StageStatus.SUCCESS
            and self.detrend != StageStatus.SUCCESS
            and self.taper != StageStatus.SUCCESS
            and self.filter != StageStatus.SUCCESS
        )

    @property
    def is_response_corrected(self) -> bool:
        """Return True when instrument response correction succeeded."""
        return self.response_correction == StageStatus.SUCCESS

    @property
    def is_baseline_corrected(self) -> bool:
        """
        Return True when baseline correction succeeded.

        This property is retained for compatibility with earlier BSMA
        code that may access ``state.is_baseline_corrected``.
        """
        return self.baseline == StageStatus.SUCCESS

    @property
    def is_detrended(self) -> bool:
        """Return True when detrending succeeded."""
        return self.detrend == StageStatus.SUCCESS

    @property
    def is_tapered(self) -> bool:
        """Return True when tapering succeeded."""
        return self.taper == StageStatus.SUCCESS

    @property
    def is_filtered(self) -> bool:
        """Return True when filtering succeeded."""
        return self.filter == StageStatus.SUCCESS

    @property
    def is_integrated(self) -> bool:
        """Return True when kinematic integration succeeded."""
        return self.integration == StageStatus.SUCCESS

    @property
    def has_strong_motion_parameters(self) -> bool:
        """Return True when strong-motion parameters are available."""
        return self.strong_motion_parameters == StageStatus.SUCCESS

    @property
    def has_response_spectrum(self) -> bool:
        """Return True when response spectrum calculation succeeded."""
        return self.response_spectrum == StageStatus.SUCCESS

    @property
    def is_qc_complete(self) -> bool:
        """Return True when QC processing completed successfully."""
        return self.qc == StageStatus.SUCCESS

    @property
    def is_report_complete(self) -> bool:
        """Return True when report generation completed successfully."""
        return self.report == StageStatus.SUCCESS

    # ==========================================================
    # Pipeline Readiness
    # ==========================================================

    @property
    def ready_for_integration(self) -> bool:
        """
        Determine whether the waveform is adequately conditioned
        for kinematic integration.

        Instrument-response correction is not forced here because
        some workflows may already provide physical acceleration.
        The actual integration plugin remains responsible for
        validating physical units.
        """
        conditioning_stages = (
            self.baseline,
            self.detrend,
            self.taper,
            self.filter,
        )

        return all(
            stage in (StageStatus.SUCCESS, StageStatus.SKIPPED)
            for stage in conditioning_stages
        )

    @property
    def ready_for_strong_motion_analysis(self) -> bool:
        """
        Return True when the waveform has successfully passed through
        the minimum conditioning required for strong-motion parameters.
        """
        return (
            self.ready_for_integration
            and self.integration == StageStatus.SUCCESS
        )

    @property
    def ready_for_response_spectrum(self) -> bool:
        """
        Return True when acceleration is available in a physically
        valid processed state for response-spectrum computation.

        Response spectra should normally be calculated from physical
        acceleration rather than arbitrary raw digital counts.
        """
        return (
            self.response_correction in (
                StageStatus.SUCCESS,
                StageStatus.SKIPPED,
            )
            and self.baseline in (
                StageStatus.SUCCESS,
                StageStatus.SKIPPED,
            )
            and self.detrend in (
                StageStatus.SUCCESS,
                StageStatus.SKIPPED,
            )
            and self.taper in (
                StageStatus.SUCCESS,
                StageStatus.SKIPPED,
            )
            and self.filter in (
                StageStatus.SUCCESS,
                StageStatus.SKIPPED,
            )
        )

    # ==========================================================
    # Serialization
    # ==========================================================

    def to_dict(self) -> dict[str, str | bool]:
        """
        Serialize processing state into a lightweight dictionary.

        The returned dictionary contains state values and selected
        convenience flags. Numerical waveform data are intentionally
        excluded.
        """
        return {
            "raw": self.raw.value,
            "response_correction": self.response_correction.value,
            "baseline": self.baseline.value,
            "detrend": self.detrend.value,
            "taper": self.taper.value,
            "filter": self.filter.value,
            "integration": self.integration.value,
            "strong_motion_parameters": (
                self.strong_motion_parameters.value
            ),
            "response_spectrum": self.response_spectrum.value,
            "qc": self.qc.value,
            "report": self.report.value,

            "is_raw": self.is_raw,
            "is_response_corrected": self.is_response_corrected,
            "is_baseline_corrected": self.is_baseline_corrected,
            "is_detrended": self.is_detrended,
            "is_tapered": self.is_tapered,
            "is_filtered": self.is_filtered,
            "is_integrated": self.is_integrated,
            "has_strong_motion_parameters": (
                self.has_strong_motion_parameters
            ),
            "has_response_spectrum": self.has_response_spectrum,
            "is_qc_complete": self.is_qc_complete,
            "is_report_complete": self.is_report_complete,
            "ready_for_integration": self.ready_for_integration,
            "ready_for_strong_motion_analysis": (
                self.ready_for_strong_motion_analysis
            ),
            "ready_for_response_spectrum": (
                self.ready_for_response_spectrum
            ),
        }

    # ==========================================================
    # Representation
    # ==========================================================

    def __repr__(self) -> str:
        """
        Return a compact diagnostic representation.
        """
        return (
            f"{self.__class__.__name__}("
            f"raw={self.raw.value}, "
            f"response={self.response_correction.value}, "
            f"baseline={self.baseline.value}, "
            f"detrend={self.detrend.value}, "
            f"taper={self.taper.value}, "
            f"filter={self.filter.value}, "
            f"integration={self.integration.value}, "
            f"strong_motion={self.strong_motion_parameters.value}, "
            f"response_spectrum={self.response_spectrum.value})"
        )