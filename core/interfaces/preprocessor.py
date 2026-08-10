"""
BMKG Strong Motion Analyzer (BSMA)

Module
------
core/interfaces/preprocessor.py

Description
-----------
Abstract interface for processing plugins used by the BSMA
signal-processing pipeline.

Every processing algorithm operating on waveform data must
implement ``PreprocessorPlugin``.

Design Principles
-----------------
- Single Responsibility Principle
- Open/Closed Principle
- Immutable ProcessingContext
- Strict domain contracts
- Explicit waveform validation
- Deterministic plugin behavior
- Production-grade error handling
- Compatible with PipelineOrchestrator
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.types.context import ProcessingContext, WaveformData
from utils.exceptions import (
    ErrorCode,
    ProcessingError,
    SeverityLevel,
)

__all__ = [
    "PreprocessorPlugin",
]


class PreprocessorPlugin(ABC):
    """
    Abstract base class for BSMA processing plugins.

    Contract
    --------
    A plugin:

    1. receives a ``ProcessingContext``;
    2. validates the required waveform;
    3. performs one well-defined processing operation;
    4. returns a new ``ProcessingContext``;
    5. must not mutate the input context;
    6. must preserve the physical unit unless the algorithm
       explicitly changes it;
    7. must preserve waveform sampling rate;
    8. must not overwrite ``raw_waveform``.

    Notes
    -----
    ``raw_waveform`` represents the original waveform and is part
    of the provenance chain. Preprocessing plugins operate on the
    processed representation, normally ``context.acceleration``.
    """

    # ==========================================================
    # Plugin Metadata
    # ==========================================================

    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """
        Return the unique human-readable plugin name.
        """
        raise NotImplementedError

    @property
    def plugin_version(self) -> str:
        """
        Return the semantic version of the plugin.
        """
        return "1.0.0"

    @property
    def plugin_description(self) -> str:
        """
        Return a concise description of the algorithm.
        """
        return ""

    # ==========================================================
    # Input Validation
    # ==========================================================

    def validate_input(
        self,
        context: ProcessingContext,
    ) -> WaveformData:
        """
        Validate and return the active processed waveform.

        Parameters
        ----------
        context
            Current immutable processing context.

        Returns
        -------
        WaveformData
            Validated acceleration/waveform object.

        Raises
        ------
        ProcessingError
            If the context or waveform is invalid.

        Notes
        -----
        The active processing waveform is deliberately restricted
        to ``context.acceleration`` when available, with
        ``context.waveform`` retained as a compatibility fallback.

        ``raw_waveform`` is never selected as a fallback because
        raw data must not silently enter a preprocessing stage.
        """

        if context is None:
            raise ProcessingError(
                message="ProcessingContext cannot be None.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": self.plugin_name,
                    "validation": "context",
                },
            )

        waveform = getattr(context, "acceleration", None)

        if waveform is None:
            waveform = getattr(context, "waveform", None)

        if waveform is None:
            raise ProcessingError(
                message=(
                    "No active processed waveform is available "
                    "for this processing stage."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": self.plugin_name,
                    "validation": "waveform",
                },
            )

        if not isinstance(waveform, WaveformData):
            raise ProcessingError(
                message=(
                    "Active waveform must be an instance of "
                    "WaveformData."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": self.plugin_name,
                    "validation": "waveform_type",
                    "received_type": type(waveform).__name__,
                },
            )

        data = np.asarray(waveform.data)

        # ------------------------------------------------------
        # Shape validation
        # ------------------------------------------------------

        if data.ndim != 1:
            raise ProcessingError(
                message=(
                    "Waveform must be one-dimensional. "
                    "BSMA preprocessing operates on one seismic "
                    "channel per processing context."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": self.plugin_name,
                    "validation": "shape",
                    "shape": data.shape,
                    "ndim": data.ndim,
                },
            )

        if data.size == 0:
            raise ProcessingError(
                message="Waveform contains zero samples.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": self.plugin_name,
                    "validation": "empty_waveform",
                },
            )

        # ------------------------------------------------------
        # Numerical validation
        # ------------------------------------------------------

        if not np.issubdtype(data.dtype, np.number):
            raise ProcessingError(
                message="Waveform data must contain numerical values.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": self.plugin_name,
                    "validation": "dtype",
                    "dtype": str(data.dtype),
                },
            )

        if not np.all(np.isfinite(data)):
            raise ProcessingError(
                message=(
                    "Waveform contains NaN or infinite values. "
                    "Numerical preprocessing is unsafe."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": self.plugin_name,
                    "validation": "finite_values",
                },
            )

        # ------------------------------------------------------
        # Sampling-rate validation
        # ------------------------------------------------------

        sampling_rate = float(waveform.sampling_rate)

        if not np.isfinite(sampling_rate):
            raise ProcessingError(
                message="Sampling rate must be finite.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": self.plugin_name,
                    "validation": "sampling_rate",
                    "sampling_rate": sampling_rate,
                },
            )

        if sampling_rate <= 0.0:
            raise ProcessingError(
                message="Sampling rate must be greater than zero.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": self.plugin_name,
                    "validation": "sampling_rate",
                    "sampling_rate": sampling_rate,
                },
            )

        # ------------------------------------------------------
        # Physical consistency
        # ------------------------------------------------------

        if not isinstance(waveform.unit, str) or not waveform.unit.strip():
            raise ProcessingError(
                message="Waveform physical unit is missing or invalid.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": self.plugin_name,
                    "validation": "unit",
                },
            )

        return waveform

    # ==========================================================
    # Processing
    # ==========================================================

    @abstractmethod
    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """
        Execute the processing algorithm.

        Parameters
        ----------
        context
            Input processing context.

        Returns
        -------
        ProcessingContext
            New processing context containing the processed result.

        Notes
        -----
        Implementations must not mutate ``context`` in-place.
        """
        raise NotImplementedError

    # ==========================================================
    # Lifecycle Hooks
    # ==========================================================

    def initialize(self) -> None:
        """
        Initialize plugin resources before pipeline execution.

        Default implementation performs no operation.
        """
        return None

    def finalize(self) -> None:
        """
        Release plugin resources after pipeline execution.

        Default implementation performs no operation.
        """
        return None

    def reset(self) -> None:
        """
        Reset plugin runtime state.

        Stateless plugins may keep the default implementation.
        """
        return None

    # ==========================================================
    # Execution Characteristics
    # ==========================================================

    def supports_parallel(self) -> bool:
        """
        Indicate whether the plugin can safely execute in parallel.

        Returns
        -------
        bool
            ``True`` when the plugin is stateless and independent
            across waveform contexts.
        """
        return True

    # ==========================================================
    # Representation
    # ==========================================================

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.plugin_name}', "
            f"version='{self.plugin_version}')"
        )