"""
BMKG Strong Motion Analyzer (BSMA)

Module
------
core.preprocessing.filter

Description
-----------
Butterworth frequency-domain filtering plugin for strong-motion
waveform processing.

The implementation uses second-order sections (SOS) for improved
numerical stability and supports causal and zero-phase filtering.

Supported filter types
----------------------
- bandpass
- bandstop
- lowpass
- highpass

Scientific notes
---------------
For offline strong-motion analysis, zero-phase filtering using
``sosfiltfilt`` is generally preferred when preservation of waveform
phase is required.

The filter configuration must satisfy:

    0 < f_low < f_high < fs / 2

for band filters, and:

    0 < f_cutoff < fs / 2

for low-pass/high-pass filters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, sosfiltfilt

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext, WaveformData
from core.types.processing_state import StageStatus
from utils.exceptions import (
    ErrorCode,
    ProcessingError,
    SeverityLevel,
)


__all__ = [
    "FilterType",
    "FilterConfig",
    "ButterworthFilterPlugin",
]


FloatArray = NDArray[np.float64]


class FilterType(str, Enum):
    """Supported Butterworth filter configurations."""

    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """
    Configuration for the Butterworth filter.

    Parameters
    ----------
    type
        Filter topology.
    freq_min
        Lower cutoff frequency in Hz.

        Used by:
        - bandpass
        - bandstop
        - highpass

    freq_max
        Upper cutoff frequency in Hz.

        Used by:
        - bandpass
        - bandstop
        - lowpass

    corners
        Butterworth filter order.

    zerophase
        If True, use forward-backward SOS filtering.
        If False, use causal SOS filtering.
    """

    type: FilterType = FilterType.BANDPASS
    freq_min: float = 0.1
    freq_max: float = 25.0
    corners: int = 4
    zerophase: bool = True

    def validate(self, sampling_rate: float) -> None:
        """
        Validate filter parameters against the Nyquist frequency.

        Raises
        ------
        ValueError
            If configuration is mathematically invalid.
        """

        if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
            raise ValueError(
                f"Sampling rate must be positive and finite, "
                f"got {sampling_rate!r}."
            )

        nyquist = sampling_rate / 2.0

        if not np.isfinite(self.freq_min):
            raise ValueError("freq_min must be finite.")

        if not np.isfinite(self.freq_max):
            raise ValueError("freq_max must be finite.")

        if self.corners < 1:
            raise ValueError(
                f"Filter order must be >= 1, got {self.corners}."
            )

        if self.type in (
            FilterType.BANDPASS,
            FilterType.BANDSTOP,
        ):
            if not (
                0.0 < self.freq_min < self.freq_max < nyquist
            ):
                raise ValueError(
                    "Band filter requires "
                    "0 < freq_min < freq_max < Nyquist. "
                    f"Received freq_min={self.freq_min}, "
                    f"freq_max={self.freq_max}, "
                    f"Nyquist={nyquist} Hz."
                )

        elif self.type == FilterType.LOWPASS:
            if not (0.0 < self.freq_max < nyquist):
                raise ValueError(
                    "Low-pass filter requires "
                    "0 < freq_max < Nyquist. "
                    f"Received freq_max={self.freq_max}, "
                    f"Nyquist={nyquist} Hz."
                )

        elif self.type == FilterType.HIGHPASS:
            if not (0.0 < self.freq_min < nyquist):
                raise ValueError(
                    "High-pass filter requires "
                    "0 < freq_min < Nyquist. "
                    f"Received freq_min={self.freq_min}, "
                    f"Nyquist={nyquist} Hz."
                )


class ButterworthFilterPlugin(PreprocessorPlugin):
    """
    Butterworth frequency filter using second-order sections.

    The plugin operates on the processed acceleration waveform while
    preserving the original raw waveform.

    Notes
    -----
    The filter does not perform instrument-response correction.
    Instrument response removal must occur before this stage.
    """

    def __init__(
        self,
        config: FilterConfig | None = None,
    ) -> None:
        self.config = config or FilterConfig()

    @property
    def plugin_name(self) -> str:
        return "ButterworthFilter"

    @property
    def plugin_description(self) -> str:
        return (
            f"Applies {self.config.type.value} Butterworth filter "
            f"with order={self.config.corners}, "
            f"zero_phase={self.config.zerophase}."
        )

    def _get_waveform(
        self,
        context: ProcessingContext,
    ) -> WaveformData:
        """
        Retrieve the current processed waveform.

        The raw waveform is deliberately excluded because raw data
        must remain immutable throughout the processing pipeline.
        """

        waveform = getattr(context, "waveform", None)

        if waveform is None:
            waveform = getattr(context, "acceleration", None)

        if waveform is None:
            raise ProcessingError(
                message=(
                    "Cannot perform filtering: "
                    "no processed waveform found in context."
                ),
                error_code=ErrorCode.PR003,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "filter",
                    "operation": "retrieve_waveform",
                },
            )

        if not isinstance(waveform, WaveformData):
            raise ProcessingError(
                message=(
                    "Filtering requires WaveformData as the "
                    "processed waveform container."
                ),
                error_code=ErrorCode.PR003,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "filter",
                    "received_type": type(waveform).__name__,
                },
            )

        return waveform

    def _design_filter(
        self,
        sampling_rate: float,
    ) -> NDArray[np.float64]:
        """
        Design a numerically stable SOS Butterworth filter.
        """

        self.config.validate(sampling_rate)

        nyquist = sampling_rate / 2.0

        if self.config.type in (
            FilterType.BANDPASS,
            FilterType.BANDSTOP,
        ):
            wn = [
                self.config.freq_min / nyquist,
                self.config.freq_max / nyquist,
            ]
        elif self.config.type == FilterType.LOWPASS:
            wn = self.config.freq_max / nyquist
        else:
            wn = self.config.freq_min / nyquist

        return butter(
            N=self.config.corners,
            Wn=wn,
            btype=self.config.type.value,
            output="sos",
        )

    def _apply_filter(
        self,
        sos: NDArray[np.float64],
        data: FloatArray,
    ) -> FloatArray:
        """
        Apply causal or zero-phase SOS filtering.
        """

        if data.ndim == 0:
            raise ValueError("Waveform data must be at least one-dimensional.")

        if self.config.zerophase:
            filtered = sosfiltfilt(
                sos,
                data,
                axis=-1,
            )
        else:
            filtered = sosfilt(
                sos,
                data,
                axis=-1,
            )

        return np.asarray(
            filtered,
            dtype=np.float64,
        )

    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """
        Execute Butterworth filtering.

        Returns
        -------
        ProcessingContext
            Updated immutable processing context.
        """

        self.validate_input(context)

        waveform = self._get_waveform(context)

        data = np.asarray(
            waveform.data,
            dtype=np.float64,
        )

        if data.size == 0:
            raise ProcessingError(
                message="Cannot perform filtering on an empty waveform.",
                error_code=ErrorCode.PR003,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "filter",
                    "operation": "filter",
                },
            )

        if not np.all(np.isfinite(data)):
            raise ProcessingError(
                message=(
                    "Waveform contains NaN or infinite values. "
                    "Filtering aborted."
                ),
                error_code=ErrorCode.PR003,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "filter",
                    "operation": "filter",
                },
            )

        sampling_rate = float(waveform.sampling_rate)

        try:
            sos = self._design_filter(sampling_rate)

            filtered_data = self._apply_filter(
                sos=sos,
                data=data,
            )

        except Exception as exc:
            raise ProcessingError(
                message="Butterworth filtering failed.",
                error_code=ErrorCode.PR003,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "filter",
                    "filter_type": self.config.type.value,
                    "freq_min_hz": self.config.freq_min,
                    "freq_max_hz": self.config.freq_max,
                    "corners": self.config.corners,
                    "zero_phase": self.config.zerophase,
                    "sampling_rate_hz": sampling_rate,
                    "nyquist_hz": sampling_rate / 2.0,
                },
                cause=exc,
            ) from exc

        updated_waveform = replace(
            waveform,
            data=filtered_data,
        )

        state = replace(
            context.processing_state,
            filter=StageStatus.SUCCESS,
        )

        # ------------------------------------------------------
        # IMPORTANT:
        # Do NOT modify raw_waveform.
        # ------------------------------------------------------

        # Filtering changes acceleration; velocity, displacement, spectra,
        # metrics, and cached products from the prior signal are no longer
        # physically valid.  ``waveform`` is a computed property, not a
        # dataclass field, so it must never be passed to ``replace``.
        new_context = context.with_acceleration(
            updated_waveform,
            clear_derived=True,
        ).with_state(
            processing_state=state,
        )

        if hasattr(new_context, "add_history"):
            new_context = new_context.add_history(
                step_name=self.plugin_name,
                details={
                    "status": "SUCCESS",
                    "filter_type": self.config.type.value,
                    "freq_min_hz": self.config.freq_min,
                    "freq_max_hz": self.config.freq_max,
                    "corners": self.config.corners,
                    "zero_phase": self.config.zerophase,
                    "sampling_rate_hz": sampling_rate,
                    "nyquist_hz": sampling_rate / 2.0,
                },
            )

        return new_context
