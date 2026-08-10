"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/preprocessing/taper.py

Description
-----------
Signal tapering plugin for suppressing edge discontinuities and
reducing spectral leakage before frequency-domain analysis.

Important
---------
Tapering modifies waveform amplitudes near the record boundaries.
Therefore the raw waveform must never be overwritten by this plugin.
All downstream numerical products derived from the previous waveform
must be invalidated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import windows

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext, WaveformData
from core.types.processing_state import StageStatus
from utils.exceptions import (
    ErrorCode,
    ProcessingError,
    SeverityLevel,
)

__all__ = [
    "TaperConfig",
    "TaperPlugin",
]


@dataclass(frozen=True, slots=True)
class TaperConfig:
    """
    Configuration for waveform tapering.

    Parameters
    ----------
    window_type
        Window family. Supported values:

        - ``"tukey"``
        - ``"hann"``
        - ``"cosine"``

    alpha
        Fraction controlling the tapered portion.

        For a Tukey window this follows the SciPy definition:

        - 0.0 -> rectangular window
        - 1.0 -> Hann window

        For Hann/cosine windows the parameter is retained for
        configuration compatibility but does not alter the
        mathematical window shape.
    """

    window_type: str = "tukey"
    alpha: float = 0.05

    def __post_init__(self) -> None:
        window_type = self.window_type.lower().strip()

        if window_type not in {"tukey", "hann", "cosine"}:
            raise ValueError(
                "Unsupported taper window. "
                "Use 'tukey', 'hann', or 'cosine'."
            )

        if not np.isfinite(self.alpha):
            raise ValueError("Taper alpha must be finite.")

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                "Taper alpha must satisfy 0 <= alpha <= 1."
            )

        object.__setattr__(self, "window_type", window_type)


class TaperPlugin(PreprocessorPlugin):
    """
    Apply a mathematically defined taper to one waveform channel.

    The plugin never modifies ``raw_waveform``. Only the processed
    acceleration representation is replaced.

    Notes
    -----
    Tapering is intended primarily for operations sensitive to
    endpoint discontinuities, particularly FFT/FAS and related
    frequency-domain calculations.

    It should not automatically be interpreted as a correction
    to the physical amplitude of the recorded ground motion.
    """

    def __init__(
        self,
        config: TaperConfig | None = None,
    ) -> None:
        self.config = config or TaperConfig()

    @property
    def plugin_name(self) -> str:
        return "Taper"

    @property
    def plugin_description(self) -> str:
        return (
            f"Applies {self.config.window_type} taper "
            f"with alpha={self.config.alpha:.3f}."
        )

    def _get_waveform(
        self,
        context: ProcessingContext,
    ) -> WaveformData:
        """
        Retrieve the processed waveform.

        Preference order:
        acceleration -> waveform.

        ``raw_waveform`` is intentionally excluded because raw data
        must remain immutable throughout preprocessing.
        """

        waveform = getattr(context, "acceleration", None)

        if waveform is None:
            waveform = getattr(context, "waveform", None)

        if waveform is None:
            raise ProcessingError(
                message=(
                    "Cannot perform taper: "
                    "no processed waveform is available."
                ),
                error_code=ErrorCode.PR002,
                severity=SeverityLevel.ERROR,
                context={"module": "taper"},
            )

        if not isinstance(waveform, WaveformData):
            raise ProcessingError(
                message=(
                    "Taper requires WaveformData as the processed "
                    "waveform representation."
                ),
                error_code=ErrorCode.PR002,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "taper",
                    "type": type(waveform).__name__,
                },
            )

        return waveform

    def _build_window(self, n_samples: int) -> np.ndarray:
        """
        Construct the requested taper window.

        Returns
        -------
        numpy.ndarray
            One-dimensional window with length ``n_samples``.
        """

        if n_samples < 1:
            raise ValueError("Number of samples must be positive.")

        window_type = self.config.window_type

        if window_type == "tukey":
            window = windows.tukey(
                n_samples,
                alpha=self.config.alpha,
            )

        elif window_type == "hann":
            window = windows.hann(
                n_samples,
                sym=True,
            )

        elif window_type == "cosine":
            window = windows.cosine(
                n_samples,
                sym=True,
            )

        else:
            # Should be unreachable because TaperConfig validates it.
            raise ValueError(
                f"Unsupported taper window: {window_type}"
            )

        return np.asarray(window, dtype=np.float64)

    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """
        Apply tapering to the processed acceleration waveform.

        The operation is non-destructive with respect to the original
        raw waveform.
        """

        self.validate_input(context)

        waveform = self._get_waveform(context)

        data = np.asarray(
            waveform.data,
            dtype=np.float64,
        )

        if data.ndim != 1:
            raise ProcessingError(
                message=(
                    "Taper expects a one-dimensional waveform "
                    "representing a single seismic channel."
                ),
                error_code=ErrorCode.PR002,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "taper",
                    "ndim": data.ndim,
                    "shape": data.shape,
                },
            )

        if data.size == 0:
            raise ProcessingError(
                message="Cannot perform taper on an empty waveform.",
                error_code=ErrorCode.PR002,
                severity=SeverityLevel.ERROR,
                context={"module": "taper"},
            )

        if not np.all(np.isfinite(data)):
            raise ProcessingError(
                message=(
                    "Cannot perform taper on waveform containing "
                    "NaN or infinite values."
                ),
                error_code=ErrorCode.PR002,
                severity=SeverityLevel.ERROR,
                context={"module": "taper"},
            )

        sampling_rate = float(waveform.sampling_rate)

        if (
            not np.isfinite(sampling_rate)
            or sampling_rate <= 0.0
        ):
            raise ProcessingError(
                message="Invalid waveform sampling rate.",
                error_code=ErrorCode.PR002,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "taper",
                    "sampling_rate": sampling_rate,
                },
            )

        try:
            window = self._build_window(data.size)

            tapered_data = np.multiply(
                data,
                window,
                dtype=np.float64,
            )

        except Exception as exc:
            raise ProcessingError(
                message="Tapering operation failed.",
                error_code=ErrorCode.PR002,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "taper",
                    "window_type": self.config.window_type,
                    "alpha": self.config.alpha,
                    "samples": data.size,
                    "sampling_rate": sampling_rate,
                },
                cause=exc,
            ) from exc

        updated_waveform = replace(
            waveform,
            data=tapered_data,
        )

        state = replace(
            context.processing_state,
            taper=StageStatus.SUCCESS,
        )

        # Taper changes acceleration.  Clear dependent quantities through
        # the context transition instead of mutating the cache in-place or
        # attempting to replace the read-only ``waveform`` property.
        new_context = context.with_acceleration(
            updated_waveform,
            clear_derived=True,
        ).with_state(
            processing_state=state,
        )

        if hasattr(new_context, "add_history"):
            return new_context.add_history(
                step_name=self.plugin_name,
                details={
                    "status": "SUCCESS",
                    "window_type": self.config.window_type,
                    "alpha": self.config.alpha,
                    "samples": data.size,
                    "sampling_rate": sampling_rate,
                },
            )

        return new_context
