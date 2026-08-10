"""
BMKG Strong Motion Analyzer (BSMA)

Module
------
core.preprocessing.baseline

Description
-----------
Baseline correction plugin for strong-motion acceleration records.

The plugin removes either:

1. Constant baseline offset (DC component), or
2. Linear baseline drift.

The correction is performed on the waveform stored in
``ProcessingContext.waveform``.

Scientific note
---------------
Baseline correction is applied to the acceleration record before
numerical integration. Residual baseline errors in acceleration can
produce severe artificial drift in velocity and displacement.

A linear correction is therefore useful for removing a first-order
instrumental/processing trend. However, it must not be interpreted as
a universal substitute for physically based strong-motion baseline
correction, especially for records containing permanent ground
displacement or significant low-frequency content.

Design principles
-----------------
- Immutable ProcessingContext
- No ObsPy dependency
- NumPy/SciPy numerical implementation
- Explicit input validation
- Preservation of waveform units
- Production-grade exception handling
- Pipeline-compatible processing state
- Processing provenance through context history
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from scipy.signal import detrend

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext, WaveformData
from core.types.processing_state import StageStatus
from utils.exceptions import (
    ErrorCode,
    ProcessingError,
    SeverityLevel,
)

__all__ = [
    "BaselineCorrectionPlugin",
]


FloatArray = NDArray[np.float64]


class BaselineCorrectionPlugin(PreprocessorPlugin):
    """
    Remove constant or linear baseline drift from an acceleration record.

    Parameters
    ----------
    method
        Baseline correction method.

        ``"constant"``
            Remove the arithmetic mean of the complete waveform.

        ``"linear"``
            Remove the least-squares linear trend from the complete
            waveform.

    Notes
    -----
    The input ``ProcessingContext`` is never modified in place.

    The corrected waveform is returned through a new
    ``ProcessingContext`` instance.

    Examples
    --------
    >>> plugin = BaselineCorrectionPlugin(method="linear")
    >>> new_context = plugin.process(context)
    """

    _VALID_METHODS = {
        "constant",
        "linear",
    }

    def __init__(
        self,
        method: str = "linear",
    ) -> None:
        """
        Initialize the baseline correction plugin.

        Parameters
        ----------
        method
            ``"constant"`` or ``"linear"``.
        """

        method = str(method).strip().lower()

        if method not in self._VALID_METHODS:
            raise ValueError(
                "Baseline method must be either "
                "'constant' or 'linear'."
            )

        self._method = method

    # ------------------------------------------------------------------
    # Plugin information
    # ------------------------------------------------------------------

    @property
    def plugin_name(self) -> str:
        """Return the canonical plugin name."""
        return "BaselineCorrection"

    @property
    def plugin_description(self) -> str:
        """Return a concise plugin description."""
        return (
            "Removes constant or linear baseline drift "
            "from the acceleration waveform."
        )

    @property
    def method(self) -> str:
        """Return the selected baseline correction method."""
        return self._method

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_waveform(
        waveform: FloatArray,
    ) -> None:
        """
        Validate waveform numerical integrity.

        Raises
        ------
        ProcessingError
            If the waveform is empty, non-finite, or has an
            unsupported dimensionality.
        """

        if waveform.ndim != 1:
            raise ProcessingError(
                message=(
                    "Baseline correction requires a "
                    "one-dimensional waveform."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "baseline",
                    "shape": waveform.shape,
                },
            )

        if waveform.size < 2:
            raise ProcessingError(
                message=(
                    "Baseline correction requires at least "
                    "two waveform samples."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "baseline",
                    "npts": int(waveform.size),
                },
            )

        if not np.isfinite(waveform).all():
            raise ProcessingError(
                message=(
                    "Waveform contains NaN or infinite values."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "baseline",
                },
            )

    # ------------------------------------------------------------------
    # Numerical correction
    # ------------------------------------------------------------------

    def _correct(
        self,
        waveform: FloatArray,
    ) -> FloatArray:
        """
        Apply the selected baseline correction.

        Parameters
        ----------
        waveform
            Input acceleration waveform.

        Returns
        -------
        numpy.ndarray
            Corrected acceleration waveform.
        """

        # Always work on a float64 copy.
        signal = np.asarray(
            waveform,
            dtype=np.float64,
        ).copy()

        if self._method == "constant":
            corrected = signal - np.mean(signal)

        elif self._method == "linear":
            corrected = detrend(
                signal,
                axis=-1,
                type="linear",
            )

        else:
            # Protected by __init__, but retained as a defensive guard.
            raise ProcessingError(
                message=(
                    f"Unsupported baseline method: "
                    f"{self._method!r}."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "baseline",
                    "method": self._method,
                },
            )

        corrected = np.asarray(
            corrected,
            dtype=np.float64,
        )

        if not np.isfinite(corrected).all():
            raise ProcessingError(
                message=(
                    "Baseline correction produced "
                    "non-finite values."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "baseline",
                    "method": self._method,
                },
            )

        return corrected

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """
        Apply baseline correction to the processing context.

        Parameters
        ----------
        context
            Current BSMA processing context.

        Returns
        -------
        ProcessingContext
            New context containing the corrected waveform.

        Raises
        ------
        ProcessingError
            If waveform validation or numerical processing fails.
        """

        # --------------------------------------------------------------
        # Common interface validation
        # --------------------------------------------------------------

        self.validate_input(context)

        # --------------------------------------------------------------
        # Obtain waveform
        # --------------------------------------------------------------

        if context.waveform is None:
            raise ProcessingError(
                message=(
                    "Cannot perform baseline correction: "
                    "waveform is not available."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "baseline",
                },
            )

        active_waveform = context.waveform

        waveform = np.asarray(
            active_waveform.data,
            dtype=np.float64,
        )

        # --------------------------------------------------------------
        # Validate numerical input
        # --------------------------------------------------------------

        self._validate_waveform(
            waveform,
        )

        # --------------------------------------------------------------
        # Apply correction
        # --------------------------------------------------------------

        try:
            corrected_waveform = self._correct(
                waveform,
            )

        except ProcessingError:
            raise

        except Exception as exc:
            raise ProcessingError(
                message=(
                    "Baseline correction failed "
                    "during numerical processing."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "baseline",
                    "method": self._method,
                    "npts": int(waveform.size),
                    "sampling_rate": (
                        float(context.sampling_rate)
                    ),
                },
                cause=exc,
            ) from exc

        # --------------------------------------------------------------
        # Update processing state
        # --------------------------------------------------------------

        state = replace(
            context.processing_state,
            baseline=StageStatus.SUCCESS,
        )

        # --------------------------------------------------------------
        # Immutable context update
        # --------------------------------------------------------------

        # A preprocessing operation changes acceleration, so all
        # kinematic, spectral, and scalar products derived from the prior
        # acceleration must be invalidated.  ``with_acceleration`` performs
        # that immutable transition; ``ProcessingContext`` deliberately has
        # no mutable ``waveform`` field or ``copy`` method.
        new_context = context.with_acceleration(
            WaveformData(
                data=corrected_waveform,
                sampling_rate=active_waveform.sampling_rate,
                unit=active_waveform.unit,
            ),
            clear_derived=True,
        ).with_state(
            processing_state=state,
        )

        # --------------------------------------------------------------
        # Processing provenance
        # --------------------------------------------------------------

        history_message = (
            "BaselineCorrection("
            f"method={self._method}, "
            f"npts={waveform.size}, "
            f"sampling_rate={context.sampling_rate:.6g} Hz"
            ")"
        )

        return new_context.add_history(
            history_message,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return an unambiguous developer representation."""
        return (
            f"{self.__class__.__name__}("
            f"method={self._method!r})"
        )
