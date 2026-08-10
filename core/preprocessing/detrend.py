"""
BMKG Strong Motion Analyzer (BSMA)

Module
------
core.preprocessing.detrend

Description
-----------
Constant and linear detrending plugin for acceleration waveforms.

The plugin removes either the constant mean component or a
least-squares linear trend from the waveform before subsequent
strong-motion processing.

Scientific note
---------------
Detrending is a numerical signal-conditioning operation. It should
not be interpreted as a substitute for instrument-response correction,
baseline correction based on strong-motion processing methodology,
or physical correction for permanent ground displacement.

Design principles
-----------------
- Immutable ProcessingContext
- No ObsPy dependency
- Explicit numerical validation
- Production-grade error handling
- Pipeline-compatible processing state
- Preservation of waveform units
- Processing provenance through context history
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

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
    "DetrendMethod",
    "DetrendConfig",
    "DetrendPlugin",
]


FloatArray = NDArray[np.float64]


class DetrendMethod(str, Enum):
    """
    Supported detrending methods.
    """

    CONSTANT = "constant"
    LINEAR = "linear"


@dataclass(slots=True, frozen=True)
class DetrendConfig:
    """
    Configuration for waveform detrending.

    Parameters
    ----------
    method
        Detrending method.

        ``CONSTANT``
            Removes the arithmetic mean.

        ``LINEAR``
            Removes the least-squares linear trend.
    """

    method: DetrendMethod = DetrendMethod.LINEAR


class DetrendPlugin(PreprocessorPlugin):
    """
    Apply constant or linear detrending to a waveform.

    The input ``ProcessingContext`` is never modified in place.
    A new context is returned containing the detrended waveform.

    Notes
    -----
    This implementation intentionally operates on the one-dimensional
    waveform stored in ``ProcessingContext.waveform``.
    """

    def __init__(
        self,
        config: DetrendConfig = DetrendConfig(),
    ) -> None:
        """
        Initialize the detrending plugin.

        Parameters
        ----------
        config
            Detrending configuration.
        """

        self._config = config

    # ------------------------------------------------------------------
    # Plugin information
    # ------------------------------------------------------------------

    @property
    def plugin_name(self) -> str:
        """Return the canonical plugin name."""
        return "Detrend"

    @property
    def plugin_description(self) -> str:
        """Return a concise plugin description."""
        return (
            "Removes constant or linear trend "
            "from the waveform."
        )

    @property
    def method(self) -> DetrendMethod:
        """Return the configured detrending method."""
        return self._config.method

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_waveform(
        waveform: FloatArray,
    ) -> None:
        """
        Validate waveform before numerical detrending.

        Raises
        ------
        ProcessingError
            If the waveform is empty, too short, non-finite,
            or not one-dimensional.
        """

        if waveform.ndim != 1:
            raise ProcessingError(
                message=(
                    "Detrending requires a one-dimensional "
                    "waveform."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "detrend",
                    "shape": waveform.shape,
                },
            )

        if waveform.size < 2:
            raise ProcessingError(
                message=(
                    "Detrending requires at least "
                    "two waveform samples."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "detrend",
                    "npts": int(waveform.size),
                },
            )

        if not np.isfinite(waveform).all():
            raise ProcessingError(
                message=(
                    "Waveform contains NaN or infinite "
                    "values."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "detrend",
                },
            )

    # ------------------------------------------------------------------
    # Numerical operation
    # ------------------------------------------------------------------

    def _detrend(
        self,
        waveform: FloatArray,
    ) -> FloatArray:
        """
        Apply the configured detrending operation.

        Parameters
        ----------
        waveform
            One-dimensional acceleration waveform.

        Returns
        -------
        numpy.ndarray
            Detrended waveform in the same physical units.
        """

        signal = np.asarray(
            waveform,
            dtype=np.float64,
        ).copy()

        corrected = detrend(
            signal,
            axis=-1,
            type=self._config.method.value,
        )

        corrected = np.asarray(
            corrected,
            dtype=np.float64,
        )

        if not np.isfinite(corrected).all():
            raise ProcessingError(
                message=(
                    "Detrending produced non-finite "
                    "values."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "detrend",
                    "method": self._config.method.value,
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
        Execute waveform detrending.

        Parameters
        ----------
        context
            Current BSMA processing context.

        Returns
        -------
        ProcessingContext
            New context containing the detrended waveform.
        """

        self.validate_input(context)

        # --------------------------------------------------------------
        # Waveform availability
        # --------------------------------------------------------------

        if context.waveform is None:
            raise ProcessingError(
                message=(
                    "Cannot detrend waveform: "
                    "waveform is not available."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "detrend",
                },
            )

        active_waveform = context.waveform

        waveform = np.asarray(
            active_waveform.data,
            dtype=np.float64,
        )

        # --------------------------------------------------------------
        # Numerical validation
        # --------------------------------------------------------------

        self._validate_waveform(
            waveform,
        )

        # --------------------------------------------------------------
        # Detrending
        # --------------------------------------------------------------

        try:
            corrected = self._detrend(
                waveform,
            )

        except ProcessingError:
            raise

        except Exception as exc:
            raise ProcessingError(
                message=(
                    "Detrending failed during numerical "
                    "processing."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "detrend",
                    "method": (
                        self._config.method.value
                    ),
                    "npts": int(waveform.size),
                    "sampling_rate": (
                        float(context.sampling_rate)
                    ),
                },
                cause=exc,
            ) from exc

        # --------------------------------------------------------------
        # Processing state
        # --------------------------------------------------------------

        state = replace(
            context.processing_state,
            detrend=StageStatus.SUCCESS,
        )

        # --------------------------------------------------------------
        # Cache handling
        # --------------------------------------------------------------
        #
        # Detrending changes the waveform and therefore invalidates
        # numerical products derived from the previous waveform.
        #
        # The cache API is intentionally handled through its own
        # interface rather than assuming individual cache fields here.
        #

        # --------------------------------------------------------------
        # Immutable context update
        # --------------------------------------------------------------
        #
        # Replacing acceleration must clear every downstream product.  Do
        # not mutate ``context.cache``: another immutable context may still
        # legitimately reference it.
        #

        new_context = context.with_acceleration(
            WaveformData(
                data=corrected,
                sampling_rate=active_waveform.sampling_rate,
                unit=active_waveform.unit,
            ),
            clear_derived=True,
        ).with_state(
            processing_state=state,
        )

        # --------------------------------------------------------------
        # Provenance
        # --------------------------------------------------------------

        history_message = (
            "Detrend("
            f"method={self._config.method.value}, "
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
        """Return developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"method={self._config.method.value!r})"
        )
