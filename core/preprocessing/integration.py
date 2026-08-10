"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/processing/integration.py

Description
-----------
Numerical integration of corrected acceleration into velocity and
displacement using cumulative trapezoidal integration.

Scientific policy
-----------------
Baseline correction is intentionally NOT performed on velocity or
displacement inside this module. Baseline correction must be handled
by the dedicated preprocessing stage before integration.

This prevents undocumented modification of the physical kinematic
relationship between acceleration, velocity, and displacement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.signal import detrend

from core.orchestrator import ProcessingStep
from core.types.context import ProcessingContext, WaveformData
from core.types.processing_state import StageStatus
from utils.exceptions import ErrorCode, ProcessingError, SeverityLevel

__all__ = [
    "IntegrationMethod",
    "IntegrationConfig",
    "KinematicIntegrationPlugin",
]

FloatArray = NDArray[np.float64]


class IntegrationMethod(str, Enum):
    """Numerical integration methods supported by BSMA."""

    CUMULATIVE_TRAPEZOID = "cumulative_trapezoid"


@dataclass(slots=True, frozen=True)
class IntegrationConfig:
    """
    Configuration for kinematic integration.

    Parameters
    ----------
    method
        Numerical integration method.

    condition_acceleration
        Optional conditioning of acceleration immediately before
        integration. Disabled by default because baseline correction
        should normally be performed by the dedicated preprocessing
        pipeline.

    acceleration_detrend
        Conditioning method when ``condition_acceleration=True``.
        Supported values are ``"constant"`` and ``"linear"``.
    """

    method: IntegrationMethod = IntegrationMethod.CUMULATIVE_TRAPEZOID
    condition_acceleration: bool = False
    acceleration_detrend: str = "linear"


class KinematicIntegrationPlugin(ProcessingStep):
    """
    Numerically integrate acceleration into velocity and displacement.

    The plugin assumes that the input acceleration has already undergone
    the required instrument-response correction, QC, baseline correction,
    tapering, and filtering upstream in the processing pipeline.

    No post-integration detrending is performed.
    """

    def __init__(
        self,
        config: IntegrationConfig | None = None,
    ) -> None:
        self._config = config or IntegrationConfig()

        if self._config.acceleration_detrend not in {
            "constant",
            "linear",
        }:
            raise ValueError(
                "acceleration_detrend must be 'constant' or 'linear'."
            )

    @property
    def name(self) -> str:
        """Canonical processing-step name."""
        return "KinematicIntegration"

    @staticmethod
    def _validate_waveform(
        acceleration: WaveformData,
    ) -> tuple[FloatArray, float]:
        """
        Validate acceleration data and sampling rate.

        Returns
        -------
        tuple
            Validated acceleration array and sampling rate.
        """

        data = np.asarray(acceleration.data, dtype=np.float64)

        if data.ndim != 1:
            raise ProcessingError(
                message="Acceleration waveform must be one-dimensional.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "integration"},
            )

        if data.size < 2:
            raise ProcessingError(
                message=(
                    "Acceleration waveform must contain at least "
                    "two samples for numerical integration."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "integration"},
            )

        if not np.all(np.isfinite(data)):
            raise ProcessingError(
                message="Acceleration waveform contains NaN or infinite values.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "integration"},
            )

        sampling_rate = float(acceleration.sampling_rate)

        if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
            raise ProcessingError(
                message="Sampling rate must be finite and greater than zero.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "integration",
                    "sampling_rate": sampling_rate,
                },
            )

        return data, sampling_rate

    def _condition_acceleration(
        self,
        acceleration: FloatArray,
    ) -> FloatArray:
        """
        Optionally condition acceleration before integration.

        This operation is intentionally applied only to acceleration,
        never to velocity or displacement.
        """

        if not self._config.condition_acceleration:
            return acceleration.copy()

        return detrend(
            acceleration,
            type=self._config.acceleration_detrend,
        ).astype(np.float64, copy=False)

    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """Integrate acceleration into velocity and displacement."""

        if not isinstance(context, ProcessingContext):
            raise ProcessingError(
                message="context must be a ProcessingContext instance.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "integration"},
            )

        acceleration = context.acceleration

        if acceleration is None:
            raise ProcessingError(
                message=(
                    "Acceleration waveform is unavailable. "
                    "Ensure preprocessing stages completed successfully."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "integration",
                    "trace_id": context.trace_id,
                },
            )

        acc_data, sampling_rate = self._validate_waveform(acceleration)

        # ----------------------------------------------------------
        # 1. Optional acceleration conditioning
        # ----------------------------------------------------------

        conditioned_acc = self._condition_acceleration(acc_data)

        dt = 1.0 / sampling_rate

        try:
            # ------------------------------------------------------
            # 2. Acceleration -> Velocity
            # ------------------------------------------------------

            velocity = cumulative_trapezoid(
                conditioned_acc,
                dx=dt,
                initial=0.0,
            )

            # ------------------------------------------------------
            # 3. Velocity -> Displacement
            # ------------------------------------------------------

            displacement = cumulative_trapezoid(
                velocity,
                dx=dt,
                initial=0.0,
            )

        except Exception as exc:
            raise ProcessingError(
                message="Numerical kinematic integration failed.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "integration",
                    "trace_id": context.trace_id,
                    "sampling_rate": sampling_rate,
                    "sample_count": acc_data.size,
                    "method": self._config.method.value,
                },
                cause=exc,
            ) from exc

        # ----------------------------------------------------------
        # 4. Numerical validation
        # ----------------------------------------------------------

        if not np.all(np.isfinite(velocity)):
            raise ProcessingError(
                message="Integration produced non-finite velocity values.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "integration"},
            )

        if not np.all(np.isfinite(displacement)):
            raise ProcessingError(
                message=(
                    "Integration produced non-finite displacement values."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "integration"},
            )

        # ----------------------------------------------------------
        # 5. Construct domain objects
        # ----------------------------------------------------------

        velocity_waveform = WaveformData(
            data=velocity,
            sampling_rate=sampling_rate,
            unit="m/s",
        )

        displacement_waveform = WaveformData(
            data=displacement,
            sampling_rate=sampling_rate,
            unit="m",
        )

        # ----------------------------------------------------------
        # 6. Processing state
        # ----------------------------------------------------------

        state = replace(
            context.processing_state,
            integration=StageStatus.SUCCESS,
        )

        # ----------------------------------------------------------
        # 7. Audit history
        # ----------------------------------------------------------

        history_details = {
            "status": "SUCCESS",
            "method": self._config.method.value,
            "sampling_rate_hz": sampling_rate,
            "sample_count": int(acc_data.size),
            "dt_s": dt,
            "condition_acceleration": (
                self._config.condition_acceleration
            ),
            "acceleration_detrend": (
                self._config.acceleration_detrend
                if self._config.condition_acceleration
                else "none"
            ),
        }

        return (
            context.with_state(
                velocity=velocity_waveform,
                displacement=displacement_waveform,
                processing_state=state,
            )
            .add_history(
                step_name=self.name,
                details=history_details,
            )
        )
