"""
BMKG Strong Motion Analyzer (BSMA)

Module: core/processing/kinematics.py

Description
-----------
Kinematic integration plugin for strong-motion waveform processing.

The plugin numerically integrates acceleration into velocity and
velocity into displacement using cumulative trapezoidal integration.

Processing flow
---------------
Acceleration
    ↓
Conditioning
    ├── optional mean removal
    └── optional linear detrending
    ↓
Velocity
    ↓
Velocity conditioning
    ├── optional mean removal
    └── optional linear detrending
    ↓
Displacement
    ↓
Displacement conditioning
    └── optional linear detrending

Design principles
-----------------
- General-purpose; no dependency on specific networks/stations.
- Compatible with MiniSEED + StationXML workflows.
- Immutable ProcessingContext.
- Float64 numerical processing.
- Explicit sampling-rate validation.
- No in-place modification of input waveform.
- Cascading cache invalidation.
- Processing-state tracking.
- Audit-history support.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
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
    "IntegrationMethod",
    "IntegrationConfig",
    "KinematicIntegrationPlugin",
]


FloatArray = NDArray[np.float64]


# ============================================================================
# Configuration
# ============================================================================


class IntegrationMethod(str, Enum):
    """
    Numerical integration methods supported by BSMA.
    """

    CUMULATIVE_TRAPEZOID = "cumulative_trapezoid"


@dataclass(slots=True, frozen=True)
class IntegrationConfig:
    """
    Configuration for kinematic integration.

    Parameters
    ----------
    method
        Numerical integration method.

    remove_mean
        Remove the mean from acceleration before the first integration
        and from velocity before the second integration.

    remove_linear_trend
        Remove a first-order linear trend from acceleration, velocity,
        and displacement.

    Notes
    -----
    Baseline correction and instrument-response correction should normally
    be handled by dedicated preprocessing stages. These conditioning
    options provide an additional numerical safeguard against residual
    low-frequency bias before integration.
    """

    method: IntegrationMethod = IntegrationMethod.CUMULATIVE_TRAPEZOID

    remove_mean: bool = True

    remove_linear_trend: bool = True


# ============================================================================
# Plugin
# ============================================================================


class KinematicIntegrationPlugin(PreprocessorPlugin):
    """
    Integrate acceleration into velocity and displacement.

    The plugin expects ``context.acceleration`` to contain a valid
    acceleration waveform in physical units.

    The input ProcessingContext is never modified in-place.
    """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def plugin_name(self) -> str:
        """Human-readable plugin name."""
        return "KinematicIntegration"

    @property
    def plugin_version(self) -> str:
        """Semantic plugin version."""
        return "2.0.0"

    @property
    def plugin_description(self) -> str:
        """Human-readable plugin description."""
        return (
            "Numerically integrates acceleration into velocity and "
            "displacement using cumulative trapezoidal integration."
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: IntegrationConfig | None = None,
    ) -> None:
        """
        Initialize the integration plugin.

        Parameters
        ----------
        config
            Integration configuration. If omitted, the scientific
            defaults defined by ``IntegrationConfig`` are used.
        """

        self._config = config or IntegrationConfig()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_acceleration(
        acceleration: WaveformData,
    ) -> FloatArray:
        """
        Validate the acceleration waveform.

        Returns
        -------
        numpy.ndarray
            Float64 acceleration array.

        Raises
        ------
        ProcessingError
            If the waveform is invalid.
        """

        data = np.asarray(
            acceleration.data,
            dtype=np.float64,
        )

        if data.ndim != 1:
            raise ProcessingError(
                message=(
                    "Acceleration waveform must be one-dimensional."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "kinematics"},
            )

        if data.size < 2:
            raise ProcessingError(
                message=(
                    "Acceleration waveform must contain at least "
                    "two samples for numerical integration."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "kinematics"},
            )

        if not np.all(np.isfinite(data)):
            raise ProcessingError(
                message=(
                    "Acceleration waveform contains NaN or "
                    "infinite values."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "kinematics"},
            )

        return data

    @staticmethod
    def _validate_sampling_rate(
        sampling_rate: float,
    ) -> float:
        """
        Validate and return sampling rate.

        Raises
        ------
        ProcessingError
            If sampling rate is invalid.
        """

        if (
            not np.isfinite(sampling_rate)
            or sampling_rate <= 0.0
        ):
            raise ProcessingError(
                message=(
                    "Sampling rate must be finite and greater "
                    "than zero."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "kinematics",
                    "sampling_rate": sampling_rate,
                },
            )

        return float(sampling_rate)

    # ------------------------------------------------------------------
    # Signal Conditioning
    # ------------------------------------------------------------------

    def _condition_signal(
        self,
        signal: FloatArray,
    ) -> FloatArray:
        """
        Apply configured numerical conditioning.

        Parameters
        ----------
        signal
            One-dimensional signal.

        Returns
        -------
        numpy.ndarray
            Conditioned float64 signal.

        Notes
        -----
        The original signal is never modified in-place.
        """

        conditioned = np.asarray(
            signal,
            dtype=np.float64,
        ).copy()

        if self._config.remove_mean:
            conditioned -= np.mean(conditioned)

        if self._config.remove_linear_trend:
            conditioned = detrend(
                conditioned,
                type="linear",
                overwrite_data=False,
            )

        return np.asarray(
            conditioned,
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Numerical Integration
    # ------------------------------------------------------------------

    @staticmethod
    def _integrate(
        signal: FloatArray,
        dt: float,
    ) -> FloatArray:
        """
        Perform cumulative trapezoidal integration.

        Parameters
        ----------
        signal
            Input waveform.

        dt
            Sampling interval in seconds.

        Returns
        -------
        numpy.ndarray
            Integrated waveform with the same number of samples.
        """

        result = cumulative_trapezoid(
            signal,
            dx=dt,
            initial=0.0,
        )

        return np.asarray(
            result,
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """
        Execute acceleration → velocity → displacement integration.

        Parameters
        ----------
        context
            Current immutable ProcessingContext.

        Returns
        -------
        ProcessingContext
            New context containing velocity and displacement.

        Raises
        ------
        ProcessingError
            If acceleration, sampling rate, or numerical processing
            is invalid.
        """

        # --------------------------------------------------------------
        # Validate generic plugin input
        # --------------------------------------------------------------

        self.validate_input(context)

        # --------------------------------------------------------------
        # Acceleration is the required source signal.
        # Do NOT silently fall back to raw_waveform here.
        #
        # This is important because integration must operate on the
        # acceleration state that has already passed the appropriate
        # preprocessing stages.
        # --------------------------------------------------------------

        acceleration = context.acceleration

        if acceleration is None:
            raise ProcessingError(
                message=(
                    "Acceleration data is unavailable. "
                    "Run waveform preparation and acceleration "
                    "preprocessing before kinematic integration."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "kinematics",
                    "trace_id": context.trace_id,
                },
            )

        # --------------------------------------------------------------
        # Validate waveform
        # --------------------------------------------------------------

        acc_data = self._validate_acceleration(
            acceleration
        )

        sampling_rate = self._validate_sampling_rate(
            acceleration.sampling_rate
        )

        dt = 1.0 / sampling_rate

        # --------------------------------------------------------------
        # Numerical processing
        # --------------------------------------------------------------

        try:
            # ==========================================================
            # 1. Acceleration conditioning
            # ==========================================================

            conditioned_acceleration = self._condition_signal(
                acc_data
            )

            # ==========================================================
            # 2. Acceleration → Velocity
            # ==========================================================

            velocity_data = self._integrate(
                conditioned_acceleration,
                dt,
            )

            # ==========================================================
            # 3. Velocity conditioning
            # ==========================================================

            conditioned_velocity = self._condition_signal(
                velocity_data
            )

            # ==========================================================
            # 4. Velocity → Displacement
            # ==========================================================

            displacement_data = self._integrate(
                conditioned_velocity,
                dt,
            )

            # ==========================================================
            # 5. Displacement conditioning
            # ==========================================================

            conditioned_displacement = self._condition_signal(
                displacement_data
            )

        except Exception as exc:
            raise ProcessingError(
                message=(
                    "Kinematic integration failed during "
                    "numerical processing."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "kinematics",
                    "trace_id": context.trace_id,
                    "sampling_rate": sampling_rate,
                    "dt": dt,
                    "method": self._config.method.value,
                },
                cause=exc,
            ) from exc

        # --------------------------------------------------------------
        # Construct physical waveform containers
        # --------------------------------------------------------------

        velocity = WaveformData(
            data=conditioned_velocity,
            sampling_rate=sampling_rate,
            unit="m/s",
        )

        displacement = WaveformData(
            data=conditioned_displacement,
            sampling_rate=sampling_rate,
            unit="m",
        )

        # --------------------------------------------------------------
        # Update processing state
        # --------------------------------------------------------------

        state = replace(
            context.processing_state,
            integration=StageStatus.SUCCESS,
        )

        # --------------------------------------------------------------
        # Cache handling
        # --------------------------------------------------------------
        #
        # Integration produces new velocity/displacement products.
        # Existing downstream quantities based on the previous
        # waveform must not remain available.
        #
        # We intentionally clear the cache before storing the newly
        # generated kinematic products.
        # --------------------------------------------------------------

        cache = context.cache

        cache.clear()

        cache.velocity = conditioned_velocity.copy()
        cache.displacement = conditioned_displacement.copy()

        # --------------------------------------------------------------
        # Build immutable context transition
        # --------------------------------------------------------------

        new_context = context.with_state(
            velocity=velocity,
            displacement=displacement,
            processing_state=state,
            cache=cache,
        )

        # --------------------------------------------------------------
        # Audit trail
        # --------------------------------------------------------------

        return new_context.add_history(
            step_name=self.plugin_name,
            details={
                "status": "SUCCESS",
                "method": self._config.method.value,
                "sampling_rate_hz": sampling_rate,
                "dt_seconds": dt,
                "remove_mean": self._config.remove_mean,
                "remove_linear_trend": (
                    self._config.remove_linear_trend
                ),
            },
        )