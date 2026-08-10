"""
BMKG Strong Motion Analyzer (BSMA)

Module: core/processing/parameters.py

Description
-----------
Strong-motion engineering parameter extraction.

This module computes fundamental time-domain strong-motion parameters
from acceleration, velocity, and displacement waveforms:

    - PGA
    - PGV
    - PGD
    - Arias Intensity
    - Cumulative Absolute Velocity (CAV)
    - Husid Curve
    - Significant Duration D5-75
    - Significant Duration D5-95
    - Mean acceleration
    - Standard deviation
    - RMS acceleration
    - Peak-to-peak acceleration
    - Acceleration variance

The implementation is intentionally general-purpose and contains
no assumptions about:

    - network
    - station
    - channel
    - instrument manufacturer
    - acquisition system
    - specific BMKG station
    - specific strong-motion network

The waveform source is represented by the domain-level
ProcessingContext and WaveformData objects.

Design principles
-----------------
- General strong-motion processing.
- No ObsPy dependency.
- No station-specific assumptions.
- Float64 numerical processing.
- Vectorized NumPy/SciPy operations.
- Immutable ProcessingContext transition.
- Numerical products stored in ProcessingCache.
- Scalar engineering parameters stored in context.metrics.
- Audit trail through ProcessingContext.history.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext
from core.types.processing_state import StageStatus
from utils.exceptions import (
    ErrorCode,
    ProcessingError,
    SeverityLevel,
)


__all__ = [
    "ParameterConfig",
    "ParameterExtractionPlugin",
]


FloatArray = NDArray[np.float64]


# ============================================================================
# Constants
# ============================================================================

DEFAULT_GRAVITY: Final[float] = 9.80665


# ============================================================================
# Configuration
# ============================================================================


@dataclass(slots=True, frozen=True)
class ParameterConfig:
    """
    Configuration for strong-motion parameter extraction.

    Parameters
    ----------
    gravity
        Standard gravitational acceleration in m/s².

    husid_start
        Lower Husid energy percentile.

    husid_end75
        Intermediate Husid energy percentile.

    husid_end95
        Upper Husid energy percentile.

    Notes
    -----
    The default configuration produces D5-75 and D5-95 significant
    durations.
    """

    gravity: float = DEFAULT_GRAVITY

    husid_start: float = 0.05
    husid_end75: float = 0.75
    husid_end95: float = 0.95

    def __post_init__(self) -> None:
        """Validate parameter-extraction configuration."""

        if not np.isfinite(self.gravity):
            raise ValueError(
                "gravity must be finite."
            )

        if self.gravity <= 0.0:
            raise ValueError(
                f"gravity must be positive. Got {self.gravity}."
            )

        if not (
            0.0
            < self.husid_start
            < self.husid_end75
            < self.husid_end95
            <= 1.0
        ):
            raise ValueError(
                "Husid thresholds must satisfy "
                "0 < start < end75 < end95 <= 1."
            )


# ============================================================================
# Plugin
# ============================================================================


class ParameterExtractionPlugin(PreprocessorPlugin):
    """
    Extract fundamental strong-motion engineering parameters.

    Required ProcessingContext products
    ------------------------------------
    acceleration
        Acceleration waveform.

    velocity
        Velocity waveform.

    displacement
        Displacement waveform.

    Generated products
    ------------------
    context.metrics
        Scalar engineering parameters and waveform statistics.

    context.cache.husid_curve
        Normalized cumulative Arias-energy curve.

    context.processing_state
        Parameter-extraction stage marked SUCCESS.

    Notes
    -----
    The plugin does not perform baseline correction, filtering,
    detrending, tapering, or integration itself.

    Those operations belong to their dedicated pipeline stages.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: ParameterConfig | None = None,
    ) -> None:
        """
        Initialize the parameter extraction plugin.

        Parameters
        ----------
        config
            Parameter extraction configuration.
        """

        self._config = config or ParameterConfig()

    # ------------------------------------------------------------------
    # Plugin metadata
    # ------------------------------------------------------------------

    @property
    def plugin_name(self) -> str:
        """Return plugin name."""
        return "ParameterExtraction"

    @property
    def plugin_version(self) -> str:
        """Return plugin semantic version."""
        return "2.0.0"

    @property
    def plugin_description(self) -> str:
        """Return human-readable plugin description."""
        return (
            "Extract PGA, PGV, PGD, Arias Intensity, CAV, "
            "Husid Curve, significant durations, and waveform statistics."
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_waveform(
        name: str,
        data: NDArray[np.floating],
    ) -> FloatArray:
        """
        Validate a waveform for numerical analysis.

        Parameters
        ----------
        name
            Human-readable waveform name.

        data
            Waveform array.

        Returns
        -------
        numpy.ndarray
            Validated float64 array.

        Raises
        ------
        ProcessingError
            If waveform dimensions, length, or numerical values
            are invalid.
        """

        array = np.asarray(
            data,
            dtype=np.float64,
        )

        if array.ndim != 1:
            raise ProcessingError(
                message=(
                    f"{name} waveform must be one-dimensional."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "parameters",
                    "waveform": name,
                },
            )

        if array.size < 2:
            raise ProcessingError(
                message=(
                    f"{name} waveform must contain at least "
                    "two samples."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "parameters",
                    "waveform": name,
                    "npts": int(array.size),
                },
            )

        if not np.all(np.isfinite(array)):
            raise ProcessingError(
                message=(
                    f"{name} waveform contains NaN or infinite values."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "parameters",
                    "waveform": name,
                },
            )

        return array

    @staticmethod
    def _validate_sampling_rate(
        sampling_rate: float,
    ) -> float:
        """
        Validate sampling rate.
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
                    "module": "parameters",
                    "sampling_rate": sampling_rate,
                },
            )

        return float(sampling_rate)

    @staticmethod
    def _validate_kinematic_consistency(
        acceleration: FloatArray,
        velocity: FloatArray,
        displacement: FloatArray,
    ) -> None:
        """
        Validate that acceleration, velocity, and displacement
        have compatible sample counts.
        """

        npts = acceleration.size

        if velocity.size != npts:
            raise ProcessingError(
                message=(
                    "Acceleration and velocity have different "
                    "sample counts."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "parameters",
                    "acceleration_npts": int(acceleration.size),
                    "velocity_npts": int(velocity.size),
                },
            )

        if displacement.size != npts:
            raise ProcessingError(
                message=(
                    "Acceleration and displacement have different "
                    "sample counts."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "parameters",
                    "acceleration_npts": int(acceleration.size),
                    "displacement_npts": int(displacement.size),
                },
            )

    # ------------------------------------------------------------------
    # Numerical utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _cumulative_integral(
        data: FloatArray,
        dt: float,
    ) -> FloatArray:
        """
        Compute cumulative trapezoidal integral.
        """

        return np.asarray(
            cumulative_trapezoid(
                data,
                dx=dt,
                initial=0.0,
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _interpolated_crossing_time(
        cumulative_curve: FloatArray,
        threshold: float,
        dt: float,
    ) -> float:
        """
        Determine the time at which a cumulative normalized curve
        reaches a specified threshold.

        Linear interpolation is used between adjacent samples.

        Parameters
        ----------
        cumulative_curve
            Monotonically increasing normalized cumulative curve.

        threshold
            Target normalized energy level.

        dt
            Sampling interval.

        Returns
        -------
        float
            Estimated crossing time in seconds.
        """

        index = int(
            np.searchsorted(
                cumulative_curve,
                threshold,
                side="left",
            )
        )

        if index <= 0:
            return 0.0

        if index >= cumulative_curve.size:
            return float(
                (cumulative_curve.size - 1) * dt
            )

        y0 = cumulative_curve[index - 1]
        y1 = cumulative_curve[index]

        t0 = (index - 1) * dt
        t1 = index * dt

        denominator = y1 - y0

        if np.isclose(denominator, 0.0):
            return float(t1)

        fraction = (
            (threshold - y0)
            / denominator
        )

        fraction = float(
            np.clip(
                fraction,
                0.0,
                1.0,
            )
        )

        return float(
            t0 + fraction * (t1 - t0)
        )

    # ------------------------------------------------------------------
    # Strong-motion calculations
    # ------------------------------------------------------------------

    def _compute_peak_parameters(
        self,
        acceleration: FloatArray,
        velocity: FloatArray,
        displacement: FloatArray,
    ) -> tuple[float, float, float]:
        """
        Compute PGA, PGV, and PGD.
        """

        pga = float(
            np.max(
                np.abs(acceleration)
            )
        )

        pgv = float(
            np.max(
                np.abs(velocity)
            )
        )

        pgd = float(
            np.max(
                np.abs(displacement)
            )
        )

        return pga, pgv, pgd

    def _compute_arias_and_husid(
        self,
        acceleration: FloatArray,
        dt: float,
    ) -> tuple[
        float,
        FloatArray,
        float,
        float,
        float,
    ]:
        """
        Compute Arias Intensity, Husid Curve, and significant durations.

        Returns
        -------
        tuple
            Arias intensity,
            Husid curve,
            D5-75,
            D5-95,
            total acceleration-energy integral.
        """

        acceleration_squared = np.square(
            acceleration,
            dtype=np.float64,
        )

        cumulative_acceleration_energy = (
            self._cumulative_integral(
                acceleration_squared,
                dt,
            )
        )

        total_energy = float(
            cumulative_acceleration_energy[-1]
        )

        if (
            not np.isfinite(total_energy)
            or total_energy <= 0.0
        ):
            husid_curve = np.zeros_like(
                acceleration,
                dtype=np.float64,
            )

            return (
                0.0,
                husid_curve,
                0.0,
                0.0,
                0.0,
            )

        arias_factor = (
            np.pi
            / (2.0 * self._config.gravity)
        )

        arias_cumulative = (
            arias_factor
            * cumulative_acceleration_energy
        )

        arias_intensity = float(
            arias_cumulative[-1]
        )

        if (
            not np.isfinite(arias_intensity)
            or arias_intensity <= 0.0
        ):
            husid_curve = np.zeros_like(
                acceleration,
                dtype=np.float64,
            )

            return (
                0.0,
                husid_curve,
                0.0,
                0.0,
                total_energy,
            )

        husid_curve = (
            arias_cumulative
            / arias_intensity
        )

        # Numerical protection against tiny floating-point
        # excursions outside [0, 1].
        husid_curve = np.clip(
            husid_curve,
            0.0,
            1.0,
        )

        t5 = self._interpolated_crossing_time(
            husid_curve,
            self._config.husid_start,
            dt,
        )

        t75 = self._interpolated_crossing_time(
            husid_curve,
            self._config.husid_end75,
            dt,
        )

        t95 = self._interpolated_crossing_time(
            husid_curve,
            self._config.husid_end95,
            dt,
        )

        d5_75 = max(
            0.0,
            t75 - t5,
        )

        d5_95 = max(
            0.0,
            t95 - t5,
        )

        return (
            arias_intensity,
            husid_curve,
            float(d5_75),
            float(d5_95),
            total_energy,
        )

    @staticmethod
    def _compute_cav(
        acceleration: FloatArray,
        dt: float,
    ) -> float:
        """
        Compute cumulative absolute velocity (CAV).

        CAV is the time integral of absolute acceleration.
        """

        absolute_acceleration = np.abs(
            acceleration
        )

        cav = np.trapezoid(
            absolute_acceleration,
            dx=dt,
        )

        return float(cav)

    @staticmethod
    def _compute_statistics(
        acceleration: FloatArray,
    ) -> dict[str, float]:
        """
        Compute basic acceleration statistics.
        """

        squared = np.square(
            acceleration,
            dtype=np.float64,
        )

        return {
            "Mean_Acceleration": float(
                np.mean(acceleration)
            ),
            "Median_Acceleration": float(
                np.median(acceleration)
            ),
            "Std_Acceleration": float(
                np.std(
                    acceleration,
                    ddof=0,
                )
            ),
            "RMS_Acceleration": float(
                np.sqrt(
                    np.mean(squared)
                )
            ),
            "Peak_to_Peak_Acceleration": float(
                np.ptp(acceleration)
            ),
            "Variance_Acceleration": float(
                np.var(
                    acceleration,
                    ddof=0,
                )
            ),
        }

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """
        Extract all configured strong-motion parameters.
        """

        self.validate_input(context)

        # --------------------------------------------------------------
        # Required kinematic states
        # --------------------------------------------------------------

        acceleration_waveform = context.acceleration
        velocity_waveform = context.velocity
        displacement_waveform = context.displacement

        if acceleration_waveform is None:
            raise ProcessingError(
                message=(
                    "Acceleration data is unavailable. "
                    "Parameter extraction requires acceleration."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "parameters",
                    "trace_id": context.trace_id,
                },
            )

        if velocity_waveform is None:
            raise ProcessingError(
                message=(
                    "Velocity data is unavailable. "
                    "Run kinematic integration first."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "parameters",
                    "trace_id": context.trace_id,
                },
            )

        if displacement_waveform is None:
            raise ProcessingError(
                message=(
                    "Displacement data is unavailable. "
                    "Run kinematic integration first."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "parameters",
                    "trace_id": context.trace_id,
                },
            )

        # --------------------------------------------------------------
        # Extract arrays
        # --------------------------------------------------------------

        acceleration = self._validate_waveform(
            "Acceleration",
            acceleration_waveform.data,
        )

        velocity = self._validate_waveform(
            "Velocity",
            velocity_waveform.data,
        )

        displacement = self._validate_waveform(
            "Displacement",
            displacement_waveform.data,
        )

        # --------------------------------------------------------------
        # Sampling-rate validation
        # --------------------------------------------------------------

        sampling_rate = self._validate_sampling_rate(
            acceleration_waveform.sampling_rate
        )

        dt = 1.0 / sampling_rate

        # --------------------------------------------------------------
        # Kinematic consistency
        # --------------------------------------------------------------

        self._validate_kinematic_consistency(
            acceleration,
            velocity,
            displacement,
        )

        # Sampling-rate consistency
        if not np.isclose(
            velocity_waveform.sampling_rate,
            sampling_rate,
            rtol=1e-6,
            atol=0.0,
        ):
            raise ProcessingError(
                message=(
                    "Velocity sampling rate is inconsistent "
                    "with acceleration sampling rate."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "parameters",
                    "trace_id": context.trace_id,
                    "acceleration_sampling_rate": sampling_rate,
                    "velocity_sampling_rate": (
                        velocity_waveform.sampling_rate
                    ),
                },
            )

        if not np.isclose(
            displacement_waveform.sampling_rate,
            sampling_rate,
            rtol=1e-6,
            atol=0.0,
        ):
            raise ProcessingError(
                message=(
                    "Displacement sampling rate is inconsistent "
                    "with acceleration sampling rate."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "parameters",
                    "trace_id": context.trace_id,
                    "acceleration_sampling_rate": sampling_rate,
                    "displacement_sampling_rate": (
                        displacement_waveform.sampling_rate
                    ),
                },
            )

        # --------------------------------------------------------------
        # Mathematical calculations
        # --------------------------------------------------------------

        try:
            # Peak parameters
            pga, pgv, pgd = (
                self._compute_peak_parameters(
                    acceleration,
                    velocity,
                    displacement,
                )
            )

            # Arias + Husid + significant durations
            (
                arias_intensity,
                husid_curve,
                d5_75,
                d5_95,
                acceleration_energy,
            ) = self._compute_arias_and_husid(
                acceleration,
                dt,
            )

            # CAV
            cav = self._compute_cav(
                acceleration,
                dt,
            )

            # Statistics
            statistics = self._compute_statistics(
                acceleration
            )

        except Exception as exc:
            raise ProcessingError(
                message=(
                    "Mathematical computation failed during "
                    "strong-motion parameter extraction."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "parameters",
                    "trace_id": context.trace_id,
                    "sampling_rate": sampling_rate,
                    "npts": int(acceleration.size),
                },
                cause=exc,
            ) from exc

        # --------------------------------------------------------------
        # Scalar metrics
        # --------------------------------------------------------------

        new_metrics = dict(
            context.metrics
        )

        new_metrics.update(
            {
                "PGA": pga,
                "PGV": pgv,
                "PGD": pgd,
                "Arias_Intensity": arias_intensity,
                "CAV": cav,
                "Significant_Duration_D5_75": d5_75,
                "Significant_Duration_D5_95": d5_95,
                "Acceleration_Energy_Integral": (
                    acceleration_energy
                ),
                **statistics,
            }
        )

        # --------------------------------------------------------------
        # Cache
        # --------------------------------------------------------------
        #
        # Cache is used for numerical arrays that downstream plugins
        # may reuse.
        #
        # Husid is therefore stored in ProcessingCache rather than
        # spectral_data.
        # --------------------------------------------------------------

        cache = context.cache

        cache.husid_curve = husid_curve.copy()

        # Strong-motion scalar products are also available in cache
        # for downstream consumers that operate directly on cache.
        cache.pga = pga
        cache.pgv = pgv
        cache.pgd = pgd
        cache.arias_intensity = arias_intensity
        cache.cumulative_absolute_velocity = cav
        cache.significant_duration_5_75 = d5_75
        cache.significant_duration_5_95 = d5_95

        cache.mean = statistics[
            "Mean_Acceleration"
        ]

        cache.standard_deviation = statistics[
            "Std_Acceleration"
        ]

        cache.root_mean_square = statistics[
            "RMS_Acceleration"
        ]

        # Energy is the unscaled integral ∫a²dt.  Arias intensity applies
        # the separate π/(2g) factor and is stored independently above.
        cache.acceleration_energy = acceleration_energy

        # --------------------------------------------------------------
        # Processing state
        # --------------------------------------------------------------
        #
        # ``strong_motion_parameters`` is the canonical lifecycle field for
        # this stage.  Mark it only after every scalar and cache product has
        # been calculated successfully.
        # --------------------------------------------------------------

        state = replace(
            context.processing_state,
            strong_motion_parameters=StageStatus.SUCCESS,
        )

        # --------------------------------------------------------------
        # Immutable context transition
        # --------------------------------------------------------------

        new_context = context.with_state(
            metrics=new_metrics,
            cache=cache,
            processing_state=state,
        )

        # --------------------------------------------------------------
        # Audit trail
        # --------------------------------------------------------------

        return new_context.add_history(
            step_name=self.plugin_name,
            details={
                "status": "SUCCESS",
                "PGA": pga,
                "PGV": pgv,
                "PGD": pgd,
                "Arias_Intensity": arias_intensity,
                "CAV": cav,
                "D5_75": d5_75,
                "D5_95": d5_95,
                "sampling_rate_hz": sampling_rate,
                "npts": int(acceleration.size),
            },
        )
