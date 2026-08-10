"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/processing/response_spectrum.py

Description
-----------
Response Spectrum computation for generic strong-motion waveform data.

This module is instrument-agnostic and network-agnostic. It operates on
the acceleration waveform contained in ProcessingContext and computes
SDOF response spectra using either:

- Nigam-Jennings exact recurrence method
- Newmark-Beta numerical integration

The module computes:

- SD  : Spectral Displacement
- PSV : Pseudo-Spectral Velocity
- PSA : Pseudo-Spectral Acceleration
- SA  : Absolute Spectral Acceleration

All processing is performed without modifying the input
ProcessingContext in-place.

Expected acceleration unit
--------------------------
The acceleration waveform is expected to be expressed in SI units:

    m/s²

Therefore:

    SD  -> m
    PSV -> m/s
    PSA -> m/s²
    SA  -> m/s²

The module does not contain any station-, network-, sensor-, or
manufacturer-specific logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from core.orchestrator import ProcessingStep
from core.sdof.newmark import solve_newmark
from core.sdof.nigam_jennings import solve_nigam_jennings
from core.types.context import ProcessingContext
from core.types.processing_state import StageStatus
from utils.exceptions import (
    ErrorCode,
    ProcessingError,
    SeverityLevel,
)

__all__ = [
    "ResponseSpectrumConfig",
    "ResponseSpectrumPlugin",
]


SolverName = Literal[
    "nigam_jennings",
    "newmark",
]


@dataclass(slots=True, frozen=True)
class ResponseSpectrumConfig:
    """
    Immutable configuration for response-spectrum computation.

    Parameters
    ----------
    periods:
        Oscillator periods in seconds.

    damping:
        Fractional critical damping ratio.

        Example:
            0.05 = 5% damping.

    solver:
        SDOF solution method.

        Supported values:
            "nigam_jennings"
            "newmark"
    """

    periods: tuple[float, ...] = field(
        default_factory=lambda: tuple(
            np.linspace(0.01, 4.0, 100, dtype=np.float64)
        )
    )

    damping: float = 0.05

    solver: SolverName = "nigam_jennings"

    def __post_init__(self) -> None:
        """
        Validate response-spectrum configuration.
        """

        if not self.periods:
            raise ValueError(
                "Response-spectrum period array must not be empty."
            )

        periods = np.asarray(self.periods, dtype=np.float64)

        if periods.ndim != 1:
            raise ValueError(
                "Response-spectrum periods must be one-dimensional."
            )

        if not np.all(np.isfinite(periods)):
            raise ValueError(
                "Response-spectrum periods must contain only finite values."
            )

        if np.any(periods <= 0.0):
            raise ValueError(
                "All response-spectrum periods must be > 0 seconds."
            )

        if self.damping <= 0.0 or self.damping >= 1.0:
            raise ValueError(
                "Damping ratio must satisfy 0 < damping < 1."
            )

        if self.solver not in {
            "nigam_jennings",
            "newmark",
        }:
            raise ValueError(
                f"Unsupported response-spectrum solver: {self.solver}"
            )


class ResponseSpectrumPlugin(ProcessingStep):
    """
    Compute elastic SDOF response spectra.

    The plugin is completely generic and does not depend on:

    - station name,
    - network code,
    - sensor manufacturer,
    - acquisition system,
    - SBSSI,
    - specific BMKG instrument.

    Input
    -----
    ProcessingContext.acceleration

    Output
    ------
    ProcessingContext.spectral_data containing:

        periods
        SD
        PSV
        PSA
        SA
        damping
        solver
        omega

    Notes
    -----
    PSA and PSV are pseudo-spectral quantities derived from SD:

        PSV = omega * SD

        PSA = omega² * SD

    SA is obtained directly from the absolute acceleration response
    returned by the selected SDOF solver.
    """

    def __init__(
        self,
        config: ResponseSpectrumConfig | None = None,
    ) -> None:
        self._config = config or ResponseSpectrumConfig()

    @property
    def name(self) -> str:
        """Canonical processing-step name."""
        return "Response_Spectrum"

    @property
    def plugin_version(self) -> str:
        """Processing-step semantic version."""
        return "1.0.0"

    # ------------------------------------------------------------------
    # Public processing interface
    # ------------------------------------------------------------------

    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """
        Compute the response spectrum from acceleration data.

        Parameters
        ----------
        context:
            Immutable BSMA processing context.

        Returns
        -------
        ProcessingContext
            New context containing the calculated response spectrum.

        Raises
        ------
        ProcessingError
            If acceleration data or sampling information is invalid,
            or if the SDOF solver fails.
        """

        self._validate_context(context)

        acceleration = context.acceleration

        if acceleration is None:
            # Defensive check; _validate_context already handles this.
            raise ProcessingError(
                message=(
                    "Acceleration waveform is unavailable for "
                    "response-spectrum computation."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                    "trace_id": context.trace_id,
                },
            )

        acc_data = np.asarray(
            acceleration.data,
            dtype=np.float64,
        )

        sampling_rate = float(
            acceleration.sampling_rate
        )

        dt = 1.0 / sampling_rate

        periods = np.asarray(
            self._config.periods,
            dtype=np.float64,
        )

        try:
            (
                sd,
                psv,
                psa,
                sa,
                omega,
            ) = self._compute_response_spectrum(
                acceleration=acc_data,
                dt=dt,
                periods=periods,
            )

        except ProcessingError:
            raise

        except Exception as exc:
            raise ProcessingError(
                message=(
                    "Response-spectrum computation failed."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                    "trace_id": context.trace_id,
                    "solver": self._config.solver,
                    "damping": self._config.damping,
                    "npts": int(acc_data.size),
                    "sampling_rate": sampling_rate,
                },
                cause=exc,
            ) from exc

        # --------------------------------------------------------------
        # Build immutable spectral-data transition
        # --------------------------------------------------------------

        spectral_data = dict(context.spectral_data)

        spectral_data.update(
            {
                "periods": periods.copy(),
                "omega": omega.copy(),
                "SD": sd.copy(),
                "PSV": psv.copy(),
                "PSA": psa.copy(),
                "SA": sa.copy(),
                "damping": float(self._config.damping),
                "solver": self._config.solver,
            }
        )

        # --------------------------------------------------------------
        # Processing state transition
        # --------------------------------------------------------------

        state = replace(
            context.processing_state,
            response_spectrum=StageStatus.SUCCESS,
        )

        # --------------------------------------------------------------
        # Audit metadata
        # --------------------------------------------------------------

        dominant_index = int(
            np.argmax(psa)
        )

        history_details = {
            "status": "SUCCESS",
            "solver": self._config.solver,
            "damping": float(self._config.damping),
            "period_count": int(periods.size),
            "period_min": float(np.min(periods)),
            "period_max": float(np.max(periods)),
            "max_psa": float(np.max(psa)),
            "max_sa": float(np.max(sa)),
            "max_sd": float(np.max(sd)),
            "max_psv": float(np.max(psv)),
            "dominant_period": float(
                periods[dominant_index]
            ),
        }

        return context.with_state(
            spectral_data=spectral_data,
            processing_state=state,
        ).add_history(
            step_name=self.name,
            details=history_details,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_context(
        self,
        context: ProcessingContext,
    ) -> None:
        """
        Validate all input conditions required by the SDOF solver.
        """

        if context is None:
            raise ProcessingError(
                message=(
                    "ProcessingContext must not be None."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                },
            )

        if context.acceleration is None:
            raise ProcessingError(
                message=(
                    "Acceleration waveform is unavailable. "
                    "Run the required preprocessing stages first."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                    "trace_id": context.trace_id,
                },
            )

        data = np.asarray(
            context.acceleration.data,
        )

        if data.ndim != 1:
            raise ProcessingError(
                message=(
                    "Response-spectrum input must be a "
                    "one-dimensional acceleration waveform."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                    "trace_id": context.trace_id,
                    "shape": tuple(data.shape),
                },
            )

        if data.size < 2:
            raise ProcessingError(
                message=(
                    "Acceleration waveform contains fewer than "
                    "two samples."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                    "trace_id": context.trace_id,
                    "npts": int(data.size),
                },
            )

        if not np.all(np.isfinite(data)):
            raise ProcessingError(
                message=(
                    "Acceleration waveform contains NaN or Inf values."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                    "trace_id": context.trace_id,
                },
            )

        sampling_rate = float(
            context.acceleration.sampling_rate
        )

        if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
            raise ProcessingError(
                message=(
                    "Acceleration sampling rate must be finite "
                    "and greater than zero."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                    "trace_id": context.trace_id,
                    "sampling_rate": sampling_rate,
                },
            )

        periods = np.asarray(
            self._config.periods,
            dtype=np.float64,
        )

        if periods.ndim != 1 or periods.size == 0:
            raise ProcessingError(
                message=(
                    "Response-spectrum period array is invalid."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                },
            )

        if not np.all(np.isfinite(periods)):
            raise ProcessingError(
                message=(
                    "Response-spectrum periods contain NaN or Inf."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                },
            )

        if np.any(periods <= 0.0):
            raise ProcessingError(
                message=(
                    "Response-spectrum periods must be greater "
                    "than zero."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                },
            )

    # ------------------------------------------------------------------
    # SDOF calculation
    # ------------------------------------------------------------------

    def _compute_response_spectrum(
        self,
        acceleration: np.ndarray,
        dt: float,
        periods: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Execute the selected SDOF solver and derive spectral parameters.

        Returns
        -------
        tuple
            SD, PSV, PSA, SA, omega
        """

        if not np.isfinite(dt) or dt <= 0.0:
            raise ProcessingError(
                message=(
                    "Invalid integration time step."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                    "dt": dt,
                },
            )

        # --------------------------------------------------------------
        # Select SDOF solver
        # --------------------------------------------------------------

        if self._config.solver == "nigam_jennings":

            displacement, velocity, absolute_acceleration = (
                solve_nigam_jennings(
                    acceleration=acceleration,
                    dt=dt,
                    periods=periods,
                    damping=self._config.damping,
                )
            )

        elif self._config.solver == "newmark":

            displacement, velocity, absolute_acceleration = (
                solve_newmark(
                    acceleration=acceleration,
                    dt=dt,
                    periods=periods,
                    damping=self._config.damping,
                )
            )

        else:
            raise ProcessingError(
                message=(
                    f"Unsupported SDOF solver: "
                    f"{self._config.solver}"
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                    "solver": self._config.solver,
                },
            )

        # --------------------------------------------------------------
        # Validate solver outputs
        # --------------------------------------------------------------

        displacement = np.asarray(
            displacement,
            dtype=np.float64,
        )

        velocity = np.asarray(
            velocity,
            dtype=np.float64,
        )

        absolute_acceleration = np.asarray(
            absolute_acceleration,
            dtype=np.float64,
        )

        expected_shape = (
            periods.size,
            acceleration.size,
        )

        if displacement.shape != expected_shape:
            raise ProcessingError(
                message=(
                    "SDOF displacement output has an unexpected "
                    f"shape: {displacement.shape}; "
                    f"expected {expected_shape}."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                    "solver": self._config.solver,
                },
            )

        if velocity.shape != expected_shape:
            raise ProcessingError(
                message=(
                    "SDOF velocity output has an unexpected shape."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                    "solver": self._config.solver,
                },
            )

        if absolute_acceleration.shape != expected_shape:
            raise ProcessingError(
                message=(
                    "SDOF absolute-acceleration output has an "
                    "unexpected shape."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                    "solver": self._config.solver,
                },
            )

        if not np.all(np.isfinite(displacement)):
            raise ProcessingError(
                message=(
                    "SDOF displacement response contains "
                    "non-finite values."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                    "solver": self._config.solver,
                },
            )

        if not np.all(np.isfinite(velocity)):
            raise ProcessingError(
                message=(
                    "SDOF velocity response contains "
                    "non-finite values."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                    "solver": self._config.solver,
                },
            )

        if not np.all(np.isfinite(absolute_acceleration)):
            raise ProcessingError(
                message=(
                    "SDOF absolute acceleration response contains "
                    "non-finite values."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "response_spectrum",
                    "solver": self._config.solver,
                },
            )

        # --------------------------------------------------------------
        # Spectral extraction
        # --------------------------------------------------------------

        sd = np.max(
            np.abs(displacement),
            axis=1,
        )

        omega = (
            2.0
            * np.pi
            / periods
        )

        psv = omega * sd

        psa = (
            omega
            * omega
            * sd
        )

        sa = np.max(
            np.abs(absolute_acceleration),
            axis=1,
        )

        # --------------------------------------------------------------
        # Final numerical validation
        # --------------------------------------------------------------

        outputs = {
            "SD": sd,
            "PSV": psv,
            "PSA": psa,
            "SA": sa,
            "omega": omega,
        }

        for name, values in outputs.items():
            if not np.all(np.isfinite(values)):
                raise ProcessingError(
                    message=(
                        f"Computed {name} spectrum contains "
                        "non-finite values."
                    ),
                    error_code=ErrorCode.RS001,
                    severity=SeverityLevel.ERROR,
                    context={
                        "module": "response_spectrum",
                        "solver": self._config.solver,
                        "quantity": name,
                    },
                )

        return (
            sd,
            psv,
            psa,
            sa,
            omega,
        )