"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/processing/spectra.py

Description
-----------
Production-grade Response Spectrum extraction plugin.

Computes:
    - Spectral Displacement (SD)
    - Pseudo-Spectral Velocity (PSV)
    - Pseudo-Spectral Acceleration (PSA)
    - Absolute Spectral Acceleration (SA)

The plugin operates on the acceleration state stored in the immutable
ProcessingContext and supports arbitrary waveform sources. Waveform
provenance and instrument metadata remain outside this processing layer.

Important
---------
T = 0 is NOT passed to the SDOF solver. It is treated separately as
the PGA anchor because a zero-period oscillator is mathematically
undefined for the SDOF formulation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from core.interfaces.preprocessor import PreprocessorPlugin
from core.sdof.newmark import solve_newmark_vectorized
from core.types.context import ProcessingContext
from core.types.processing_state import StageStatus
from utils.exceptions import (
    ErrorCode,
    ProcessingError,
    SeverityLevel,
)

__all__ = [
    "SpectraConfig",
    "ResponseSpectrumPlugin",
]

FloatArray = NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class SpectraConfig:
    """
    Configuration for engineering response spectrum extraction.

    Parameters
    ----------
    damping:
        Critical damping ratio. Default is 5%.

    period_min:
        Minimum oscillator period in seconds.

    period_max:
        Maximum oscillator period in seconds.

    period_steps:
        Number of logarithmically spaced oscillator periods.

    include_pga:
        If True, prepend T=0 as a PGA anchor. The zero-period value
        is not passed to the SDOF solver.
    """

    damping: float = 0.05
    period_min: float = 0.01
    period_max: float = 10.0
    period_steps: int = 100
    include_pga: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.damping):
            raise ValueError("Damping must be finite.")

        if not 0.0 < self.damping < 1.0:
            raise ValueError(
                f"Damping must be within (0, 1). Got {self.damping}."
            )

        if not np.isfinite(self.period_min):
            raise ValueError("period_min must be finite.")

        if not np.isfinite(self.period_max):
            raise ValueError("period_max must be finite.")

        if self.period_min <= 0.0:
            raise ValueError(
                f"period_min must be > 0. Got {self.period_min}."
            )

        if self.period_max <= self.period_min:
            raise ValueError(
                "period_max must be greater than period_min."
            )

        if self.period_steps < 2:
            raise ValueError(
                "period_steps must be >= 2."
            )


class ResponseSpectrumPlugin(PreprocessorPlugin):
    """
    Compute engineering response spectra from acceleration.

    The plugin is source-independent. It does not assume a particular
    station, network, instrument, XML format, or waveform provider.

    Input
    -----
    context.acceleration.data

    Output
    ------
    context.spectral_data containing:

        periods
        frequency
        angular_frequency
        SD
        PSV
        PSA
        SA
        PGA
        damping
    """

    def __init__(
        self,
        config: SpectraConfig | None = None,
    ) -> None:
        self._config = config or SpectraConfig()

        oscillator_periods = np.logspace(
            np.log10(self._config.period_min),
            np.log10(self._config.period_max),
            self._config.period_steps,
            dtype=np.float64,
        )

        if self._config.include_pga:
            self._periods = np.concatenate(
                (
                    np.array([0.0], dtype=np.float64),
                    oscillator_periods,
                )
            )
        else:
            self._periods = oscillator_periods

    # ------------------------------------------------------------------
    # Plugin metadata
    # ------------------------------------------------------------------

    @property
    def plugin_name(self) -> str:
        """Return canonical plugin name."""
        return "ResponseSpectrum"

    @property
    def plugin_version(self) -> str:
        """Return plugin version."""
        return "2.0.0"

    @property
    def plugin_description(self) -> str:
        """Return human-readable plugin description."""
        return (
            "Computes SD, PSV, PSA, and absolute SA using "
            "a vectorized Newmark-Beta SDOF solver."
        )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """
        Compute response spectrum from the acceleration state.

        Returns
        -------
        ProcessingContext
            New immutable context containing response-spectrum data.
        """

        self.validate_input(context)

        acceleration_state = context.acceleration

        if acceleration_state is None:
            raise ProcessingError(
                message=(
                    "Acceleration state is unavailable. "
                    "Response spectrum requires acceleration data."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "spectra",
                    "trace_id": context.trace_id,
                },
            )

        acceleration = np.asarray(
            acceleration_state.data,
            dtype=np.float64,
        )

        # --------------------------------------------------------------
        # Fundamental validation
        # --------------------------------------------------------------

        if acceleration.ndim != 1:
            raise ProcessingError(
                message="Acceleration data must be a one-dimensional array.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "spectra",
                    "trace_id": context.trace_id,
                    "ndim": acceleration.ndim,
                },
            )

        if acceleration.size < 2:
            raise ProcessingError(
                message=(
                    "At least two acceleration samples are required "
                    "for response-spectrum computation."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "spectra",
                    "trace_id": context.trace_id,
                    "npts": int(acceleration.size),
                },
            )

        if not np.all(np.isfinite(acceleration)):
            raise ProcessingError(
                message=(
                    "Acceleration contains NaN or infinite values."
                ),
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "spectra",
                    "trace_id": context.trace_id,
                },
            )

        sampling_rate = float(context.sampling_rate)

        if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
            raise ProcessingError(
                message="Sampling rate must be finite and greater than zero.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "spectra",
                    "trace_id": context.trace_id,
                    "sampling_rate": sampling_rate,
                },
            )

        dt = 1.0 / sampling_rate

        # --------------------------------------------------------------
        # PGA anchor
        # --------------------------------------------------------------

        pga = float(np.max(np.abs(acceleration)))

        # --------------------------------------------------------------
        # SDOF solver
        #
        # IMPORTANT:
        # T=0 is excluded from the solver.
        # --------------------------------------------------------------

        oscillator_mask = self._periods > 0.0
        oscillator_periods = self._periods[oscillator_mask]

        try:
            u, v, a_abs = solve_newmark_vectorized(
                acceleration=acceleration,
                dt=dt,
                periods=oscillator_periods,
                damping=self._config.damping,
            )

        except Exception as exc:
            raise ProcessingError(
                message=(
                    "Vectorized Newmark-Beta response-spectrum "
                    "calculation failed."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "spectra",
                    "trace_id": context.trace_id,
                    "solver": "newmark_vectorized",
                    "damping": self._config.damping,
                    "period_min": self._config.period_min,
                    "period_max": self._config.period_max,
                    "period_steps": self._config.period_steps,
                },
                cause=exc,
            ) from exc

        # --------------------------------------------------------------
        # Shape validation
        # --------------------------------------------------------------

        expected_shape = (
            oscillator_periods.size,
            acceleration.size,
        )

        if u.shape != expected_shape:
            raise ProcessingError(
                message=(
                    "SDOF displacement output has an unexpected shape."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "spectra",
                    "trace_id": context.trace_id,
                    "expected_shape": expected_shape,
                    "actual_shape": u.shape,
                },
            )

        if v.shape != expected_shape:
            raise ProcessingError(
                message=(
                    "SDOF velocity output has an unexpected shape."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "spectra",
                    "trace_id": context.trace_id,
                    "expected_shape": expected_shape,
                    "actual_shape": v.shape,
                },
            )

        if a_abs.shape != expected_shape:
            raise ProcessingError(
                message=(
                    "SDOF absolute-acceleration output has "
                    "an unexpected shape."
                ),
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "spectra",
                    "trace_id": context.trace_id,
                    "expected_shape": expected_shape,
                    "actual_shape": a_abs.shape,
                },
            )

        # --------------------------------------------------------------
        # Spectral displacement
        # --------------------------------------------------------------

        sd_osc = np.max(
            np.abs(u),
            axis=1,
        )

        # --------------------------------------------------------------
        # Circular frequency
        # --------------------------------------------------------------

        omega_osc = (
            2.0
            * np.pi
            / oscillator_periods
        )

        frequency_osc = 1.0 / oscillator_periods

        # --------------------------------------------------------------
        # Pseudo-spectral velocity
        #
        # PSV = omega * SD
        # --------------------------------------------------------------

        psv_osc = omega_osc * sd_osc

        # --------------------------------------------------------------
        # Pseudo-spectral acceleration
        #
        # PSA = omega² * SD
        #
        # This is intentionally distinguished from absolute SA.
        # --------------------------------------------------------------

        psa_osc = (
            omega_osc
            * omega_osc
            * sd_osc
        )

        # --------------------------------------------------------------
        # Absolute spectral acceleration
        #
        # SA = max(|absolute acceleration|)
        # --------------------------------------------------------------

        sa_osc = np.max(
            np.abs(a_abs),
            axis=1,
        )

        # --------------------------------------------------------------
        # Add PGA anchor at T=0
        # --------------------------------------------------------------

        if self._config.include_pga:
            periods = self._periods.copy()

            frequency = np.concatenate(
                (
                    np.array([np.inf], dtype=np.float64),
                    frequency_osc,
                )
            )

            angular_frequency = np.concatenate(
                (
                    np.array([np.inf], dtype=np.float64),
                    omega_osc,
                )
            )

            sd = np.concatenate(
                (
                    np.array([0.0], dtype=np.float64),
                    sd_osc,
                )
            )

            psv = np.concatenate(
                (
                    np.array([0.0], dtype=np.float64),
                    psv_osc,
                )
            )

            psa = np.concatenate(
                (
                    np.array([pga], dtype=np.float64),
                    psa_osc,
                )
            )

            sa = np.concatenate(
                (
                    np.array([pga], dtype=np.float64),
                    sa_osc,
                )
            )

        else:
            periods = oscillator_periods.copy()
            frequency = frequency_osc
            angular_frequency = omega_osc
            sd = sd_osc
            psv = psv_osc
            psa = psa_osc
            sa = sa_osc

        # --------------------------------------------------------------
        # Final numerical validation
        # --------------------------------------------------------------

        spectral_arrays = {
            "periods": periods,
            "frequency": frequency,
            "angular_frequency": angular_frequency,
            "SD": sd,
            "PSV": psv,
            "PSA": psa,
            "SA": sa,
        }

        for name, array in spectral_arrays.items():
            if not np.all(np.isfinite(array)):
                # frequency and angular_frequency contain +inf at
                # the optional T=0 PGA anchor by definition.
                if name in {"frequency", "angular_frequency"}:
                    if np.all(
                        np.isfinite(array[1:])
                    ):
                        continue

                raise ProcessingError(
                    message=(
                        f"Response-spectrum output '{name}' "
                        "contains non-finite values."
                    ),
                    error_code=ErrorCode.RS001,
                    severity=SeverityLevel.ERROR,
                    context={
                        "module": "spectra",
                        "trace_id": context.trace_id,
                        "array": name,
                    },
                )

        # --------------------------------------------------------------
        # Store results in ProcessingContext
        # --------------------------------------------------------------

        new_spectral_data = dict(
            context.spectral_data
        )

        new_spectral_data.update(
            {
                "periods": periods,
                "frequency": frequency,
                "angular_frequency": angular_frequency,
                "SD": sd,
                "PSV": psv,
                "PSA": psa,
                "SA": sa,
                "PGA": pga,
                "damping": float(self._config.damping),
                "solver": "newmark_vectorized",
                "period_min": float(
                    self._config.period_min
                ),
                "period_max": float(
                    self._config.period_max
                ),
                "period_steps": int(
                    self._config.period_steps
                ),
            }
        )

        # --------------------------------------------------------------
        # Processing state
        # --------------------------------------------------------------

        state = replace(
            context.processing_state,
            spectra=StageStatus.SUCCESS,
        )

        # --------------------------------------------------------------
        # Audit trail
        # --------------------------------------------------------------

        dominant_index = int(
            np.argmax(psa)
        )

        dominant_period = float(
            periods[dominant_index]
        )

        max_psa = float(
            np.max(psa)
        )

        return (
            context.with_state(
                spectral_data=new_spectral_data,
                processing_state=state,
            )
            .add_history(
                step_name=self.plugin_name,
                details={
                    "status": "SUCCESS",
                    "version": self.plugin_version,
                    "solver": "newmark_vectorized",
                    "damping": float(
                        self._config.damping
                    ),
                    "period_min": float(
                        self._config.period_min
                    ),
                    "period_max": float(
                        self._config.period_max
                    ),
                    "period_steps": int(
                        self._config.period_steps
                    ),
                    "include_pga": bool(
                        self._config.include_pga
                    ),
                    "PGA": pga,
                    "max_PSA": max_psa,
                    "dominant_period": dominant_period,
                },
            )
        )