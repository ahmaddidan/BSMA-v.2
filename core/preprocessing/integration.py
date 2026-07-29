"""
BMKG Strong Motion Analyzer (BSMA)

Integration Plugin
==================

Numerically integrates acceleration into velocity and displacement.

Scientific References
---------------------
- Boore (2001)
- Boore et al. (2002)
- COSMOS Strong Motion Processing Manual
- PEER NGA Processing Guidelines
- SciPy Documentation

Design Principles
-----------------
- Immutable ProcessingContext
- Production-grade typing
- Cache-aware
- Numerical stability
- Pipeline compatible
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from dataclasses import replace
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext
from core.types.processing_state import StageStatus

__all__ = [
    "IntegrationMethod",
    "IntegrationConfig",
    "IntegrationPlugin",
]

FloatArray = NDArray[np.float64]


class IntegrationMethod(str, Enum):
    """
    Numerical integration methods.

    Currently only cumulative trapezoidal integration is
    implemented because it provides excellent numerical
    stability for strong-motion processing.
    """

    CUMULATIVE_TRAPEZOID = "cumulative_trapezoid"


@dataclass(slots=True, frozen=True)
class IntegrationConfig:
    """
    Configuration for numerical integration.

    Parameters
    ----------
    method
        Numerical integration algorithm.

    remove_mean_before_integrating
        Remove residual DC offset before integration.

    remove_linear_trend
        Remove first-order trend before integration.
    """

    method: IntegrationMethod = (
        IntegrationMethod.CUMULATIVE_TRAPEZOID
    )

    remove_mean_before_integrating: bool = True

    remove_linear_trend: bool = False


class IntegrationPlugin(PreprocessorPlugin):
    """
    Velocity and displacement integration plugin.

    Input
    -----
    Acceleration

    Output
    ------
    Velocity
    Displacement
    """

    def __init__(
        self,
        config: IntegrationConfig = IntegrationConfig(),
    ) -> None:

        self._config = config

    @property
    def plugin_name(self) -> str:
        return "Integration"

    @property
    def plugin_description(self) -> str:
        return (
            "Acceleration to velocity/displacement integration."
        )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _validate(
        self,
        waveform: FloatArray,
        sampling_rate: float,
    ) -> None:
        """
        Validate numerical input before integration.
        """

        if waveform.size < 2:
            raise ValueError(
                "Waveform must contain at least two samples."
            )

        if np.isnan(waveform).any():
            raise ValueError(
                "Waveform contains NaN values."
            )

        if np.isinf(waveform).any():
            raise ValueError(
                "Waveform contains infinite values."
            )

        if sampling_rate <= 0.0:
            raise ValueError(
                "Sampling rate must be positive."
            )

    # ---------------------------------------------------------
    # Signal conditioning
    # ---------------------------------------------------------

    def _prepare_signal(
        self,
        waveform: FloatArray,
    ) -> FloatArray:
        """
        Apply optional preprocessing before integration.
        """

        signal = waveform.astype(
            np.float64,
            copy=True,
        )

        if self._config.remove_mean_before_integrating:
            signal -= np.mean(signal)

        if self._config.remove_linear_trend:

            x = np.arange(
                signal.size,
                dtype=np.float64,
            )

            coeff = np.polyfit(
                x,
                signal,
                1,
            )

            signal -= np.polyval(
                coeff,
                x,
            )

        return signal
    # ---------------------------------------------------------
    # Processing
    # ---------------------------------------------------------

    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """
        Integrate acceleration into velocity and displacement.

        Returns
        -------
        ProcessingContext
            New immutable ProcessingContext containing updated
            cache and processing state.
        """

        #
        # ---------------------------------------------
        # Common validation
        # ---------------------------------------------
        #

        self.validate_input(context)

        waveform = context.waveform
        assert waveform is not None

        sampling_rate = context.sampling_rate

        self._validate(
            waveform,
            sampling_rate,
        )

        #
        # ---------------------------------------------
        # Prepare signal
        # ---------------------------------------------
        #

        acceleration = self._prepare_signal(
            waveform,
        )

        dt = 1.0 / sampling_rate

        #
        # ---------------------------------------------
        # Velocity
        # ---------------------------------------------
        #

        velocity = cumulative_trapezoid(
            acceleration,
            dx=dt,
            initial=0.0,
        )

        #
        # ---------------------------------------------
        # Displacement
        # ---------------------------------------------
        #

        displacement = cumulative_trapezoid(
            velocity,
            dx=dt,
            initial=0.0,
        )

        #
        # ---------------------------------------------
        # Preserve dtype
        # ---------------------------------------------
        #

        velocity = velocity.astype(
            waveform.dtype,
            copy=False,
        )

        displacement = displacement.astype(
            waveform.dtype,
            copy=False,
        )

        #
        # ---------------------------------------------
        # Copy cache
        # ---------------------------------------------
        #

        cache = deepcopy(
            context.cache,
        )

        #
        # Velocity / displacement
        #

        cache.velocity = velocity
        cache.displacement = displacement

        #
        # Clear quantities depending on integration
        #

        cache.pgv = None
        cache.pgd = None

        cache.response_periods = None
        cache.spectral_acceleration = None
        cache.spectral_velocity = None
        cache.spectral_displacement = None

        cache.pseudo_spectral_velocity = None
        cache.pseudo_spectral_acceleration = None

        #
        # ---------------------------------------------
        # Processing state
        # ---------------------------------------------
        #

        state = replace(
            context.processing_state,
            integration=StageStatus.SUCCESS,
        )

        #
        # ---------------------------------------------
        # Immutable return
        # ---------------------------------------------
        #

        history = (
            "Integration("
            f"method={self._config.method.value}, "
            f"remove_mean={self._config.remove_mean_before_integrating}, "
            f"remove_trend={self._config.remove_linear_trend}"
            ")"
        )

        return (
            context.copy(
                cache=cache,
                processing_state=state,
            )
            .add_history(
                history,
            )
        )

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(
        self,
    ) -> str:

        cfg = self._config

        return (
            f"{self.__class__.__name__}("
            f"method={cfg.method.value}, "
            f"remove_mean={cfg.remove_mean_before_integrating}, "
            f"remove_trend={cfg.remove_linear_trend})"
        )