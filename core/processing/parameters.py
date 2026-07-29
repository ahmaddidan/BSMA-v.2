"""
BMKG Strong Motion Analyzer (BSMA)

Strong Motion Parameter Extraction Plugin
=========================================

Computes engineering strong-motion parameters from processed
waveforms and stores them in ProcessingCache.

Computed Parameters
-------------------
- PGA
- PGV
- PGD
- Arias Intensity
- Cumulative Absolute Velocity (CAV)
- Husid Curve
- Significant Duration D5-75
- Significant Duration D5-95
- Waveform Statistics

Scientific References
---------------------
Arias (1970)
Trifunac & Brady (1975)
Boore (2001)
COSMOS V2
PEER NGA
USGS ShakeMap
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import simpson

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext
from core.types.processing_state import StageStatus

FloatArray = NDArray[np.float64]

__all__ = [
    "ParameterConfig",
    "ParameterExtractionPlugin",
]


@dataclass(slots=True, frozen=True)
class ParameterConfig:
    """
    Configuration for strong-motion parameter extraction.
    """

    gravity: float = 9.80665

    husid_start: float = 0.05

    husid_end: float = 0.75

    husid_end95: float = 0.95


class ParameterExtractionPlugin(PreprocessorPlugin):
    """
    Extract engineering strong-motion parameters.
    """

    def __init__(
        self,
        config: ParameterConfig = ParameterConfig(),
    ) -> None:

        self._config = config

    @property
    def plugin_name(self) -> str:

        return "ParameterExtraction"

    @property
    def plugin_description(self) -> str:

        return (
            "Compute engineering strong-motion parameters."
        )
    # ---------------------------------------------------------
    # Configuration Validation
    # ---------------------------------------------------------

    def _validate_config(self) -> None:
        """
        Validate parameter extraction configuration.
        """

        cfg = self._config

        if cfg.gravity <= 0.0:
            raise ValueError(
                "Gravity must be positive."
            )

        if not (
            0.0
            < cfg.husid_start
            < cfg.husid_end
            <= 1.0
        ):
            raise ValueError(
                "Require: "
                "0 < husid_start < husid_end <= 1."
            )

        if not (
            0.0
            < cfg.husid_start
            < cfg.husid_end95
            <= 1.0
        ):
            raise ValueError(
                "Require: "
                "0 < husid_start < husid_end95 <= 1."
            )
    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _validate(
        self,
        context: ProcessingContext,
    ) -> None:

        waveform = context.waveform
        velocity = context.cache.velocity
        displacement = context.cache.displacement

        if waveform is None:
            raise ValueError(
                "Waveform is None."
            )

        if len(waveform) < 2:
            raise ValueError(
                "Waveform must contain at least two samples."
            )

        if context.sampling_rate <= 0.0:
            raise ValueError(
                "Sampling rate must be positive."
            )

        if velocity is None:
            raise ValueError(
                "Velocity not available. "
                "Run IntegrationPlugin first."
            )

        if displacement is None:
            raise ValueError(
                "Displacement not available. "
                "Run IntegrationPlugin first."
            )

        if len(velocity) != len(waveform):
            raise ValueError(
                "Velocity length mismatch."
            )

        if len(displacement) != len(waveform):
            raise ValueError(
                "Displacement length mismatch."
            )

        if np.isnan(waveform).any():
            raise ValueError(
                "Waveform contains NaN."
            )

        if np.isinf(waveform).any():
            raise ValueError(
                "Waveform contains Inf."
            )
        if np.isnan(velocity).any():
            raise ValueError(
                "Velocity contains NaN."
            )

        if np.isnan(displacement).any():
            raise ValueError(
                "Displacement contains NaN."
            )

        if np.isinf(velocity).any():
            raise ValueError(
                "Velocity contains Inf."
            )

        if np.isinf(displacement).any():
            raise ValueError(
                "Displacement contains Inf."
            )
    # ---------------------------------------------------------
    # Numerical Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _peak(
        signal: FloatArray,
    ) -> float:

        return float(
            np.max(
                np.abs(signal)
            )
        )

    def _arias(
        self,
        acceleration: FloatArray,
        dt: float,
    ) -> float:

        integral = simpson(
            acceleration ** 2,
            dx=dt,
        )

        return float(
            np.pi
            / (2.0 * self._config.gravity)
            * integral
        )

    @staticmethod
    def _cav(
        acceleration: FloatArray,
        dt: float,
    ) -> float:

        return float(
            simpson(
                np.abs(acceleration),
                dx=dt,
            )
        )

    @staticmethod
    def _waveform_statistics(
        signal: FloatArray,
        dt: float,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        """
        Compute basic waveform statistics.
        """

        mean = float(
            np.mean(signal)
        )

        standard_deviation = float(
            np.std(
                signal,
                ddof=0,
            )
        )

        root_mean_square = float(
            np.sqrt(
                np.mean(
                    signal ** 2
                )
            )
        )

        energy = float(
            np.sum(
                signal ** 2
            )
            * dt
        )

        return (
            mean,
            standard_deviation,
            root_mean_square,
            energy,
        )

    @staticmethod
    def _husid(
        acceleration: FloatArray,
        dt: float,
    ) -> FloatArray:

        energy = cumulative_trapezoid(
            acceleration ** 2,
            dx=dt,
            initial=0.0,
        )

        total = energy[-1]

        if total <= 0.0:

            return np.zeros_like(
                energy
            )

        return energy / total

    @staticmethod
    def _duration(
        husid: FloatArray,
        dt: float,
        start_level: float,
        end_level: float,
    ) -> float:
        """
        Compute significant duration using linear interpolation.
        """

        time = (
            np.arange(
                husid.size,
                dtype=np.float64,
            )
            * dt
        )

        t_start = np.interp(
            start_level,
            husid,
            time,
        )

        t_end = np.interp(
            end_level,
            husid,
            time,
        )

        return float(
            t_end - t_start
        )
    # ---------------------------------------------------------
    # Processing
    # ---------------------------------------------------------

    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """
        Compute engineering strong-motion parameters.

        Parameters
        ----------
        context
            Immutable ProcessingContext.

        Returns
        -------
        ProcessingContext
            New immutable context containing updated cache.
        """

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        self.validate_input(
            context,
        )

        self._validate_config()

        self._validate(
            context,
        )

        waveform = context.waveform
        velocity = context.cache.velocity
        displacement = context.cache.displacement

        assert waveform is not None
        assert velocity is not None
        assert displacement is not None

        dt = (
            1.0
            / context.sampling_rate
        )

        # -----------------------------------------------------
        # Waveform Statistics
        # -----------------------------------------------------

        (
            mean,
            standard_deviation,
            root_mean_square,
            energy,
        ) = self._waveform_statistics(
            waveform,
            dt,
        )

        # -----------------------------------------------------
        # Peak Ground Motion
        # -----------------------------------------------------

        pga = self._peak(
            waveform,
        )

        pgv = self._peak(
            velocity,
        )

        pgd = self._peak(
            displacement,
        )

        # -----------------------------------------------------
        # Arias Intensity
        # -----------------------------------------------------

        arias = self._arias(
            waveform,
            dt,
        )

        # -----------------------------------------------------
        # Cumulative Absolute Velocity
        # -----------------------------------------------------

        cav = self._cav(
            waveform,
            dt,
        )

        # -----------------------------------------------------
        # Husid Curve
        # -----------------------------------------------------

        husid = self._husid(
            waveform,
            dt,
        )

        # -----------------------------------------------------
        # Significant Duration
        # -----------------------------------------------------

        duration_5_75 = self._duration(
            husid,
            dt,
            self._config.husid_start,
            self._config.husid_end,
        )

        duration_5_95 = self._duration(
            husid,
            dt,
            self._config.husid_start,
            self._config.husid_end95,
        )

        # -----------------------------------------------------
        # Copy Cache
        # -----------------------------------------------------

        cache = deepcopy(
            context.cache,
        )

        # -----------------------------------------------------
        # Time Domain Products
        # -----------------------------------------------------

        cache.husid_curve = husid

        # -----------------------------------------------------
        # Strong Motion Parameters
        # -----------------------------------------------------

        cache.pga = pga

        cache.pgv = pgv

        cache.pgd = pgd

        cache.arias_intensity = arias

        cache.cumulative_absolute_velocity = cav

        cache.significant_duration_5_75 = (
            duration_5_75
        )

        cache.significant_duration_5_95 = (
            duration_5_95
        )

        # -----------------------------------------------------
        # Waveform Statistics
        # -----------------------------------------------------

        cache.mean = mean

        cache.standard_deviation = (
            standard_deviation
        )

        cache.root_mean_square = (
            root_mean_square
        )

        cache.energy = energy
        # -----------------------------------------------------
        # Processing State
        # -----------------------------------------------------

        state = replace(
            context.processing_state,
            parameters=StageStatus.SUCCESS,
        )

        # -----------------------------------------------------
        # Processing History
        # -----------------------------------------------------

        history = (
            "ParameterExtraction("
            f"PGA={pga:.6g}, "
            f"PGV={pgv:.6g}, "
            f"PGD={pgd:.6g}, "
            f"Arias={arias:.6g}, "
            f"CAV={cav:.6g}, "
            f"D5-75={duration_5_75:.4f}s, "
            f"D5-95={duration_5_95:.4f}s"
            f")"
        )

        # -----------------------------------------------------
        # Return Immutable Context
        # -----------------------------------------------------

        return (
            context.copy(
                cache=cache,
                processing_state=state,
            )
            .add_history(
                history
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
            f"gravity={cfg.gravity}, "
            f"D5-75={cfg.husid_start:.2f}-{cfg.husid_end:.2f}, "
            f"D5-95={cfg.husid_start:.2f}-{cfg.husid_end95:.2f}"
            f")"
        )