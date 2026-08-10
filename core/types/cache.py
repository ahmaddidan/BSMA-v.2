"""
BMKG Strong Motion Analyzer (BSMA)

Module
------
core/types/cache.py

Description
-----------
Processing cache for numerical products generated during
strong-motion waveform processing.

The cache stores derived numerical quantities associated with
one ProcessingContext. Products are grouped according to their
physical and computational dependencies.

Dependency hierarchy
--------------------

    Acceleration
        |
        +-- Time-domain products
        |      +-- Velocity
        |      +-- Displacement
        |      +-- Husid
        |      +-- Strong-motion parameters
        |
        +-- Frequency-domain products
        |      +-- FFT
        |      +-- FAS
        |      +-- PSD
        |
        +-- Response-spectrum products
               +-- SD
               +-- SV
               +-- SA
               +-- PSV
               +-- PSA

Notes
-----
This cache is intended for a single waveform/component.

Station-level quantities requiring multiple components, such as
HVSR and RotD50/RotD100, should be handled by higher-level
analysis objects rather than this cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "ProcessingCache",
]


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class ProcessingCache:
    """
    Numerical cache associated with one waveform/component.

    A value of ``None`` means that the corresponding product has
    not been computed or has been invalidated.

    The cache is mutable by design. ProcessingContext itself may
    remain functionally immutable while receiving a deep-copied
    cache during processing.
    """

    # ==========================================================
    # Time Domain
    # ==========================================================

    velocity: FloatArray | None = None
    displacement: FloatArray | None = None
    husid_curve: FloatArray | None = None

    # ==========================================================
    # Strong Motion Parameters
    # ==========================================================

    pga: float | None = None
    pgv: float | None = None
    pgd: float | None = None

    arias_intensity: float | None = None
    acceleration_energy: float | None = None
    cumulative_absolute_velocity: float | None = None

    significant_duration_5_75: float | None = None
    significant_duration_5_95: float | None = None

    # ==========================================================
    # Waveform Statistics
    # ==========================================================

    mean: float | None = None
    standard_deviation: float | None = None
    root_mean_square: float | None = None

    # ==========================================================
    # Frequency Domain
    # ==========================================================

    fft_frequency: FloatArray | None = None
    fft_amplitude: FloatArray | None = None
    fft_phase: FloatArray | None = None

    fourier_amplitude_spectrum: FloatArray | None = None
    power_spectral_density: FloatArray | None = None

    # ==========================================================
    # Response Spectrum
    # ==========================================================

    response_periods: FloatArray | None = None

    spectral_displacement: FloatArray | None = None
    spectral_velocity: FloatArray | None = None
    spectral_acceleration: FloatArray | None = None

    pseudo_spectral_velocity: FloatArray | None = None
    pseudo_spectral_acceleration: FloatArray | None = None

    # ==========================================================
    # Availability Properties
    # ==========================================================

    @property
    def has_velocity(self) -> bool:
        """Return True when velocity has been computed."""
        return self.velocity is not None

    @property
    def has_displacement(self) -> bool:
        """Return True when displacement has been computed."""
        return self.displacement is not None

    @property
    def has_time_domain(self) -> bool:
        """
        Return True when the principal integrated time-domain
        products are available.
        """
        return (
            self.velocity is not None
            and self.displacement is not None
        )

    @property
    def has_husid(self) -> bool:
        """Return True when the Husid curve is available."""
        return self.husid_curve is not None

    @property
    def has_waveform_statistics(self) -> bool:
        """
        Return True when all primary waveform statistics exist.
        """
        return all(
            value is not None
            for value in (
                self.mean,
                self.standard_deviation,
                self.root_mean_square,
            )
        )

    @property
    def has_strong_motion_parameters(self) -> bool:
        """
        Return True when the complete principal strong-motion
        parameter set is available.
        """
        return all(
            value is not None
            for value in (
                self.pga,
                self.pgv,
                self.pgd,
                self.arias_intensity,
                self.cumulative_absolute_velocity,
                self.significant_duration_5_75,
                self.significant_duration_5_95,
            )
        )

    @property
    def has_fft(self) -> bool:
        """
        Return True when the FFT frequency and amplitude arrays
        are both available.
        """
        return (
            self.fft_frequency is not None
            and self.fft_amplitude is not None
        )

    @property
    def has_frequency_domain(self) -> bool:
        """
        Return True when the principal FFT representation exists.
        """
        return self.has_fft

    @property
    def has_response_spectrum(self) -> bool:
        """
        Return True when the complete response-spectrum dataset
        is available.

        The required quantities are:

        - response periods
        - spectral displacement (SD)
        - spectral velocity (SV)
        - spectral acceleration (SA)
        - pseudo spectral velocity (PSV)
        - pseudo spectral acceleration (PSA)

        This prevents downstream code from treating a partial
        response spectrum as a valid complete spectrum.
        """
        return all(
            value is not None
            for value in (
                self.response_periods,
                self.spectral_displacement,
                self.spectral_velocity,
                self.spectral_acceleration,
                self.pseudo_spectral_velocity,
                self.pseudo_spectral_acceleration,
            )
        )

    @property
    def is_empty(self) -> bool:
        """
        Return True when no cached numerical product exists.
        """
        return all(
            value is None
            for value in self.__dataclass_fields__
        )

    # ==========================================================
    # Cache Invalidation
    # ==========================================================

    def clear(self) -> None:
        """
        Invalidate the entire cache.

        This must be called whenever the waveform itself is
        fundamentally replaced or modified.
        """
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, None)

    def clear_time_domain(self) -> None:
        """
        Invalidate all products derived from the acceleration
        time history.

        Because frequency-domain and response-spectrum products
        are also derived from the waveform, they are invalidated
        transitively.
        """
        # Integrated signals
        self.velocity = None
        self.displacement = None
        self.husid_curve = None

        # Strong-motion parameters
        self.pga = None
        self.pgv = None
        self.pgd = None
        self.arias_intensity = None
        self.acceleration_energy = None
        self.cumulative_absolute_velocity = None
        self.significant_duration_5_75 = None
        self.significant_duration_5_95 = None

        # Statistics
        self.mean = None
        self.standard_deviation = None
        self.root_mean_square = None

        # Downstream products
        self.clear_frequency_domain()
        self.clear_response_spectrum()

    def clear_frequency_domain(self) -> None:
        """
        Invalidate all frequency-domain products.
        """
        self.fft_frequency = None
        self.fft_amplitude = None
        self.fft_phase = None
        self.fourier_amplitude_spectrum = None
        self.power_spectral_density = None

    def clear_response_spectrum(self) -> None:
        """
        Invalidate all response-spectrum products.
        """
        self.response_periods = None
        self.spectral_displacement = None
        self.spectral_velocity = None
        self.spectral_acceleration = None
        self.pseudo_spectral_velocity = None
        self.pseudo_spectral_acceleration = None

    # ==========================================================
    # Serialization
    # ==========================================================

    def to_dict(self) -> dict[str, bool]:
        """
        Return a lightweight availability summary.

        Numerical arrays are deliberately excluded.
        """
        return {
            "empty": self.is_empty,
            "time_domain": self.has_time_domain,
            "waveform_statistics": self.has_waveform_statistics,
            "strong_motion": self.has_strong_motion_parameters,
            "frequency_domain": self.has_frequency_domain,
            "response_spectrum": self.has_response_spectrum,
            "husid_curve": self.has_husid,
        }

    # ==========================================================
    # Representation
    # ==========================================================

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"time_domain={self.has_time_domain}, "
            f"strong_motion={self.has_strong_motion_parameters}, "
            f"frequency_domain={self.has_frequency_domain}, "
            f"response_spectrum={self.has_response_spectrum})"
        )
