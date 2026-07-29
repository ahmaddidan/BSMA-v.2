"""
Unit tests for ProcessingCache.

BMKG Strong Motion Analyzer (BSMA)
"""

from __future__ import annotations

import numpy as np

from core.types.cache import ProcessingCache


def create_array(size: int = 100) -> np.ndarray:
    """Create synthetic float64 array."""
    return np.linspace(0.0, 1.0, size, dtype=np.float64)


# ---------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------


def test_default_cache_is_empty():
    cache = ProcessingCache()

    assert cache.velocity is None
    assert cache.displacement is None

    assert cache.fft_frequency is None
    assert cache.fft_amplitude is None
    assert cache.fft_phase is None

    assert cache.response_periods is None
    assert cache.spectral_acceleration is None

    assert cache.arias_intensity is None
    assert cache.cumulative_absolute_velocity is None

    assert cache.pga is None
    assert cache.pgv is None
    assert cache.pgd is None

    assert cache.husid_curve is None

    assert cache.has_velocity is False
    assert cache.has_displacement is False
    assert cache.has_fft is False
    assert cache.has_response_spectrum is False
    assert cache.has_strong_motion_parameters is False


# ---------------------------------------------------------------------
# Velocity / Displacement
# ---------------------------------------------------------------------


def test_store_velocity():
    cache = ProcessingCache()

    velocity = create_array()

    cache.velocity = velocity

    assert cache.has_velocity
    assert cache.velocity is velocity


def test_store_displacement():
    cache = ProcessingCache()

    displacement = create_array()

    cache.displacement = displacement

    assert cache.has_displacement
    assert cache.displacement is displacement


# ---------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------


def test_store_fft():
    cache = ProcessingCache()

    freq = np.linspace(0.0, 25.0, 51)
    amp = np.random.random(51)

    cache.fft_frequency = freq
    cache.fft_amplitude = amp

    assert cache.has_fft
    assert cache.fft_frequency is freq
    assert cache.fft_amplitude is amp


# ---------------------------------------------------------------------
# Strong-motion parameters
# ---------------------------------------------------------------------


def test_store_strong_motion_parameters():
    cache = ProcessingCache()

    cache.arias_intensity = 1.25
    cache.cumulative_absolute_velocity = 0.81

    cache.pga = 0.34
    cache.pgv = 18.4
    cache.pgd = 7.2

    assert cache.has_strong_motion_parameters

    assert cache.arias_intensity == 1.25
    assert cache.cumulative_absolute_velocity == 0.81

    assert cache.pga == 0.34
    assert cache.pgv == 18.4
    assert cache.pgd == 7.2


# ---------------------------------------------------------------------
# Response spectrum
# ---------------------------------------------------------------------


def test_response_spectrum_flag():
    cache = ProcessingCache()

    cache.response_periods = np.linspace(0.01, 5.0, 100)
    cache.spectral_acceleration = np.random.random(100)

    assert cache.has_response_spectrum


# ---------------------------------------------------------------------
# Husid curve
# ---------------------------------------------------------------------


def test_store_husid_curve():
    cache = ProcessingCache()

    husid = np.linspace(0.0, 1.0, 100)

    cache.husid_curve = husid

    assert cache.husid_curve is husid


# ---------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------


def test_clear():
    cache = ProcessingCache()

    cache.velocity = create_array()
    cache.displacement = create_array()

    cache.fft_frequency = np.linspace(0, 20, 50)
    cache.fft_amplitude = np.random.random(50)

    cache.arias_intensity = 1.5
    cache.cumulative_absolute_velocity = 2.0

    cache.pga = 0.45
    cache.pgv = 22.0
    cache.pgd = 10.0

    cache.husid_curve = create_array()

    cache.clear()

    assert cache.velocity is None
    assert cache.displacement is None

    assert cache.fft_frequency is None
    assert cache.fft_amplitude is None
    assert cache.fft_phase is None

    assert cache.arias_intensity is None
    assert cache.cumulative_absolute_velocity is None

    assert cache.pga is None
    assert cache.pgv is None
    assert cache.pgd is None

    assert cache.husid_curve is None

    assert cache.has_velocity is False
    assert cache.has_displacement is False
    assert cache.has_fft is False
    assert cache.has_response_spectrum is False
    assert cache.has_strong_motion_parameters is False


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


def test_to_dict():
    cache = ProcessingCache()

    cache.velocity = create_array()

    status = cache.to_dict()

    assert isinstance(status, dict)

    assert status["velocity"] is True
    assert status["displacement"] is False
    assert status["fft"] is False
    assert status["response_spectrum"] is False
    assert status["strong_motion"] is False


# ---------------------------------------------------------------------
# References
# ---------------------------------------------------------------------


def test_cache_keeps_reference():
    cache = ProcessingCache()

    arr = create_array()

    cache.velocity = arr

    arr[0] = 999.0

    assert cache.velocity[0] == 999.0


def test_multiple_fields_independent():
    cache = ProcessingCache()

    v = create_array()
    d = create_array() * 2

    cache.velocity = v
    cache.displacement = d

    assert np.array_equal(cache.velocity, v)
    assert np.array_equal(cache.displacement, d)

    assert not np.array_equal(cache.velocity, cache.displacement)