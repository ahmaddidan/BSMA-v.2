"""
BMKG Strong Motion Analyzer (BSMA)

Module: core/processing/advanced_analysis.py

Description
-----------
General-purpose advanced strong-motion analysis algorithms.

This module contains pure numerical functions for waveform analysis.
It is intentionally independent from:

- ObsPy
- MiniSEED
- StationXML
- specific seismic networks
- specific stations
- specific instruments
- GUI components
- ProcessingContext

The functions operate only on numerical waveform arrays and sampling
information.

Supported analyses
------------------
- Husid curve
- Significant duration (D5-75 / D5-95)
- Fourier Amplitude Spectrum (FAS)

Scientific principle
--------------------
Instrument response removal, unit conversion, detrending, tapering,
filtering, and other preprocessing operations MUST be performed by
their respective pipeline stages before calling these functions.

This module therefore does not silently modify the waveform.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid


FloatArray = NDArray[np.float64]

__all__ = [
    "compute_husid_curve",
    "compute_significant_duration",
    "compute_husid_and_duration",
    "compute_fas",
]


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

_HUSID_LOW_PERCENT: Final[float] = 0.05
_HUSID_HIGH_PERCENT: Final[float] = 0.95


# ===========================================================================
# Validation
# ===========================================================================


def _validate_waveform(
    data: NDArray[np.floating] | NDArray[np.integer],
    fs: float,
) -> FloatArray:
    """
    Validate and normalize a waveform for numerical processing.

    Parameters
    ----------
    data
        One-dimensional waveform samples.

    fs
        Sampling frequency in Hz.

    Returns
    -------
    numpy.ndarray
        A float64 copy of the waveform.

    Raises
    ------
    ValueError
        If the sampling rate or waveform is invalid.
    """

    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(
            f"Sampling frequency must be finite and > 0 Hz; got {fs!r}."
        )

    array = np.asarray(data, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError(
            f"Waveform must be one-dimensional; got ndim={array.ndim}."
        )

    if array.size < 2:
        raise ValueError(
            "Waveform must contain at least two samples."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "Waveform contains NaN or infinite values."
        )

    return array


# ===========================================================================
# Husid Curve
# ===========================================================================


def compute_husid_curve(
    data: NDArray[np.floating] | NDArray[np.integer],
    fs: float,
) -> FloatArray:
    """
    Compute the normalized Husid energy curve.

    The Husid curve is defined from cumulative squared acceleration:

        H(t) =
            integral_0^t a(tau)^2 dtau
            --------------------------------
            integral_0^T a(tau)^2 dtau

    Trapezoidal numerical integration is used instead of a simple
    rectangular cumulative sum.

    Parameters
    ----------
    data
        Acceleration waveform.

    fs
        Sampling frequency in Hz.

    Returns
    -------
    numpy.ndarray
        Normalized Husid curve ranging approximately from 0 to 1.

    Notes
    -----
    The waveform is NOT detrended, filtered, tapered, or otherwise
    modified inside this function.
    """

    waveform = _validate_waveform(data, fs)
    dt = 1.0 / fs

    squared_acceleration = np.square(waveform)

    cumulative_energy = cumulative_trapezoid(
        squared_acceleration,
        dx=dt,
        initial=0.0,
    )

    total_energy = float(cumulative_energy[-1])

    if not np.isfinite(total_energy) or total_energy <= 0.0:
        return np.zeros_like(waveform)

    husid = cumulative_energy / total_energy

    # Numerical protection against tiny floating-point excursions.
    husid = np.clip(husid, 0.0, 1.0)

    return husid.astype(np.float64, copy=False)


# ===========================================================================
# Significant Duration
# ===========================================================================


def _crossing_time(
    husid: FloatArray,
    time: FloatArray,
    threshold: float,
) -> float:
    """
    Determine the time at which the Husid curve crosses a threshold.

    Linear interpolation is used between the two surrounding samples.

    Parameters
    ----------
    husid
        Normalized Husid curve.

    time
        Time vector corresponding to ``husid``.

    threshold
        Threshold between 0 and 1.

    Returns
    -------
    float
        Estimated threshold-crossing time in seconds.
    """

    indices = np.flatnonzero(husid >= threshold)

    if indices.size == 0:
        return float(time[-1])

    index = int(indices[0])

    if index == 0:
        return float(time[0])

    h0 = float(husid[index - 1])
    h1 = float(husid[index])
    t0 = float(time[index - 1])
    t1 = float(time[index])

    denominator = h1 - h0

    if np.isclose(denominator, 0.0):
        return t1

    fraction = (threshold - h0) / denominator

    return float(
        t0 + fraction * (t1 - t0)
    )


def compute_significant_duration(
    husid: FloatArray,
    fs: float,
    low_percent: float = _HUSID_LOW_PERCENT,
    high_percent: float = _HUSID_HIGH_PERCENT,
) -> tuple[float, float, float]:
    """
    Compute significant duration from a Husid curve.

    Parameters
    ----------
    husid
        Normalized Husid curve.

    fs
        Sampling frequency in Hz.

    low_percent
        Lower energy threshold as a fraction.
        Default is 0.05 (5%).

    high_percent
        Upper energy threshold as a fraction.
        Default is 0.95 (95%).

    Returns
    -------
    tuple
        ``(t_low, t_high, duration)`` in seconds.

    Raises
    ------
    ValueError
        If thresholds or sampling rate are invalid.
    """

    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(
            f"Sampling frequency must be > 0 Hz; got {fs!r}."
        )

    if not (
        0.0 <= low_percent < high_percent <= 1.0
    ):
        raise ValueError(
            "Duration thresholds must satisfy "
            "0 <= low_percent < high_percent <= 1."
        )

    husid_array = np.asarray(
        husid,
        dtype=np.float64,
    )

    if husid_array.ndim != 1:
        raise ValueError(
            "Husid curve must be one-dimensional."
        )

    if husid_array.size < 2:
        raise ValueError(
            "Husid curve must contain at least two samples."
        )

    if not np.all(np.isfinite(husid_array)):
        raise ValueError(
            "Husid curve contains NaN or infinite values."
        )

    time = np.arange(
        husid_array.size,
        dtype=np.float64,
    ) / fs

    t_low = _crossing_time(
        husid_array,
        time,
        low_percent,
    )

    t_high = _crossing_time(
        husid_array,
        time,
        high_percent,
    )

    duration = max(
        0.0,
        t_high - t_low,
    )

    return (
        float(t_low),
        float(t_high),
        float(duration),
    )


# ===========================================================================
# Combined Husid + Duration
# ===========================================================================


def compute_husid_and_duration(
    data: NDArray[np.floating] | NDArray[np.integer],
    fs: float,
    low_percent: float = 0.05,
    high_percent: float = 0.95,
) -> tuple[FloatArray, float, float, float]:
    """
    Compute Husid curve and significant duration.

    Parameters
    ----------
    data
        Acceleration waveform.

    fs
        Sampling frequency in Hz.

    low_percent
        Lower Husid threshold. Default: 5%.

    high_percent
        Upper Husid threshold. Default: 95%.

    Returns
    -------
    tuple
        ``(husid, t_low, t_high, duration)``.

    Notes
    -----
    This function is retained as a compatibility wrapper for the
    previous BSMA API.
    """

    waveform = _validate_waveform(data, fs)

    husid = compute_husid_curve(
        waveform,
        fs,
    )

    if np.allclose(husid, 0.0):
        return (
            np.zeros_like(waveform),
            0.0,
            0.0,
            0.0,
        )

    t_low, t_high, duration = compute_significant_duration(
        husid,
        fs,
        low_percent=low_percent,
        high_percent=high_percent,
    )

    return (
        husid,
        t_low,
        t_high,
        duration,
    )


# ===========================================================================
# Fourier Amplitude Spectrum
# ===========================================================================


def compute_fas(
    data: NDArray[np.floating] | NDArray[np.integer],
    fs: float,
) -> tuple[FloatArray, FloatArray]:
    """
    Compute the one-sided Fourier Amplitude Spectrum (FAS).

    Parameters
    ----------
    data
        One-dimensional waveform.

    fs
        Sampling frequency in Hz.

    Returns
    -------
    frequencies : numpy.ndarray
        One-sided frequency vector in Hz.

    amplitude : numpy.ndarray
        One-sided Fourier amplitude spectrum.

    Notes
    -----
    The function does not apply:

    - detrending
    - tapering
    - filtering
    - instrument correction

    Those operations belong to separate processing stages.

    The returned spectrum uses one-sided amplitude normalization:

        A(f) = |FFT(x)| / N

    with the non-DC and non-Nyquist components multiplied by 2.

    This convention preserves the amplitude contribution represented
    by the positive-frequency half of a real-valued signal.
    """

    waveform = _validate_waveform(data, fs)

    n_samples = waveform.size
    dt = 1.0 / fs

    spectrum = np.fft.rfft(waveform)

    frequencies = np.fft.rfftfreq(
        n_samples,
        d=dt,
    )

    amplitude = np.abs(spectrum) / n_samples

    # One-sided amplitude correction.
    if n_samples > 2:
        amplitude[1:-1] *= 2.0

    return (
        frequencies.astype(np.float64, copy=False),
        amplitude.astype(np.float64, copy=False),
    )