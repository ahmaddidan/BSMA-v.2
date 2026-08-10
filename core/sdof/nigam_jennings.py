"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/sdof/nigam_jennings.py

Description
-----------
Exact analytical SDOF oscillator solver using the Nigam-Jennings
piecewise-linear excitation method.

Reference
---------
Nigam, N. C., & Jennings, P. C. (1968).
Calculation of response spectra from strong-motion earthquake records.
Bulletin of the Seismological Society of America, 58(2), 909-922.

Mathematical model
------------------
For a linear SDOF oscillator subjected to ground acceleration:

    u¨ + 2*zeta*omega*u˙ + omega²*u = -a_g(t)

where

    u      : relative displacement
    u˙     : relative velocity
    u¨     : relative acceleration
    a_g    : ground acceleration
    omega  : natural circular frequency
    zeta   : damping ratio

The ground acceleration between two consecutive samples is assumed
to vary linearly. The corresponding state transition is evaluated
analytically.

Design principles
-----------------
- Generic for all valid strong-motion waveforms
- Independent of station/network/channel identity
- No dependency on ObsPy
- No dependency on MiniSEED/SAC/XML
- Vectorized across oscillator periods
- Explicit numerical validation
- Supports T=0 as the PGA anchor
- Deterministic output
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias


__all__ = ["solve_nigam_jennings"]


FloatArray: TypeAlias = NDArray[np.float64]


def solve_nigam_jennings(
    acceleration: FloatArray,
    dt: float,
    periods: FloatArray | None = None,
    damping: float = 0.05,
    *,
    T: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Solve linear SDOF oscillators using the Nigam-Jennings method.

    Parameters
    ----------
    acceleration
        One-dimensional ground-acceleration time series.

    dt
        Sampling interval in seconds.

    periods
        One-dimensional oscillator periods in seconds.

        T = 0 is permitted and represents the zero-period/PGA anchor.

    damping
        Critical damping ratio.

        Typical engineering value:

            0.05

        corresponds to 5% critical damping.

    Returns
    -------
    u
        Relative displacement response.

        Shape:
            (n_periods, n_samples)

    v
        Relative velocity response.

        Shape:
            (n_periods, n_samples)

    a_abs
        Absolute acceleration response.

        Shape:
            (n_periods, n_samples)

    Notes
    -----
    The excitation is assumed to vary linearly between consecutive
    acceleration samples.

    The state vector is:

        x = [u, v]^T

    and the recursive solution is:

        x_(i+1)
            = A x_i + B1 p_i + B2 p_(i+1)

    where:

        p_i = -a_g(i)

    For a zero-period oscillator:

        u = 0
        v = 0
        a_abs = a_g

    which provides the PGA anchor without numerical division by
    T = 0.

    The solver always returns array responses. A scalar period input is
    preserved as a length-one period array.
    """

    # ==================================================================
    # 1. Normalize inputs
    # ==================================================================

    acceleration = np.asarray(
        acceleration,
        dtype=np.float64,
    )

    if periods is None:
        if T is None:
            raise TypeError(
                "solve_nigam_jennings() missing required argument: 'periods' or 'T'"
            )
        periods = T
    elif T is not None:
        raise TypeError(
            "solve_nigam_jennings() received both 'periods' and 'T'"
        )

    periods = np.asarray(
        periods,
        dtype=np.float64,
    )

    scalar_period = periods.ndim == 0

    if scalar_period:
        periods = periods.reshape(1)

    if acceleration.ndim != 1:
        raise ValueError(
            "acceleration must be a one-dimensional array."
        )

    if periods.ndim != 1:
        raise ValueError(
            "periods must be a one-dimensional array."
        )

    if acceleration.size < 2:
        raise ValueError(
            "acceleration must contain at least two samples."
        )

    if periods.size == 0:
        raise ValueError(
            "periods must contain at least one value."
        )

    # ==================================================================
    # 2. Validate numerical inputs
    # ==================================================================

    if not np.isfinite(acceleration).all():
        raise ValueError(
            "acceleration contains NaN or infinite values."
        )

    if not np.isfinite(periods).all():
        raise ValueError(
            "periods contains NaN or infinite values."
        )

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(
            f"dt must be finite and > 0. Got {dt!r}."
        )

    if not np.isfinite(damping):
        raise ValueError(
            f"damping must be finite. Got {damping!r}."
        )

    if damping < 0.0 or damping >= 1.0:
        raise ValueError(
            "damping must satisfy 0 <= damping < 1."
        )

    if np.any(periods < 0.0):
        raise ValueError(
            "All oscillator periods must satisfy T >= 0."
        )

    # ==================================================================
    # 3. Allocate global outputs
    # ==================================================================

    n_samples = acceleration.size
    n_periods = periods.size

    u = np.zeros(
        (n_periods, n_samples),
        dtype=np.float64,
    )

    v = np.zeros(
        (n_periods, n_samples),
        dtype=np.float64,
    )

    a_abs = np.zeros(
        (n_periods, n_samples),
        dtype=np.float64,
    )

    # ==================================================================
    # 4. Handle zero-period oscillator
    #
    # T -> 0 response corresponds to ground acceleration.
    #
    # No division by zero is permitted.
    # ==================================================================

    zero_period = periods == 0.0
    valid_period = periods > 0.0

    if np.any(zero_period):
        a_abs[zero_period, :] = acceleration

    if not np.any(valid_period):
        return u, v, a_abs

    # ==================================================================
    # 5. Oscillator parameters
    # ==================================================================

    oscillator_periods = periods[valid_period]

    omega = (
        2.0
        * np.pi
        / oscillator_periods
    )

    omega_squared = omega**2

    sqrt_term = np.sqrt(
        1.0 - damping**2
    )

    damped_frequency = (
        omega
        * sqrt_term
    )

    # ==================================================================
    # 6. Trigonometric/exponential terms
    # ==================================================================

    exponential = np.exp(
        -damping
        * omega
        * dt
    )

    sine = np.sin(
        damped_frequency
        * dt
    )

    cosine = np.cos(
        damped_frequency
        * dt
    )

    zeta_over_sqrt = (
        damping
        / sqrt_term
    )

    omega_over_sqrt = (
        omega
        / sqrt_term
    )

    # ==================================================================
    # 7. State-transition matrix A
    #
    # x_(i+1) = A*x_i + ...
    # ==================================================================

    A11 = exponential * (
        zeta_over_sqrt * sine
        + cosine
    )

    A12 = exponential * (
        sine
        / damped_frequency
    )

    A21 = -exponential * (
        omega_over_sqrt
        * sine
    )

    A22 = exponential * (
        cosine
        - zeta_over_sqrt * sine
    )

    # ==================================================================
    # 8. Particular-solution coefficients
    #
    # The excitation is represented as piecewise-linear between
    # a_i and a_(i+1).
    #
    # p_i = -a_g(i)
    # ==================================================================

    term_1 = (
        2.0
        * damping
        / (omega * dt)
    )

    term_2 = (
        (1.0 - 2.0 * damping**2)
        / (damped_frequency * dt)
        - zeta_over_sqrt
    )

    term_3 = (
        (2.0 * damping**2 - 1.0)
        / (damped_frequency * dt)
    )

    B11 = (
        term_1
        + exponential
        * (
            term_2 * sine
            - (1.0 + term_1) * cosine
        )
    ) / omega_squared

    B12 = (
        1.0
        - term_1
        + exponential
        * (
            term_3 * sine
            + term_1 * cosine
        )
    ) / omega_squared

    term_4 = (
        omega_over_sqrt
        + damping
        / (
            dt
            * sqrt_term
        )
    )

    term_5 = (
        damping
        / (
            dt
            * sqrt_term
        )
    )

    B21 = (
        -1.0 / dt
        + exponential
        * (
            term_4 * sine
            + cosine / dt
        )
    ) / omega_squared

    B22 = (
        1.0 / dt
        - exponential
        * (
            term_5 * sine
            + cosine / dt
        )
    ) / omega_squared

    # ==================================================================
    # 9. Validate transition coefficients
    # ==================================================================

    coefficient_arrays = (
        A11,
        A12,
        A21,
        A22,
        B11,
        B12,
        B21,
        B22,
    )

    if not all(
        np.isfinite(array).all()
        for array in coefficient_arrays
    ):
        raise FloatingPointError(
            "Nigam-Jennings generated non-finite "
            "state-transition coefficients."
        )

    # ==================================================================
    # 10. Allocate valid-period workspaces
    # ==================================================================

    n_valid = oscillator_periods.size

    relative_displacement = np.zeros(
        (n_valid, n_samples),
        dtype=np.float64,
    )

    relative_velocity = np.zeros(
        (n_valid, n_samples),
        dtype=np.float64,
    )

    # ==================================================================
    # 11. Time stepping
    # ==================================================================

    for index in range(n_samples - 1):

        u_i = relative_displacement[:, index]
        v_i = relative_velocity[:, index]

        p_i = -acceleration[index]
        p_i1 = -acceleration[index + 1]

        relative_displacement[:, index + 1] = (
            A11 * u_i
            + A12 * v_i
            + B11 * p_i
            + B12 * p_i1
        )

        relative_velocity[:, index + 1] = (
            A21 * u_i
            + A22 * v_i
            + B21 * p_i
            + B22 * p_i1
        )

    # ==================================================================
    # 12. Absolute acceleration
    #
    # Equation of motion:
    #
    #     u¨ + 2*zeta*omega*u˙ + omega²*u = -a_g
    #
    # Therefore:
    #
    #     a_abs = u¨ + a_g
    #            = -2*zeta*omega*u˙ - omega²*u
    #
    # ==================================================================

    absolute_acceleration = -(
        2.0
        * damping
        * omega[:, np.newaxis]
        * relative_velocity
    ) - (
        omega_squared[:, np.newaxis]
        * relative_displacement
    )

    # ==================================================================
    # 13. Numerical sanity checks
    # ==================================================================

    if not np.isfinite(relative_displacement).all():
        raise FloatingPointError(
            "Nigam-Jennings produced NaN or infinite "
            "relative displacement values."
        )

    if not np.isfinite(relative_velocity).all():
        raise FloatingPointError(
            "Nigam-Jennings produced NaN or infinite "
            "relative velocity values."
        )

    if not np.isfinite(absolute_acceleration).all():
        raise FloatingPointError(
            "Nigam-Jennings produced NaN or infinite "
            "absolute acceleration values."
        )

    # ==================================================================
    # 14. Insert valid-period responses into global arrays
    # ==================================================================

    u[valid_period, :] = relative_displacement
    v[valid_period, :] = relative_velocity
    a_abs[valid_period, :] = absolute_acceleration

    if scalar_period:
        period = float(periods[0])

        if period == 0.0:
            sd = 0.0
            psv = 0.0
            psa = float(np.max(np.abs(a_abs[0])))
            return sd, psv, psa

        sd = float(np.max(np.abs(u[0])))
        omega = 2.0 * np.pi / period
        psv = omega * sd
        psa = omega * omega * sd
        return sd, psv, psa

    return u, v, a_abs