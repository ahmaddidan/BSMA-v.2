"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/sdof/newmark.py

Description
-----------
Linear SDOF oscillator solver using the Newmark-Beta average acceleration
method (beta=1/4, gamma=1/2).

The solver accepts acceleration time series in physical units and computes:

    - Relative displacement, u(t)
    - Relative velocity, v(t)
    - Absolute acceleration, a_abs(t)

The governing equation is:

    u¨ + 2*zeta*omega*u˙ + omega²*u = -a_g(t)

where:

    u       = relative displacement of the oscillator
    u˙      = relative velocity
    u¨      = relative acceleration
    a_g(t)  = ground acceleration
    omega   = natural circular frequency
    zeta    = damping ratio

For T -> 0, the oscillator response approaches the ground acceleration,
therefore the zero-period response is explicitly supported as the PGA
anchor.

Design goals
------------
- Python 3.12 compatible
- NumPy vectorized across oscillator periods
- Newmark average acceleration method
- No dependency on station/network-specific identifiers
- Works with arbitrary valid strong-motion acceleration arrays
- Deterministic numerical behavior
- Explicit validation of all numerical inputs
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias


__all__ = ["solve_newmark", "solve_newmark_vectorized"]


FloatArray: TypeAlias = NDArray[np.float64]


def solve_newmark(
    acceleration: FloatArray,
    dt: float,
    periods: FloatArray | None = None,
    damping: float = 0.05,
    *,
    T: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Solve a set of linear SDOF oscillators using Newmark-Beta.

    Parameters
    ----------
    acceleration
        Ground acceleration time series.

        Shape:
            (n_samples,)

        Unit:
            Any internally consistent acceleration unit, e.g.
            m/s², cm/s² (gal), or g.

    dt
        Sampling interval in seconds.

    periods
        Oscillator natural periods in seconds.

        Periods must satisfy:

            T >= 0

        A period of exactly zero is allowed and represents the
        rigid/zero-period PGA anchor.

    damping
        Critical damping ratio.

        Typical engineering value:

            damping = 0.05

        corresponding to 5% critical damping.

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
    The implemented Newmark parameters are:

        beta  = 1/4
        gamma = 1/2

    This is the constant-average-acceleration formulation.

    For unit mass:

        m = 1

    therefore:

        k = omega²
        c = 2*zeta*omega

    The zero-period oscillator is handled explicitly:

        u = 0
        v = 0
        a_abs = a_ground

    The solver always returns array responses. A scalar period input is
    preserved as a length-one period array.
    """

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------

    acceleration = np.asarray(acceleration, dtype=np.float64)
    if periods is None:
        if T is None:
            raise TypeError(
                "solve_newmark() missing required argument: 'periods' or 'T'"
            )
        periods = T
    elif T is not None:
        raise TypeError(
            "solve_newmark() received both 'periods' and 'T'"
        )

    periods = np.asarray(periods, dtype=np.float64)

    scalar_period = periods.ndim == 0

    # Support scalar periods for convenience.
    if scalar_period:
        periods = periods.reshape(1)

    # Ensure one-dimensional waveform.
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
            "periods must contain at least one oscillator period."
        )

    if not np.isfinite(acceleration).all():
        raise ValueError(
            "acceleration contains NaN or infinite values."
        )

    if not np.isfinite(periods).all():
        raise ValueError(
            "periods contains NaN or infinite values."
        )

    # ------------------------------------------------------------------
    # Numerical parameter validation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Output allocation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Zero-period oscillator / PGA anchor
    # ------------------------------------------------------------------

    zero_period = periods == 0.0
    valid_period = periods > 0.0

    if np.any(zero_period):
        a_abs[zero_period, :] = acceleration

    # ------------------------------------------------------------------
    # No finite-period oscillators
    # ------------------------------------------------------------------

    if not np.any(valid_period):
        return u, v, a_abs

    # ------------------------------------------------------------------
    # SDOF properties
    # ------------------------------------------------------------------

    oscillator_periods = periods[valid_period]

    omega = 2.0 * np.pi / oscillator_periods

    # Unit mass formulation:
    #
    #     m = 1
    #     c = 2*zeta*omega
    #     k = omega²

    damping_coefficient = 2.0 * damping * omega
    stiffness = omega**2

    # ------------------------------------------------------------------
    # Newmark-Beta constants
    #
    # beta  = 1/4
    # gamma = 1/2
    # ------------------------------------------------------------------

    beta = 0.25
    gamma = 0.5

    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    # Effective stiffness:
    #
    #     k_eff = k + a0*m + a1*c
    #
    # with m = 1.

    effective_stiffness = (
        stiffness
        + a0
        + a1 * damping_coefficient
    )

    if not np.isfinite(effective_stiffness).all():
        raise FloatingPointError(
            "Non-finite effective stiffness encountered in Newmark solver."
        )

    if np.any(effective_stiffness <= 0.0):
        raise FloatingPointError(
            "Non-positive effective stiffness encountered in Newmark solver."
        )

    # ------------------------------------------------------------------
    # Working arrays
    # ------------------------------------------------------------------

    n_valid = oscillator_periods.size

    relative_displacement = np.zeros(
        (n_valid, n_samples),
        dtype=np.float64,
    )

    relative_velocity = np.zeros(
        (n_valid, n_samples),
        dtype=np.float64,
    )

    relative_acceleration = np.zeros(
        (n_valid, n_samples),
        dtype=np.float64,
    )

    # Initial conditions:
    #
    #     u(0) = 0
    #     v(0) = 0
    #
    # From the equation of motion:
    #
    #     u¨(0) = -a_g(0)

    relative_acceleration[:, 0] = -acceleration[0]

    # ------------------------------------------------------------------
    # Newmark time integration
    # ------------------------------------------------------------------

    for index in range(n_samples - 1):

        u_i = relative_displacement[:, index]
        v_i = relative_velocity[:, index]
        a_i = relative_acceleration[:, index]

        ground_acceleration_next = acceleration[index + 1]

        # External effective load:
        #
        # P_eff =
        #     -a_g(i+1)
        #     + m*(a0*u_i + a2*v_i + a3*a_i)
        #     + c*(a1*u_i + a4*v_i + a5*a_i)

        effective_load = (
            -ground_acceleration_next
            + (
                a0 * u_i
                + a2 * v_i
                + a3 * a_i
            )
            + damping_coefficient
            * (
                a1 * u_i
                + a4 * v_i
                + a5 * a_i
            )
        )

        u_next = (
            effective_load
            / effective_stiffness
        )

        a_next = (
            a0 * (u_next - u_i)
            - a2 * v_i
            - a3 * a_i
        )

        v_next = (
            v_i
            + dt
            * (
                (1.0 - gamma) * a_i
                + gamma * a_next
            )
        )

        relative_displacement[:, index + 1] = u_next
        relative_velocity[:, index + 1] = v_next
        relative_acceleration[:, index + 1] = a_next

    # ------------------------------------------------------------------
    # Absolute acceleration
    #
    #     a_abs = u¨ + a_g
    #
    # where u¨ is relative acceleration.
    # ------------------------------------------------------------------

    absolute_acceleration = (
        relative_acceleration
        + acceleration[np.newaxis, :]
    )

    # ------------------------------------------------------------------
    # Numerical sanity check
    # ------------------------------------------------------------------

    if not np.isfinite(relative_displacement).all():
        raise FloatingPointError(
            "NaN or infinite values detected in displacement response."
        )

    if not np.isfinite(relative_velocity).all():
        raise FloatingPointError(
            "NaN or infinite values detected in velocity response."
        )

    if not np.isfinite(absolute_acceleration).all():
        raise FloatingPointError(
            "NaN or infinite values detected in absolute acceleration response."
        )

    # ------------------------------------------------------------------
    # Copy valid-period results into global output arrays
    # ------------------------------------------------------------------

    u[valid_period, :] = relative_displacement
    v[valid_period, :] = relative_velocity
    a_abs[valid_period, :] = absolute_acceleration

    return u, v, a_abs


solve_newmark_vectorized = solve_newmark