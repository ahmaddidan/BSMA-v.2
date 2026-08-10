"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/types/context.py

Domain Types
============

Immutable Single Source of Truth (SSOT) for waveform processing.

This module defines:

    WaveformData
        Immutable physical container for one time-series waveform.

    ProcessingContext
        Immutable state container shared across the complete BSMA
        processing pipeline.

Design Principles
-----------------
- Single Source of Truth (SSOT)
- No ObsPy dependency
- Immutable context transitions
- Numerical safety
- Explicit distinction between raw and processed waveform
- Float64 numerical representation
- Full audit/provenance support
- Compatible with preprocessing, integration, strong-motion
  parameter extraction, response-spectrum analysis, and reporting
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .cache import ProcessingCache
from .processing_state import ProcessingState


__all__ = [
    "WaveformData",
    "ProcessingContext",
]


FloatArray = NDArray[np.float64]


# ============================================================================
# WaveformData
# ============================================================================


@dataclass(frozen=True, slots=True)
class WaveformData:
    """
    Immutable physical container for one seismic time series.

    Parameters
    ----------
    data
        One-dimensional waveform samples.
        Data are internally normalized to ``float64``.

    sampling_rate
        Sampling frequency in Hz.

    unit
        Physical unit of the waveform, for example:

        - ``"counts"``
        - ``"m/s^2"``
        - ``"m/s"``
        - ``"m"``

    Notes
    -----
    The dataclass itself is frozen, but NumPy arrays are mutable objects.
    Therefore the input array is copied during initialization and exposed
    through a read-only NumPy array.

    This prevents accidental in-place modification such as::

        waveform.data[:] = 0

    from silently modifying the processing state.
    """

    data: FloatArray
    sampling_rate: float
    unit: str

    def __post_init__(self) -> None:
        # ------------------------------------------------------------------
        # Validate sampling rate
        # ------------------------------------------------------------------
        sampling_rate = float(self.sampling_rate)

        if not np.isfinite(sampling_rate):
            raise ValueError(
                "sampling_rate must be finite."
            )

        if sampling_rate <= 0.0:
            raise ValueError(
                f"sampling_rate must be > 0 Hz, got {sampling_rate!r}."
            )

        # ------------------------------------------------------------------
        # Validate unit
        # ------------------------------------------------------------------
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError(
                "unit must be a non-empty string."
            )

        # ------------------------------------------------------------------
        # Normalize waveform to float64.
        #
        # A copy is intentional. It prevents external references to the
        # original NumPy array from modifying the domain object.
        # ------------------------------------------------------------------
        array = np.asarray(self.data, dtype=np.float64)

        if array.ndim != 1:
            raise ValueError(
                "WaveformData.data must be a one-dimensional array."
            )

        if array.size == 0:
            raise ValueError(
                "WaveformData.data cannot be empty."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                "WaveformData.data contains NaN or infinite values."
            )

        # Defensive copy + read-only flag.
        immutable_array = np.array(
            array,
            dtype=np.float64,
            copy=True,
        )
        immutable_array.setflags(write=False)

        object.__setattr__(
            self,
            "data",
            immutable_array,
        )

        object.__setattr__(
            self,
            "sampling_rate",
            sampling_rate,
        )

        object.__setattr__(
            self,
            "unit",
            self.unit.strip(),
        )

    # ----------------------------------------------------------------------
    # Convenience properties
    # ----------------------------------------------------------------------

    @property
    def npts(self) -> int:
        """Number of waveform samples."""
        return int(self.data.size)

    @property
    def duration(self) -> float:
        """
        Waveform duration in seconds.

        Defined as::

            T = (N - 1) / fs

        where ``N`` is the number of samples and ``fs`` is the sampling rate.

        This corresponds to the elapsed time between the first and last
        sample rather than the number of sample intervals rounded upward.
        """
        if self.npts <= 1:
            return 0.0

        return float((self.npts - 1) / self.sampling_rate)

    @property
    def dt(self) -> float:
        """Sampling interval in seconds."""
        return float(1.0 / self.sampling_rate)

    @property
    def is_finite(self) -> bool:
        """Return True when all samples are finite."""
        return bool(np.all(np.isfinite(self.data)))

    @property
    def peak(self) -> float:
        """Absolute peak amplitude."""
        return float(np.max(np.abs(self.data)))

    @property
    def mean(self) -> float:
        """Arithmetic mean of the waveform."""
        return float(np.mean(self.data))

    @property
    def rms(self) -> float:
        """Root-mean-square amplitude."""
        return float(
            np.sqrt(np.mean(np.square(self.data)))
        )

    def copy_with(
        self,
        *,
        data: FloatArray | None = None,
        sampling_rate: float | None = None,
        unit: str | None = None,
    ) -> WaveformData:
        """
        Create a new immutable waveform with selected fields replaced.

        The original waveform is never modified.
        """
        return WaveformData(
            data=self.data if data is None else data,
            sampling_rate=(
                self.sampling_rate
                if sampling_rate is None
                else sampling_rate
            ),
            unit=self.unit if unit is None else unit,
        )


# ============================================================================
# ProcessingContext
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProcessingContext:
    """
    Immutable state container for the BSMA processing pipeline.

    Architecture
    ------------
    The context distinguishes between:

    ``raw_waveform``
        Original waveform immediately after ingestion/calibration.
        This is the immutable reference and MUST NOT be overwritten.

    ``acceleration``
        Current working acceleration after preprocessing.

    ``velocity``
        Integrated velocity.

    ``displacement``
        Integrated displacement.

    This separation is fundamental for reproducibility. Every derived
    product can always be traced back to the original waveform.

    Notes
    -----
    Every state transition must return a NEW ``ProcessingContext``.
    """

    # ======================================================================
    # Identity
    # ======================================================================

    trace_id: str

    # ======================================================================
    # Metadata
    # ======================================================================

    metadata: Mapping[str, Any]

    # ======================================================================
    # Immutable raw waveform
    # ======================================================================

    raw_waveform: WaveformData

    # ======================================================================
    # Processing products
    # ======================================================================

    acceleration: WaveformData | None = None
    velocity: WaveformData | None = None
    displacement: WaveformData | None = None

    # ======================================================================
    # Engineering analysis products
    # ======================================================================

    spectral_data: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    metrics: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    # ======================================================================
    # Processing ecosystem
    # ======================================================================

    processing_state: ProcessingState = field(
        default_factory=ProcessingState
    )

    cache: ProcessingCache = field(
        default_factory=ProcessingCache
    )

    # QC report is intentionally loosely coupled to the domain context.
    qc: Any = None

    # ======================================================================
    # Provenance / audit trail
    # ======================================================================

    history: tuple[dict[str, Any], ...] = field(
        default_factory=tuple
    )

    # ======================================================================
    # Configuration
    # ======================================================================

    config: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    # ======================================================================
    # Validation
    # ======================================================================

    def __post_init__(self) -> None:
        """
        Validate and normalize the context.

        The context must always contain a valid raw waveform because
        ``raw_waveform`` is the SSOT reference for the complete processing
        chain.
        """

        if not isinstance(self.trace_id, str):
            raise TypeError(
                "trace_id must be a string."
            )

        if not self.trace_id.strip():
            raise ValueError(
                "trace_id cannot be empty."
            )

        if not isinstance(self.raw_waveform, WaveformData):
            raise TypeError(
                "raw_waveform must be an instance of WaveformData."
            )

        if self.acceleration is not None:
            if not isinstance(self.acceleration, WaveformData):
                raise TypeError(
                    "acceleration must be WaveformData or None."
                )

            self._validate_sampling_compatibility(
                self.raw_waveform,
                self.acceleration,
                "acceleration",
            )

        if self.velocity is not None:
            if not isinstance(self.velocity, WaveformData):
                raise TypeError(
                    "velocity must be WaveformData or None."
                )

        if self.displacement is not None:
            if not isinstance(self.displacement, WaveformData):
                raise TypeError(
                    "displacement must be WaveformData or None."
                )

        # Normalize mutable mappings to immutable mapping proxies.
        object.__setattr__(
            self,
            "metadata",
            self._freeze_mapping(self.metadata),
        )

        object.__setattr__(
            self,
            "spectral_data",
            self._freeze_mapping(self.spectral_data),
        )

        object.__setattr__(
            self,
            "metrics",
            self._freeze_mapping(self.metrics),
        )

        object.__setattr__(
            self,
            "config",
            self._freeze_mapping(self.config),
        )

    # ======================================================================
    # Single Source of Truth / Working Waveform
    # ======================================================================

    @property
    def waveform(self) -> WaveformData:
        """
        Return the current working waveform.

        Priority:

        1. Processed acceleration
        2. Raw waveform

        This property provides compatibility with preprocessing plugins that
        operate on ``context.waveform`` while preserving ``raw_waveform`` as
        the immutable SSOT.

        Returns
        -------
        WaveformData
            Current waveform being processed.
        """
        if self.acceleration is not None:
            return self.acceleration

        return self.raw_waveform

    @property
    def working_waveform(self) -> WaveformData:
        """
        Explicit alias for the current working waveform.

        This is preferable in new code because it makes the processing
        semantics unambiguous.
        """
        return self.waveform

    # ======================================================================
    # Basic waveform properties
    # ======================================================================

    @property
    def sampling_rate(self) -> float:
        """
        Current working waveform sampling rate in Hz.

        The raw and processed waveform should normally have identical
        sampling rates. The working waveform is used here because all
        downstream numerical processing operates on the current signal.
        """
        return self.waveform.sampling_rate

    @property
    def dt(self) -> float:
        """Current sampling interval in seconds."""
        return self.waveform.dt

    @property
    def npts(self) -> int:
        """Number of samples in the current working waveform."""
        return self.waveform.npts

    @property
    def duration(self) -> float:
        """Elapsed duration of the current working waveform in seconds."""
        return self.waveform.duration

    # ======================================================================
    # State inspection
    # ======================================================================

    @property
    def has_acceleration(self) -> bool:
        """Return True if processed acceleration is available."""
        return self.acceleration is not None

    @property
    def has_velocity(self) -> bool:
        """Return True if velocity has been computed."""
        return self.velocity is not None

    @property
    def has_displacement(self) -> bool:
        """Return True if displacement has been computed."""
        return self.displacement is not None

    @property
    def has_kinematic_products(self) -> bool:
        """
        Return True when both velocity and displacement are available.
        """
        return (
            self.velocity is not None
            and self.displacement is not None
        )

    # ======================================================================
    # Immutable state transitions
    # ======================================================================

    def with_state(
        self,
        **kwargs: Any,
    ) -> ProcessingContext:
        """
        Return a new context with selected fields replaced.

        The current context is never modified.

        Examples
        --------
        ::

            new_context = context.with_state(
                acceleration=processed_acceleration,
                processing_state=new_state,
            )
        """
        return replace(self, **kwargs)

    def with_acceleration(
        self,
        acceleration: WaveformData,
        *,
        clear_derived: bool = True,
    ) -> ProcessingContext:
        """
        Replace the working acceleration.

        Parameters
        ----------
        acceleration
            New processed acceleration.

        clear_derived
            If True, velocity, displacement and numerical caches are
            invalidated because they are mathematically derived from the
            acceleration waveform.

        Notes
        -----
        This method is the preferred transition for preprocessing plugins.

        Any operation such as baseline correction, filtering or tapering
        changes the acceleration and therefore invalidates all downstream
        kinematic and spectral products.
        """

        if not isinstance(acceleration, WaveformData):
            raise TypeError(
                "acceleration must be a WaveformData instance."
            )

        self._validate_sampling_compatibility(
            self.raw_waveform,
            acceleration,
            "acceleration",
        )

        if clear_derived:
            cache = ProcessingCache()

            return replace(
                self,
                acceleration=acceleration,
                velocity=None,
                displacement=None,
                spectral_data=MappingProxyType({}),
                metrics=MappingProxyType({}),
                cache=cache,
            )

        return replace(
            self,
            acceleration=acceleration,
        )

    def with_kinematic_products(
        self,
        *,
        velocity: WaveformData,
        displacement: WaveformData,
    ) -> ProcessingContext:
        """
        Return a new context containing velocity and displacement.

        Sampling rate and number of samples must remain compatible with
        the acceleration waveform.
        """

        if not isinstance(velocity, WaveformData):
            raise TypeError(
                "velocity must be a WaveformData instance."
            )

        if not isinstance(displacement, WaveformData):
            raise TypeError(
                "displacement must be a WaveformData instance."
            )

        reference = self.waveform

        self._validate_sampling_compatibility(
            reference,
            velocity,
            "velocity",
        )

        self._validate_sampling_compatibility(
            reference,
            displacement,
            "displacement",
        )

        if velocity.npts != reference.npts:
            raise ValueError(
                "Velocity and acceleration must contain the same "
                f"number of samples. Got {velocity.npts} and "
                f"{reference.npts}."
            )

        if displacement.npts != reference.npts:
            raise ValueError(
                "Displacement and acceleration must contain the same "
                f"number of samples. Got {displacement.npts} and "
                f"{reference.npts}."
            )

        return replace(
            self,
            velocity=velocity,
            displacement=displacement,
        )

    # ======================================================================
    # Audit trail
    # ======================================================================

    def add_history(
        self,
        step_name: str,
        details: Mapping[str, Any] | None = None,
    ) -> ProcessingContext:
        """
        Append a processing event to the immutable audit trail.

        Parameters
        ----------
        step_name
            Name of processing operation.

        details
            Additional serializable metadata.

        Returns
        -------
        ProcessingContext
            New context containing the additional history record.
        """

        if not isinstance(step_name, str) or not step_name.strip():
            raise ValueError(
                "step_name must be a non-empty string."
            )

        record: dict[str, Any] = {
            "step": step_name.strip(),
        }

        if details:
            record.update(dict(details))

        return replace(
            self,
            history=self.history + (record,),
        )

    # ======================================================================
    # Cache handling
    # ======================================================================

    def clear_cache(self) -> ProcessingContext:
        """
        Return a new context with a completely empty processing cache.
        """
        return replace(
            self,
            cache=ProcessingCache(),
        )

    # ======================================================================
    # Metadata helpers
    # ======================================================================

    def with_metadata(
        self,
        **updates: Any,
    ) -> ProcessingContext:
        """
        Return a new context with metadata updates.
        """
        metadata = dict(self.metadata)
        metadata.update(updates)

        return replace(
            self,
            metadata=MappingProxyType(metadata),
        )

    def with_metrics(
        self,
        **updates: Any,
    ) -> ProcessingContext:
        """
        Return a new context with engineering metrics updated.
        """
        metrics = dict(self.metrics)
        metrics.update(updates)

        return replace(
            self,
            metrics=MappingProxyType(metrics),
        )

    def with_spectral_data(
        self,
        **updates: Any,
    ) -> ProcessingContext:
        """
        Return a new context with spectral-analysis products updated.
        """
        spectral_data = dict(self.spectral_data)
        spectral_data.update(updates)

        return replace(
            self,
            spectral_data=MappingProxyType(spectral_data),
        )

    # ======================================================================
    # Internal utilities
    # ======================================================================

    @staticmethod
    def _freeze_mapping(
        value: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        """
        Convert a mapping into a shallow immutable mapping.

        Numerical arrays inside mappings are not recursively copied here.
        Large numerical products should remain in ``ProcessingCache`` rather
        than being embedded in metadata dictionaries.
        """
        if value is None:
            return MappingProxyType({})

        if isinstance(value, MappingProxyType):
            return value

        return MappingProxyType(dict(value))

    @staticmethod
    def _validate_sampling_compatibility(
        reference: WaveformData,
        candidate: WaveformData,
        name: str,
    ) -> None:
        """
        Ensure candidate waveform has compatible sampling characteristics.

        A derived waveform cannot silently use a different sampling rate
        from its source because that would invalidate dt-dependent
        integration and spectral calculations.
        """

        if not np.isclose(
            candidate.sampling_rate,
            reference.sampling_rate,
            rtol=1e-7,
            atol=1e-12,
        ):
            raise ValueError(
                f"{name} sampling rate is incompatible with reference: "
                f"{candidate.sampling_rate} Hz vs "
                f"{reference.sampling_rate} Hz."
            )

    # ======================================================================
    # Representation
    # ======================================================================

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"trace_id={self.trace_id!r}, "
            f"npts={self.npts}, "
            f"sampling_rate={self.sampling_rate:.6g} Hz, "
            f"acceleration={self.has_acceleration}, "
            f"velocity={self.has_velocity}, "
            f"displacement={self.has_displacement})"
        )