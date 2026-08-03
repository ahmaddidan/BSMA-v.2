"""
BMKG Strong Motion Analyzer (BSMA)

Plugin Interface
================

Defines the abstract contract for every preprocessing plugin used by
the BSMA processing pipeline.

Every processing algorithm (baseline correction, filtering,
tapering, integration, parameter extraction, response spectrum,
etc.) MUST inherit from PreprocessorPlugin.

Design Principles
-----------------
- Single Responsibility Principle
- Open/Closed Principle
- Immutable ProcessingContext
- Production-grade type hints
- Independent plugin execution
- Compatible with PipelineOrchestrator
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from core.types.context import ProcessingContext
from utils.exceptions import ProcessingError

__all__ = [
    "PreprocessorPlugin",
]


class PreprocessorPlugin(ABC):
    """
    Base class for every preprocessing plugin.

    A plugin receives a ProcessingContext and MUST return a new
    ProcessingContext.

    Plugins must never modify the input context in-place.
    """

    # ------------------------------------------------------------
    # Plugin Metadata
    # ------------------------------------------------------------

    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """Human-readable plugin name."""
        raise NotImplementedError

    @property
    def plugin_version(self) -> str:
        """Plugin semantic version."""
        return "1.0.0"

    @property
    def plugin_description(self) -> str:
        """Human-readable description."""
        return ""

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    def validate_input(self, context: ProcessingContext) -> None:
        """Validate input context before processing safely."""
        # Ambil data gelombang secara aman dari berbagai alternatif atribut
        wf = getattr(context, "waveform", None)
        if wf is None:
            wf = getattr(context, "raw_waveform", None)
        if wf is None:
            wf = getattr(context, "acceleration", None)

        if wf is None:
            raise ProcessingError("ProcessingContext tidak memiliki data gelombang yang valid.")

        # Ekstrak array numpy (mendukung objek WaveformData maupun numpy array langsung)
        data_arr = wf.data if hasattr(wf, "data") else wf

        if data_arr is None:
            raise ProcessingError("Array data gelombang bernilai None.")

        if hasattr(data_arr, "size") and data_arr.size == 0:
            raise ProcessingError("Array data gelombang kosong (size == 0).")
        elif not hasattr(data_arr, "size") and len(data_arr) == 0:
            raise ProcessingError("Array data gelombang kosong (len == 0).")

        # Validasi sampling rate jika tersedia di konteks
        sr = getattr(context, "sampling_rate", None)
        if sr is None and hasattr(wf, "sampling_rate"):
            sr = wf.sampling_rate

        if sr is not None and sr <= 0:
            raise ValueError("Sampling rate must be positive.")

    # ------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------

    @abstractmethod
    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """Execute the processing algorithm."""
        raise NotImplementedError

    # ------------------------------------------------------------
    # Optional lifecycle hooks
    # ------------------------------------------------------------

    def initialize(self) -> None:
        return None

    def finalize(self) -> None:
        return None

    # ------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------

    def supports_parallel(self) -> bool:
        return True

    def reset(self) -> None:
        return None

    # ------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.plugin_name}', "
            f"version='{self.plugin_version}')"
        )