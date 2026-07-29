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

__all__ = [
    "PreprocessorPlugin",
]


class PreprocessorPlugin(ABC):
    """
    Base class for every preprocessing plugin.

    A plugin receives a ProcessingContext and MUST return a new
    ProcessingContext.

    Plugins must never modify the input context in-place.

    Examples
    --------

    >>> plugin = BaselineCorrectionPlugin()
    >>> new_context = plugin.process(context)

    """

    # ------------------------------------------------------------
    # Plugin Metadata
    # ------------------------------------------------------------

    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """
        Human-readable plugin name.

        Examples
        --------
        BaselineCorrectionPlugin
        ButterworthFilterPlugin
        IntegrationPlugin
        """
        raise NotImplementedError

    @property
    def plugin_version(self) -> str:
        """
        Plugin semantic version.

        Override if necessary.
        """
        return "1.0.0"

    @property
    def plugin_description(self) -> str:
        """
        Human-readable description.

        Override in derived plugins.
        """
        return ""

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    def validate_input(
        self,
        context: ProcessingContext,
    ) -> None:
        """
        Validate ProcessingContext before execution.

        Default implementation performs only minimal validation.

        Derived plugins are encouraged to extend this method.
        """

        if context.waveform is None:
            raise ValueError(
                "ProcessingContext.waveform is None."
            )

        if context.waveform.size == 0:
            raise ValueError(
                "Waveform contains no samples."
            )

        if context.sampling_rate <= 0:
            raise ValueError(
                "Sampling rate must be positive."
            )
    # ------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------

    @abstractmethod
    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """
        Execute the processing algorithm.

        Parameters
        ----------
        context
            Current immutable processing context.

        Returns
        -------
        ProcessingContext
            A NEW ProcessingContext instance.

        Notes
        -----
        Implementations MUST NOT modify the input context in-place.
        """
        raise NotImplementedError

    # ------------------------------------------------------------
    # Optional lifecycle hooks
    # ------------------------------------------------------------

    def initialize(self) -> None:
        """
        Optional initialization hook.

        Override if the plugin needs to allocate resources,
        initialize lookup tables, or prepare reusable objects.
        """
        return None

    def finalize(self) -> None:
        """
        Optional cleanup hook.

        Override if the plugin holds external resources that
        should be released after processing.
        """
        return None

    # ------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------

    def supports_parallel(self) -> bool:
        """
        Indicates whether the plugin is safe to execute in a
        parallel processing environment.

        Default
        -------
        True
        """
        return True

    def reset(self) -> None:
        """
        Reset internal plugin state.

        Stateless plugins do not need to override this method.
        """
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