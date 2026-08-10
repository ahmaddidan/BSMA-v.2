"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/pipeline.py

Description
-----------
Central execution engine and fluent builder for the BSMA processing pipeline.

Architectural principles
------------------------
- Single Source of Truth (SSOT) through ProcessingContext.
- Functional state transitions: every plugin must return a new ProcessingContext.
- Deterministic sequential execution.
- Fail-fast execution by default.
- Optional non-fail-fast execution.
- Progress callback support.
- Compatible with both ProcessingStep.name and
  PreprocessorPlugin.plugin_name interfaces.
- No mutation of ProcessingContext by the orchestrator.
- Production-grade logging and validation.

NOTE
----
`core/orchestrator.py` is the canonical implementation of the pipeline
execution engine. This module provides the higher-level fluent builder and
a compatibility facade around that engine.

The architecture intentionally avoids maintaining two independent pipeline
execution implementations.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterator, Sequence, TypeAlias

from core.orchestrator import (
    PipelineOrchestrator as _CorePipelineOrchestrator,
    ProcessingStep,
)
from core.types.context import ProcessingContext
from utils.logger import setup_logger


__all__ = [
    "PipelineOrchestrator",
    "PipelineBuilder",
    "ProgressCallback",
]


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

ProgressCallback: TypeAlias = Callable[[int, int, str], None]


# ============================================================================
# PIPELINE ORCHESTRATOR
# ============================================================================


class PipelineOrchestrator(_CorePipelineOrchestrator):
    """
    High-level BSMA pipeline orchestrator.

    This class delegates actual execution to the canonical orchestrator
    defined in :mod:`core.orchestrator`.

    Parameters
    ----------
    plugins:
        Ordered collection of processing plugins.

    logger:
        Optional Python logger. If omitted, the BSMA logger is created.

    halt_on_error:
        If ``True``, the first processing failure raises immediately.
        If ``False``, execution stops and the last valid context is returned.

    Notes
    -----
    Plugins must expose:

    - ``process(context) -> ProcessingContext``
    - either ``name`` or ``plugin_name``

    The returned object MUST be a ``ProcessingContext``.
    """

    def __init__(
        self,
        plugins: Sequence[ProcessingStep],
        logger: logging.Logger | None = None,
        halt_on_error: bool = True,
    ) -> None:
        super().__init__(
            steps=tuple(plugins),
            logger=logger or setup_logger(__name__),
            halt_on_error=halt_on_error,
        )

    @property
    def plugins(self) -> tuple[ProcessingStep, ...]:
        """
        Return the immutable plugin sequence.
        """
        return self.steps

    @property
    def size(self) -> int:
        """
        Return the number of registered processing plugins.
        """
        return len(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[ProcessingStep]:
        return iter(self.steps)

    @staticmethod
    def _plugin_name(plugin: ProcessingStep) -> str:
        """
        Resolve the canonical human-readable plugin name.

        Supports both interfaces used in BSMA:

        - ``ProcessingStep.name``
        - ``PreprocessorPlugin.plugin_name``
        """
        name = getattr(plugin, "name", None)

        if isinstance(name, str) and name.strip():
            return name

        plugin_name = getattr(plugin, "plugin_name", None)

        if isinstance(plugin_name, str) and plugin_name.strip():
            return plugin_name

        raise TypeError(
            f"Invalid processing plugin {plugin!r}: "
            "plugin must expose a non-empty 'name' or 'plugin_name'."
        )

    @classmethod
    def validate_plugins(
        cls,
        plugins: Sequence[ProcessingStep],
    ) -> None:
        """
        Validate plugin architecture before execution.

        Raises
        ------
        TypeError
            If a plugin does not expose a callable ``process`` method or
            a valid plugin name.

        ValueError
            If duplicate plugin names are detected.
        """
        names: set[str] = set()

        for plugin in plugins:
            process = getattr(plugin, "process", None)

            if not callable(process):
                raise TypeError(
                    f"Invalid processing plugin {plugin!r}: "
                    "missing callable 'process(context)' method."
                )

            name = cls._plugin_name(plugin)

            if name in names:
                raise ValueError(
                    f"Duplicate plugin name detected: {name!r}"
                )

            names.add(name)

    def run(
        self,
        context: ProcessingContext,
        identifier: str = "unknown",
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> ProcessingContext:
        """
        Execute the complete pipeline.

        Parameters
        ----------
        context:
            Initial immutable ``ProcessingContext``.

        identifier:
            Identifier used for logging/tracing.

        progress_callback:
            Optional callback receiving:

            ``(current_index, total_plugins, plugin_name)``

        Returns
        -------
        ProcessingContext
            Final valid processing context.

        Notes
        -----
        Execution remains sequential and deterministic. The callback is
        observational only and is never allowed to modify the context.
        """
        if not isinstance(context, ProcessingContext):
            raise TypeError(
                "context must be an instance of ProcessingContext."
            )

        total = len(self.steps)

        if total == 0:
            self.logger.warning(
                "Pipeline executed with zero processing plugins.",
                extra={
                    "bsma_context": {
                        "identifier": identifier,
                        "total_steps": 0,
                    }
                },
            )
            return context

        current = context

        for index, plugin in enumerate(self.steps, start=1):
            plugin_name = self._plugin_name(plugin)

            if progress_callback is not None:
                progress_callback(
                    index,
                    total,
                    plugin_name,
                )

            self.logger.debug(
                "Executing pipeline plugin %d/%d: %s",
                index,
                total,
                plugin_name,
                extra={
                    "bsma_context": {
                        "identifier": identifier,
                        "plugin": plugin_name,
                        "plugin_index": index,
                        "total_plugins": total,
                    }
                },
            )

            # The canonical implementation in core.orchestrator performs
            # the actual processing and architectural return-value check.
            #
            # The compatibility wrapper deliberately does not mutate the
            # ProcessingContext itself.
            current = self._run_single_plugin(
                plugin=plugin,
                context=current,
                identifier=identifier,
                index=index,
                total=total,
            )

        self.logger.info(
            "BSMA pipeline completed successfully.",
            extra={
                "bsma_context": {
                    "identifier": identifier,
                    "total_plugins": total,
                }
            },
        )

        return current

    def _run_single_plugin(
        self,
        plugin: ProcessingStep,
        context: ProcessingContext,
        identifier: str,
        index: int,
        total: int,
    ) -> ProcessingContext:
        """
        Execute one plugin and preserve pipeline-level error semantics.

        This method exists so the high-level facade can provide the same
        ``plugins`` terminology as the rest of the BSMA architecture while
        preserving the immutable ProcessingContext contract.
        """
        plugin_name = self._plugin_name(plugin)

        try:
            new_context = plugin.process(context)

            if not isinstance(new_context, ProcessingContext):
                raise TypeError(
                    f"Plugin '{plugin_name}' returned "
                    f"{type(new_context).__name__}; "
                    "expected ProcessingContext."
                )

            if new_context is context:
                self.logger.warning(
                    "Plugin '%s' returned the same ProcessingContext "
                    "instance. Plugins should perform immutable state "
                    "transitions and return a new context.",
                    plugin_name,
                    extra={
                        "bsma_context": {
                            "identifier": identifier,
                            "plugin": plugin_name,
                            "plugin_index": index,
                            "total_plugins": total,
                        }
                    },
                )

            return new_context

        except Exception as exc:
            self.logger.error(
                "Pipeline execution failed at plugin '%s'.",
                plugin_name,
                exc_info=True,
                extra={
                    "bsma_context": {
                        "identifier": identifier,
                        "failed_plugin": plugin_name,
                        "plugin_index": index,
                        "total_plugins": total,
                    }
                },
            )

            if self.halt_on_error:
                raise

            self.logger.warning(
                "halt_on_error=False: stopping remaining pipeline "
                "execution and returning the last valid context.",
                extra={
                    "bsma_context": {
                        "identifier": identifier,
                        "failed_plugin": plugin_name,
                    }
                },
            )

            return context


# ============================================================================
# PIPELINE BUILDER
# ============================================================================


class PipelineBuilder:
    """
    Fluent builder for constructing a BSMA processing pipeline.

    Example
    -------
    >>> pipeline = (
    ...     PipelineBuilder()
    ...     .add(BaselineCorrectionPlugin())
    ...     .add(TaperPlugin())
    ...     .add(FilterPlugin())
    ...     .add(KinematicIntegrationPlugin())
    ...     .add(ParameterExtractionPlugin())
    ...     .add(ResponseSpectrumPlugin())
    ...     .build()
    ... )
    """

    __slots__ = (
        "_plugins",
        "_logger",
        "_halt_on_error",
    )

    def __init__(
        self,
        logger: logging.Logger | None = None,
        halt_on_error: bool = True,
    ) -> None:
        self._plugins: list[ProcessingStep] = []
        self._logger = logger
        self._halt_on_error = halt_on_error

    def add(self, plugin: ProcessingStep) -> PipelineBuilder:
        """
        Add one processing plugin.

        Validation of the plugin interface is performed immediately so
        architectural errors are detected during pipeline construction
        rather than during waveform processing.
        """
        if plugin is None:
            raise TypeError("plugin cannot be None.")

        process = getattr(plugin, "process", None)

        if not callable(process):
            raise TypeError(
                f"Invalid plugin {plugin!r}: "
                "plugin must provide callable process(context)."
            )

        PipelineOrchestrator._plugin_name(plugin)

        self._plugins.append(plugin)
        return self

    def add_many(
        self,
        plugins: Sequence[ProcessingStep],
    ) -> PipelineBuilder:
        """
        Add multiple processing plugins while preserving their order.
        """
        for plugin in plugins:
            self.add(plugin)

        return self

    def clear(self) -> PipelineBuilder:
        """
        Remove all currently registered plugins.
        """
        self._plugins.clear()
        return self

    def build(self) -> PipelineOrchestrator:
        """
        Validate and construct an immutable pipeline.
        """
        plugins = tuple(self._plugins)

        PipelineOrchestrator.validate_plugins(plugins)

        return PipelineOrchestrator(
            plugins=plugins,
            logger=self._logger,
            halt_on_error=self._halt_on_error,
        )

    @property
    def plugins(self) -> tuple[ProcessingStep, ...]:
        """
        Return currently registered plugins as an immutable tuple.
        """
        return tuple(self._plugins)

    @property
    def size(self) -> int:
        """
        Return the number of registered plugins.
        """
        return len(self._plugins)

    def __len__(self) -> int:
        return len(self._plugins)

    def __iter__(self) -> Iterator[ProcessingStep]:
        return iter(self._plugins)