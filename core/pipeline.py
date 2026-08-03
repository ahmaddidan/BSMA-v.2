"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/pipeline.py

Description: Central execution engine of the BSMA preprocessing pipeline.
Features fluent builder pattern, progress callbacks, and fail-fast execution.
"""
from __future__ import annotations
import logging
from dataclasses import replace
from typing import Callable, Iterable, Iterator, Sequence

from core.orchestrator import ProcessingStep
from core.types.context import ProcessingContext
from utils.exceptions import ProcessingError, SeverityLevel
from utils.logger import setup_logger

__all__ = [
    "PipelineOrchestrator",
    "PipelineBuilder",
]

ProgressCallback = Callable[[int, int, str], None]


class PipelineOrchestrator:
    """
    Executes preprocessing plugins sequentially.
    Every plugin receives a ProcessingContext and returns a NEW ProcessingContext.
    """

    def __init__(
        self,
        plugins: Sequence[ProcessingStep],
        logger: logging.Logger | None = None,
        halt_on_error: bool = True
    ) -> None:
        self._plugins = tuple(plugins)
        self.logger = logger or setup_logger(__name__)
        self.halt_on_error = halt_on_error
        self._validate_plugins()

    @property
    def plugins(self) -> tuple[ProcessingStep, ...]:
        return self._plugins

    @property
    def size(self) -> int:
        return len(self._plugins)

    def __len__(self) -> int:
        return len(self._plugins)

    def __iter__(self) -> Iterator[ProcessingStep]:
        return iter(self._plugins)

    def _validate_plugins(self) -> None:
        """
        Validate plugin list using robust attribute checking.
        """
        names: set[str] = set()

        for plugin in self._plugins:
            has_process = hasattr(plugin, "process") and callable(getattr(plugin, "process"))
            has_name = hasattr(plugin, "name") or hasattr(plugin, "plugin_name")

            if not (has_process and has_name):
                raise TypeError(
                    f"{plugin!r} bukan merupakan ProcessingStep yang valid (harus memiliki metode 'process' dan atribut nama)."
                )

            if hasattr(plugin, "name"):
                plugin_name = plugin.name
            else:
                plugin_name = plugin.plugin_name

            if plugin_name in names:
                raise ValueError(
                    f"Duplicate plugin name: {plugin_name}"
                )

            names.add(plugin_name)
            
    def run(
        self,
        context: ProcessingContext,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> ProcessingContext:
        """Execute all plugins sequentially."""
        current = context
        total = len(self._plugins)

        for index, plugin in enumerate(self._plugins, start=1):
            # Ambil nama plugin secara aman (mendukung 'name' atau 'plugin_name')
            p_name = getattr(plugin, "name", getattr(plugin, "plugin_name", "UnknownPlugin"))

            if progress_callback is not None:
                progress_callback(index, total, p_name)

            try:
                self.logger.debug(f"Executing plugin: {p_name}")
                new_context = plugin.process(current)
                
                if not isinstance(new_context, ProcessingContext):
                    raise TypeError(f"{p_name}.process() must return ProcessingContext.")
                
                current = new_context
                
            except Exception as exc:
                self.logger.error(
                    f"Pipeline execution failed at {p_name}.",
                    exc_info=True,
                    extra={"bsma_context": {"failed_plugin": p_name}}
                )
                
                if self.halt_on_error:
                    raise
                else:
                    self.logger.warning("halt_on_error=False, stopping pipeline execution and returning last valid context.")
                    break

        return current


class PipelineBuilder:
    """
    Fluent builder for preprocessing pipelines.
    """

    def __init__(self, logger: logging.Logger | None = None, halt_on_error: bool = True) -> None:
        self._plugins: list[ProcessingStep] = []
        self._logger = logger
        self._halt_on_error = halt_on_error

    def add(self, plugin: ProcessingStep) -> "PipelineBuilder":
        self._plugins.append(plugin)
        return self

    def clear(self) -> "PipelineBuilder":
        self._plugins.clear()
        return self

    def build(self) -> PipelineOrchestrator:
        return PipelineOrchestrator(
            plugins=tuple(self._plugins),
            logger=self._logger,
            halt_on_error=self._halt_on_error
        )