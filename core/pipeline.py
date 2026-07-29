"""
BMKG Strong Motion Analyzer (BSMA)

Pipeline Orchestrator
=====================

Central execution engine of the BSMA preprocessing pipeline.

Responsibilities
----------------
- Execute preprocessing plugins sequentially
- Preserve immutable ProcessingContext
- Maintain processing history
- Perform fail-fast execution
- Validate plugins
- Integrate Quality Control
- Support progress callbacks
- Provide backward compatibility

Architecture
------------
ProcessingContext
        │
        ▼
PipelineOrchestrator
        │
        ▼
Plugin 1
        │
        ▼
Plugin 2
        │
        ▼
Plugin N
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import Sequence

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext

__all__ = [
    "PipelineOrchestrator",
    "PreprocessingPipeline",
]


ProgressCallback = Callable[
    [int, int, str],
    None,
]


class PipelineOrchestrator:
    """
    Executes preprocessing plugins sequentially.

    Notes
    -----
    Pipeline is intentionally stateless.

    Every plugin receives a ProcessingContext and
    returns a NEW ProcessingContext.

    No plugin may mutate the previous context.
    """

    def __init__(
        self,
        plugins: Sequence[PreprocessorPlugin],
    ) -> None:

        self._plugins = tuple(plugins)

        self._validate_plugins()

    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    def plugins(
        self,
    ) -> tuple[PreprocessorPlugin, ...]:
        """
        Registered plugins.
        """
        return self._plugins

    @property
    def size(self) -> int:
        """
        Number of plugins.
        """
        return len(self._plugins)

    def __len__(self) -> int:
        return len(self._plugins)

    def __iter__(
        self,
    ) -> Iterator[PreprocessorPlugin]:
        return iter(self._plugins)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _validate_plugins(
        self,
    ) -> None:
        """
        Validate plugin list.
        """

        names: set[str] = set()

        for plugin in self._plugins:

            if not isinstance(
                plugin,
                PreprocessorPlugin,
            ):
                raise TypeError(
                    f"{plugin!r} is not "
                    "a PreprocessorPlugin."
                )

            if plugin.plugin_name in names:

                raise ValueError(
                    f"Duplicate plugin name: "
                    f"{plugin.plugin_name}"
                )

            names.add(plugin.plugin_name)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def run(
        self,
        context: ProcessingContext,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> ProcessingContext:
        """
        Execute all plugins.

        Parameters
        ----------
        context
            Initial ProcessingContext.

        progress_callback
            Optional callback.

        Returns
        -------
        ProcessingContext
        """

        current = context

        total = len(self._plugins)

        for index, plugin in enumerate(
            self._plugins,
            start=1,
        ):

            if progress_callback is not None:

                progress_callback(
                    index,
                    total,
                    plugin.plugin_name,
                )

            current = self._run_plugin(
                plugin,
                current,
            )

            if current.qc.has_fatal_error:

                break

        return current
    # ---------------------------------------------------------
    # Internal Execution
    # ---------------------------------------------------------

    def _run_plugin(
        self,
        plugin: PreprocessorPlugin,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """
        Execute a single preprocessing plugin.

        This method guarantees:

        - initialize() is called
        - validate_input() is executed
        - process() returns ProcessingContext
        - finalize() is always called
        - exceptions are converted into QC failures
        """

        try:
            plugin.initialize()

            plugin.validate_input(context)

            new_context = plugin.process(context)

            if not isinstance(
                new_context,
                ProcessingContext,
            ):
                raise TypeError(
                    f"{plugin.plugin_name}.process() "
                    "must return ProcessingContext."
                )

            plugin.finalize()

            return new_context

        except Exception as exc:

            plugin.finalize()

            #
            # Fail-safe:
            # Any unexpected exception is recorded inside
            # the QC report so the pipeline stops gracefully.
            #

            from core.types.qc import (
                QCResult,
                QCSeverity,
                QCStatus,
            )

            report = context.qc

            report.add_result(
                QCResult(
                    status=QCStatus.FAIL,
                    severity=QCSeverity.CRITICAL,
                    validator_name=plugin.plugin_name,
                    message=str(exc),
                )
            )

            return context.copy(qc=report)

    # ---------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------

    def append(
        self,
        plugin: PreprocessorPlugin,
    ) -> "PipelineOrchestrator":
        """
        Return NEW pipeline with one additional plugin.
        """

        return PipelineOrchestrator(
            self._plugins + (plugin,)
        )

    def extend(
        self,
        plugins: Iterable[
            PreprocessorPlugin
        ],
    ) -> "PipelineOrchestrator":
        """
        Return NEW pipeline with additional plugins.
        """

        return PipelineOrchestrator(
            self._plugins + tuple(plugins)
        )

    def names(
        self,
    ) -> tuple[str, ...]:
        """
        Plugin names.
        """

        return tuple(
            p.plugin_name
            for p in self._plugins
        )

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(
        self,
    ) -> str:

        return (
            f"{self.__class__.__name__}"
            f"(plugins={len(self)})"
        )


# ==========================================================
# Pipeline Builder
# ==========================================================


class PipelineBuilder:
    """
    Fluent builder for preprocessing pipelines.

    Example
    -------

    pipeline = (
        PipelineBuilder()
            .add(BaselinePlugin())
            .add(FilterPlugin())
            .add(TaperPlugin())
            .build()
    )
    """

    def __init__(self) -> None:

        self._plugins: list[
            PreprocessorPlugin
        ] = []

    def add(
        self,
        plugin: PreprocessorPlugin,
    ) -> "PipelineBuilder":

        self._plugins.append(plugin)

        return self

    def clear(
        self,
    ) -> "PipelineBuilder":

        self._plugins.clear()

        return self

    def build(
        self,
    ) -> PipelineOrchestrator:

        return PipelineOrchestrator(
            tuple(self._plugins)
        )


# ==========================================================
# Backward Compatibility
# ==========================================================

#
# Existing code:
#
# from core.pipeline import PreprocessingPipeline
#
# continues to work.
#

PreprocessingPipeline = PipelineOrchestrator