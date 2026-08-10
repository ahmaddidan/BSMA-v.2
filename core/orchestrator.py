"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/orchestrator.py

Description
-----------
Master Pipeline Orchestrator for seismic waveform processing.

The orchestrator is intentionally domain-generic. It does not contain
any assumptions about:

- network
- station
- location
- channel
- sensor manufacturer
- MiniSEED source
- XML metadata source
- earthquake/event identity
- specific BMKG station

All waveform and metadata information must already be represented by
the ProcessingContext created by the I/O/domain layer.

Architectural principles
------------------------
- Single Source of Truth (SSOT)
- Immutable ProcessingContext transitions
- Functional pipeline composition
- Deterministic execution order
- Fail-fast support
- Optional non-fatal execution mode
- No scientific computation inside the orchestrator
- No waveform mutation inside the orchestrator
- Compatible with all processing plugins
"""

from __future__ import annotations

import logging
import time
from typing import Protocol, Sequence, runtime_checkable

from core.types.context import ProcessingContext


__all__ = [
    "ProcessingStep",
    "PipelineOrchestrator",
]


# ============================================================================
# PROCESSING STEP CONTRACT
# ============================================================================


@runtime_checkable
class ProcessingStep(Protocol):
    """
    Structural interface for every BSMA processing step.

    A processing step receives a ProcessingContext and MUST return
    a new ProcessingContext.

    The processing step itself may implement either:

        name

    or, for compatibility with the lower-level PreprocessorPlugin
    interface:

        plugin_name

    The orchestrator resolves the public name through ``step_name``.
    """

    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """
        Execute the processing operation.

        Parameters
        ----------
        context
            Current immutable processing context.

        Returns
        -------
        ProcessingContext
            New processing context after the operation.

        Notes
        -----
        Implementations MUST NOT mutate the supplied context in-place.
        """
        ...


# ============================================================================
# PIPELINE ORCHESTRATOR
# ============================================================================


class PipelineOrchestrator:
    """
    Production-grade deterministic seismic processing pipeline.

    The orchestrator performs only pipeline coordination.

    Scientific algorithms belong to processing plugins, for example:

    - baseline correction
    - detrending
    - tapering
    - filtering
    - integration
    - strong-motion parameter extraction
    - Fourier amplitude spectrum
    - Husid curve
    - significant duration
    - response spectrum
    - engineering metrics

    The orchestrator itself is intentionally agnostic to the waveform
    source and seismic station identity.

    Parameters
    ----------
    steps
        Ordered processing steps.

    logger
        Optional logger instance.

    halt_on_error
        If True, processing stops immediately when a step fails.

        If False, the failed step is recorded and the pipeline returns
        the last valid ProcessingContext.
    """

    def __init__(
        self,
        steps: Sequence[ProcessingStep],
        logger: logging.Logger | None = None,
        halt_on_error: bool = True,
    ) -> None:

        if steps is None:
            raise ValueError(
                "steps must not be None."
            )

        self.steps: tuple[ProcessingStep, ...] = tuple(steps)

        self.logger = (
            logger
            if logger is not None
            else logging.getLogger(
                "BSMA.PipelineOrchestrator"
            )
        )

        self.halt_on_error = bool(halt_on_error)

        self._validate_steps()

        self.logger.info(
            "PipelineOrchestrator initialized.",
            extra={
                "bsma_context": {
                    "total_steps": len(self.steps),
                    "halt_on_error": self.halt_on_error,
                }
            },
        )

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def _validate_steps(self) -> None:
        """
        Validate pipeline step contracts before execution.

        Scientific computation is not performed here.
        """

        for index, step in enumerate(self.steps):

            if step is None:
                raise TypeError(
                    f"Pipeline step at index {index} is None."
                )

            process_method = getattr(
                step,
                "process",
                None,
            )

            if not callable(process_method):
                raise TypeError(
                    "Invalid processing step at index "
                    f"{index}: object of type "
                    f"{type(step).__name__!r} does not implement "
                    "a callable process(context) method."
                )

    # ========================================================================
    # STEP NAME RESOLUTION
    # ========================================================================

    @staticmethod
    def _get_step_name(
        step: ProcessingStep,
        index: int,
    ) -> str:
        """
        Resolve a stable human-readable processing-step name.

        Supported plugin conventions
        -----------------------------
        1. ``name``
        2. ``plugin_name``

        This allows the orchestrator to work with both the generic
        ProcessingStep contract and the PreprocessorPlugin hierarchy
        without duplicating processing logic.
        """

        name = getattr(
            step,
            "name",
            None,
        )

        if callable(name):
            name = name()

        if isinstance(name, str) and name.strip():
            return name.strip()

        plugin_name = getattr(
            step,
            "plugin_name",
            None,
        )

        if callable(plugin_name):
            plugin_name = plugin_name()

        if (
            isinstance(plugin_name, str)
            and plugin_name.strip()
        ):
            return plugin_name.strip()

        return (
            f"{step.__class__.__name__}"
            f"[{index}]"
        )

    # ========================================================================
    # SINGLE STEP EXECUTION
    # ========================================================================

    def _execute_step(
        self,
        step: ProcessingStep,
        context: ProcessingContext,
        step_index: int,
        identifier: str,
    ) -> ProcessingContext:
        """
        Execute one processing step and validate its transition.

        This method does not modify ``context``.
        """

        step_name = self._get_step_name(
            step,
            step_index,
        )

        start_time = time.perf_counter()

        self.logger.debug(
            "Starting processing step.",
            extra={
                "bsma_context": {
                    "identifier": identifier,
                    "step_index": step_index,
                    "step_name": step_name,
                }
            },
        )

        try:
            next_context = step.process(
                context
            )

        except Exception:
            elapsed_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

            self.logger.exception(
                "Processing step failed.",
                extra={
                    "bsma_context": {
                        "identifier": identifier,
                        "step_index": step_index,
                        "step_name": step_name,
                        "execution_time_ms": round(
                            elapsed_ms,
                            3,
                        ),
                    }
                },
            )

            raise

        # --------------------------------------------------------------------
        # Architectural contract validation
        # --------------------------------------------------------------------

        if next_context is None:
            raise RuntimeError(
                "Architectural contract violation: "
                f"processing step '{step_name}' returned None. "
                "Every processing step must return a "
                "ProcessingContext."
            )

        if not isinstance(
            next_context,
            ProcessingContext,
        ):
            raise TypeError(
                "Architectural contract violation: "
                f"processing step '{step_name}' returned "
                f"{type(next_context).__name__}, expected "
                "ProcessingContext."
            )

        # --------------------------------------------------------------------
        # Prevent accidental in-place replacement
        # --------------------------------------------------------------------

        if next_context is context:
            raise RuntimeError(
                "Architectural contract violation: "
                f"processing step '{step_name}' returned "
                "the exact same ProcessingContext instance. "
                "Processing steps must create an immutable "
                "state transition."
            )

        elapsed_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        # --------------------------------------------------------------------
        # Audit trail
        # --------------------------------------------------------------------

        next_context = next_context.add_history(
            step_name=step_name,
            details={
                "status": "SUCCESS",
                "step_index": step_index,
                "execution_time_ms": round(
                    elapsed_ms,
                    3,
                ),
            },
        )

        self.logger.debug(
            "Processing step completed.",
            extra={
                "bsma_context": {
                    "identifier": identifier,
                    "step_index": step_index,
                    "step_name": step_name,
                    "execution_time_ms": round(
                        elapsed_ms,
                        3,
                    ),
                }
            },
        )

        return next_context

    # ========================================================================
    # PIPELINE EXECUTION
    # ========================================================================

    def run(
        self,
        initial_context: ProcessingContext,
        identifier: str = "unknown",
    ) -> ProcessingContext:
        """
        Execute the complete processing pipeline.

        Parameters
        ----------
        initial_context
            Initial ProcessingContext produced by the reader/domain layer.

        identifier
            Optional execution identifier used only for logging and tracing.

        Returns
        -------
        ProcessingContext
            Last valid context produced by the pipeline.

        Raises
        ------
        TypeError
            If ``initial_context`` is not a ProcessingContext.

        RuntimeError
            If a processing step violates the pipeline contract or
            ``halt_on_error=True`` and a processing step fails.

        Notes
        -----
        The orchestrator never modifies waveform arrays directly.

        State transitions are delegated entirely to processing plugins.
        """

        if not isinstance(
            initial_context,
            ProcessingContext,
        ):
            raise TypeError(
                "initial_context must be a ProcessingContext."
            )

        # --------------------------------------------------------------------
        # Empty pipeline
        # --------------------------------------------------------------------

        if not self.steps:
            self.logger.warning(
                "Pipeline contains no processing steps.",
                extra={
                    "bsma_context": {
                        "identifier": identifier,
                    }
                },
            )

            return initial_context

        current_context = initial_context

        total_start = time.perf_counter()

        completed_steps = 0
        failed_step: str | None = None

        self.logger.info(
            "Starting seismic processing pipeline.",
            extra={
                "bsma_context": {
                    "identifier": identifier,
                    "total_steps": len(self.steps),
                    "halt_on_error": self.halt_on_error,
                }
            },
        )

        # ====================================================================
        # SEQUENTIAL EXECUTION
        # ====================================================================

        for step_index, step in enumerate(
            self.steps,
            start=1,
        ):
            step_name = self._get_step_name(
                step,
                step_index,
            )

            try:
                current_context = self._execute_step(
                    step=step,
                    context=current_context,
                    step_index=step_index,
                    identifier=identifier,
                )

                completed_steps += 1

            except Exception as exc:

                failed_step = step_name

                elapsed_ms = (
                    time.perf_counter()
                    - total_start
                ) * 1000.0

                self.logger.error(
                    "Pipeline processing step failed.",
                    extra={
                        "bsma_context": {
                            "identifier": identifier,
                            "failed_step": step_name,
                            "step_index": step_index,
                            "completed_steps": completed_steps,
                            "total_steps": len(self.steps),
                            "execution_time_ms": round(
                                elapsed_ms,
                                3,
                            ),
                        }
                    },
                )

                # ------------------------------------------------------------
                # FAIL-FAST MODE
                # ------------------------------------------------------------

                if self.halt_on_error:
                    raise RuntimeError(
                        "Pipeline execution failed at "
                        f"step '{step_name}'."
                    ) from exc

                # ------------------------------------------------------------
                # NON-FATAL MODE
                #
                # Important:
                # The failed step is NOT counted as completed.
                # The context returned is the last valid context.
                # ------------------------------------------------------------

                self.logger.warning(
                    "Pipeline stopped after failed step because "
                    "halt_on_error=False.",
                    extra={
                        "bsma_context": {
                            "identifier": identifier,
                            "failed_step": step_name,
                            "completed_steps": completed_steps,
                            "total_steps": len(self.steps),
                        }
                    },
                )

                break

        # ====================================================================
        # FINAL PIPELINE METRICS
        # ====================================================================

        total_elapsed_ms = (
            time.perf_counter()
            - total_start
        ) * 1000.0

        pipeline_status = (
            "FAILED"
            if failed_step is not None
            else "SUCCESS"
        )

        # --------------------------------------------------------------------
        # Add pipeline-level audit record.
        # --------------------------------------------------------------------

        current_context = current_context.add_history(
            step_name="PipelineOrchestrator",
            details={
                "status": pipeline_status,
                "identifier": identifier,
                "completed_steps": completed_steps,
                "total_steps": len(self.steps),
                "execution_time_ms": round(
                    total_elapsed_ms,
                    3,
                ),
                "failed_step": failed_step,
            },
        )

        self.logger.info(
            "Seismic processing pipeline completed.",
            extra={
                "bsma_context": {
                    "identifier": identifier,
                    "status": pipeline_status,
                    "completed_steps": completed_steps,
                    "total_steps": len(self.steps),
                    "failed_step": failed_step,
                    "total_execution_time_ms": round(
                        total_elapsed_ms,
                        3,
                    ),
                }
            },
        )

        return current_context

    # ========================================================================
    # CONVENIENCE API
    # ========================================================================

    def __len__(self) -> int:
        """Return the number of configured processing steps."""
        return len(self.steps)

    def __iter__(self):
        """Iterate through configured processing steps."""
        return iter(self.steps)

    def __repr__(self) -> str:
        """Return a concise pipeline representation."""
        step_names = [
            self._get_step_name(
                step,
                index,
            )
            for index, step in enumerate(
                self.steps,
                start=1,
            )
        ]

        return (
            f"{self.__class__.__name__}("
            f"steps={step_names!r}, "
            f"halt_on_error={self.halt_on_error!r}"
            f")"
        )