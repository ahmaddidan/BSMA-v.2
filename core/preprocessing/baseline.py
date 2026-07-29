"""
BMKG Strong Motion Analyzer (BSMA)

Baseline Correction Plugin
==========================

Remove baseline offset from acceleration waveform.

Supported methods
-----------------
- Mean removal
- Median removal
- Constant offset removal

Design
------
- Immutable ProcessingContext
- Production-grade typing
- Pipeline compatible
- Cache-aware
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext
from core.types.processing_state import StageStatus

__all__ = [
    "BaselineMethod",
    "BaselineConfig",
    "BaselineCorrectionPlugin",
]


class BaselineMethod(str, Enum):
    """Available baseline correction methods."""

    MEAN = "mean"
    MEDIAN = "median"
    CONSTANT = "constant"


@dataclass(slots=True, frozen=True)
class BaselineConfig:
    """
    Configuration for baseline correction.
    """

    method: BaselineMethod = BaselineMethod.MEAN

    constant: float = 0.0


class BaselineCorrectionPlugin(PreprocessorPlugin):
    """
    Baseline correction plugin.
    """

    def __init__(
        self,
        config: BaselineConfig = BaselineConfig(),
    ) -> None:

        self._config = config

    @property
    def plugin_name(self) -> str:
        return "BaselineCorrection"

    @property
    def plugin_description(self) -> str:
        return (
            "Remove baseline offset from waveform."
        )

    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:

        self.validate_input(context)

        waveform = context.waveform

        assert waveform is not None

        #
        # -----------------------------------------------------
        # Baseline correction
        # -----------------------------------------------------
        #

        if self._config.method is BaselineMethod.MEAN:

            corrected = waveform - np.mean(waveform)

        elif self._config.method is BaselineMethod.MEDIAN:

            corrected = waveform - np.median(waveform)

        elif self._config.method is BaselineMethod.CONSTANT:

            corrected = waveform - self._config.constant

        else:

            raise ValueError(
                f"Unsupported baseline method: "
                f"{self._config.method}"
            )

        #
        # -----------------------------------------------------
        # Processing state
        # -----------------------------------------------------
        #

        state = context.processing_state

        state.baseline = StageStatus.SUCCESS

        #
        # -----------------------------------------------------
        # Clear derived cache
        # -----------------------------------------------------
        #

        cache = context.cache

        cache.clear()

        #
        # -----------------------------------------------------
        # Return immutable context
        # -----------------------------------------------------
        #

        return (
            context.copy(
                waveform=corrected,
                processing_state=state,
                cache=cache,
            )
            .add_history(
                f"{self.plugin_name}"
                f"({self._config.method.value})"
            )
        )