"""
BMKG Strong Motion Analyzer (BSMA)

Linear / Polynomial Detrending Plugin
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.signal import detrend

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext
from core.types.processing_state import StageStatus

__all__ = [
    "DetrendMethod",
    "DetrendConfig",
    "DetrendPlugin",
]


class DetrendMethod(str, Enum):
    CONSTANT = "constant"
    LINEAR = "linear"


@dataclass(slots=True, frozen=True)
class DetrendConfig:

    method: DetrendMethod = DetrendMethod.LINEAR


class DetrendPlugin(PreprocessorPlugin):

    def __init__(
        self,
        config: DetrendConfig = DetrendConfig(),
    ) -> None:

        self._config = config

    @property
    def plugin_name(self) -> str:

        return "Detrend"

    @property
    def plugin_description(self) -> str:

        return "Remove linear trend."

    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:

        self.validate_input(context)

        waveform = context.waveform

        assert waveform is not None

        corrected = detrend(
            waveform,
            type=self._config.method.value,
        )

        state = context.processing_state
        state.detrend = StageStatus.SUCCESS

        cache = context.cache
        cache.clear()

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