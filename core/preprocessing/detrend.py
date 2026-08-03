"""
BMKG Strong Motion Analyzer (BSMA)

Linear / Polynomial Detrending Plugin
=====================================

Removes the mean or linear trend from the signal to stabilize
subsequent integration processes and fix baseline wandering.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.signal import detrend

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext
from core.types.processing_state import StageStatus

__all__ = [
    "DetrendMethod",
    "DetrendConfig",
    "DetrendPlugin",
]

FloatArray = NDArray[np.float64]


class DetrendMethod(str, Enum):
    CONSTANT = "constant"
    LINEAR = "linear"


@dataclass(slots=True, frozen=True)
class DetrendConfig:
    """
    Configuration for detrending.
    """
    method: DetrendMethod = DetrendMethod.LINEAR


class DetrendPlugin(PreprocessorPlugin):
    """
    Apply constant or linear detrending.
    """

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
        return "Remove constant or linear trend."

    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """
        Execute detrending process safely across all channels.
        """
        self.validate_input(context)

        waveform = context.waveform
        assert waveform is not None

        if waveform.size == 0:
            raise ValueError("Waveform is empty.")

        # SciPy's detrend handles axis=-1 by default, safe for multi-channel
        corrected = detrend(
            waveform,
            type=self._config.method.value,
            axis=-1
        )
        corrected = corrected.astype(waveform.dtype, copy=False)

        # State management
        state = replace(
            context.processing_state,
            detrend=StageStatus.SUCCESS,
        )

        # Cache invalidation
        cache = deepcopy(context.cache)
        cache.clear()

        history = f"{self.plugin_name}({self._config.method.value})"

        return (
            context.copy(
                waveform=corrected,
                processing_state=state,
                cache=cache,
            )
            .add_history(history)
        )