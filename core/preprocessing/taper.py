"""
BMKG Strong Motion Analyzer (BSMA)

Taper Plugin
============

Applies a Tukey taper window to the waveform prior to filtering
or numerical integration.

The taper minimizes spectral leakage caused by discontinuities
at the beginning and end of the signal.

References
----------
- Harris (1978)
- ObsPy
- COSMOS Strong Motion Processing
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import tukey

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext
from core.types.processing_state import StageStatus

__all__ = [
    "TaperConfig",
    "TaperPlugin",
]


@dataclass(slots=True, frozen=True)
class TaperConfig:
    """
    Configuration for Tukey taper.

    Parameters
    ----------
    alpha
        Fraction of cosine taper.

        Typical values

        0.05 (default)
        0.10
    """

    alpha: float = 0.05


class TaperPlugin(PreprocessorPlugin):
    """
    Apply Tukey taper.
    """

    def __init__(
        self,
        config: TaperConfig = TaperConfig(),
    ) -> None:

        self._config = config

    @property
    def plugin_name(self) -> str:
        return "Taper"

    @property
    def plugin_description(self) -> str:
        return "Apply Tukey taper."

    def process(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:

        self.validate_input(context)

        waveform = context.waveform

        assert waveform is not None

        window = tukey(
            waveform.size,
            alpha=self._config.alpha,
        )

        tapered = waveform * window

        state = context.processing_state
        state.taper = StageStatus.SUCCESS

        cache = context.cache
        cache.clear()

        return (
            context.copy(
                waveform=tapered,
                processing_state=state,
                cache=cache,
            )
            .add_history(
                f"{self.plugin_name}"
                f"(alpha={self._config.alpha:.3f})"
            )
        )