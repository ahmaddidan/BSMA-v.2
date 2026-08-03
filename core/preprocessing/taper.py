"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/preprocessing/taper.py
Description: Tapering plugin to suppress spectral leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from scipy.signal import windows

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext, WaveformData
from core.types.processing_state import StageStatus
from utils.exceptions import ErrorCode, ProcessingError, SeverityLevel

__all__ = ["TaperPlugin", "TaperConfig"]


@dataclass(frozen=True)
class TaperConfig:
    alpha: float = 0.05
    window_type: str = "hann"


class TaperPlugin(PreprocessorPlugin):
    """
    Plugin Tapering untuk mereduksi kebocoran spektral (spectral leakage) 
    pada awal dan akhir rekaman sinyal gempa.
    """

    def __init__(self, config: TaperConfig | None = None) -> None:
        self.config = config or TaperConfig()

    @property
    def plugin_name(self) -> str:
        return "Taper"

    @property
    def plugin_description(self) -> str:
        return f"Applies {self.config.window_type} taper with alpha={self.config.alpha}."

    def process(self, context: ProcessingContext) -> ProcessingContext:
        self.validate_input(context)

        # Ambil data gelombang secara aman dari berbagai alternatif atribut konteks
        wf_container = getattr(context, "waveform", None)
        if wf_container is None:
            wf_container = getattr(context, "raw_waveform", None)
        if wf_container is None:
            wf_container = getattr(context, "acceleration", None)

        if wf_container is None:
            raise ProcessingError(
                message="Cannot perform taper: no waveform found in context.",
                error_code=ErrorCode.PR002,
                severity=SeverityLevel.ERROR,
                context={"module": "taper"}
            )

        if isinstance(wf_container, WaveformData):
            data_array = wf_container.data
            sr = wf_container.sampling_rate
            unit = wf_container.unit
        elif hasattr(wf_container, "data"):
            data_array = wf_container.data
            sr = getattr(wf_container, "sampling_rate", 100.0)
            unit = getattr(wf_container, "unit", "m/s^2")
        else:
            data_array = np.asarray(wf_container)
            sr = getattr(context, "sampling_rate", 100.0)
            unit = "m/s^2"

        if data_array is None or data_array.size == 0:
            raise ProcessingError(
                message="Cannot perform taper on an empty array.",
                error_code=ErrorCode.PR002,
                severity=SeverityLevel.ERROR,
                context={"module": "taper"}
            )

        try:
            n = len(data_array)
            # Hitung jumlah sampel untuk jendela taper
            n_taper = int(np.floor(self.config.alpha * n))
            if n_taper > 0:
                # Menggunakan jendela tukey / hann sesuai konfigurasi
                window = windows.tukey(n, alpha=2.0 * self.config.alpha)
                tapered_data = (data_array * window).astype(np.float64)
            else:
                tapered_data = data_array.astype(np.float64)
        except Exception as e:
            raise ProcessingError(
                message="Tapering application failed mathematically.",
                error_code=ErrorCode.PR002,
                severity=SeverityLevel.ERROR,
                context={"module": "taper"},
                cause=e
            ) from e

        # Buat wadah WaveformData yang diperbarui
        if isinstance(wf_container, WaveformData):
            updated_waveform = replace(wf_container, data=tapered_data)
        else:
            updated_waveform = WaveformData(data=tapered_data, sampling_rate=sr, unit=unit)

        # Transisi State Immutable
        state = replace(
            context.processing_state,
            taper=StageStatus.SUCCESS
        )

        update_kwargs = {
            "processing_state": state,
            "acceleration": updated_waveform,
        }
        if hasattr(context, "waveform"):
            update_kwargs["waveform"] = updated_waveform
        if hasattr(context, "raw_waveform"):
            update_kwargs["raw_waveform"] = updated_waveform

        new_context = replace(context, **update_kwargs)

        if hasattr(new_context, "add_history"):
            return new_context.add_history(
                step_name="Taper",
                details={"status": "SUCCESS", "alpha": self.config.alpha}
            )
        return new_context