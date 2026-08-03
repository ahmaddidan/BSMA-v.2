"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/preprocessing/filter.py
Description: Butterworth filter plugin for waveform frequency filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from scipy.signal import butter, sosfiltfilt

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext, WaveformData
from core.types.processing_state import StageStatus
from utils.exceptions import ErrorCode, ProcessingError, SeverityLevel

__all__ = ["ButterworthFilterPlugin", "FilterConfig"]


@dataclass(frozen=True)
class FilterConfig:
    type: str = "bandpass"  # 'bandpass', 'lowpass', 'highpass', 'stop'
    freq_min: float = 0.1
    freq_max: float = 25.0
    corners: int = 4
    zerophase: bool = True


class ButterworthFilterPlugin(PreprocessorPlugin):
    """
    Plugin Filter Butterworth untuk menyaring frekuensi sinyal gempa 
    menggunakan Cascaded Second-Order Sections (SOS) agar numerik stabil.
    """

    def __init__(self, config: FilterConfig | None = None) -> None:
        self.config = config or FilterConfig()

    @property
    def plugin_name(self) -> str:
        return "ButterworthFilter"

    @property
    def plugin_description(self) -> str:
        return f"Applies {self.config.type} Butterworth filter ({self.config.freq_min}-{self.config.freq_max} Hz)."

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
                message="Cannot perform filtering: no waveform found in context.",
                error_code=ErrorCode.PR003,
                severity=SeverityLevel.ERROR,
                context={"module": "filter"}
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
                message="Cannot perform filter on an empty array.",
                error_code=ErrorCode.PR003,
                severity=SeverityLevel.ERROR,
                context={"module": "filter"}
            )

        try:
            nyquist = 0.5 * sr
            ftype = self.config.type.lower()

            if ftype == "bandpass":
                low = self.config.freq_min / nyquist
                high = self.config.freq_max / nyquist
                Wn = [low, high]
                btype = "bandpass"
            elif ftype == "lowpass":
                Wn = self.config.freq_max / nyquist
                btype = "lowpass"
            elif ftype == "highpass":
                Wn = self.config.freq_min / nyquist
                btype = "highpass"
            else:
                raise ValueError(f"Unsupported filter type: {self.config.type}")

            # Desain filter SOS
            sos = butter(self.config.corners, Wn, btype=btype, output="sos")

            # Terapkan filter (zerophase menggunakan sosfiltfilt atau sosfilt biasa)
            if self.config.zerophase:
                filtered_data = sosfiltfilt(sos, data_array).astype(np.float64)
            else:
                from scipy.signal import sosfilt
                filtered_data, _ = sosfilt(sos, data_array)
                filtered_data = filtered_data.astype(np.float64)

        except Exception as e:
            raise ProcessingError(
                message="Butterworth filtering failed mathematically.",
                error_code=ErrorCode.PR003,
                severity=SeverityLevel.ERROR,
                context={"module": "filter"},
                cause=e
            ) from e

        # Buat wadah WaveformData yang diperbarui
        if isinstance(wf_container, WaveformData):
            updated_waveform = replace(wf_container, data=filtered_data)
        else:
            updated_waveform = WaveformData(data=filtered_data, sampling_rate=sr, unit=unit)

        # Transisi State Immutable
        state = replace(
            context.processing_state,
            filter=StageStatus.SUCCESS
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
                step_name="ButterworthFilter",
                details={"status": "SUCCESS", "type": self.config.type}
            )
        return new_context