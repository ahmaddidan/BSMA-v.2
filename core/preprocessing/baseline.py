"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/preprocessing/baseline.py
Description: Pre-event baseline correction plugin.
Removes the DC offset (constant) or linear drift from the waveform 
to center the signal and prevent parabolic integration drift.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.signal import detrend

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext, WaveformData
from core.types.processing_state import StageStatus
from utils.exceptions import ErrorCode, ProcessingError, SeverityLevel

__all__ = ["BaselineCorrectionPlugin"]


class BaselineCorrectionPlugin(PreprocessorPlugin):
    """
    Koreksi baseline tingkat produksi (Tanpa dependensi ObsPy).
    Secara matematis mereduksi DC offset dan linear drift menggunakan 
    Least Squares fitting untuk mencegah anomali integrasi ganda.
    """

    def __init__(self, method: str = "linear") -> None:
        """
        Parameters
        ----------
        method : str
            'linear' : Menghilangkan tren linear (y = mx + c) - Standard Seismologi.
            'constant': Hanya menghilangkan mean (y = c).
        """
        if method not in ("constant", "linear"):
            raise ValueError("Baseline method must be strictly 'constant' or 'linear'.")
        self.method = method

    @property
    def plugin_name(self) -> str:
        return "BaselineCorrection"

    @property
    def plugin_description(self) -> str:
        return f"Removes {self.method} drift from the waveform using SciPy C-optimized detrend."

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Mengeksekusi koreksi baseline pada matriks numpy secara fungsional murni.
        """
        self.validate_input(context)

        # Ambil data gelombang secara aman dari berbagai alternatif atribut konteks
        wf_container = getattr(context, "waveform", None)
        if wf_container is None:
            wf_container = getattr(context, "raw_waveform", None)
        if wf_container is None:
            wf_container = getattr(context, "acceleration", None)

        if wf_container is None:
            raise ProcessingError(
                message="Cannot perform baseline correction: no waveform found in context.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "baseline"}
            )

        # Ekstrak array numpy mentah
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
                message="Cannot perform baseline correction on an empty array.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "baseline"}
            )

        # Eksekusi Vektorisasi Matematis (Menggunakan SciPy murni)
        try:
            corrected_data = detrend(data_array, type=self.method).astype(np.float64)
        except Exception as e:
            raise ProcessingError(
                message=f"Baseline correction ({self.method}) failed mathematically.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "baseline", "shape": data_array.shape},
                cause=e
            ) from e

        # Buat kembali wadah WaveformData yang diperbarui
        if isinstance(wf_container, WaveformData):
            updated_waveform = replace(wf_container, data=corrected_data)
        else:
            updated_waveform = WaveformData(data=corrected_data, sampling_rate=sr, unit=unit)

        # Transisi State Immutable
        state = replace(
            context.processing_state,
            baseline=StageStatus.SUCCESS
        )

        history_msg = f"BaselineCorrection(method='{self.method}')"

        # Siapkan argumen pembaruan konteks
        update_kwargs = {
            "processing_state": state,
            "acceleration": updated_waveform,
        }
        if hasattr(context, "waveform"):
            update_kwargs["waveform"] = updated_waveform
        if hasattr(context, "raw_waveform"):
            update_kwargs["raw_waveform"] = updated_waveform

        # Gunakan replace() langsung pada context karena ProcessingContext berupa dataclass
        new_context = replace(context, **update_kwargs)
        
        # Tambahkan history jika method-nya tersedia, atau kembalikan langsung
        if hasattr(new_context, "add_history"):
            return new_context.add_history(
                step_name="BaselineCorrection",
                details={"status": "SUCCESS", "method": self.method}
            )
        return new_context