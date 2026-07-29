"""
BMKG Strong Motion Analyzer (BSMA)
core/preprocessing/filter.py
"""
import time
from typing import Literal, Optional
from dataclasses import dataclass, replace
import numpy as np
from scipy import signal

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext, ProcessingStep, SeverityLevel

@dataclass(frozen=True)
class FilterConfig:
    type: Literal["bandpass", "lowpass", "highpass", "bandstop"] = "bandpass"
    freq_min: Optional[float] = 0.1   # Hz (Cutoff bawah / batas bawah bandstop)
    freq_max: Optional[float] = 25.0  # Hz (Cutoff atas / batas atas bandstop)
    corners: int = 4                  # Orde filter
    zerophase: bool = True            # Forward-backward filtering

class ButterworthFilterPlugin(PreprocessorPlugin):
    """
    Plugin untuk menyaring frekuensi menggunakan Butterworth filter.
    Menggunakan desain SOS (Second-Order Sections) untuk stabilitas numerik.
    """
    def __init__(self, config: FilterConfig):
        self._config = config

    @property
    def plugin_name(self) -> str:
        return "ButterworthFilterPlugin"

    def process(self, context: ProcessingContext) -> ProcessingContext:
        data = context.data.copy()
        fs = context.metadata.sampling_rate
        nyquist = 0.5 * fs

        # 1. Validasi Frekuensi Nyquist
        if self._config.freq_max and self._config.freq_max >= nyquist:
            new_qc = context.qc_report.add_message(
                SeverityLevel.WARNING,
                f"freq_max ({self._config.freq_max} Hz) melebihi batas Nyquist ({nyquist} Hz). Disesuaikan ke {nyquist * 0.99} Hz."
            )
            freq_max = nyquist * 0.99
            context = replace(context, qc_report=new_qc)
        else:
            freq_max = self._config.freq_max

        # 2. Desain Filter SOS
        if self._config.type == "bandpass":
            Wn = [self._config.freq_min / nyquist, freq_max / nyquist]
            btype = "bandpass"
        elif self._config.type == "lowpass":
            Wn = freq_max / nyquist
            btype = "lowpass"
        elif self._config.type == "highpass":
            Wn = self._config.freq_min / nyquist
            btype = "highpass"
        elif self._config.type == "bandstop":
            Wn = [self._config.freq_min / nyquist, freq_max / nyquist]
            btype = "bandstop"
        else:
            raise ValueError(f"Tipe filter tidak didukung: {self._config.type}")

        sos = signal.butter(
            N=self._config.corners, 
            Wn=Wn, 
            btype=btype, 
            output='sos'
        )

        # 3. Terapkan Filter
        if self._config.zerophase:
            filtered_data = signal.sosfiltfilt(sos, data)
        else:
            filtered_data = signal.sosfilt(sos, data)

        # 4. Catat Provenance
        step = ProcessingStep(
            name=self.plugin_name,
            config=self._config,
            timestamp=time.time()
        )

        return replace(context, data=filtered_data, history=context.history + (step,))