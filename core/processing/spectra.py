"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/processing/spectra.py
Description: Response Spectrum Extraction Plugin.
Computes PSA, PSV, and SD across standard engineering periods.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import numpy as np
from numpy.typing import NDArray

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext
from core.types.processing_state import StageStatus
from core.sdof.newmark import solve_newmark_vectorized
from utils.exceptions import ErrorCode, SeverityLevel, ProcessingError

FloatArray = NDArray[np.float64]

__all__ = [
    "SpectraConfig",
    "ResponseSpectrumPlugin",
]

@dataclass(slots=True, frozen=True)
class SpectraConfig:
    """
    Konfigurasi ekstraksi spektrum respons (T, damping).
    Default menggunakan standar rekayasa gempa 100 periode logaritmik.
    """
    damping: float = 0.05
    period_min: float = 0.01
    period_max: float = 10.0
    period_steps: int = 100

class ResponseSpectrumPlugin(PreprocessorPlugin):
    """
    Membangkitkan kurva Response Spectrum (PSA, PSV, SD).
    """

    def __init__(self, config: SpectraConfig = SpectraConfig()) -> None:
        self._config = config
        # Pre-komputasi array periode secara logaritmik (lebih rapat di frekuensi tinggi)
        self._periods = np.logspace(
            np.log10(self._config.period_min),
            np.log10(self._config.period_max),
            self._config.period_steps,
            dtype=np.float64
        )
        # Tambahkan periode 0.0 (PGA anchor) di indeks pertama
        self._periods = np.insert(self._periods, 0, 0.0)

    @property
    def plugin_name(self) -> str:
        return "ResponseSpectrum"

    @property
    def plugin_description(self) -> str:
        return "Computes PSA, PSV, and SD Response Spectra via Vectorized Newmark-Beta."

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Mengeksekusi solver SDOF dan menyimpan matriks spektrum ke dalam cache.
        """
        self.validate_input(context)
        waveform = context.waveform

        if waveform is None or len(waveform) < 2:
            raise ProcessingError(
                message="Valid waveform required for spectrum generation.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "spectra"}
            )

        dt = 1.0 / context.sampling_rate

        # 1. Eksekusi Vectorized SDOF Solver (O(1) time complexity reduction)
        u, v, a_abs = solve_newmark_vectorized(
            acceleration=waveform,
            dt=dt,
            periods=self._periods,
            damping=self._config.damping
        )

        # 2. Ekstraksi Spectral Ordinates (Nilai Max Absolut per Periode)
        # Spectral Displacement (SD)
        sd = np.max(np.abs(u), axis=1)
        
        # Spectral Velocity (SV / Pseudo-Spectral Velocity PSV)
        # Catatan Geofisika: Standar rekayasa menggunakan PSV = omega * SD
        # untuk stabilitas spektral, bukan max(|v|).
        omega = np.zeros_like(self._periods)
        valid = self._periods > 0.0
        omega[valid] = 2.0 * np.pi / self._periods[valid]
        psv = omega * sd

        # Spectral Acceleration (SA / Pseudo-Spectral Acceleration PSA)
        # PSA = omega^2 * SD. Namun standar PEER/USGS mengizinkan pengambilan
        # nilai puncak dari percepatan absolut max(|A_abs|) untuk ketepatan riil.
        psa = np.max(np.abs(a_abs), axis=1)

        # 3. Penyiapan Dictionary untuk Cache
        spectra_data = {
            "periods": self._periods,
            "psa": psa,
            "psv": psv,
            "sd": sd,
            "damping": self._config.damping
        }

        # 4. Modifikasi Immutable Cache (Asumsi field `response_spectra` ditambahkan di cache.py)
        cache = deepcopy(context.cache)
        cache.response_spectra = spectra_data

        state = replace(
            context.processing_state,
            spectra=StageStatus.SUCCESS,
        )

        history = (
            f"ResponseSpectrum("
            f"damping={self._config.damping*100:.1f}%, "
            f"periods={self._config.period_steps}, "
            f"PSA_max={np.max(psa):.4g})"
        )

        return context.copy(
            cache=cache,
            processing_state=state,
        ).add_history(history)