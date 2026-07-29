"""
BMKG Strong Motion Analyzer (BSMA)
core/processing/spectra.py
"""
import time
from dataclasses import dataclass, replace
import numpy as np

from core.interfaces.preprocessor import PreprocessorPlugin
from core.types.context import ProcessingContext, ProcessingStep, ResponseSpectra, SeverityLevel

@dataclass(frozen=True)
class SpectraConfig:
    damping: float = 0.05  # Redaman standar 5%
    period_min: float = 0.01
    period_max: float = 10.0
    num_periods: int = 100
    period_spacing: str = "log"  # "linear" atau "log"

class NigamJenningsSpectraPlugin(PreprocessorPlugin):
    """
    Kalkulator Spektrum Respons menggunakan metode analitik eksak Nigam & Jennings (1968).
    Menjamin stabilitas komputasi fisis untuk rentang frekuensi sistem SDOF.
    """
    def __init__(self, config: SpectraConfig = SpectraConfig()):
        self._config = config

    @property
    def plugin_name(self) -> str:
        return "NigamJenningsSpectraPlugin"

    def _generate_periods(self) -> np.ndarray:
        if self._config.period_spacing == "log":
            return np.logspace(
                np.log10(self._config.period_min), 
                np.log10(self._config.period_max), 
                self._config.num_periods
            )
        return np.linspace(
            self._config.period_min, 
            self._config.period_max, 
            self._config.num_periods
        )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        acc = context.data
        dt = 1.0 / context.metadata.sampling_rate
        periods = self._generate_periods()
        damping = self._config.damping

        # Array untuk menyimpan nilai spektral maksimum
        SD = np.zeros_like(periods)
        SV = np.zeros_like(periods)
        SA = np.zeros_like(periods)
        
        # Iterasi atas setiap periode (sistem SDOF)
        for i, T in enumerate(periods):
            omega = 2.0 * np.pi / T
            omega2 = omega ** 2
            omegaD = omega * np.sqrt(1.0 - damping**2)
            
            # Konstanta eksponensial dan trigonometri
            e = np.exp(-damping * omega * dt)
            sin_wdt = np.sin(omegaD * dt)
            cos_wdt = np.cos(omegaD * dt)

            # Elemen matriks transisi [A] untuk state space
            A11 = e * (cos_wdt + (damping * omega / omegaD) * sin_wdt)
            A12 = (e / omegaD) * sin_wdt
            A21 = -e * (omega2 / omegaD) * sin_wdt
            A22 = e * (cos_wdt - (damping * omega / omegaD) * sin_wdt)

            # Menghindari pembagian nol untuk term statis
            omega3 = omega ** 3
            
            # Elemen matriks beban [B] 
            # (Diasumsikan percepatan berubah linear antar interval dt)
            hc = (2.0 * damping) / (omega3 * dt)
            
            B11 = e * (((2.0 * damping**2 - 1.0) / (omega2 * omegaD)) * sin_wdt + 
                       (2.0 * damping / omega3) * cos_wdt) + (1.0 / omega2) - hc
            B12 = -e * (((2.0 * damping**2 - 1.0) / (omega2 * omegaD)) * sin_wdt + 
                        (2.0 * damping / omega3) * cos_wdt) + hc
            B21 = e * (((damping * omega) / omegaD) * sin_wdt + cos_wdt) / omega2 - 1.0 / (omega2 * dt)
            B22 = -e * (((damping * omega) / omegaD) * sin_wdt + cos_wdt) / omega2 + 1.0 / (omega2 * dt)

            # Inisialisasi state (perpindahan dan kecepatan awal 0)
            u = np.zeros_like(acc)
            v = np.zeros_like(acc)
            
            # Iterasi sekuensial waktu (Time History Analysis)
            # v[k+1] dan u[k+1] bergantung pada state sebelumnya
            for k in range(len(acc) - 1):
                u[k+1] = A11 * u[k] + A12 * v[k] + B11 * acc[k] + B12 * acc[k+1]
                v[k+1] = A21 * u[k] + A22 * v[k] + B21 * acc[k] + B22 * acc[k+1]
            
            # Hitung absolute acceleration dari persamaan gerak
            acc_abs = -2.0 * damping * omega * v - omega2 * u

            # Ekstrak nilai puncak (Peak values)
            SD[i] = np.max(np.abs(u))
            SV[i] = np.max(np.abs(v))
            SA[i] = np.max(np.abs(acc_abs))

        # Pseudo-spectral values
        PSV = SD * (2.0 * np.pi / periods)
        PSA = PSV * (2.0 * np.pi / periods)

        spectra_result = ResponseSpectra(
            periods=periods, sd=SD, sv=SV, sa=SA, psv=PSV, psa=PSA
        )

        step = ProcessingStep(
            name=self.plugin_name,
            config=self._config,
            timestamp=time.time()
        )

        return replace(context, spectra=spectra_result, history=context.history + (step,))