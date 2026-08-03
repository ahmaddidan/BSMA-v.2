"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/processing/response_spectrum.py

Description: Concrete ProcessingStep for Response Spectrum Computation.
Orchestrates SDOF solvers (Nigam-Jennings exact method or Newmark-Beta) 
to compute PSA, PSV, and SD curves. Fully vectorized and immutable.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from core.orchestrator import ProcessingStep
from core.types.context import ProcessingContext
from core.types.processing_state import StageStatus
from utils.exceptions import ErrorCode, ProcessingError, SeverityLevel

# Mengimpor mesin SDOF yang sudah kita buat sebelumnya
from core.sdof.nigam_jennings import solve_nigam_jennings
from core.sdof.newmark import solve_newmark

__all__ = ["ResponseSpectrumConfig", "ResponseSpectrumPlugin"]


@dataclass(slots=True, frozen=True)
class ResponseSpectrumConfig:
    """
    Konfigurasi untuk kalkulasi Spektrum Respons.
    """
    # Menggunakan tuple agar tipe data tetap immutable (frozen)
    periods: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.linspace(0.01, 4.0, 100).tolist())
    )
    damping: float = 0.05
    solver: Literal["nigam_jennings", "newmark"] = "nigam_jennings"

    def __post_init__(self) -> None:
        if self.damping <= 0.0 or self.damping >= 1.0:
            raise ValueError(f"Redaman (damping) harus berada di antara (0, 1). Menerima: {self.damping}")
        if not self.periods:
            raise ValueError("Array periode tidak boleh kosong.")
        if any(p <= 0.0 for p in self.periods):
            raise ValueError("Seluruh nilai periode harus lebih besar dari 0.")


class ResponseSpectrumPlugin(ProcessingStep):
    """
    Mengeksekusi komputasi spektrum respons secara fungsional murni.
    Menerapkan relasi Pseudo-Spectral secara matematis (PSA = w^2 * SD).
    """

    def __init__(self, config: ResponseSpectrumConfig | None = None) -> None:
        self._config = config or ResponseSpectrumConfig()

    @property
    def name(self) -> str:
        return f"Response_Spectrum_{self._config.solver.capitalize()}"

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Eksekusi kalkulasi spektrum respons terhadap osilator SDOF.
        """
        if context.acceleration is None:
            raise ProcessingError(
                message="Data akselerasi tidak ditemukan. Pastikan baseline/tapering telah berjalan.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "response_spectrum", "trace_id": context.trace_id}
            )

        acc_data = context.acceleration.data
        if acc_data.size < 2:
            raise ProcessingError(
                message="Ukuran array akselerasi terlalu kecil untuk simulasi SDOF.",
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={"module": "response_spectrum", "npts": acc_data.size}
            )

        dt = 1.0 / context.sampling_rate
        periods_array = np.array(self._config.periods, dtype=np.float64)

        try:
            # 1. Injeksi Dinamis Mesin SDOF
            if self._config.solver == "nigam_jennings":
                u, v, a_abs = solve_nigam_jennings(
                    acceleration=acc_data,
                    dt=dt,
                    periods=periods_array,
                    damping=self._config.damping
                )
            elif self._config.solver == "newmark":
                u, v, a_abs = solve_newmark(
                    acceleration=acc_data,
                    dt=dt,
                    periods=periods_array,
                    damping=self._config.damping
                )
            else:
                raise ValueError(f"Solver tidak dikenal: {self._config.solver}")

            # 2. Ekstraksi Spectral Displacement (SD) absolut maksimum
            # u memiliki shape (n_periods, n_samples)
            sd = np.max(np.abs(u), axis=1)

            # 3. Kalkulus Fundamental (Relasi Pseudo-Spectral)
            # w = 2*pi / T
            omega = 2.0 * np.pi / periods_array
            omega_sq = omega ** 2

            psv = omega * sd
            psa = omega_sq * sd

            # Ekstraksi Spectral Acceleration Aktual (Absolute Acceleration)
            sa = np.max(np.abs(a_abs), axis=1)

        except Exception as e:
            raise ProcessingError(
                message=f"Kegagalan komputasi SDOF ({self._config.solver}).",
                error_code=ErrorCode.RS001,
                severity=SeverityLevel.ERROR,
                context={"module": "response_spectrum", "solver": self._config.solver},
                cause=e
            ) from e

        # 4. Immutability Transition: Merangkai State Akhir
        new_spectral_data = dict(context.spectral_data)
        new_spectral_data.update({
            "periods": periods_array,
            "SD": sd,
            "PSV": psv,
            "PSA": psa,
            "SA": sa,
            "damping": self._config.damping
        })

        # Update processing state secara aman jika field-nya ada, abaikan jika tidak
        try:
            state = replace(context.processing_state, response_spectrum=StageStatus.SUCCESS)
        except TypeError:
            try:
                state = replace(context.processing_state, spectrum=StageStatus.SUCCESS)
            except TypeError:
                state = context.processing_state

        return context.with_state(
            spectral_data=new_spectral_data,
            processing_state=state
        ).add_history(
            step_name=self.name,
            details={
                "status": "SUCCESS",
                "solver": self._config.solver,
                "damping": self._config.damping,
                "max_psa": float(np.max(psa)),
                "dominant_period": float(periods_array[np.argmax(psa)])
            }
        )