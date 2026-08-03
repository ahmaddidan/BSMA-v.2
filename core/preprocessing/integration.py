"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/processing/integration.py

Description: Concrete ProcessingStep for Kinematic Integration.
Numerically integrates acceleration into velocity and displacement
with strict C-backend anti-drift conditioning.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import detrend

from core.orchestrator import ProcessingStep
from core.types.context import ProcessingContext, WaveformData
from core.types.processing_state import StageStatus
from utils.exceptions import ErrorCode, ProcessingError, SeverityLevel

__all__ = [
    "IntegrationMethod",
    "IntegrationConfig",
    "KinematicIntegrationPlugin",
]


class IntegrationMethod(str, Enum):
    CUMULATIVE_TRAPEZOID = "cumulative_trapezoid"


@dataclass(slots=True, frozen=True)
class IntegrationConfig:
    method: IntegrationMethod = IntegrationMethod.CUMULATIVE_TRAPEZOID
    remove_mean: bool = True
    remove_linear_trend: bool = True


class KinematicIntegrationPlugin(ProcessingStep):
    """
    Velocity and displacement integration plugin.
    Operates strictly on immutable WaveformData to prevent memory bloat.
    """

    def __init__(self, config: IntegrationConfig | None = None) -> None:
        self._config = config or IntegrationConfig()

    @property
    def name(self) -> str:
        return "Kinematic_Integration"

    def _condition_signal(self, signal: np.ndarray) -> np.ndarray:
        """Applies demean and/or linear detrending using SciPy C-backend."""
        # Salin array agar tidak memutasi data asli secara in-place
        conditioned = signal.astype(np.float64, copy=True)
        
        if self._config.remove_mean:
            conditioned -= np.mean(conditioned)
            
        if self._config.remove_linear_trend:
            conditioned = detrend(conditioned, type='linear', overwrite_data=True)
            
        return conditioned

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Integrate acceleration into velocity and displacement."""
        
        if context.acceleration is None:
            raise ProcessingError(
                message="Akselerasi tidak ditemukan. Pastikan baseline/filter telah berjalan.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "integration", "trace_id": context.trace_id}
            )

        acc_data = context.acceleration.data
        sampling_rate = context.sampling_rate

        if acc_data.size < 2:
            raise ProcessingError(
                message="Ukuran waveform terlalu kecil untuk integrasi.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "integration"}
            )

        dt = 1.0 / sampling_rate

        try:
            # 1. Acceleration Conditioning
            acc_conditioned = self._condition_signal(acc_data)

            # 2. Velocity Integration & Anti-Drift
            vel_data = cumulative_trapezoid(acc_conditioned, dx=dt, initial=0.0)
            vel_conditioned = self._condition_signal(vel_data)

            # 3. Displacement Integration & Anti-Drift
            disp_data = cumulative_trapezoid(vel_conditioned, dx=dt, initial=0.0)
            disp_conditioned = self._condition_signal(disp_data)

        except Exception as e:
            raise ProcessingError(
                message="Kalkulus integrasi kinematika gagal.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "integration", "trace_id": context.trace_id},
                cause=e
            ) from e

        # 4. Konstruksi WaveformData Baru
        new_vel = WaveformData(data=vel_conditioned, sampling_rate=sampling_rate, unit="m/s")
        new_disp = WaveformData(data=disp_conditioned, sampling_rate=sampling_rate, unit="m")

        # 5. Immutability Transition
        state = replace(context.processing_state, integration=StageStatus.SUCCESS)

        history_msg = (
            f"KinematicIntegration("
            f"method={self._config.method.value}, "
            f"remove_mean={self._config.remove_mean}, "
            f"remove_trend={self._config.remove_linear_trend})"
        )

        return context.with_state(
            velocity=new_vel,
            displacement=new_disp,
            processing_state=state
        ).add_history(
            step_name=self.name,
            details={"status": "SUCCESS", "config": history_msg}
        )