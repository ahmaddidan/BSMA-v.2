"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/processing/parameters.py

Description: Concrete ProcessingStep for Strong Motion Parameters Extraction.
Computes PGA, PGV, PGD, Arias Intensity, CAV, Husid Curve, Significant Durations,
and waveform statistics using strictly vectorized C-backend operations.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid

from core.orchestrator import ProcessingStep
from core.types.context import ProcessingContext
from core.types.processing_state import StageStatus
from utils.exceptions import ErrorCode, ProcessingError, SeverityLevel

__all__ = ["ParameterConfig", "ParameterExtractionPlugin"]


@dataclass(slots=True, frozen=True)
class ParameterConfig:
    """
    Configuration for strong-motion parameter extraction.
    """
    gravity: float = 9.80665
    husid_start: float = 0.05
    husid_end75: float = 0.75
    husid_end95: float = 0.95

    def __post_init__(self) -> None:
        if self.gravity <= 0.0:
            raise ValueError(f"Gravity must be positive. Got: {self.gravity}")
        if not (0.0 < self.husid_start < self.husid_end75 < self.husid_end95 <= 1.0):
            raise ValueError("Husid thresholds must strictly follow: 0 < start < end75 < end95 <= 1.0")


class ParameterExtractionPlugin(ProcessingStep):
    """
    Extract engineering strong-motion parameters with mathematical rigor 
    and O(log N) bisection search performance. No deepcopy overhead.
    """

    def __init__(self, config: ParameterConfig | None = None) -> None:
        self._config = config or ParameterConfig()

    @property
    def name(self) -> str:
        return "Parameter_Extraction"

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # 1. Structural Validation
        if context.acceleration is None or context.velocity is None or context.displacement is None:
            raise ProcessingError(
                message="Kinematic states (Acc, Vel, Disp) are incomplete. Run IntegrationPlugin first.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "parameters", "trace_id": context.trace_id}
            )

        acc = context.acceleration.data
        vel = context.velocity.data
        disp = context.displacement.data

        # 2. Strict Mathematical Validation (DRY approach)
        for name, arr in [("Acceleration", acc), ("Velocity", vel), ("Displacement", disp)]:
            if arr.size < 2:
                raise ProcessingError(
                    message=f"Array {name} is too short.",
                    error_code=ErrorCode.PR001,
                    severity=SeverityLevel.ERROR,
                    context={"module": "parameters"}
                )
            if not np.isfinite(arr).all():
                raise ProcessingError(
                    message=f"Non-finite values (NaN/Inf) detected in {name}.",
                    error_code=ErrorCode.PR001,
                    severity=SeverityLevel.ERROR,
                    context={"module": "parameters", "trace_id": context.trace_id}
                )

        dt = 1.0 / context.sampling_rate

        try:
            # 3. Peak Absolute Values
            pga, pgv, pgd = float(np.max(np.abs(acc))), float(np.max(np.abs(vel))), float(np.max(np.abs(disp)))

            # 4. Energy Integrals (Arias, Husid, CAV)
            acc_sq = acc ** 2
            arias_cum = (np.pi / (2.0 * self._config.gravity)) * cumulative_trapezoid(acc_sq, dx=dt, initial=0.0)
            arias_intensity = float(arias_cum[-1])
            
            cav = float(cumulative_trapezoid(np.abs(acc), dx=dt, initial=0.0)[-1])

            # 5. Husid Curve & O(log N) Significant Durations
            d5_75 = d5_95 = 0.0
            husid_curve = np.zeros_like(arias_cum)

            if arias_intensity > 0.0:
                husid_curve = arias_cum / arias_intensity
                
                # NumPy C-API bisection search for extreme performance
                idx_5 = int(np.searchsorted(husid_curve, self._config.husid_start))
                idx_75 = int(np.searchsorted(husid_curve, self._config.husid_end75))
                idx_95 = int(np.searchsorted(husid_curve, self._config.husid_end95))
                
                # Boundary protection
                idx_75 = min(idx_75, arias_cum.size - 1)
                idx_95 = min(idx_95, arias_cum.size - 1)
                
                d5_75 = float((idx_75 - idx_5) * dt)
                d5_95 = float((idx_95 - idx_5) * dt)

            # 6. Waveform Statistics
            mean_acc = float(np.mean(acc))
            std_acc = float(np.std(acc, ddof=0))
            rms_acc = float(np.sqrt(np.mean(acc_sq)))

        except Exception as e:
            raise ProcessingError(
                message="Mathematical computation failed during parameter extraction.",
                error_code=ErrorCode.PR001,
                severity=SeverityLevel.ERROR,
                context={"module": "parameters", "trace_id": context.trace_id},
                cause=e
            ) from e

        # 7. Immutability Transition: Merangkai State Akhir tanpa Deepcopy
        new_metrics = dict(context.metrics)
        new_metrics.update({
            "PGA": pga,
            "PGV": pgv,
            "PGD": pgd,
            "Arias_Intensity": arias_intensity,
            "CAV": cav,
            "Significant_Duration_D5_75": d5_75,
            "Significant_Duration_D5_95": d5_95,
            "Mean_Acceleration": mean_acc,
            "Std_Acceleration": std_acc,
            "RMS_Acceleration": rms_acc,
        })

        # Inject spectral/time-domain products into the spectral_data mapping
        new_spectral_data = dict(context.spectral_data)
        new_spectral_data["Husid_Curve"] = husid_curve

        # Menggunakan atribut state yang benar ('parameter_extraction')
        # Update processing state secara aman jika field-nya ada, abaikan jika tidak
        try:
            state = replace(context.processing_state, parameter_extraction=StageStatus.SUCCESS)
        except TypeError:
            try:
                state = replace(context.processing_state, parameters=StageStatus.SUCCESS)
            except TypeError:
                state = context.processing_state
        
        return context.with_state(
            metrics=new_metrics,
            spectral_data=new_spectral_data,
            processing_state=state
        ).add_history(
            step_name=self.name,
            details={"status": "SUCCESS", "PGA": pga, "PGV": pgv, "PGD": pgd}
        )