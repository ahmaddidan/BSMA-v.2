"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/orchestrator.py

Description: Master Pipeline Orchestrator for Seismic Data Processing.
Enforces the Single Source of Truth (SSOT) by strictly routing an immutable 
ProcessingContext through a sequence of geophysical operators.

Architectural Standard: Functional Composition (No side-effects)
"""

from __future__ import annotations

import logging
import time
from typing import Protocol, Sequence, Any, runtime_checkable

# Mengimpor ProcessingContext asli dari domain types
from core.types.context import ProcessingContext

__all__ = ["ProcessingStep", "PipelineOrchestrator"]


@runtime_checkable
class ProcessingStep(Protocol):
    """
    Interface mutlak untuk seluruh plugin pemrosesan BSMA (Taper, Filter, Integrator, dll).
    Setiap plugin wajib mematuhi kontrak ini.
    """
    
    @property
    def name(self) -> str:
        """Mengembalikan nama unik dari tahapan pemrosesan (e.g., 'Integration_Trapezoidal')."""
        ...
        
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Mengeksekusi algoritma geofisika secara matematis.
        WAJIB mengembalikan instance ProcessingContext baru (immutable transition).
        """
        ...


class PipelineOrchestrator:
    """
    Production-grade Seismic Pipeline Orchestrator.
    Menjalankan urutan ProcessingStep secara deterministik dengan proteksi kegagalan.
    """

    def __init__(
        self, 
        steps: Sequence[ProcessingStep], 
        logger: logging.Logger | None = None,
        halt_on_error: bool = True
    ) -> None:
        """
        Inisialisasi Orchestrator.

        Parameters
        ----------
        steps : Sequence[ProcessingStep]
            Daftar urutan langkah pemrosesan (pipeline).
        logger : logging.Logger | None, optional
            Injeksi logger.
        halt_on_error : bool, optional
            Jika True, pipeline akan berhenti total saat terjadi error pada salah satu step (Fail-Fast).
            Jika False, error dicatat dan mengembalikan konteks terakhir yang berhasil diproses.
        """
        self.steps = tuple(steps)  # Immutable sequence
        self.logger = logger or logging.getLogger(__name__)
        self.halt_on_error = halt_on_error

        self.logger.info(
            "PipelineOrchestrator diinisialisasi.",
            extra={"bsma_context": {"total_steps": len(self.steps), "halt_on_error": self.halt_on_error}}
        )

    def run(self, initial_context: ProcessingContext, identifier: str = "unknown") -> ProcessingContext:
        """
        Mengeksekusi seluruh pipeline secara sekuensial pada konteks awal.

        Parameters
        ----------
        initial_context : ProcessingContext
            State awal dari data seismik.
        identifier : str
            Identitas proses untuk keperluan tracing (e.g., Event ID atau Stream ID).

        Returns
        -------
        ProcessingContext
            State akhir (C_n) setelah melalui komposisi f(C).
        """
        if not self.steps:
            self.logger.warning("Pipeline dieksekusi tanpa tahapan (empty steps).")
            return initial_context

        current_context = initial_context
        total_start_time = time.perf_counter()

        self.logger.debug(
            f"Memulai eksekusi pipeline untuk identifier: {identifier}.",
            extra={"bsma_context": {"identifier": identifier}}
        )

        for step in self.steps:
            step_start = time.perf_counter()
            step_name = step.name
            
            try:
                self.logger.debug(f"Menjalankan tahapan: {step_name}")
                
                # Eksekusi fisis/matematis (State Transition: C_{i+1} = T(C_i))
                next_context = step.process(current_context)
                
                # Validasi arsitektural: Pastikan plugin mengembalikan konteks
                if next_context is None:
                    raise RuntimeError(f"Pelanggaran Arsitektur: {step_name} mengembalikan None, bukan ProcessingContext.")

                elapsed_ms = (time.perf_counter() - step_start) * 1000.0
                
                # Rekam metrik deterministik ke dalam konteks baru
                current_context = next_context.add_history(
                    step_name=step_name,
                    details={"status": "SUCCESS", "execution_time_ms": round(elapsed_ms, 3)}
                )

            except Exception as e:
                elapsed_ms = (time.perf_counter() - step_start) * 1000.0
                err_msg = f"Kegagalan fatal pada tahapan '{step_name}': {str(e)}"
                
                self.logger.error(
                    err_msg,
                    exc_info=True,
                    extra={
                        "bsma_context": {
                            "identifier": identifier,
                            "failed_step": step_name,
                            "execution_time_ms": round(elapsed_ms, 3)
                        }
                    }
                )

                if self.halt_on_error:
                    raise RuntimeError(err_msg) from e
                
                # Jika tidak halt, hentikan iterasi dan kembalikan state terakhir yang valid
                self.logger.warning(f"halt_on_error=False, menghentikan sisa eksekusi pipeline untuk {identifier}.")
                break

        total_elapsed_ms = (time.perf_counter() - total_start_time) * 1000.0
        
        self.logger.info(
            f"Pipeline selesai dieksekusi untuk {identifier}.",
            extra={
                "bsma_context": {
                    "identifier": identifier,
                    "total_execution_time_ms": round(total_elapsed_ms, 3),
                    "steps_completed": len(self.steps)
                }
            }
        )

        return current_context