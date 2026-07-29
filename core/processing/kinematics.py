"""
BMKG Strong Motion Analyzer (BSMA)
core/processing/kinematics.py
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from dataclasses import replace
from core.types.context import ProcessingContext, SeverityLevel
from core.interfaces.preprocessor import PreprocessorPlugin

class KinematicIntegrationPlugin(PreprocessorPlugin):
    """
    Plugin untuk melakukan integrasi numerik dari percepatan ke kecepatan,
    dan dari kecepatan ke perpindahan menggunakan metode trapesium.
    """
    @property
    def plugin_name(self) -> str:
        return "Kinematic Integration"

    def process(self, context: ProcessingContext) -> ProcessingContext:
        try:
            # Menghitung interval waktu (dt)
            dt = 1.0 / context.metadata.sampling_rate
            
            # Integrasi ke-1: Percepatan -> Kecepatan
            # cumulative_trapezoid mempertahankan panjang array dengan parameter initial=0.0
            velocity = cumulative_trapezoid(context.data, dx=dt, initial=0.0)
            
            # Integrasi ke-2: Kecepatan -> Perpindahan
            displacement = cumulative_trapezoid(velocity, dx=dt, initial=0.0)
            
            # Mencatat keberhasilan ke dalam sistem QC
            new_qc = context.qc_report.add_message(
                SeverityLevel.INFO,
                f"{self.plugin_name}: Integrasi numerik sukses dieksekusi (dt={dt}s)."
            )
            
            # Mengembalikan objek Context baru yang immutable
            return replace(
                context, 
                velocity=velocity, 
                displacement=displacement, 
                qc_report=new_qc
            )
            
        except Exception as e:
            # Fail-fast trigger jika integrasi gagal
            new_qc = context.qc_report.add_message(
                SeverityLevel.ERROR,
                f"{self.plugin_name}: Gagal melakukan integrasi numerik - {str(e)}"
            )
            return replace(context, qc_report=new_qc)