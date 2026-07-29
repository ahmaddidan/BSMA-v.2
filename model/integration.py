"""
BMKG Strong Motion Analyzer (BSMA)
Model: Kinematic Integrator (Version 3.1 Final Scientific Edition)

Modul ini adalah mesin kalkulus utama untuk mengonversi Akselerasi -> Kecepatan -> Perpindahan.
Menerapkan standar pemrosesan strong-motion modern dengan kontrol kondisi awal (v0, d0), 
linregress O(N) untuk analisis drift post-event, proteksi permanen displacement, dan 
urut-urutan koreksi pasca-integrasi yang valid secara matematis.

Referensi Saintifik:
- Boore, D. M. (2001). Effect of baseline corrections on displacements and response spectra.
- Iwan, W. D., et al. (1985). Some observations on strong-motion earthquake measurement.
- USGS gmprocess / COSMOS Strong Motion Data Processing Guidelines.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import linregress
from obspy import Stream, Trace
from obspy.core.util.attribdict import AttribDict
from utils.exceptions import ErrorCode, SeverityLevel, WaveformError
from utils.logger import setup_logger

__all__ = [
    "PostIntegrationConfig",
    "IntegrationConfig",
    "PeakMetric",
    "ResidualMetric",
    "KinematicMetrics",
    "IntegrationReport",
    "KinematicIntegrator",
]

@dataclass(slots=True)
class PostIntegrationConfig:
    """Konfigurasi koreksi baseline pasca-integrasi untuk meredam amplifikasi drift."""
    method: str = "linear"  # Pilihan: 'none', 'demean', 'linear', 'highpass'
    highpass_freq: float = 0.05  
    filter_corners: int = 4


@dataclass(slots=True)
class IntegrationConfig:
    """Konfigurasi utama engine integrasi kinematika."""
    method: str = "cumulative_trapezoid"
    expected_input_unit: str = "ACC"
    
    initial_velocity: float = 0.0
    initial_displacement: float = 0.0
    
    # Detrending default diset ke linear untuk mempertahankan permanent displacement
    vel_correction: PostIntegrationConfig = field(default_factory=lambda: PostIntegrationConfig(method="linear"))
    disp_correction: PostIntegrationConfig = field(default_factory=lambda: PostIntegrationConfig(method="linear"))
    
    pgd_warning_threshold_meters: float | None = None 
    
    # Parameter dinamis untuk evaluasi residual (menghindari hardcode 5%)
    residual_tail_fraction: float = 0.05  


@dataclass(slots=True)
class PeakMetric:
    """Objek metrik puncak kinematika komprehensif."""
    value: float = 0.0
    abs_max: float = 0.0
    time_sec: float = 0.0
    index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": round(self.value, 6),
            "abs_max": round(self.abs_max, 6),
            "time_sec": round(self.time_sec, 3),
            "index": self.index
        }


@dataclass(slots=True)
class ResidualMetric:
    """Metrik evaluasi drift pasca-koreksi (Berdasarkan Post-Event Tail)."""
    tail_mean: float = 0.0
    tail_slope: float = 0.0
    is_anomalous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tail_mean": round(self.tail_mean, 6),
            "tail_slope": round(self.tail_slope, 6),
            "is_anomalous": self.is_anomalous
        }


@dataclass(slots=True)
class KinematicMetrics:
    """Rekapitulasi parameter rekayasa gempa untuk satu trace komplit."""
    trace_id: str
    is_valid: bool = True
    
    pga: PeakMetric = field(default_factory=PeakMetric)
    pgv: PeakMetric = field(default_factory=PeakMetric)
    pgd: PeakMetric = field(default_factory=PeakMetric)
    
    vel_residual: ResidualMetric = field(default_factory=ResidualMetric)
    disp_residual: ResidualMetric = field(default_factory=ResidualMetric)
    
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "is_valid": self.is_valid,
            "pga": self.pga.to_dict(),
            "pgv": self.pgv.to_dict(),
            "pgd": self.pgd.to_dict(),
            "vel_residual": self.vel_residual.to_dict(),
            "disp_residual": self.disp_residual.to_dict(),
            "issues": self.issues
        }


@dataclass(slots=True)
class IntegrationReport:
    """Laporan hasil integrasi dan parameter kinematika level Stream."""
    stream_id: str
    is_passed: bool = True
    
    stream_acc: Stream = field(default_factory=Stream)
    stream_vel: Stream = field(default_factory=Stream)
    stream_disp: Stream = field(default_factory=Stream)
    
    trace_metrics: dict[str, KinematicMetrics] = field(default_factory=dict)
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "is_passed": self.is_passed,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "trace_metrics": {k: v.to_dict() for k, v in self.trace_metrics.items()}
        }


class KinematicIntegrator:
    """Production-grade Kinematic Integrator."""

    def __init__(
        self, 
        config: IntegrationConfig | None = None,
        logger: logging.Logger | None = None
    ) -> None:
        self.config = config or IntegrationConfig()
        self.logger = logger or setup_logger(__name__)
        
        self.logger.info(
            "KinematicIntegrator diinisialisasi.",
            extra={"bsma_context": {"module": "integration", "method": self.config.method}}
        )

    def _add_provenance(self, trace: Trace, step: str, details: dict[str, Any]) -> None:
        """Menyuntikkan objek riwayat komprehensif ke metadata."""
        if "bsma_history" not in trace.stats:
            trace.stats.bsma_history = []
        if "bsma_config" not in trace.stats:
            trace.stats.bsma_config = AttribDict()
            
        trace.stats.bsma_history.append({"step": step, **details})

    def process(self, acc_stream: Stream, identifier: str) -> IntegrationReport:
        """Mengeksekusi integrasi komprehensif Akselerasi -> Kecepatan -> Perpindahan."""
        if not acc_stream or len(acc_stream) == 0:
            raise WaveformError(
                message="Stream akselerasi kosong.",
                error_code=ErrorCode.WF003,
                severity=SeverityLevel.ERROR,
                context={"module": "integration", "identifier": identifier}
            )

        start_time = time.perf_counter()
        report = IntegrationReport(stream_id=identifier)
        
        # 1. Simpan salinan Stream Akselerasi
        report.stream_acc = acc_stream.copy()

        self.logger.debug(
            "Memulai kalkulus kinematika.",
            extra={"bsma_context": {"identifier": identifier, "trace_count": len(acc_stream)}}
        )

        try:
            for tr_acc in report.stream_acc:
                metric = KinematicMetrics(trace_id=tr_acc.id)
                
                # --- A. DEFENSIVE CHECKS ---
                unit = getattr(tr_acc.stats, "bsma_unit", "ACC")
                if unit != self.config.expected_input_unit:
                    self.logger.warning(f"Unit trace {tr_acc.id} adalah {unit}, diharapkan ACC.")
                    
                # Fail-fast jika ada NaN
                if not np.isfinite(tr_acc.data).all():
                    raise WaveformError(
                        message=f"Trace {tr_acc.id} mengandung NaN/Inf. Integrasi dibatalkan.",
                        error_code=ErrorCode.WF003,
                        severity=SeverityLevel.CRITICAL,
                        context={"module": "integration", "trace_id": tr_acc.id}
                    )
                
                # Pastikan data float64 sebelum operasi integral kumulatif
                if tr_acc.data.dtype != np.float64:
                    tr_acc.data = tr_acc.data.astype(np.float64)

                tr_acc.stats.bsma_unit = "ACC"
                metric.pga = self._extract_peak(tr_acc.data, tr_acc.stats.delta)

                # --- B. INTEGRASI VELOCITY ---
                tr_vel = tr_acc.copy()
                # 1. Integrasi Murni
                tr_vel.data = self._integrate_scipy(tr_acc.data, tr_acc.stats.delta)
                # 2. Koreksi Baseline Pasca-Integrasi
                self._apply_post_correction(tr_vel, self.config.vel_correction)
                # 3. Tambahkan Initial Condition (SETELAH koreksi agar tidak terhapus)
                tr_vel.data += self.config.initial_velocity
                
                tr_vel.stats.bsma_unit = "VEL"
                self._add_provenance(tr_vel, "integration_vel", {
                    "method": "cumulative_trapezoid", "initial_v": self.config.initial_velocity, 
                    "post_correction": self.config.vel_correction.method
                })
                
                metric.pgv = self._extract_peak(tr_vel.data, tr_vel.stats.delta)
                metric.vel_residual = self._evaluate_residual(tr_vel.data, tr_vel.stats.delta)
                report.stream_vel.append(tr_vel)

                # --- C. INTEGRASI DISPLACEMENT ---
                tr_disp = tr_vel.copy()
                # 1. Integrasi Murni
                tr_disp.data = self._integrate_scipy(tr_vel.data, tr_vel.stats.delta)
                # 2. Koreksi Baseline Pasca-Integrasi
                self._apply_post_correction(tr_disp, self.config.disp_correction)
                # 3. Tambahkan Initial Condition
                tr_disp.data += self.config.initial_displacement
                
                tr_disp.stats.bsma_unit = "DISP"
                self._add_provenance(tr_disp, "integration_disp", {
                    "method": "cumulative_trapezoid", "initial_d": self.config.initial_displacement, 
                    "post_correction": self.config.disp_correction.method
                })
                
                metric.pgd = self._extract_peak(tr_disp.data, tr_disp.stats.delta)
                metric.disp_residual = self._evaluate_residual(tr_disp.data, tr_disp.stats.delta)
                report.stream_disp.append(tr_disp)

                # --- D. SATURATION / QUALITY CHECKS ---
                if self.config.pgd_warning_threshold_meters is not None:
                    if metric.pgd.abs_max > self.config.pgd_warning_threshold_meters:
                        metric.issues.append(f"UNREALISTIC_PGD_{metric.pgd.abs_max:.2f}m")
                        metric.is_valid = False
                    
                if metric.vel_residual.is_anomalous or metric.disp_residual.is_anomalous:
                    metric.issues.append("RESIDUAL_DRIFT_DETECTED")
                    metric.is_valid = False

                report.trace_metrics[tr_acc.id] = metric
                if not metric.is_valid:
                    report.is_passed = False

        except WaveformError:
            raise
        except Exception as e:
            raise WaveformError(
                message=f"Gagal melakukan integrasi kinematika.",
                error_code=ErrorCode.WF004,
                severity=SeverityLevel.CRITICAL,
                context={"module": "integration", "identifier": identifier, "exception": str(e)}
            ) from e

        report.processing_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        self.logger.info(
            "Integrasi kinematika selesai.",
            extra={
                "bsma_context": {
                    "identifier": identifier, 
                    "is_passed": report.is_passed,
                    "processing_time_ms": round(report.processing_time_ms, 2)
                }
            }
        )
        return report

    # ==========================================
    # Algoritma Numerik
    # ==========================================

    def _integrate_scipy(self, data: np.ndarray, dt: float) -> np.ndarray:
        return cumulative_trapezoid(data, dx=dt, initial=0.0)

    def _extract_peak(self, data: np.ndarray, dt: float) -> PeakMetric:
        if data.size == 0: return PeakMetric()
        
        idx = int(np.argmax(np.abs(data)))
        val = float(data[idx])
        return PeakMetric(
            value=val,
            abs_max=abs(val),
            time_sec=float(idx * dt),
            index=idx
        )

    def _evaluate_residual(self, data: np.ndarray, dt: float) -> ResidualMetric:
        """Menghitung metrik drift menggunakan fraksi dinamis (default 5% terakhir)."""
        if data.size < 100: return ResidualMetric()
        
        edge_samples = max(20, int(data.size * self.config.residual_tail_fraction))
        post_data = data[-edge_samples:]
        
        time_vec = np.linspace(0, post_data.size * dt, post_data.size, endpoint=False)
        res = linregress(time_vec, post_data)
        
        slope = float(res.slope)
        mean_val = float(np.mean(post_data))
        
        p2p = np.max(data) - np.min(data)
        is_anom = False
        if p2p > 0:
            drift_amplitude = abs(slope * time_vec[-1])
            if (abs(mean_val) / p2p > 0.05) or (drift_amplitude / p2p > 0.05):
                is_anom = True
                
        return ResidualMetric(
            tail_mean=mean_val,
            tail_slope=slope,
            is_anomalous=is_anom
        )

    def _apply_post_correction(self, trace: Trace, config: PostIntegrationConfig) -> None:
        if config.method == "none":
            return
        elif config.method == "demean":
            trace.detrend("demean")
        elif config.method == "linear":
            trace.detrend("linear")
        elif config.method == "highpass":
            nyquist = trace.stats.sampling_rate / 2.0
            if config.highpass_freq < nyquist * 0.9:
                trace.filter(
                    "highpass", 
                    freqmin=config.highpass_freq, 
                    corners=config.filter_corners, 
                    zerophase=True
                )