"""
BMKG Strong Motion Analyzer (BSMA)
Model: Physical Unit & Amplitude Validator (Version 3.0 Production-Grade Scientific)

Modul ini bertindak sebagai gerbang validasi akhir untuk memastikan bahwa data seismik 
telah sukses dikonversi ke satuan fisik melalui Inventory (Instrument Correction).
Dilengkapi dengan pemisahan status FAILED vs WARNING, daftar peringatan multi-isu, 
pemeriksaan NaN/Inf pertahanan lapis kedua, dan provenance validasi.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar
import numpy as np
from obspy import Stream, Trace
from obspy.core.util.attribdict import AttribDict
from utils.exceptions import ErrorCode, SeverityLevel, WaveformError
from utils.logger import setup_logger

__all__ = [
    "PhysicalUnit",
    "ValidationStatus",
    "UnitValidatorConfig",
    "TraceUnitMetrics",
    "UnitValidationReport",
    "PhysicalUnitValidator",
]

class PhysicalUnit(Enum):
    """Enumerasi standar satuan fisik output instrumen."""
    M_S2 = "m/s^2"  # Akselerasi (SI)
    M_S = "m/s"     # Kecepatan
    M = "m"         # Perpindahan
    UNKNOWN = "unknown"


class ValidationStatus(Enum):
    """Status hasil validasi trace."""
    VALID = "VALID"
    WARNING = "WARNING"
    FAILED = "FAILED"


@dataclass(slots=True)
class UnitValidatorConfig:
    """Konfigurasi parameter untuk validator unit."""
    expected_unit: PhysicalUnit = PhysicalUnit.M_S2
    
    # Kebijakan ketat: Tolak trace jika belum melalui instrument correction
    require_instrument_correction: bool = True
    
    # Ambang batas peringatan amplitudo (Warning, bukan Failure)
    # Gempa besar (misal: Chi-Chi, Tohoku) bisa menghasilkan PGA > 1g (9.8 m/s^2) hingga ~30 m/s^2.
    # Nilai di atas ini masuk kategori WARNING (perlu perhatian khusus, tapi tetap diproses).
    warn_acceleration_ms2: float = 20.0  # ~2g
    warn_velocity_ms: float = 2.0        # 2 m/s
    warn_displacement_m: float = 1.0     # 1 meter


@dataclass(slots=True)
class TraceUnitMetrics:
    """Metrik hasil validasi unit dan amplitudo untuk satu trace."""
    trace_id: str
    status: ValidationStatus = ValidationStatus.VALID
    is_valid: bool = True  # True jika status VALID atau WARNING (bukan FAILED)
    current_unit: str = ""
    original_unit: str = ""
    instrument_corrected: bool = False
    dtype: str = ""
    
    # Statistik Amplitudo Komprehensif
    peak_amplitude: float = 0.0
    rms_amplitude: float = 0.0
    median_amplitude: float = 0.0
    p99_amplitude: float = 0.0  # 99th percentile untuk meredam false-alarm single spike
    
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "status": self.status.value,
            "is_valid": self.is_valid,
            "current_unit": self.current_unit,
            "original_unit": self.original_unit,
            "instrument_corrected": self.instrument_corrected,
            "dtype": self.dtype,
            "peak_amplitude": round(self.peak_amplitude, 6),
            "rms_amplitude": round(self.rms_amplitude, 6),
            "median_amplitude": round(self.median_amplitude, 6),
            "p99_amplitude": round(self.p99_amplitude, 6),
            "warnings": self.warnings,
            "failures": self.failures
        }


@dataclass(slots=True)
class UnitValidationReport:
    """Laporan komprehensif validasi unit level Stream."""
    stream_id: str
    is_passed: bool = True
    trace_metrics: dict[str, TraceUnitMetrics] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    validator_version: str = "3.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "is_passed": self.is_passed,
            "validator_version": self.validator_version,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "trace_metrics": {k: v.to_dict() for k, v in self.trace_metrics.items()}
        }


class PhysicalUnitValidator:
    """
    Production-grade Physical Unit & Amplitude Validator (Scientific Edition).
    Memisahkan status FAILED (kerusakan fatal) dan WARNING (amplitudo ekstrem valid).
    """

    def __init__(
        self, 
        config: UnitValidatorConfig | None = None,
        logger: logging.Logger | None = None
    ) -> None:
        self.config = config or UnitValidatorConfig()
        self.logger = logger or setup_logger(__name__)
        
        self.logger.info(
            "PhysicalUnitValidator diinisialisasi.",
            extra={"bsma_context": {"module": "unit_validator", "expected_unit": self.config.expected_unit.value}}
        )

    def _add_provenance(self, trace: Trace, step: str, details: dict[str, Any]) -> None:
        if "bsma_history" not in trace.stats:
            trace.stats.bsma_history = []
        if "bsma_config" not in trace.stats:
            trace.stats.bsma_config = AttribDict()
        trace.stats.bsma_history.append({"step": step, **details})

    def process(self, stream: Stream, identifier: str) -> tuple[Stream, UnitValidationReport]:
        """
        Memvalidasi satuan fisik dan batas kewarasan amplitudo seluruh trace di dalam stream.
        """
        if not stream or len(stream) == 0:
            raise WaveformError(
                message="Stream kosong, validasi unit dibatalkan.",
                error_code=ErrorCode.WF003,
                severity=SeverityLevel.ERROR,
                context={"module": "unit_validator", "identifier": identifier}
            )

        start_time = time.perf_counter()
        st_proc = stream.copy()
        report = UnitValidationReport(stream_id=identifier)

        self.logger.debug(
            "Memulai validasi unit fisik dan amplitudo.",
            extra={"bsma_context": {"identifier": identifier, "trace_count": len(st_proc)}}
        )

        try:
            for trace in st_proc:
                metrics = self._validate_trace(trace)
                report.trace_metrics[trace.id] = metrics
                
                if not metrics.is_valid:
                    report.is_passed = False
                    
        except WaveformError:
            raise
        except Exception as e:
            raise WaveformError(
                message="Gagal melakukan validasi unit fisik.",
                error_code=ErrorCode.WF004,
                severity=SeverityLevel.CRITICAL,
                context={"module": "unit_validator", "identifier": identifier, "exception": str(e)}
            ) from e

        report.processing_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        self.logger.info(
            "Validasi unit fisik selesai.",
            extra={"bsma_context": {"identifier": identifier, "is_passed": report.is_passed}}
        )
        return st_proc, report

    def _validate_trace(self, trace: Trace) -> TraceUnitMetrics:
        """Mengevaluasi status dekonvolusi, metadata unit, dan uji kewarasan amplitudo multi-metrik."""
        stats = trace.stats
        metrics = TraceUnitMetrics(trace_id=trace.id)
        metrics.dtype = str(trace.data.dtype)

        # 0. Pertahanan Lapis Kedua: Cek NaN / Inf
        if trace.data.size == 0 or not np.isfinite(trace.data).all():
            metrics.status = ValidationStatus.FAILED
            metrics.is_valid = False
            metrics.failures.append("Data trace mengandung nilai NaN, Inf, atau kosong.")
            return metrics

        # 1. Periksa status dekonvolusi instrumen
        is_corrected = getattr(stats, "bsma_instrument_corrected", False)
        metrics.instrument_corrected = is_corrected

        if self.config.require_instrument_correction and not is_corrected:
            metrics.status = ValidationStatus.FAILED
            metrics.is_valid = False
            metrics.failures.append("Trace belum melalui dekonvolusi instrumen (bsma_instrument_corrected bernilai False).")
            return metrics

        # 2. Periksa kesesuaian unit saat ini & Metadata Missing Check
        current_unit = getattr(stats, "bsma_current_unit", None)
        original_unit = getattr(stats, "bsma_original_unit", "raw_counts")
        metrics.original_unit = original_unit

        if current_unit is None:
            metrics.status = ValidationStatus.FAILED
            metrics.is_valid = False
            metrics.failures.append("Metadata unit fisik hilang (bsma_current_unit tidak ditemukan pada trace stats).")
            current_unit = PhysicalUnit.UNKNOWN.value
        
        metrics.current_unit = current_unit

        if current_unit == PhysicalUnit.UNKNOWN.value:
            metrics.status = ValidationStatus.FAILED
            metrics.is_valid = False
            metrics.failures.append("Satuan fisik trace terdeteksi sebagai UNKNOWN.")
            return metrics

        if current_unit != self.config.expected_unit.value:
            metrics.warnings.append(f"Satuan saat ini ({current_unit}) berbeda dari target yang diharapkan ({self.config.expected_unit.value}).")

        # 3. Kalkulasi Statistik Amplitudo Komprehensif
        data = trace.data
        metrics.peak_amplitude = float(np.max(np.abs(data)))
        metrics.rms_amplitude = float(np.sqrt(np.mean(data**2)))
        metrics.median_amplitude = float(np.median(np.abs(data)))
        metrics.p99_amplitude = float(np.percentile(np.abs(data), 99))

        # 4. Amplitude Sanity Check (Berbasis WARNING, bukan FAILED, kecuali tidak masuk akal mutlak)
        if current_unit == PhysicalUnit.M_S2.value:
            if metrics.peak_amplitude > self.config.warn_acceleration_ms2:
                metrics.warnings.append(f"Amplitudo akselerasi sangat tinggi ({metrics.peak_amplitude:.2f} m/s^2 > limit warning). Potensi gempa besar atau transien ekstrem.")
        elif current_unit == PhysicalUnit.M_S.value:
            if metrics.peak_amplitude > self.config.warn_velocity_ms:
                metrics.warnings.append(f"Amplitudo kecepatan sangat tinggi ({metrics.peak_amplitude:.2f} m/s > limit warning).")
        elif current_unit == PhysicalUnit.M.value:
            if metrics.peak_amplitude > self.config.warn_displacement_m:
                metrics.warnings.append(f"Amplitudo perpindahan sangat tinggi ({metrics.peak_amplitude:.2f} m > limit warning).")

        # Tentukan status akhir berdasarkan daftar kegagalan (failures)
        if metrics.failures:
            metrics.status = ValidationStatus.FAILED
            metrics.is_valid = False
        elif metrics.warnings:
            metrics.status = ValidationStatus.WARNING
            metrics.is_valid = True  # Warning tetap dianggap sah untuk diproses (Valid)
        else:
            metrics.status = ValidationStatus.VALID
            metrics.is_valid = True

        # 5. Tandai Metadata Provenance Validasi Berhasil
        stats.bsma_unit_validated = True
        
        self._add_provenance(trace, "physical_unit_validation", {
            "status": metrics.status.value,
            "unit": current_unit,
            "peak_amplitude": metrics.peak_amplitude,
            "warnings": metrics.warnings,
            "failures": metrics.failures
        })

        return metrics