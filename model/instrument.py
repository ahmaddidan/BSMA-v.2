"""
BMKG Strong Motion Analyzer (BSMA)
Model: Instrument Response Corrector (Version 2.0 Scientific Hardened)

Modul ini bertanggung jawab untuk mendekonvolusi respons instrumen (instrument response removal)
menggunakan metadata StationXML (Inventory). Dilengkapi dengan validasi Epoch, pre-filter adaptif,
water-level opsional, pemeriksaan NaN/Inf pasca-dekonvolusi, dan InstrumentReport.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar
import numpy as np
from obspy import Stream, Trace, Inventory
from obspy.core.util.attribdict import AttribDict
from utils.exceptions import ErrorCode, SeverityLevel, WaveformError
from utils.logger import setup_logger

__all__ = [
    "InstrumentOutput",
    "InstrumentConfig",
    "InstrumentTraceMetrics",
    "InstrumentReport",
    "InstrumentCorrector",
]

class InstrumentOutput(Enum):
    """Enumerasi standar target output dekonvolusi instrumen."""
    ACC = "ACC"   # m/s^2
    VEL = "VEL"   # m/s
    DISP = "DISP" # m


@dataclass(slots=True)
class InstrumentConfig:
    """Konfigurasi parameter untuk koreksi respons instrumen."""
    output_unit: InstrumentOutput = InstrumentOutput.ACC
    water_level: float | None = 60.0  # Opsional, dapat diset None jika tidak diinginkan
    strict_matching: bool = True      # True = Gagal jika Inventory/Epoch tidak cocok. False = Lewati.
    taper: bool = True                # Tapering otomatis sebelum dekonvolusi
    taper_fraction: float = 0.05      # Fraksi taper (5%)
    
    # Pre-filter adaptif opsional (jika None, dihitung otomatis aman terhadap Nyquist)
    pre_filt: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class InstrumentTraceMetrics:
    """Metrik hasil dekonvolusi untuk satu trace spesifik."""
    trace_id: str
    success: bool = True
    output_unit: str = ""
    epoch_used: str = "Unknown"
    processing_time_ms: float = 0.0
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "success": self.success,
            "output_unit": self.output_unit,
            "epoch_used": self.epoch_used,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "warning": self.warning
        }


@dataclass(slots=True)
class InstrumentReport:
    """Laporan komprehensif koreksi instrumen level Stream."""
    stream_id: str
    is_passed: bool = True
    trace_metrics: dict[str, InstrumentTraceMetrics] = field(default_factory=dict)
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "is_passed": self.is_passed,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "trace_metrics": {k: v.to_dict() for k, v in self.trace_metrics.items()}
        }


class InstrumentCorrector:
    """
    Production-grade Instrument Response Corrector.
    Menjamin dekonvolusi berbasis StationXML yang valid secara epoch dan numerik.
    """

    def __init__(
        self, 
        config: InstrumentConfig | None = None,
        logger: logging.Logger | None = None
    ) -> None:
        self.config = config or InstrumentConfig()
        self.logger = logger or setup_logger(__name__)
        
        self._validate_config()
        
        self.logger.info(
            "InstrumentCorrector diinisialisasi.",
            extra={
                "bsma_context": {
                    "module": "instrument",
                    "output_unit": self.config.output_unit.value,
                    "strict_matching": self.config.strict_matching
                }
            }
        )

    def _validate_config(self) -> None:
        if self.config.water_level is not None and self.config.water_level <= 0:
            raise ValueError("Water level harus bernilai positif (dB).")

    def _add_provenance(self, trace: Trace, step: str, details: dict[str, Any]) -> None:
        if "bsma_history" not in trace.stats:
            trace.stats.bsma_history = []
        if "bsma_config" not in trace.stats:
            trace.stats.bsma_config = AttribDict()
        trace.stats.bsma_history.append({"step": step, **details})

    def process(self, stream: Stream, inventory: Inventory | None, identifier: str) -> tuple[Stream, InstrumentReport]:
        """
        Mengeksekusi dekonvolusi respons instrumen pada seluruh trace di dalam stream.
        """
        if not stream or len(stream) == 0:
            raise WaveformError(
                message="Stream kosong, koreksi instrumen dibatalkan.",
                error_code=ErrorCode.WF003,
                severity=SeverityLevel.ERROR,
                context={"module": "instrument", "identifier": identifier}
            )

        start_time = time.perf_counter()
        report = InstrumentReport(stream_id=identifier)

        if inventory is None:
            if self.config.strict_matching:
                raise WaveformError(
                    message="Inventory (StationXML) tidak disediakan pada mode strict.",
                    error_code=ErrorCode.WF004,
                    severity=SeverityLevel.CRITICAL,
                    context={"module": "instrument", "identifier": identifier}
                )
            else:
                self.logger.warning("Inventory tidak tersedia. Melewati koreksi respons instrumen.")
                return stream.copy(), report

        st_proc = stream.copy()

        self.logger.debug(
            "Memulai koreksi respons instrumen.",
            extra={"bsma_context": {"identifier": identifier, "trace_count": len(st_proc)}}
        )

        try:
            for trace in st_proc:
                t_start = time.perf_counter()
                metrics = InstrumentTraceMetrics(trace_id=trace.id, output_unit=self.config.output_unit.value)
                
                success, warning_msg, epoch_str = self._remove_response_trace(trace, inventory)
                metrics.success = success
                metrics.warning = warning_msg
                metrics.epoch_used = epoch_str
                metrics.processing_time_ms = (time.perf_counter() - t_start) * 1000.0
                
                report.trace_metrics[trace.id] = metrics
                if not success:
                    report.is_passed = False
                    
        except WaveformError:
            raise
        except Exception as e:
            raise WaveformError(
                message="Gagal mengeksekusi koreksi respons instrumen.",
                error_code=ErrorCode.WF004,
                severity=SeverityLevel.CRITICAL,
                context={"module": "instrument", "identifier": identifier, "exception": str(e)}
            ) from e

        report.processing_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        self.logger.info(
            "Koreksi respons instrumen selesai.",
            extra={"bsma_context": {"identifier": identifier, "is_passed": report.is_passed}}
        )
        return st_proc, report

    def _get_adaptive_pre_filt(self, trace: Trace) -> tuple[float, float, float, float]:
        """Menghitung pre-filter adaptif yang aman terhadap batas Nyquist trace."""
        if self.config.pre_filt is not None:
            return self.config.pre_filt
            
        nyquist = trace.stats.sampling_rate / 2.0
        f1 = 0.001
        f2 = 0.005
        f3 = min(40.0, nyquist * 0.9)
        f4 = min(45.0, nyquist * 0.95)
        
        # Pastikan urutan frekuensi logis
        if f3 >= f4:
            f3 = nyquist * 0.85
            f4 = nyquist * 0.95
            
        return (f1, f2, f3, f4)

    def _verify_epoch_matching(self, trace: Trace, inventory: Inventory) -> str:
        """Memeriksa apakah trace berada dalam rentang epoch stasiun di Inventory."""
        try:
            net = trace.stats.network
            sta = trace.stats.station
            loc = trace.stats.location
            chan = trace.stats.channel
            t_tr = trace.stats.starttime
            
            # Cari respons melalui fasilitas ObsPy select / get_response
            # ObsPy akan mencocokkan epoch secara otomatis
            resp = inventory.get_response(f"{net}.{sta}.{loc}.{chan}", t_tr)
            return f"Matched (Epoch valid untuk {t_tr.isoformat()})"
        except Exception:
            return "Epoch mismatch / Response not found in Inventory"

    def _remove_response_trace(self, trace: Trace, inventory: Inventory) -> tuple[bool, str | None, str]:
        """Mengeksekusi remove_response dengan validasi epoch dan pengecekan NaN pasca-proses."""
        epoch_status = self._verify_epoch_matching(trace, inventory)
        if "mismatch" in epoch_status.lower() and self.config.strict_matching:
            return False, epoch_status, epoch_status

        pre_filt = self._get_adaptive_pre_filt(trace)

        try:
            if trace.data.dtype != np.float64:
                trace.data = trace.data.astype(np.float64)

            # Eksekusi dekonvolusi ObsPy
            trace.remove_response(
                inventory=inventory,
                output=self.config.output_unit.value,
                pre_filt=pre_filt,
                water_level=self.config.water_level,
                taper=self.config.taper,
                taper_fraction=self.config.taper_fraction,
                zero_mean=True
            )

            # Validasi numerik pasca-proses: Apakah menghasilkan NaN atau Inf?
            if not np.isfinite(trace.data).all():
                return False, "Hasil remove_response menghasilkan nilai NaN atau Inf (kemungkinan water_level atau pre_filt tidak sesuai).", epoch_status

            # Tandai metadata provenance
            trace.stats.bsma_instrument_corrected = True
            trace.stats.bsma_original_unit = "raw_counts"
            trace.stats.bsma_current_unit = self.config.output_unit.value
            
            self._add_provenance(trace, "instrument_response_removal", {
                "output_unit": self.config.output_unit.value,
                "water_level": self.config.water_level,
                "pre_filt": pre_filt,
                "epoch_status": epoch_status
            })

            return True, None, epoch_status

        except Exception as e:
            err_msg = str(e)
            if self.config.strict_matching:
                raise WaveformError(
                    message=f"Gagal dekonvolusi trace {trace.id}: {err_msg}",
                    error_code=ErrorCode.WF004,
                    severity=SeverityLevel.CRITICAL,
                    context={"module": "instrument", "trace_id": trace.id}
                ) from e
            else:
                return False, err_msg, epoch_status