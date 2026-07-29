"""
BMKG Strong Motion Analyzer (BSMA)
Model: Response Spectrum Calculator (Version 4.0 Nigam-Jennings Enabled)

Orkestrator kalkulasi spektrum respons yang menggunakan solver eksak Nigam & Jennings (1969)
sebagai standar utama, sejalan dengan praktik terbaik observatorium gempa internasional.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar
import numpy as np
from obspy import Stream, Trace
from obspy.core.util.attribdict import AttribDict
from utils.exceptions import ErrorCode, SeverityLevel, WaveformError
from utils.logger import setup_logger
from core.sdof.nigam_jennings import solve_nigam_jennings
from core.sdof.newmark import solve_newmark

__all__ = [
    "ResponseSpectrumConfig",
    "SpectrumData",
    "ResponseSpectrumReport",
    "ResponseSpectrumCalculator",
]

@dataclass(slots=True)
class ResponseSpectrumConfig:
    """Konfigurasi parameter untuk kalkulasi Response Spectrum."""
    damping: float = 0.05
    periods: np.ndarray | list[float] | None = None
    t_min: float = 0.01
    t_max: float = 10.0
    num_periods: int = 100
    solver_algorithm: str = "nigam_jennings"  # Default ke Nigam-Jennings (Gold Standard)
    strict_prerequisites: bool = False


@dataclass(slots=True)
class SpectrumData:
    """Kontainer hasil spektrum respons untuk satu trace."""
    trace_id: str
    periods: list[float] = field(default_factory=list)
    psa: list[float] = field(default_factory=list)
    psv: list[float] = field(default_factory=list)
    sd: list[float] = field(default_factory=list)
    
    max_psa: float = 0.0
    max_psv: float = 0.0
    max_sd: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "max_psa": round(self.max_psa, 6),
            "max_psv": round(self.max_psv, 6),
            "max_sd": round(self.max_sd, 6),
            "warnings": self.warnings,
            "periods": self.periods,
            "psa": [round(v, 6) for v in self.psa],
            "psv": [round(v, 6) for v in self.psv],
            "sd": [round(v, 6) for v in self.sd]
        }


@dataclass(slots=True)
class ResponseSpectrumReport:
    """Laporan komprehensif spektral respons level Stream."""
    stream_id: str
    damping: float
    is_passed: bool = True
    spectrums: dict[str, SpectrumData] = field(default_factory=dict)
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "damping": self.damping,
            "is_passed": self.is_passed,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "spectrums": {k: v.to_dict() for k, v in self.spectrums.items()}
        }


class ResponseSpectrumCalculator:
    """Production-grade Response Spectrum Calculator (Nigam-Jennings Powered)."""

    SUPPORTED_ALGORITHMS: ClassVar[tuple[str, ...]] = ("nigam_jennings", "newmark")

    def __init__(
        self, 
        config: ResponseSpectrumConfig | None = None,
        logger: logging.Logger | None = None
    ) -> None:
        self.config = config or ResponseSpectrumConfig()
        self.logger = logger or setup_logger(__name__)
        
        self._validate_config()
        
        self.logger.info(
            "ResponseSpectrumCalculator diinisialisasi.",
            extra={"bsma_context": {"module": "response_spectrum", "damping": self.config.damping, "solver": self.config.solver_algorithm}}
        )

    def _validate_config(self) -> None:
        c = self.config
        if not (0.0 <= c.damping < 1.0):
            raise ValueError("Damping ratio harus berada di antara 0.0 dan 1.0.")
        if c.solver_algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Solver '{c.solver_algorithm}' tidak didukung. Pilih dari: {self.SUPPORTED_ALGORITHMS}")
        
        if c.periods is not None:
            arr = np.asarray(c.periods, dtype=np.float64)
            if len(arr) == 0:
                raise ValueError("Array periods tidak boleh kosong.")
            if not np.all(np.diff(arr) > 0):
                raise ValueError("Array periods harus berurutan secara monoton naik (ascending).")
        else:
            if c.t_min <= 0 or c.t_max <= c.t_min:
                raise ValueError("Rentang periode T_min dan T_max tidak valid.")

    def _get_periods_array(self) -> np.ndarray:
        if self.config.periods is not None:
            return np.asarray(self.config.periods, dtype=np.float64)
        return np.logspace(np.log10(self.config.t_min), np.log10(self.config.t_max), self.config.num_periods)

    def _add_provenance(self, trace: Trace, step: str, details: dict[str, Any]) -> None:
        if "bsma_history" not in trace.stats:
            trace.stats.bsma_history = []
        if "bsma_config" not in trace.stats:
            trace.stats.bsma_config = AttribDict()
        trace.stats.bsma_history.append({"step": step, **details})

    def process(self, stream: Stream, identifier: str) -> ResponseSpectrumReport:
        """Menghitung spektral respons struktur dengan solver pilihan (Default: Nigam-Jennings)."""
        if not stream or len(stream) == 0:
            raise WaveformError(
                message="Stream kosong, kalkulasi response spectrum dibatalkan.",
                error_code=ErrorCode.WF003,
                severity=SeverityLevel.ERROR,
                context={"module": "response_spectrum", "identifier": identifier}
            )

        start_time = time.perf_counter()
        report = ResponseSpectrumReport(stream_id=identifier, damping=self.config.damping)
        periods = self._get_periods_array()

        try:
            for trace in stream:
                trace_warnings = []
                stats = trace.stats

                # 1. Validasi Unit Fisik Wajib (m/s^2)
                unit = getattr(stats, "bsma_current_unit", getattr(stats, "bsma_unit", "unknown"))
                if unit != "m/s^2":
                    raise WaveformError(
                        message=f"Trace {trace.id} memiliki satuan '{unit}', response spectrum menuntut 'm/s^2'.",
                        error_code=ErrorCode.WF004,
                        severity=SeverityLevel.ERROR,
                        context={"module": "response_spectrum", "trace_id": trace.id, "unit": unit}
                    )

                # 2. Pertahanan Lapis Kedua: Cek NaN / Inf
                if trace.data.size == 0 or not np.isfinite(trace.data).all():
                    raise WaveformError(
                        message=f"Trace {trace.id} mengandung NaN, Inf, atau data kosong.",
                        error_code=ErrorCode.WF003,
                        severity=SeverityLevel.CRITICAL,
                        context={"module": "response_spectrum", "trace_id": trace.id}
                    )

                # 3. Pengecekan Prasyarat Pipeline
                history = getattr(stats, "bsma_history", [])
                has_baseline_step = any(isinstance(h, dict) and "baseline" in h.get("step", "") for h in history) or \
                                    any(isinstance(h, str) and "baseline" in h for h in history)
                
                if not has_baseline_step:
                    msg = "Trace belum melewati tahap koreksi baseline. Spektrum rentan terhadap drift frekuensi rendah."
                    if self.config.strict_prerequisites:
                        raise WaveformError(message=msg, error_code=ErrorCode.WF004, severity=SeverityLevel.ERROR, context={"trace_id": trace.id})
                    else:
                        trace_warnings.append(msg)

                # 4. Validasi Sampling Terhadap Periode Minimum (Nyquist / Aliasing Check)
                dt = stats.delta
                min_period = periods[0]
                if dt > (min_period / 10.0):
                    msg = f"Sampling interval (dt={dt}s) terlalu kasar untuk periode minimum T={min_period}s (rekomendasi dt <= T/10)."
                    trace_warnings.append(msg)

                if trace.data.dtype != np.float64:
                    trace.data = trace.data.astype(np.float64)

                acc = trace.data
                pga = float(np.max(np.abs(acc)))

                psa_list, psv_list, sd_list = [], [], []

                for T in periods:
                    if self.config.solver_algorithm == "nigam_jennings":
                        sd_val, psv_val, psa_val = solve_nigam_jennings(acc, dt, T, self.config.damping)
                    else:
                        sd_val, psv_val, psa_val = solve_newmark(acc, dt, T, self.config.damping)
                        
                    sd_list.append(sd_val)
                    psv_list.append(psv_val)
                    psa_list.append(psa_val)

                # 5. Uji Konsistensi PGA (T mendekati 0 harus mendekati PGA)
                if len(psa_list) > 0:
                    diff_pga = abs(psa_list[0] - pga)
                    if diff_pga > (0.1 * pga) and pga > 1e-3:
                        trace_warnings.append(f"Deviasi PSA pada T terkecil ({psa_list[0]:.4f}) cukup jauh dari PGA ({pga:.4f}).")

                spec_data = SpectrumData(
                    trace_id=trace.id,
                    periods=periods.tolist(),
                    psa=psa_list,
                    psv=psv_list,
                    sd=sd_list,
                    max_psa=float(np.max(psa_list)) if psa_list else 0.0,
                    max_psv=float(np.max(psv_list)) if psv_list else 0.0,
                    max_sd=float(np.max(sd_list)) if sd_list else 0.0,
                    warnings=trace_warnings
                )
                report.spectrums[trace.id] = spec_data

                self._add_provenance(trace, "response_spectrum_calculated", {
                    "damping": self.config.damping,
                    "algorithm": self.config.solver_algorithm,
                    "num_periods": len(periods),
                    "max_psa": spec_data.max_psa,
                    "warnings_count": len(trace_warnings)
                })

        except WaveformError:
            raise
        except Exception as e:
            raise WaveformError(
                message="Gagal menghitung response spectrum.",
                error_code=ErrorCode.WF004,
                severity=SeverityLevel.SeverityLevel if 'SeverityLevel' in globals() else SeverityLevel.CRITICAL,
                context={"module": "response_spectrum", "identifier": identifier, "exception": str(e)}
            ) from e

        report.processing_time_ms = (time.perf_counter() - start_time) * 1000.0
        return report