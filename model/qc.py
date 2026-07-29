"""
BMKG Strong Motion Analyzer (BSMA)
Model: Quality Control (QC) Analyzer (Version 2.0 Professional Standard)

Modul ini melakukan inspeksi kualitas data seismik secara statistik dan fisis.
Dilengkapi dengan STA/LTA SNR, Rolling Variance Flatline, Relaxed Plateau Clipping,
Severity Enums, Proportional Quality Scoring, dan rekapitulasi durasi Stream.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar
import numpy as np
from obspy import Stream, Trace
from scipy.signal import medfilt
from scipy.ndimage import uniform_filter1d
from utils.exceptions import ErrorCode, SeverityLevel, WaveformError
from utils.logger import setup_logger

__all__ = [
    "QCSeverity",
    "QCIssue",
    "TraceStatistics",
    "TraceQCMetrics",
    "StreamQCReport",
    "QCAnalyzer",
]

# Konstanta Numerik
MAD_SCALE_FACTOR: float = 0.67448975

# Batas maksimum/minimum resolusi ADC standar
ADC_LIMITS = {
    16: (32767, -32768),
    24: (8388607, -8388608),
    32: (2147483647, -2147483648)
}

class QCSeverity(Enum):
    WARNING = 1   
    ERROR = 2     
    CRITICAL = 3  

@dataclass(slots=True)
class QCIssue:
    """Representasi anomali kualitas data spesifik."""
    name: str
    severity: QCSeverity
    deduction: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, 
            "severity": self.severity.name,
            "deduction": round(self.deduction, 2)
        }

@dataclass(slots=True)
class TraceStatistics:
    """Objek komprehensif untuk menyimpan metrik statistik dasar Trace."""
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    rms: float = 0.0
    peak: float = 0.0
    peak_to_peak: float = 0.0
    variance: float = 0.0
    mad: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": round(self.mean, 6),
            "median": round(self.median, 6),
            "std": round(self.std, 6),
            "rms": round(self.rms, 6),
            "peak": round(self.peak, 6),
            "peak_to_peak": round(self.peak_to_peak, 6),
            "variance": round(self.variance, 6),
            "mad": round(self.mad, 6),
        }

@dataclass(slots=True)
class TraceQCMetrics:
    """Metrik hasil Quality Control untuk satu Trace spesifik."""
    trace_id: str
    is_valid: bool = True
    quality_score: int = 100  
    
    clipping_percent: float = 0.0
    flatline_duration_sec: float = 0.0
    spike_count: int = 0
    drift_percent: float = 0.0
    offset_percent: float = 0.0
    snr_estimate_db: float = 0.0
    
    issues: list[QCIssue] = field(default_factory=list)
    statistics: TraceStatistics = field(default_factory=TraceStatistics)
    
    def calculate_quality_score(self) -> None:
        """Kalkulasi skor akhir (0-100) berdasarkan total penalti."""
        score = 100.0
        has_critical = False
        
        for issue in self.issues:
            if issue.severity == QCSeverity.CRITICAL:
                has_critical = True
                break
            score -= issue.deduction
                
        self.quality_score = 0 if has_critical else max(0, int(round(score)))
        self.is_valid = self.quality_score >= 60

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "is_valid": self.is_valid,
            "quality_score": self.quality_score,
            "clipping_percent": round(self.clipping_percent, 4),
            "flatline_duration_sec": round(self.flatline_duration_sec, 4),
            "spike_count": self.spike_count,
            "drift_percent": round(self.drift_percent, 4),
            "offset_percent": round(self.offset_percent, 4),
            "snr_estimate_db": round(self.snr_estimate_db, 2),
            "issues": [iss.to_dict() for iss in self.issues],
            "statistics": self.statistics.to_dict(),
        }

@dataclass(slots=True)
class StreamQCReport:
    """Laporan komprehensif QC untuk satu Stream penuh."""
    stream_id: str
    is_passed: bool = True
    trace_metrics: dict[str, TraceQCMetrics] = field(default_factory=dict)
    
    gap_count: int = 0
    overlap_count: int = 0
    gap_duration_sec: float = 0.0
    overlap_duration_sec: float = 0.0
    average_quality_score: float = 0.0
    
    @property
    def total_traces(self) -> int:
        return len(self.trace_metrics)
        
    @property
    def failed_trace_count(self) -> int:
        return sum(not m.is_valid for m in self.trace_metrics.values())
        
    def finalize_metrics(self) -> None:
        """Menghitung ringkasan metrik tingkat stream."""
        if self.trace_metrics:
            self.average_quality_score = float(np.mean([m.quality_score for m in self.trace_metrics.values()]))


class QCAnalyzer:
    """Production-grade Quality Control analyzer untuk data strong motion."""
    
    PENALTY_CLIPPING_PER_PCT: ClassVar[float] = 10.0
    PENALTY_FLATLINE_PER_SEC: ClassVar[float] = 15.0
    PENALTY_SPIKE_PER_COUNT: ClassVar[float] = 2.0
    PENALTY_OFFSET_PER_PCT: ClassVar[float] = 5.0
    PENALTY_DRIFT_PER_PCT: ClassVar[float] = 5.0

    MAX_PENALTY_CLIPPING: ClassVar[float] = 40.0
    MAX_PENALTY_FLATLINE: ClassVar[float] = 40.0
    MAX_PENALTY_SPIKE: ClassVar[float] = 30.0
    MAX_PENALTY_OFFSET: ClassVar[float] = 20.0
    MAX_PENALTY_DRIFT: ClassVar[float] = 20.0

    def __init__(
        self, 
        logger: logging.Logger | None = None,
        flatline_std_factor: float = 1e-4,
        mad_spike_multiplier: float = 10.0,
        offset_limit_percent: float = 2.0,
        drift_limit_percent: float = 5.0
    ) -> None:
        self.logger = logger or setup_logger(__name__)
        
        self.flatline_std_factor = flatline_std_factor
        self.mad_spike_multiplier = mad_spike_multiplier
        self.offset_limit_percent = offset_limit_percent
        self.drift_limit_percent = drift_limit_percent

    def analyze_stream(self, stream: Stream, identifier: str) -> StreamQCReport:
        if not stream or len(stream) == 0:
            raise WaveformError(
                message="Stream kosong, QC dibatalkan.",
                error_code=ErrorCode.WF003,
                severity=SeverityLevel.ERROR,
                context={"module": "qc_analyzer", "identifier": identifier}
            )
            
        report = StreamQCReport(stream_id=identifier)
        self._analyze_stream_continuity(stream, report)
        
        for trace in stream:
            try:
                metrics = self.analyze_trace(trace)
                report.trace_metrics[trace.id] = metrics
                if not metrics.is_valid:
                    report.is_passed = False
            except Exception as e:
                report.is_passed = False
                self.logger.warning("QC gagal pada trace spesifik.", exc_info=True, 
                                    extra={"bsma_context": {"trace_id": trace.id, "error": str(e)}})
                                    
        report.finalize_metrics()
        return report

    def analyze_trace(self, trace: Trace) -> TraceQCMetrics:
        raw_data = trace.data
        data = raw_data.astype(np.float64)
        sr = float(trace.stats.sampling_rate)
        
        metrics = TraceQCMetrics(trace_id=trace.id)
        
        # 1. Pertahanan Awal
        if data.size == 0:
            metrics.issues.append(QCIssue("EMPTY_TRACE", QCSeverity.CRITICAL))
            metrics.calculate_quality_score()
            return metrics
            
        if sr <= 0:
            metrics.issues.append(QCIssue("INVALID_SAMPLING_RATE", QCSeverity.CRITICAL))
            metrics.calculate_quality_score()
            return metrics

        if not np.all(np.isfinite(data)):
            metrics.issues.append(QCIssue("NON_FINITE_DATA", QCSeverity.CRITICAL))
            metrics.calculate_quality_score()
            return metrics

        metrics.statistics = self._calculate_statistics(data)
        metrics.snr_estimate_db = self.estimate_snr_sta_lta(data, sr)

        # 2. Deteksi Clipping Adaptif (Relaxed RLE)
        clip_pct = self.detect_clipping_relaxed_rle(data)
        metrics.clipping_percent = clip_pct
        if clip_pct > 0.0:
            deduction = min(clip_pct * self.PENALTY_CLIPPING_PER_PCT, self.MAX_PENALTY_CLIPPING)
            sev = QCSeverity.ERROR if clip_pct >= 1.0 else QCSeverity.WARNING
            metrics.issues.append(QCIssue("CLIPPING_PLATEAU", sev, deduction))
            
        # 3. Deteksi Saturasi ADC Cerdas (Metadata / Tipe Memori)
        if self.detect_adc_saturation(trace):
            metrics.issues.append(QCIssue("ADC_SATURATION", QCSeverity.ERROR, 30.0))

        # 4. Deteksi Flatline via Rolling Variance
        flat_dur = self.detect_flatline_rolling_variance(data, sr, metrics.statistics.mad)
        metrics.flatline_duration_sec = flat_dur
        if flat_dur > 0.0:
            deduction = min(flat_dur * self.PENALTY_FLATLINE_PER_SEC, self.MAX_PENALTY_FLATLINE)
            sev = QCSeverity.CRITICAL if flat_dur >= 2.0 else QCSeverity.ERROR
            metrics.issues.append(QCIssue("FLATLINE", sev, deduction))

        # 5. Deteksi Spike (Turunan + MAD)
        spk_count = self.detect_spikes_mad(data)
        metrics.spike_count = spk_count
        if spk_count > 0:
            deduction = min(spk_count * self.PENALTY_SPIKE_PER_COUNT, self.MAX_PENALTY_SPIKE)
            sev = QCSeverity.ERROR if spk_count > 10 else QCSeverity.WARNING
            metrics.issues.append(QCIssue("SPIKE", sev, deduction))
            
        # 6. Deteksi Baseline Offset (Menggunakan Median)
        p2p = metrics.statistics.peak_to_peak
        if p2p > 0:
            offset_pct = (abs(metrics.statistics.median) / p2p) * 100.0
            metrics.offset_percent = offset_pct
            if offset_pct > self.offset_limit_percent:
                deduction = min((offset_pct - self.offset_limit_percent) * self.PENALTY_OFFSET_PER_PCT, self.MAX_PENALTY_OFFSET)
                metrics.issues.append(QCIssue("BASELINE_OFFSET", QCSeverity.WARNING, deduction))
            
        # 7. Deteksi Baseline Drift (Median Filtered Linear Regression)
        drift_pct = self.detect_robust_drift(data, sr, p2p)
        metrics.drift_percent = drift_pct
        if drift_pct > self.drift_limit_percent:
            deduction = min((drift_pct - self.drift_limit_percent) * self.PENALTY_DRIFT_PER_PCT, self.MAX_PENALTY_DRIFT)
            metrics.issues.append(QCIssue("BASELINE_DRIFT", QCSeverity.WARNING, deduction))
            
        metrics.calculate_quality_score()
        return metrics

    def _analyze_stream_continuity(self, stream: Stream, report: StreamQCReport) -> None:
        try:
            gaps = stream.get_gaps()
            for g in gaps:
                dur = g[6]
                if dur > 0:
                    report.gap_count += 1
                    report.gap_duration_sec += dur
                elif dur < 0:
                    report.overlap_count += 1
                    report.overlap_duration_sec += abs(dur)
        except Exception as e:
            self.logger.warning("Gagal mengekstrak kontinuitas.", exc_info=True)

    def _calculate_statistics(self, data: np.ndarray) -> TraceStatistics:
        mean_val = np.mean(data)
        median_val = np.median(data)
        return TraceStatistics(
            mean=float(mean_val),
            median=float(median_val),
            std=float(np.std(data)),
            rms=float(np.sqrt(np.mean(data**2))),
            peak=float(np.max(np.abs(data))),
            peak_to_peak=float(np.max(data) - np.min(data)),
            variance=float(np.var(data)),
            mad=float(np.median(np.abs(data - median_val)))
        )

    # ==========================================
    # Algoritma Saintifik (Vektorisasi NumPy & SciPy)
    # ==========================================
    
    def detect_clipping_relaxed_rle(self, data: np.ndarray) -> float:
        """Deteksi clipping menggunakan RLE dengan pelonggaran batas (1% dari rentang dinamis)."""
        if data.size < 2: return 0.0

        p2p = np.max(data) - np.min(data)
        if np.isclose(p2p, 0.0): return 0.0
        
        # Toleransi 1% untuk mencari batas "dekat" ekstrem
        upper_thresh = np.max(data) - (0.01 * p2p)
        lower_thresh = np.min(data) + (0.01 * p2p)
        
        is_extreme = (data >= upper_thresh) | (data <= lower_thresh)
        diff_is_zero = np.concatenate(([False], np.isclose(np.diff(data), 0.0, atol=1e-8)))
        
        # Harus berada di batas ekstrem DAN variasi turunannya nol (plateau murni)
        is_plateau_extreme = is_extreme & diff_is_zero
        
        padded = np.concatenate(([False], is_plateau_extreme, [False]))
        transitions = np.flatnonzero(padded[1:] != padded[:-1])
        run_lengths = transitions[1::2] - transitions[::2]
        
        plateaus = run_lengths[run_lengths >= 1]
        if plateaus.size == 0: return 0.0
            
        total_clipped_samples = int(np.sum(plateaus + 1))
        return float((total_clipped_samples / data.size) * 100.0)
        
    def detect_adc_saturation(self, trace: Trace) -> bool:
        raw_data = trace.data
        if raw_data.dtype.kind not in ('i', 'u'): return False 
            
        max_val, min_val = np.max(raw_data), np.min(raw_data)
        fmt = getattr(trace.stats, "_format", "").upper()
        
        if fmt == "MSEED":
            encoding = getattr(trace.stats.mseed, "encoding", "")
            if "INT16" in encoding: limits = ADC_LIMITS[16]
            elif "INT32" in encoding: limits = ADC_LIMITS[32]
            else: limits = ADC_LIMITS[24]
        else:
            if raw_data.dtype.itemsize == 2: limits = ADC_LIMITS[16]
            elif raw_data.dtype.itemsize == 4: limits = ADC_LIMITS[32]
            else: return False

        return bool(np.isclose(max_val, limits[0], atol=1e-3) or np.isclose(min_val, limits[1], atol=1e-3))
        
    def detect_flatline_rolling_variance(self, data: np.ndarray, sr: float, mad_val: float) -> float:
        """Deteksi flatline/sensor macet menggunakan rolling variance O(N)."""
        win = max(2, int(sr * 0.5)) 
        if data.size < win: return 0.0

        # E[X^2] - (E[X])^2
        c1 = uniform_filter1d(data, size=win)
        c2 = uniform_filter1d(data**2, size=win)
        rolling_var = np.clip(c2 - c1**2, 0, None) # Cegah nilai negatif floating point
        
        dynamic_eps = max(self.flatline_std_factor * (mad_val**2), 1e-12)
        is_flat = rolling_var < dynamic_eps
        
        padded = np.concatenate(([False], is_flat, [False]))
        transitions = np.flatnonzero(padded[1:] != padded[:-1])
        consecutive_lengths = transitions[1::2] - transitions[::2]
        
        if consecutive_lengths.size == 0: return 0.0
        return float(np.max(consecutive_lengths) / sr)
        
    def detect_spikes_mad(self, data: np.ndarray) -> int:
        if data.size < 2: return 0

        diff_data = np.diff(data)
        med = np.median(diff_data)
        mad = np.median(np.abs(diff_data - med))
        
        if np.isclose(mad, 0.0): return 0

        modified_z_scores = MAD_SCALE_FACTOR * (diff_data - med) / mad
        spike_indices = np.abs(modified_z_scores) > self.mad_spike_multiplier
        return int(np.sum(spike_indices))

    def detect_robust_drift(self, data: np.ndarray, sr: float, p2p: float) -> float:
        if data.size < 2 or np.isclose(p2p, 0.0): return 0.0
        
        # Mencegah overload CPU (maksimal kernel window = 201 sampel)
        kernel_size = min(int(sr), 201)
        if kernel_size % 2 == 0: kernel_size += 1
            
        filtered_data = medfilt(data, kernel_size=kernel_size)
            
        time_vector = np.linspace(0, filtered_data.size / sr, filtered_data.size, endpoint=False)
        slope, _ = np.polyfit(time_vector, filtered_data, 1)
        
        total_drift_amplitude = abs(slope * time_vector[-1])
        return float((total_drift_amplitude / p2p) * 100.0)

    def estimate_snr_sta_lta(self, data: np.ndarray, sr: float) -> float:
        """Estimasi SNR menggunakan pendekatan STA/LTA seismologis klasik."""
        if data.size < int(sr * 5): return 0.0 
        
        sta_len = max(1, int(sr * 1.0))  # 1 detik Short-Time Average
        lta_len = max(1, int(sr * 5.0))  # 5 detik Long-Time Average
        
        data_sq = data ** 2
        sta = uniform_filter1d(data_sq, size=sta_len)
        lta = uniform_filter1d(data_sq, size=lta_len)
        
        lta[lta == 0] = 1e-10
        
        signal_power = np.max(sta)
        noise_power = np.min(lta)
        
        if np.isclose(noise_power, 0.0): return 999.9
        return float(10 * np.log10(signal_power / noise_power))