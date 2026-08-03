"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/preprocessing/qc.py
Description: Quality Control (QC) Analyzer (Version 1.3 Scientific & Production Mastery)
Melakukan inspeksi kualitas data seismik secara statistik dan fisis.
Dilengkapi dengan adaptivitas resolusi instrumen, metrik fisis yang dinormalisasi,
Severity Levels, dan Quality Scoring (0-100).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Tuple
import numpy as np
from obspy import Stream, Trace

# Mengasumsikan utilitas BSMA Error sudah ada di utils/exceptions.py
from utils.exceptions import ErrorCode, SeverityLevel, WaveformError
from utils.logger import setup_logger

__all__ = [
    "QCIssue",
    "TraceStatistics",
    "TraceQCMetrics",
    "StreamQCReport",
    "QCAnalyzer",
]

# Konstanta aproksimasi distribusi normal untuk Modified Z-Score
MAD_SCALE_FACTOR: float = 0.67448975

# Batas maksimum/minimum resolusi ADC standar (khusus tipe data integer)
ADC_LIMITS = {
    16: (32767, -32768),
    24: (8388607, -8388608),
    32: (2147483647, -2147483648)
}

class QCSeverity:
    WARNING = "WARNING"      # Anomali ringan (potong 15 poin)
    ERROR = "ERROR"          # Anomali berat, data masih bisa diproses dengan hati-hati (potong 40 poin)
    CRITICAL = "CRITICAL"    # Data rusak parah, tidak layak analisis (Skor = 0)

@dataclass(slots=True)
class QCIssue:
    """Representasi anomali kualitas data spesifik."""
    name: str
    severity: str

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "severity": self.severity}

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

    def to_dict(self) -> Dict[str, float]:
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
    quality_score: int = 100  # Skala 0-100
    
    # Anomaly Flags
    has_clipping: bool = False
    has_adc_saturation: bool = False
    has_flatline: bool = False
    has_spikes: bool = False
    has_offset: bool = False
    has_drift: bool = False

    # Anomaly Quantifications
    clipping_percent: float = 0.0
    flatline_duration_sec: float = 0.0
    spike_count: int = 0
    drift_percent: float = 0.0
    offset_percent: float = 0.0
    snr_estimate_db: float = 0.0

    # Tagging isu untuk kebutuhan antarmuka (GUI)
    issues: List[QCIssue] = field(default_factory=list)
    
    # Statistik Komprehensif
    statistics: TraceStatistics = field(default_factory=TraceStatistics)

    def calculate_quality_score(self) -> None:
        """Kalkulasi skor akhir berdasarkan severity list issue yang terkumpul."""
        score = 100
        for issue in self.issues:
            if issue.severity == QCSeverity.CRITICAL:
                score = 0
                self.is_valid = False
                break
            elif issue.severity == QCSeverity.ERROR:
                score -= 40
            elif issue.severity == QCSeverity.WARNING:
                score -= 15
                
        self.quality_score = max(0, score)
        self.is_valid = self.quality_score >= 60  # Batas kelulusan dinamis

    def to_dict(self) -> Dict[str, Any]:
        """Konversi metrik ke dictionary untuk kebutuhan logging/GUI."""
        return {
            "trace_id": self.trace_id,
            "is_valid": self.is_valid,
            "quality_score": self.quality_score,
            "has_clipping": self.has_clipping,
            "has_adc_saturation": self.has_adc_saturation,
            "has_flatline": self.has_flatline,
            "has_spikes": self.has_spikes,
            "has_offset": self.has_offset,
            "has_drift": self.has_drift,
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
    trace_metrics: Dict[str, TraceQCMetrics] = field(default_factory=dict)
    
    # Stream-level metrics
    gap_count: int = 0
    overlap_count: int = 0

    @property
    def total_traces(self) -> int:
        return len(self.trace_metrics)

    @property
    def failed_trace_count(self) -> int:
        return sum(not m.is_valid for m in self.trace_metrics.values())

class QCAnalyzer:
    """
    Production-grade Quality Control analyzer untuk data strong motion.
    Menggunakan metrik relatif fisis dan vektorisasi murni NumPy.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        clipping_percent_limit: float = 1.0,
        clipping_rtol: float = 1e-4,
        flatline_time_threshold_sec: float = 1.0,
        flatline_std_factor: float = 1e-5,
        mad_spike_multiplier: float = 10.0,
        offset_limit_percent: float = 2.0,
        drift_limit_percent: float = 5.0
    ) -> None:
        """Inisialisasi QC Analyzer dengan parameter adaptif."""
        self.logger = logger or setup_logger(__name__)
        
        self.clipping_percent_limit = clipping_percent_limit
        self.clipping_rtol = clipping_rtol
        self.flatline_time_threshold_sec = flatline_time_threshold_sec
        self.flatline_std_factor = flatline_std_factor
        self.mad_spike_multiplier = mad_spike_multiplier
        self.offset_limit_percent = offset_limit_percent
        self.drift_limit_percent = drift_limit_percent
        
        self.logger.info(
            "QCAnalyzer diinisialisasi.",
            extra={"bsma_context": {"module": "qc_analyzer"}}
        )

    def analyze_stream(self, stream: Stream, identifier: str) -> StreamQCReport:
        """Mengeksekusi pipeline QC pada level Stream."""
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
                self.logger.warning(
                    "QC gagal pada trace spesifik.",
                    exc_info=True,
                    extra={
                        "bsma_context": {
                            "module": "qc_analyzer",
                            "trace_id": trace.id,
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    }
                )

        self.logger.info(
            "Analisis QC Stream selesai.",
            extra={
                "bsma_context": {
                    "identifier": identifier,
                    "is_passed": report.is_passed,
                    "failed_traces": report.failed_trace_count,
                    "gap_count": report.gap_count,
                    "overlap_count": report.overlap_count,
                }
            }
        )
        return report

    def analyze_trace(self, trace: Trace) -> TraceQCMetrics:
        """Mengeksekusi QC saintifik pada satu Trace."""
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

        # Kalkulasi Metrik & SNR
        metrics.statistics = self._calculate_statistics(data)
        metrics.snr_estimate_db = self.estimate_seismological_snr(data, sr)

        # 2. Deteksi Clipping Adaptif (Plateau + Derivative)
        has_clip, clip_pct = self.detect_clipping_plateau(data)
        metrics.has_clipping = has_clip
        metrics.clipping_percent = clip_pct
        if has_clip:
            metrics.issues.append(QCIssue("CLIPPING_PLATEAU", QCSeverity.ERROR))

        # 3. Deteksi Saturasi ADC Cerdas (Hanya jika data adalah Integer)
        if raw_data.dtype.kind in ('i', 'u'):
            has_adc_sat = self.detect_adc_saturation(raw_data)
            metrics.has_adc_saturation = has_adc_sat
            if has_adc_sat:
                metrics.issues.append(QCIssue("ADC_SATURATION", QCSeverity.ERROR))

        # 4. Deteksi Flatline Adaptif (Berdasarkan Std Deviasi)
        has_flat, flat_dur = self.detect_adaptive_flatline(data, sr, metrics.statistics.std)
        metrics.has_flatline = has_flat
        metrics.flatline_duration_sec = flat_dur
        if has_flat:
            metrics.issues.append(QCIssue("FLATLINE", QCSeverity.CRITICAL))

        # 5. Deteksi Spike (Turunan + MAD)
        spk_count = self.detect_spikes_mad(data)
        metrics.has_spikes = bool(spk_count > 0)
        metrics.spike_count = spk_count
        if spk_count > 0:
            sev = QCSeverity.ERROR if spk_count > 10 else QCSeverity.WARNING
            metrics.issues.append(QCIssue("SPIKE", sev))

        # 6. Deteksi Baseline Offset (Dinormalisasi terhadap Peak-to-Peak)
        p2p = metrics.statistics.peak_to_peak
        if p2p > 0:
            offset_pct = (abs(metrics.statistics.mean) / p2p) * 100.0
            metrics.offset_percent = offset_pct
            metrics.has_offset = offset_pct > self.offset_limit_percent
            if metrics.has_offset:
                metrics.issues.append(QCIssue("BASELINE_OFFSET", QCSeverity.WARNING))

        # 7. Deteksi Baseline Drift (Total kemiringan dinormalisasi)
        drift_pct = self.detect_normalized_drift(data, sr, p2p)
        metrics.drift_percent = drift_pct
        metrics.has_drift = drift_pct > self.drift_limit_percent
        if metrics.has_drift:
            metrics.issues.append(QCIssue("BASELINE_DRIFT", QCSeverity.WARNING))

        # Finalisasi Skor
        metrics.calculate_quality_score()
        return metrics

    def _analyze_stream_continuity(self, stream: Stream, report: StreamQCReport) -> None:
        """Mengevaluasi gap dan overlap fisis (dalam detik) antar trace."""
        try:
            gaps = stream.get_gaps()
            report.gap_count = sum(1 for g in gaps if g[6] > 0)
            report.overlap_count = sum(1 for g in gaps if g[6] < 0)
        except Exception as e:
            self.logger.warning("Gagal mengekstrak kontinuitas (gap/overlap).", exc_info=True)

    def _calculate_statistics(self, data: np.ndarray) -> TraceStatistics:
        """Menghitung metrik saintifik dasar secara aman."""
        if data.size == 0:
            return TraceStatistics()
            
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
    # Algoritma Deteksi & Estimasi (NumPy)
    # ==========================================

    def detect_clipping_plateau(self, data: np.ndarray) -> Tuple[bool, float]:
        """Mendeteksi clipping dengan mencari plateau menggunakan rtol adaptif."""
        if data.size < 2:
            return False, 0.0

        max_val = np.max(data)
        min_val = np.min(data)

        # Toleransi relatif agar robust terhadap variasi dinamis
        is_max = np.isclose(data, max_val, rtol=self.clipping_rtol, atol=1e-8)
        is_min = np.isclose(data, min_val, rtol=self.clipping_rtol, atol=1e-8)
        is_extreme = is_max | is_min

        padded = np.concatenate(([False], is_extreme, [False]))
        transitions = np.flatnonzero(padded[1:] != padded[:-1])
        consecutive_lengths = transitions[1::2] - transitions[::2]

        if len(consecutive_lengths) == 0:
            return False, 0.0

        plateaus = consecutive_lengths[consecutive_lengths >= 2]
        plateau_samples = int(np.sum(plateaus))
        clip_pct = (plateau_samples / data.size) * 100.0

        has_clipping = clip_pct >= self.clipping_percent_limit
        return bool(has_clipping), float(clip_pct)

    def detect_adc_saturation(self, raw_data: np.ndarray) -> bool:
        """Mendeteksi jika sinyal mencapai batas limit arsitektur bit ADC."""
        if raw_data.size == 0:
            return False

        max_val = np.max(raw_data)
        min_val = np.min(raw_data)

        for limits in ADC_LIMITS.values():
            if np.isclose(max_val, limits[0], atol=1e-3) or np.isclose(min_val, limits[1], atol=1e-3):
                return True
        return False

    def detect_adaptive_flatline(self, data: np.ndarray, sampling_rate: float, std_dev: float) -> Tuple[bool, float]:
        """Mendeteksi dead channel (flatline) menggunakan np.diff dan EPS tolerance."""
        if data.size < 2 or sampling_rate <= 0:
            return False, 0.0

        diffs = np.diff(data)
        # Menggunakan np.abs(diffs) < EPS agar instrumen modern (float noise) tetap tertangkap
        adaptive_eps = max(self.flatline_eps, self.flatline_std_factor * std_dev)
        is_flat = np.abs(diffs) < adaptive_eps

        if not np.any(is_flat):
            return False, 0.0

        padded = np.concatenate(([False], is_flat, [False]))
        transitions = np.flatnonzero(padded[1:] != padded[:-1])
        consecutive_lengths = transitions[1::2] - transitions[::2]

        max_consecutive_zeros = np.max(consecutive_lengths)
        flatline_duration = float(max_consecutive_zeros / sampling_rate)
        has_flatline = flatline_duration >= self.flatline_time_threshold_sec
        return bool(has_flatline), flatline_duration

    def detect_spikes_mad(self, data: np.ndarray) -> int:
        """Mendeteksi spike dengan mengevaluasi turunan pertama (first derivative)."""
        if data.size < 2:
            return 0

        # Menganalisis kecepatan perubahan (gradient), bukan amplitudo statis
        diff_data = np.diff(data)
        med = np.median(diff_data)
        mad = np.median(np.abs(diff_data - med))

        # Proteksi pembagian nol (flatline atau noise mendekati nol mutlak)
        if np.isclose(mad, 0.0):
            return 0

        modified_z_scores = MAD_SCALE_FACTOR * (diff_data - med) / mad
        spike_indices = np.abs(modified_z_scores) > self.mad_spike_multiplier
        spike_count = int(np.sum(spike_indices))

        return spike_count

    def detect_normalized_drift(self, data: np.ndarray, sampling_rate: float, peak_to_peak: float) -> float:
        """Mendeteksi indikasi linear drift menggunakan regresi polinomial derajat 1."""
        if data.size < 2 or np.isclose(peak_to_peak, 0.0):
            return 0.0

        time_vector = np.linspace(0, data.size / sampling_rate, data.size, endpoint=False)
        # Menghasilkan kemiringan (slope) dan intercept
        slope, _ = np.polyfit(time_vector, data, 1)

        total_duration = data.size / sampling_rate
        total_drift = abs(slope * total_duration)
        drift_pct = (total_drift / peak_to_peak) * 100.0

        return float(drift_pct)

    def estimate_seismological_snr(self, data: np.ndarray, sampling_rate: float) -> float:
        """
        Estimasi kasaran Signal-to-Noise Ratio (dB).
        Membagi data ke dalam blok 1-detik. Asumsi:
        - Noise floor: Blok dengan Root Mean Square (RMS) terendah.
        - Signal peak: Blok dengan RMS tertinggi.
        """
        if data.size < int(sampling_rate):
            return 0.0 # Durasi terlalu pendek untuk estimasi SNR

        window_size = int(sampling_rate)
        # Reshape data menjadi blok-blok 1 detik (membuang sisa di akhir)
        num_windows = data.size // window_size
        truncated_data = data[:num_windows * window_size]
        reshaped = truncated_data.reshape((num_windows, window_size))

        # Hitung RMS untuk setiap jendela
        rms_windows = np.sqrt(np.mean(reshaped**2, axis=1))
        noise_rms = np.min(rms_windows)
        signal_rms = np.max(rms_windows)

        if np.isclose(noise_rms, 0.0):
            return 999.9 # Asumsi SNR 'tak terhingga' atau pure artificial signal

        snr_db = 20 * np.log10(signal_rms / noise_rms)
        return float(snr_db)