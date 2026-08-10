"""
BMKG Strong Motion Analyzer (BSMA)

Module: core/preprocessing/qc.py

Scientific Quality Control Analyzer
------------------------------------

Performs quality-control analysis for strong-motion waveforms.

The QC stage is intentionally designed for RAW waveform inspection
before baseline correction, filtering, tapering, and integration.

Implemented checks
------------------
- Empty / invalid waveform
- Non-finite samples
- Sampling-rate validation
- Gap / overlap detection
- Clipping / plateau detection
- ADC saturation detection when ADC limits are known
- Adaptive flatline detection
- Impulsive spike detection using MAD-based modified Z-score
- Baseline offset estimation
- Baseline drift estimation
- Approximate noise/SNR estimation
- Quality scoring

Important
---------
QC metrics are diagnostic indicators. They must not be interpreted
as replacements for engineering/seismological judgement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from obspy import Stream, Trace

from utils.exceptions import (
    ErrorCode,
    SeverityLevel,
    WaveformError,
)
from utils.logger import setup_logger


__all__ = [
    "QCSeverity",
    "QCIssue",
    "TraceStatistics",
    "TraceQCMetrics",
    "StreamQCReport",
    "QCAnalyzer",
]


FloatArray = NDArray[np.float64]

# Standard normal-consistency factor for MAD.
#
# modified_z = 0.67448975 * (x - median) / MAD
#
# For a Gaussian distribution:
# MAD ≈ 0.6745 * sigma
MAD_SCALE_FACTOR = 0.67448975


class QCSeverity(str, Enum):
    """Severity classification for QC anomalies."""

    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(slots=True)
class QCIssue:
    """Single quality-control issue."""

    name: str
    severity: QCSeverity
    message: str = ""

    def to_dict(self) -> dict[str, str]:
        """Serialize the QC issue."""
        return {
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
        }


@dataclass(slots=True)
class TraceStatistics:
    """Basic statistical properties of one waveform trace."""

    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    rms: float = 0.0
    peak: float = 0.0
    peak_to_peak: float = 0.0
    variance: float = 0.0
    mad: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialize statistics."""
        return {
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "rms": self.rms,
            "peak": self.peak,
            "peak_to_peak": self.peak_to_peak,
            "variance": self.variance,
            "mad": self.mad,
        }


@dataclass(slots=True)
class TraceQCMetrics:
    """QC metrics associated with one seismic trace."""

    trace_id: str

    is_valid: bool = True
    quality_score: int = 100

    # ---------------------------------------------------------
    # Anomaly flags
    # ---------------------------------------------------------

    has_clipping: bool = False
    has_adc_saturation: bool = False
    has_flatline: bool = False
    has_spikes: bool = False
    has_offset: bool = False
    has_drift: bool = False

    # ---------------------------------------------------------
    # Quantitative diagnostics
    # ---------------------------------------------------------

    clipping_percent: float = 0.0
    flatline_duration_sec: float = 0.0
    spike_count: int = 0

    drift_percent: float = 0.0
    offset_percent: float = 0.0

    snr_estimate_db: float | None = None

    # ---------------------------------------------------------
    # Diagnostic information
    # ---------------------------------------------------------

    issues: list[QCIssue] = field(default_factory=list)
    statistics: TraceStatistics = field(
        default_factory=TraceStatistics
    )

    def calculate_quality_score(self) -> None:
        """
        Calculate diagnostic quality score.

        Scoring:
            WARNING  -> -15
            ERROR    -> -40
            CRITICAL -> 0

        Multiple issues are cumulative except CRITICAL.
        """

        score = 100

        for issue in self.issues:

            if issue.severity is QCSeverity.CRITICAL:
                score = 0
                break

            if issue.severity is QCSeverity.ERROR:
                score -= 40

            elif issue.severity is QCSeverity.WARNING:
                score -= 15

        self.quality_score = max(0, min(100, score))

        # A score below 60 is considered QC-failed.
        self.is_valid = self.quality_score >= 60

    def to_dict(self) -> dict[str, Any]:
        """Serialize QC metrics."""

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
            "clipping_percent": self.clipping_percent,
            "flatline_duration_sec": self.flatline_duration_sec,
            "spike_count": self.spike_count,
            "drift_percent": self.drift_percent,
            "offset_percent": self.offset_percent,
            "snr_estimate_db": self.snr_estimate_db,
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
            "statistics": self.statistics.to_dict(),
        }


@dataclass(slots=True)
class StreamQCReport:
    """QC report for an entire ObsPy Stream."""

    stream_id: str

    is_passed: bool = True

    trace_metrics: dict[str, TraceQCMetrics] = field(
        default_factory=dict
    )

    gap_count: int = 0
    overlap_count: int = 0

    max_gap_duration_sec: float = 0.0
    max_overlap_duration_sec: float = 0.0

    @property
    def total_traces(self) -> int:
        """Number of traces analyzed."""
        return len(self.trace_metrics)

    @property
    def failed_trace_count(self) -> int:
        """Number of traces failing QC."""
        return sum(
            not metrics.is_valid
            for metrics in self.trace_metrics.values()
        )


class QCAnalyzer:
    """
    Scientific QC analyzer for strong-motion waveforms.

    Parameters
    ----------
    clipping_percent_limit
        Minimum percentage of samples contained in extreme
        plateaus before clipping is reported.

    clipping_min_duration_sec
        Minimum duration of an extreme plateau.

    flatline_time_threshold_sec
        Minimum duration of a low-variation segment.

    flatline_std_factor
        Relative tolerance for adaptive flatline detection.

    flatline_eps
        Absolute minimum difference tolerance.

    mad_spike_multiplier
        Modified-Z threshold for impulsive spike detection.

    offset_limit_percent
        Maximum normalized baseline offset before warning.

    drift_limit_percent
        Maximum normalized baseline drift before warning.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        clipping_percent_limit: float = 1.0,
        clipping_rtol: float = 1e-4,
        clipping_min_duration_sec: float = 0.02,
        flatline_time_threshold_sec: float = 1.0,
        flatline_std_factor: float = 1e-5,
        flatline_eps: float = 1e-12,
        mad_spike_multiplier: float = 6.0,
        offset_limit_percent: float = 2.0,
        drift_limit_percent: float = 5.0,
    ) -> None:

        if clipping_percent_limit < 0:
            raise ValueError(
                "clipping_percent_limit must be >= 0."
            )

        if clipping_rtol <= 0:
            raise ValueError(
                "clipping_rtol must be > 0."
            )

        if clipping_min_duration_sec <= 0:
            raise ValueError(
                "clipping_min_duration_sec must be > 0."
            )

        if flatline_time_threshold_sec <= 0:
            raise ValueError(
                "flatline_time_threshold_sec must be > 0."
            )

        if flatline_std_factor < 0:
            raise ValueError(
                "flatline_std_factor must be >= 0."
            )

        if flatline_eps < 0:
            raise ValueError(
                "flatline_eps must be >= 0."
            )

        if mad_spike_multiplier <= 0:
            raise ValueError(
                "mad_spike_multiplier must be > 0."
            )

        self.logger = logger or setup_logger(__name__)

        self.clipping_percent_limit = clipping_percent_limit
        self.clipping_rtol = clipping_rtol
        self.clipping_min_duration_sec = (
            clipping_min_duration_sec
        )

        self.flatline_time_threshold_sec = (
            flatline_time_threshold_sec
        )
        self.flatline_std_factor = flatline_std_factor
        self.flatline_eps = flatline_eps

        self.mad_spike_multiplier = (
            mad_spike_multiplier
        )

        self.offset_limit_percent = offset_limit_percent
        self.drift_limit_percent = drift_limit_percent

        self.logger.info(
            "QCAnalyzer initialized.",
            extra={
                "bsma_context": {
                    "module": "qc_analyzer"
                }
            },
        )

    # ==========================================================
    # STREAM QC
    # ==========================================================

    def analyze_stream(
        self,
        stream: Stream,
        identifier: str,
    ) -> StreamQCReport:
        """Analyze QC for an entire ObsPy Stream."""

        if not stream:
            raise WaveformError(
                message="Stream kosong, QC dibatalkan.",
                error_code=ErrorCode.WF003,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "qc_analyzer",
                    "identifier": identifier,
                },
            )

        report = StreamQCReport(
            stream_id=identifier
        )

        self._analyze_stream_continuity(
            stream,
            report,
        )

        for trace in stream:

            try:
                metrics = self.analyze_trace(trace)

                report.trace_metrics[
                    trace.id
                ] = metrics

                if not metrics.is_valid:
                    report.is_passed = False

            except Exception as exc:

                report.is_passed = False

                self.logger.warning(
                    "QC failed for trace.",
                    exc_info=True,
                    extra={
                        "bsma_context": {
                            "module": "qc_analyzer",
                            "trace_id": trace.id,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    },
                )

        self.logger.info(
            "Stream QC completed.",
            extra={
                "bsma_context": {
                    "identifier": identifier,
                    "is_passed": report.is_passed,
                    "failed_traces": (
                        report.failed_trace_count
                    ),
                    "gap_count": report.gap_count,
                    "overlap_count": report.overlap_count,
                }
            },
        )

        return report

    # ==========================================================
    # TRACE QC
    # ==========================================================

    def analyze_trace(
        self,
        trace: Trace,
    ) -> TraceQCMetrics:
        """Perform scientific QC on one trace."""

        raw_data = np.asarray(trace.data)

        metrics = TraceQCMetrics(
            trace_id=trace.id
        )

        # ------------------------------------------------------
        # Basic validation
        # ------------------------------------------------------

        if raw_data.size == 0:

            metrics.issues.append(
                QCIssue(
                    "EMPTY_TRACE",
                    QCSeverity.CRITICAL,
                    "Trace contains no samples.",
                )
            )

            metrics.calculate_quality_score()
            return metrics

        try:
            sampling_rate = float(
                trace.stats.sampling_rate
            )
        except (
            AttributeError,
            TypeError,
            ValueError,
        ):

            sampling_rate = 0.0

        if sampling_rate <= 0:

            metrics.issues.append(
                QCIssue(
                    "INVALID_SAMPLING_RATE",
                    QCSeverity.CRITICAL,
                    "Sampling rate must be positive.",
                )
            )

            metrics.calculate_quality_score()
            return metrics

        # Convert only after checking original dtype.
        data = raw_data.astype(
            np.float64,
            copy=False,
        )

        if not np.all(np.isfinite(data)):

            metrics.issues.append(
                QCIssue(
                    "NON_FINITE_DATA",
                    QCSeverity.CRITICAL,
                    "Waveform contains NaN or infinite values.",
                )
            )

            metrics.calculate_quality_score()
            return metrics

        # ------------------------------------------------------
        # Statistics
        # ------------------------------------------------------

        metrics.statistics = (
            self._calculate_statistics(data)
        )

        # ------------------------------------------------------
        # Clipping
        # ------------------------------------------------------

        has_clip, clip_pct = (
            self.detect_clipping_plateau(
                data,
                sampling_rate,
            )
        )

        metrics.has_clipping = has_clip
        metrics.clipping_percent = clip_pct

        if has_clip:
            metrics.issues.append(
                QCIssue(
                    "CLIPPING_PLATEAU",
                    QCSeverity.ERROR,
                    (
                        "Extreme-value plateau detected "
                        "for a physically significant duration."
                    ),
                )
            )

        # ------------------------------------------------------
        # ADC saturation
        # ------------------------------------------------------

        if raw_data.dtype.kind in ("i", "u"):

            has_saturation = (
                self.detect_adc_saturation(
                    raw_data
                )
            )

            metrics.has_adc_saturation = (
                has_saturation
            )

            if has_saturation:

                metrics.issues.append(
                    QCIssue(
                        "ADC_SATURATION",
                        QCSeverity.ERROR,
                        "Integer waveform reaches its known ADC limit.",
                    )
                )

        # ------------------------------------------------------
        # Flatline
        # ------------------------------------------------------

        has_flat, flat_duration = (
            self.detect_adaptive_flatline(
                data,
                sampling_rate,
                metrics.statistics.std,
            )
        )

        metrics.has_flatline = has_flat
        metrics.flatline_duration_sec = (
            flat_duration
        )

        if has_flat:
            # Pre-event and post-event quiet windows are normal in strong
            # motion records.  A flat segment is critical only when it
            # consumes essentially the whole record; otherwise it remains a
            # visible warning for the analyst.
            trace_duration = max((data.size - 1) / sampling_rate, 0.0)
            severity = (
                QCSeverity.CRITICAL
                if trace_duration > 0.0 and flat_duration >= 0.9 * trace_duration
                else QCSeverity.WARNING
            )
            metrics.issues.append(
                QCIssue(
                    "FLATLINE",
                    severity,
                    (
                        "Low-variation waveform segment "
                        "exceeds the flatline duration threshold."
                    ),
                )
            )

        # ------------------------------------------------------
        # Spike detection
        # ------------------------------------------------------

        spike_count = self.detect_spikes_mad(
            data
        )

        metrics.spike_count = spike_count
        metrics.has_spikes = spike_count > 0

        if spike_count > 0:

            # A global MAD threshold can mark many legitimate high-gradient
            # samples during the strong-motion phase.  Escalate only when
            # more than half of all sample-to-sample differences are
            # anomalous; smaller fractions are reported for analyst review.
            spike_fraction = spike_count / max(data.size - 1, 1)
            severity = (
                QCSeverity.ERROR
                if spike_fraction > 0.5
                else QCSeverity.WARNING
            )

            metrics.issues.append(
                QCIssue(
                    "IMPULSIVE_SPIKE",
                    severity,
                    (
                        f"{spike_count} impulsive "
                        "samples detected using MAD."
                    ),
                )
            )

        # ------------------------------------------------------
        # Baseline offset
        # ------------------------------------------------------

        amplitude_scale = (
            self._robust_amplitude_scale(data)
        )

        if amplitude_scale > 0:

            offset_pct = (
                abs(metrics.statistics.mean)
                / amplitude_scale
                * 100.0
            )

            metrics.offset_percent = (
                float(offset_pct)
            )

            metrics.has_offset = (
                offset_pct
                > self.offset_limit_percent
            )

            if metrics.has_offset:

                metrics.issues.append(
                    QCIssue(
                        "BASELINE_OFFSET",
                        QCSeverity.WARNING,
                        (
                            "Mean offset exceeds the "
                            "configured robust amplitude threshold."
                        ),
                    )
                )

        # ------------------------------------------------------
        # Baseline drift
        # ------------------------------------------------------

        drift_pct = (
            self.detect_normalized_drift(
                data,
                sampling_rate,
                amplitude_scale,
            )
        )

        metrics.drift_percent = drift_pct
        metrics.has_drift = (
            drift_pct
            > self.drift_limit_percent
        )

        if metrics.has_drift:

            metrics.issues.append(
                QCIssue(
                    "BASELINE_DRIFT",
                    QCSeverity.WARNING,
                    (
                        "Linear baseline drift exceeds "
                        "the configured threshold."
                    ),
                )
            )

        # ------------------------------------------------------
        # SNR estimate
        # ------------------------------------------------------

        metrics.snr_estimate_db = (
            self.estimate_seismological_snr(
                data,
                sampling_rate,
            )
        )

        # ------------------------------------------------------
        # Final score
        # ------------------------------------------------------

        metrics.calculate_quality_score()

        return metrics

    # ==========================================================
    # STREAM CONTINUITY
    # ==========================================================

    def _analyze_stream_continuity(
        self,
        stream: Stream,
        report: StreamQCReport,
    ) -> None:
        """
        Analyze temporal gaps and overlaps.

        ObsPy get_gaps() returns entries containing the
        temporal difference in seconds at index 6.
        """

        try:

            gaps = stream.get_gaps()

            for gap in gaps:

                duration = float(gap[6])

                if duration > 0:

                    report.gap_count += 1

                    report.max_gap_duration_sec = max(
                        report.max_gap_duration_sec,
                        duration,
                    )

                elif duration < 0:

                    report.overlap_count += 1

                    report.max_overlap_duration_sec = max(
                        report.max_overlap_duration_sec,
                        abs(duration),
                    )

        except Exception:

            self.logger.warning(
                "Failed to analyze stream continuity.",
                exc_info=True,
            )

    # ==========================================================
    # STATISTICS
    # ==========================================================

    def _calculate_statistics(
        self,
        data: FloatArray,
    ) -> TraceStatistics:

        median = float(np.median(data))

        return TraceStatistics(
            mean=float(np.mean(data)),
            median=median,
            std=float(np.std(data)),
            rms=float(
                np.sqrt(np.mean(data ** 2))
            ),
            peak=float(
                np.max(np.abs(data))
            ),
            peak_to_peak=float(
                np.ptp(data)
            ),
            variance=float(
                np.var(data)
            ),
            mad=float(
                np.median(
                    np.abs(data - median)
                )
            ),
        )

    # ==========================================================
    # CLIPPING
    # ==========================================================

    def detect_clipping_plateau(
        self,
        data: FloatArray,
        sampling_rate: float,
    ) -> tuple[bool, float]:
        """
        Detect sustained plateaus near extreme amplitudes.

        A single maximum sample is NOT interpreted as clipping.

        The plateau must:
        - contain consecutive samples;
        - remain close to a global extreme;
        - persist for a minimum physical duration.
        """

        if data.size < 3:
            return False, 0.0

        max_value = float(np.max(data))
        min_value = float(np.min(data))

        max_mask = np.isclose(
            data,
            max_value,
            rtol=self.clipping_rtol,
            atol=self.clipping_rtol
            * max(abs(max_value), 1.0),
        )

        min_mask = np.isclose(
            data,
            min_value,
            rtol=self.clipping_rtol,
            atol=self.clipping_rtol
            * max(abs(min_value), 1.0),
        )

        extreme_mask = max_mask | min_mask

        min_samples = max(
            2,
            int(
                np.ceil(
                    self.clipping_min_duration_sec
                    * sampling_rate
                )
            ),
        )

        plateau_samples = self._count_runs(
            extreme_mask,
            min_samples,
        )

        clipping_percent = (
            plateau_samples
            / data.size
            * 100.0
        )

        return (
            clipping_percent
            >= self.clipping_percent_limit,
            float(clipping_percent),
        )

    # ==========================================================
    # ADC SATURATION
    # ==========================================================

    def detect_adc_saturation(
        self,
        raw_data: np.ndarray,
    ) -> bool:
        """
        Detect integer-domain saturation.

        The method only uses the actual dtype limits.

        It intentionally does NOT assume that every integer
        waveform originated from a specific ADC bit depth.
        """

        if raw_data.size == 0:
            return False

        if raw_data.dtype.kind not in ("i", "u"):
            return False

        info = np.iinfo(raw_data.dtype)

        max_value = np.max(raw_data)
        min_value = np.min(raw_data)

        return bool(
            max_value >= info.max
            or min_value <= info.min
        )

    # ==========================================================
    # FLATLINE
    # ==========================================================

    def detect_adaptive_flatline(
        self,
        data: FloatArray,
        sampling_rate: float,
        std_dev: float,
    ) -> tuple[bool, float]:
        """
        Detect sustained low-variation segments.

        The tolerance combines:
        - absolute numerical tolerance;
        - data-dependent tolerance.
        """

        if data.size < 2:
            return False, 0.0

        if sampling_rate <= 0:
            return False, 0.0

        differences = np.diff(data)

        adaptive_eps = max(
            self.flatline_eps,
            self.flatline_std_factor
            * max(std_dev, self.flatline_eps),
        )

        flat_mask = (
            np.abs(differences)
            <= adaptive_eps
        )

        min_samples = max(
            1,
            int(
                np.ceil(
                    self.flatline_time_threshold_sec
                    * sampling_rate
                )
            ),
        )

        max_run = self._max_run_length(
            flat_mask
        )

        # np.diff has N-1 samples. A run of k equal
        # differences corresponds to approximately
        # k / fs seconds of constant behavior.
        duration = (
            max_run / sampling_rate
        )

        has_flatline = (
            max_run >= min_samples
        )

        return bool(has_flatline), float(duration)

    # ==========================================================
    # SPIKES
    # ==========================================================

    def detect_spikes_mad(
        self,
        data: FloatArray,
    ) -> int:
        """
        Detect impulsive spikes using first differences.

        Modified Z-score:

            z = 0.67448975 * (x - median) / MAD

        A threshold of 6 is used by default.
        """

        if data.size < 3:
            return 0

        differences = np.diff(data)

        median = np.median(differences)

        mad = np.median(
            np.abs(differences - median)
        )

        if np.isclose(mad, 0.0):

            # Fall back to robust standardization only
            # when a non-zero spread exists.
            return 0

        modified_z = (
            MAD_SCALE_FACTOR
            * (differences - median)
            / mad
        )

        return int(
            np.count_nonzero(
                np.abs(modified_z)
                > self.mad_spike_multiplier
            )
        )

    # ==========================================================
    # BASELINE DRIFT
    # ==========================================================

    def detect_normalized_drift(
        self,
        data: FloatArray,
        sampling_rate: float,
        amplitude_scale: float,
    ) -> float:
        """
        Estimate linear baseline drift.

        The fitted slope is converted to total amplitude
        excursion over the record and normalized by a
        robust amplitude scale.
        """

        if data.size < 2:
            return 0.0

        if sampling_rate <= 0:
            return 0.0

        if amplitude_scale <= 0:
            return 0.0

        time = (
            np.arange(data.size, dtype=np.float64)
            / sampling_rate
        )

        slope, _ = np.polyfit(
            time,
            data,
            deg=1,
        )

        duration = (
            (data.size - 1)
            / sampling_rate
        )

        total_drift = abs(
            float(slope) * duration
        )

        drift_percent = (
            total_drift
            / amplitude_scale
            * 100.0
        )

        return float(drift_percent)

    # ==========================================================
    # SNR
    # ==========================================================

    def estimate_seismological_snr(
        self,
        data: FloatArray,
        sampling_rate: float,
    ) -> float | None:
        """
        Estimate an RMS dynamic-range ratio.

        IMPORTANT
        ---------
        Without an independently defined pre-event noise
        window, this is only an approximate SNR indicator.

        The lowest-RMS 1-second window is treated as a
        noise proxy and the highest-RMS window as a signal
        proxy.
        """

        if sampling_rate <= 0:
            return None

        window_size = int(
            round(sampling_rate)
        )

        if window_size < 2:
            return None

        if data.size < 2 * window_size:
            return None

        n_windows = (
            data.size // window_size
        )

        trimmed = data[
            : n_windows * window_size
        ]

        windows = trimmed.reshape(
            n_windows,
            window_size,
        )

        rms = np.sqrt(
            np.mean(
                windows ** 2,
                axis=1,
            )
        )

        if rms.size < 2:
            return None

        noise_rms = float(
            np.percentile(rms, 10)
        )

        signal_rms = float(
            np.percentile(rms, 90)
        )

        if noise_rms <= np.finfo(float).eps:
            return None

        if signal_rms <= noise_rms:
            return 0.0

        snr_db = (
            20.0
            * np.log10(
                signal_rms / noise_rms
            )
        )

        return float(snr_db)

    # ==========================================================
    # ROBUST AMPLITUDE SCALE
    # ==========================================================

    @staticmethod
    def _robust_amplitude_scale(
        data: FloatArray,
    ) -> float:
        """
        Robust amplitude scale based on P95-P5.

        This is less sensitive to isolated extreme samples
        than peak-to-peak amplitude.
        """

        if data.size == 0:
            return 0.0

        p95 = np.percentile(data, 95)
        p05 = np.percentile(data, 5)

        return float(p95 - p05)

    # ==========================================================
    # RUN-LENGTH HELPERS
    # ==========================================================

    @staticmethod
    def _count_runs(
        mask: NDArray[np.bool_],
        min_length: int,
    ) -> int:
        """Count samples belonging to runs >= min_length."""

        if mask.size == 0:
            return 0

        padded = np.concatenate(
            (
                np.array([False]),
                mask,
                np.array([False]),
            )
        )

        changes = np.flatnonzero(
            padded[1:] != padded[:-1]
        )

        if changes.size < 2:
            return 0

        starts = changes[::2]
        ends = changes[1::2]

        lengths = ends - starts

        valid = lengths >= min_length

        return int(
            np.sum(lengths[valid])
        )

    @staticmethod
    def _max_run_length(
        mask: NDArray[np.bool_],
    ) -> int:
        """Return maximum consecutive True run."""

        if mask.size == 0:
            return 0

        padded = np.concatenate(
            (
                np.array([False]),
                mask,
                np.array([False]),
            )
        )

        changes = np.flatnonzero(
            padded[1:] != padded[:-1]
        )

        if changes.size < 2:
            return 0

        starts = changes[::2]
        ends = changes[1::2]

        lengths = ends - starts

        if lengths.size == 0:
            return 0

        return int(np.max(lengths))
