"""Application service for one-station strong-motion analysis.

The service is the boundary between ObsPy objects used for acquisition and
the dependency-free BSMA domain pipeline.  It intentionally contains no UI,
plotting, file export, or Streamlit state.

Operational safety policy
-------------------------
* Instrument response removal is never silently ignored.
* A record without StationXML is accepted only when the caller explicitly
  declares its physical acceleration unit.
* A gap, overlap, or failed QC gate stops station analysis by default.
* Each processing step returns an immutable :class:`ProcessingContext`.
"""

from __future__ import annotations

import logging
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from obspy import Inventory, Stream, Trace

from core.pipeline import PipelineBuilder, PipelineOrchestrator, ProgressCallback
from core.preprocessing.baseline import BaselineCorrectionPlugin
from core.preprocessing.filter import (
    ButterworthFilterPlugin,
    FilterConfig,
    FilterType,
)
from core.preprocessing.integration import (
    IntegrationConfig,
    KinematicIntegrationPlugin,
)
from core.preprocessing.qc import QCAnalyzer, StreamQCReport, TraceQCMetrics
from core.preprocessing.taper import TaperConfig, TaperPlugin
from core.processing.parameters import ParameterConfig, ParameterExtractionPlugin
from core.processing.response_spectrum import (
    ResponseSpectrumConfig,
    ResponseSpectrumPlugin,
    SolverName,
)
from core.types.context import ProcessingContext, WaveformData
from core.types.processing_state import ProcessingState, StageStatus

__all__ = [
    "AnalysisConfiguration",
    "AnalysisService",
    "AnalysisServiceError",
    "extract_summary_data",
    "process_station_stream",
]


_ACCELERATION_UNIT_FACTORS: Mapping[str, float] = {
    "m/s^2": 1.0,
    "m/s2": 1.0,
    "m s-2": 1.0,
    "ms-2": 1.0,
    "gal": 0.01,
    "cm/s^2": 0.01,
    "cm/s2": 0.01,
    "cm s-2": 0.01,
}

BSMA_ENGINE_VERSION = "1.1.0"


class AnalysisServiceError(RuntimeError):
    """Raised when a station cannot safely enter or finish analysis."""


@dataclass(frozen=True, slots=True)
class AnalysisConfiguration:
    """Configuration for the standard offline strong-motion workflow.

    The default sequence is linear baseline correction, a 5% Tukey taper,
    zero-phase fourth-order band-pass filtering, cumulative-trapezoid
    integration, parameter extraction, and a 5%-damped response spectrum.
    It is an analysis recipe, not a replacement for event-specific review of
    the usable frequency band.

    ``input_unit`` is deliberately required when ``inventory`` is absent.
    This prevents raw ADC counts from being mislabeled as m/s².
    """

    baseline_method: str = "linear"
    taper_alpha: float = 0.05
    filter_type: FilterType | str = FilterType.BANDPASS
    freq_min_hz: float = 0.1
    freq_max_hz: float = 25.0
    filter_order: int = 4
    zero_phase: bool = True
    damping_ratio: float = 0.05
    response_periods: tuple[float, ...] | None = None
    response_solver: SolverName = "nigam_jennings"
    response_water_level: float | None = 60.0
    response_pre_filt: tuple[float, float, float, float] | None = None
    input_unit: str | None = None
    input_mode: str = "physical_acceleration"
    adaptive_filter: bool = True
    reject_qc_failure: bool = True
    reject_continuity_issues: bool = True

    def __post_init__(self) -> None:
        baseline_method = self.baseline_method.strip().lower()
        if baseline_method not in {"constant", "linear"}:
            raise ValueError("baseline_method must be 'constant' or 'linear'.")

        try:
            filter_type = FilterType(self.filter_type)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported filter_type: {self.filter_type!r}."
            ) from exc

        if not 0.0 <= self.taper_alpha <= 1.0:
            raise ValueError("taper_alpha must be within [0, 1].")
        if not (0.0 < self.freq_min_hz < self.freq_max_hz):
            raise ValueError("Require 0 < freq_min_hz < freq_max_hz.")
        if self.filter_order < 1:
            raise ValueError("filter_order must be at least 1.")
        if not 0.0 <= self.damping_ratio < 1.0:
            raise ValueError("damping_ratio must be within [0, 1).")
        if self.response_solver not in {"nigam_jennings", "newmark"}:
            raise ValueError("Unsupported response-spectrum solver.")
        if self.response_water_level is not None and self.response_water_level <= 0:
            raise ValueError("response_water_level must be positive or None.")

        if self.response_pre_filt is not None:
            if len(self.response_pre_filt) != 4:
                raise ValueError("response_pre_filt must contain four frequencies.")
            if not all(np.isfinite(value) and value > 0 for value in self.response_pre_filt):
                raise ValueError("response_pre_filt frequencies must be finite and positive.")
            if tuple(sorted(self.response_pre_filt)) != self.response_pre_filt:
                raise ValueError("response_pre_filt frequencies must be strictly increasing.")

        input_unit = self.input_unit.strip().lower() if self.input_unit else None
        if input_unit is not None and input_unit not in _ACCELERATION_UNIT_FACTORS:
            raise ValueError(
                "input_unit must be a declared acceleration unit: "
                f"{sorted(_ACCELERATION_UNIT_FACTORS)}."
            )
        input_mode = self.input_mode.strip().lower()
        if input_mode not in {"raw_counts", "physical_acceleration"}:
            raise ValueError("input_mode must be 'raw_counts' or 'physical_acceleration'.")
        if input_mode == "physical_acceleration" and input_unit is None:
            # Kept as a runtime error for backwards-compatible construction,
            # but the mode is still explicit in every constructed recipe.
            pass

        object.__setattr__(self, "baseline_method", baseline_method)
        object.__setattr__(self, "filter_type", filter_type)
        object.__setattr__(self, "input_unit", input_unit)
        object.__setattr__(self, "input_mode", input_mode)

        # Delegate the period and damping validation to the domain model at
        # configuration time, rather than discovering an invalid recipe while
        # a station is being processed.
        ResponseSpectrumConfig(
            periods=self.response_periods
            if self.response_periods is not None
            else ResponseSpectrumConfig().periods,
            damping=self.damping_ratio,
            solver=self.response_solver,
        )


class AnalysisService:
    """Coordinate QC, response correction, and the BSMA domain pipeline."""

    def __init__(
        self,
        configuration: AnalysisConfiguration | None = None,
        *,
        logger: logging.Logger | None = None,
        qc_analyzer: QCAnalyzer | None = None,
    ) -> None:
        self.configuration = configuration or AnalysisConfiguration()
        self.logger = logger or logging.getLogger(__name__)
        self.qc_analyzer = qc_analyzer or QCAnalyzer(logger=self.logger)

    def build_pipeline(self, *, filter_config: FilterConfig | None = None) -> PipelineOrchestrator:
        """Create the deterministic mathematical pipeline for one trace."""
        config = self.configuration
        spectrum_config = ResponseSpectrumConfig(
            periods=config.response_periods
            if config.response_periods is not None
            else ResponseSpectrumConfig().periods,
            damping=config.damping_ratio,
            solver=config.response_solver,
        )

        return (
            PipelineBuilder(logger=self.logger, halt_on_error=True)
            .add(BaselineCorrectionPlugin(method=config.baseline_method))
            .add(TaperPlugin(config=TaperConfig(alpha=config.taper_alpha)))
            .add(
                ButterworthFilterPlugin(
                    config=filter_config or FilterConfig(
                        type=config.filter_type,
                        freq_min=config.freq_min_hz,
                        freq_max=config.freq_max_hz,
                        corners=config.filter_order,
                        zerophase=config.zero_phase,
                    )
                )
            )
            # The acceleration has already been baseline-corrected and
            # filtered. A second hidden detrend before integration would make
            # the physical processing recipe ambiguous, so it is disabled.
            .add(KinematicIntegrationPlugin(config=IntegrationConfig()))
            .add(ParameterExtractionPlugin(config=ParameterConfig()))
            .add(ResponseSpectrumPlugin(config=spectrum_config))
            .build()
        )

    def process_station_stream(
        self,
        station_stream: Stream,
        inventory: Inventory | None = None,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, ProcessingContext]:
        """Process one station stream and return contexts keyed by channel.

        The input ``Stream`` is never mutated.  Segmented duplicate channels
        are rejected rather than silently overwriting one segment with another;
        the ingestion layer must first resolve continuity explicitly.
        """
        self._validate_station_stream(station_stream)
        qc_report = self.qc_analyzer.analyze_stream(
            station_stream,
            identifier=self._stream_identifier(station_stream),
        )
        self._validate_qc_gate(qc_report)

        contexts: dict[str, ProcessingContext] = {}

        for trace in station_stream:
            channel = str(trace.stats.channel)
            try:
                filter_config, filter_decision = self._recommend_filter(trace, qc_report.trace_metrics[trace.id])
                pipeline = self.build_pipeline(filter_config=filter_config)
                context = self._create_initial_context(
                    trace=trace,
                    inventory=inventory,
                    qc_metrics=qc_report.trace_metrics[trace.id],
                ).add_history("FilterRecommendation", filter_decision)
                contexts[channel] = pipeline.run(
                    context,
                    identifier=trace.id,
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                self.logger.exception("BSMA analysis failed for trace %s.", trace.id)
                raise AnalysisServiceError(
                    f"Analysis failed for trace {trace.id}: {exc}"
                ) from exc

        return contexts

    def _validate_station_stream(self, stream: Stream) -> None:
        if not isinstance(stream, Stream):
            raise TypeError("station_stream must be an ObsPy Stream.")
        if not stream:
            raise AnalysisServiceError("Cannot analyse an empty station stream.")

        stations = {str(trace.stats.station) for trace in stream}
        if len(stations) != 1:
            raise AnalysisServiceError(
                "AnalysisService accepts exactly one station per request; "
                f"received {sorted(stations)}."
            )

        channels: set[str] = set()
        for trace in stream:
            channel = str(trace.stats.channel)
            if channel in channels:
                raise AnalysisServiceError(
                    f"Duplicate channel {channel!r}. Resolve stream segments "
                    "and their gaps/overlaps before analysis."
                )
            channels.add(channel)

            data = np.asarray(trace.data)
            if np.ma.isMaskedArray(data) or data.ndim != 1 or data.size < 2:
                raise AnalysisServiceError(
                    f"Trace {trace.id} must be an unmasked one-dimensional "
                    "waveform containing at least two samples."
                )
            if not np.all(np.isfinite(data)):
                raise AnalysisServiceError(f"Trace {trace.id} contains NaN or Inf.")
            if not np.isfinite(trace.stats.sampling_rate) or trace.stats.sampling_rate <= 0:
                raise AnalysisServiceError(f"Trace {trace.id} has an invalid sampling rate.")

    def _validate_qc_gate(self, report: StreamQCReport) -> None:
        if self.configuration.reject_continuity_issues and (
            report.gap_count or report.overlap_count
        ):
            raise AnalysisServiceError(
                "Waveform continuity check failed: "
                f"{report.gap_count} gap(s), {report.overlap_count} overlap(s)."
            )
        if self.configuration.reject_qc_failure and not report.is_passed:
            failed = [
                trace_id
                for trace_id, metrics in report.trace_metrics.items()
                if not metrics.is_valid
            ]
            raise AnalysisServiceError(
                "Raw waveform QC failed for trace(s): " + ", ".join(failed)
            )

    def _create_initial_context(
        self,
        *,
        trace: Trace,
        inventory: Inventory | None,
        qc_metrics: TraceQCMetrics,
    ) -> ProcessingContext:
        raw_data = np.asarray(trace.data, dtype=np.float64)
        sampling_rate = float(trace.stats.sampling_rate)

        if self.configuration.input_mode == "raw_counts" and inventory is None:
            raise AnalysisServiceError(
                f"Raw-count mode requires matching StationXML for {trace.id}; "
                "do not process counts as physical acceleration."
            )
        if self.configuration.input_mode == "physical_acceleration" and inventory is not None:
            raise AnalysisServiceError(
                "Physical-acceleration mode forbids instrument response removal. "
                "Select raw-count mode only when the input samples are ADC counts."
            )

        if inventory is not None:
            raw_waveform = WaveformData(
                data=raw_data,
                sampling_rate=sampling_rate,
                unit="counts",
            )
            acceleration_data = self._remove_response(trace, inventory)
            response_state = StageStatus.SUCCESS
            response_details: Mapping[str, Any] = {
                "method": "obspy_remove_response",
                "output_unit": "m/s^2",
                "water_level": self.configuration.response_water_level,
            }
        else:
            if self.configuration.input_unit is None:
                raise AnalysisServiceError(
                    f"StationXML is absent for {trace.id}. Declare input_unit "
                    "explicitly to confirm that the samples are acceleration."
                )
            raw_waveform = WaveformData(
                data=raw_data,
                sampling_rate=sampling_rate,
                unit=self.configuration.input_unit,
            )
            acceleration_data = raw_data * _ACCELERATION_UNIT_FACTORS[
                self.configuration.input_unit
            ]
            response_state = StageStatus.SKIPPED
            response_details = {
                "method": "caller_declared_physical_acceleration",
                "input_unit": self.configuration.input_unit,
                "output_unit": "m/s^2",
            }

        acceleration = WaveformData(
            data=acceleration_data,
            sampling_rate=sampling_rate,
            unit="m/s^2",
        )
        state = replace(
            ProcessingState(),
            response_correction=response_state,
            qc=StageStatus.SUCCESS,
        )
        context = ProcessingContext(
            trace_id=trace.id,
            metadata=dict(trace.stats),
            raw_waveform=raw_waveform,
            acceleration=acceleration,
            processing_state=state,
            qc=qc_metrics,
            config={
                "freq_min_hz": self.configuration.freq_min_hz,
                "freq_max_hz": self.configuration.freq_max_hz,
                "damping_ratio": self.configuration.damping_ratio,
                "input_mode": self.configuration.input_mode,
                "input_unit": raw_waveform.unit,
                "engine_version": BSMA_ENGINE_VERSION,
            },
        )
        return (
            context.add_history(
                "ScientificProvenance",
                {
                    "engine_version": BSMA_ENGINE_VERSION,
                    "processed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_file": trace.stats.get("bsma_source_file", "unknown"),
                    "source_sha256": trace.stats.get("bsma_source_sha256", "unknown"),
                    "sample_sha256": hashlib.sha256(raw_data.tobytes()).hexdigest(),
                    "input_mode": self.configuration.input_mode,
                    "input_unit": raw_waveform.unit,
                },
            )
            .add_history("RawQC", qc_metrics.to_dict())
            .add_history("InstrumentResponse", dict(response_details))
        )

    def _recommend_filter(
        self,
        trace: Trace,
        qc_metrics: TraceQCMetrics,
    ) -> tuple[FilterConfig, dict[str, Any]]:
        """Derive a conservative, auditable band from sampling and SNR.

        This is a screening recommendation, not a replacement for an
        event-specific noise-spectrum review.  The high corner is always
        constrained below 80% of Nyquist; the low corner rises for weak SNR
        to suppress low-frequency drift in subsequent integration.
        """
        sampling_rate = float(trace.stats.sampling_rate)
        nyquist = sampling_rate / 2.0
        requested_low = self.configuration.freq_min_hz
        requested_high = self.configuration.freq_max_hz
        snr_db = qc_metrics.snr_estimate_db
        recommended_low = requested_low
        rationale = "manual configuration retained"
        if self.configuration.adaptive_filter:
            if snr_db is None:
                recommended_low = max(requested_low, 0.20)
                rationale = "SNR unavailable; conservative 0.20 Hz floor applied"
            elif snr_db < 10.0:
                recommended_low = max(requested_low, 0.40)
                rationale = "low SNR (<10 dB); 0.40 Hz floor applied"
            elif snr_db < 20.0:
                recommended_low = max(requested_low, 0.20)
                rationale = "moderate SNR (10-20 dB); 0.20 Hz floor applied"
            else:
                recommended_low = max(requested_low, 0.10)
                rationale = "good SNR (>=20 dB); requested low corner retained"
        recommended_high = min(requested_high, nyquist * 0.80)
        if recommended_low >= recommended_high:
            recommended_low = max(0.001, recommended_high * 0.25)
            rationale += "; low corner reduced to preserve a valid band"
        config = FilterConfig(
            type=self.configuration.filter_type,
            freq_min=float(recommended_low),
            freq_max=float(recommended_high),
            corners=self.configuration.filter_order,
            zerophase=self.configuration.zero_phase,
        )
        return config, {
            "status": "RECOMMENDED" if self.configuration.adaptive_filter else "MANUAL",
            "method": "sampling_rate_nyquist_and_snr_screening",
            "sampling_rate_hz": sampling_rate,
            "nyquist_hz": nyquist,
            "snr_db": snr_db,
            "requested_low_hz": requested_low,
            "requested_high_hz": requested_high,
            "selected_low_hz": config.freq_min,
            "selected_high_hz": config.freq_max,
            "rationale": rationale,
        }

    def _remove_response(self, trace: Trace, inventory: Inventory) -> np.ndarray:
        corrected = trace.copy()
        sampling_rate = float(corrected.stats.sampling_rate)
        pre_filt = self._response_prefilter(sampling_rate)
        try:
            corrected.remove_response(
                inventory=inventory,
                output="ACC",
                water_level=self.configuration.response_water_level,
                pre_filt=pre_filt,
            )
        except Exception as exc:
            raise AnalysisServiceError(
                f"Instrument response removal failed for {trace.id}."
            ) from exc

        data = np.asarray(corrected.data, dtype=np.float64)
        if data.ndim != 1 or data.size < 2 or not np.all(np.isfinite(data)):
            raise AnalysisServiceError(
                f"Response removal produced invalid acceleration for {trace.id}."
            )
        return data

    def _response_prefilter(
        self,
        sampling_rate: float,
    ) -> tuple[float, float, float, float]:
        nyquist = sampling_rate / 2.0
        configured = self.configuration.response_pre_filt
        if configured is not None:
            if configured[-1] >= nyquist:
                raise AnalysisServiceError(
                    "response_pre_filt must end below the trace Nyquist frequency "
                    f"({nyquist:g} Hz)."
                )
            return configured

        # A response-removal pre-filter should be broader than the retained
        # analysis band but must stay below Nyquist.  It prevents division by
        # a poorly constrained response near 0 Hz and at the digitizer limit.
        f1 = max(0.001, self.configuration.freq_min_hz * 0.5)
        f2 = self.configuration.freq_min_hz
        f3 = min(self.configuration.freq_max_hz, nyquist * 0.80)
        f4 = min(nyquist * 0.95, f3 * 1.20)
        if not 0.0 < f1 < f2 < f3 < f4 < nyquist:
            raise AnalysisServiceError(
                "Cannot derive a valid response pre-filter from the requested "
                f"analysis band and sampling rate ({sampling_rate:g} Hz)."
            )
        return f1, f2, f3, f4

    @staticmethod
    def _stream_identifier(stream: Stream) -> str:
        return f"{stream[0].stats.network}.{stream[0].stats.station}"


def process_station_stream(
    station_stream: Stream,
    inventory: Inventory | None = None,
    *,
    configuration: AnalysisConfiguration | None = None,
    logger: logging.Logger | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, ProcessingContext]:
    """Convenience wrapper for UI and CLI callers."""
    return AnalysisService(configuration, logger=logger).process_station_stream(
        station_stream,
        inventory,
        progress_callback=progress_callback,
    )


def extract_summary_data(
    station_code: str,
    contexts: Mapping[str, ProcessingContext],
) -> list[dict[str, Any]]:
    """Return presentation-neutral rows for one station's processed traces.

    SIG classification and document formatting deliberately remain in the
    reporting layer; this service only exposes computed physical quantities.
    """
    rows: list[dict[str, Any]] = []
    for channel, context in contexts.items():
        metrics = context.metrics
        qc = context.qc
        rows.append(
            {
                "station": station_code,
                "channel": channel,
                "pga_gal": float(metrics.get("PGA", 0.0)) * 100.0,
                "pgv_cm_s": float(metrics.get("PGV", 0.0)) * 100.0,
                "pgd_cm": float(metrics.get("PGD", 0.0)) * 100.0,
                "arias_intensity_m_s": float(
                    metrics.get("Arias_Intensity", 0.0)
                ),
                "significant_duration_d5_95_s": float(
                    metrics.get("Significant_Duration_D5_95", 0.0)
                ),
                "qc_valid": bool(qc.is_valid) if qc is not None else None,
                "qc_score": int(qc.quality_score) if qc is not None else None,
            }
        )
    return rows
