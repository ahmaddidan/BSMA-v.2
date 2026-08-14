"""Application service for BSMA operational exports.

This layer owns export orchestration.  The dashboard only requests exports;
it does not perform CSV serialization, chart rendering, or PDF assembly.
"""

from __future__ import annotations

import tempfile
import re
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.io.exporter import ResultExporter
from core.io.pdf_exporter import generate_station_pdf
from core.types.context import ProcessingContext
from services.analysis_service import extract_summary_data
from utils.pdf_exporter import export_station_report, generate_4panel_waveform

__all__ = ["ExportService"]


class ExportService:
    """Generate CSV and PDF outputs from completed processing contexts."""

    def export_summary_csv(
        self,
        rows: list[Mapping[str, Any]],
        output_path: str | Path,
    ) -> Path:
        """Write presentation-neutral summary rows to a UTF-8 CSV file."""
        return ResultExporter.export_to_csv(rows, output_path)

    def export_batch_csv(
        self,
        contexts_by_station: Mapping[str, Mapping[str, ProcessingContext]],
        output_path: str | Path,
    ) -> Path:
        """Export all successfully processed station components to CSV."""
        rows: list[dict[str, Any]] = []
        for station, contexts in contexts_by_station.items():
            rows.extend(extract_summary_data(station, contexts))
        return self.export_summary_csv(rows, output_path)

    def export_station_pdf(
        self,
        station_code: str,
        contexts: Mapping[str, ProcessingContext],
        output_path: str | Path,
        *,
        event_info: Mapping[str, Any] | None = None,
        logo_path: str | Path | None = None,
    ) -> Path:
        """Generate an operational station PDF with waveform and PSA plots.

        The report uses the component with the largest PGA for headline
        parameters and includes PSA curves for every supplied component.
        """
        if not contexts:
            raise ValueError("contexts must contain at least one component.")

        strongest = max(
            contexts.values(),
            key=lambda context: float(context.metrics.get("PGA", 0.0)),
        )
        pga_gal = float(strongest.metrics.get("PGA", 0.0)) * 100.0
        sig_status = self._sig_label(pga_gal)

        # The full operational report includes the executive summary, SIG
        # interpretation, parameter table, spectrum, and every component's
        # waveform.  Retain the compact generator only when callers need
        # custom event metadata or branding.
        if not event_info and logo_path is None:
            safe_station_code = re.sub(r'[<>:"/\\|?*]+', "_", station_code)
            return export_station_report(
                safe_station_code,
                dict(contexts),
                Path(output_path).parent,
                output_path=output_path,
            )

        with tempfile.TemporaryDirectory(prefix="bsma_pdf_") as directory:
            plot_paths = self._render_plots(contexts, Path(directory))
            return generate_station_pdf(
                metadata=strongest.metadata,
                params=strongest.metrics,
                sig_status=sig_status,
                plot_paths=plot_paths,
                output_pdf_path=output_path,
                event_info=event_info,
                logo_path=logo_path,
            )

    def export_station_waveform_pngs(
        self,
        station_code: str,
        contexts: Mapping[str, ProcessingContext],
        output_directory: str | Path,
    ) -> list[Path]:
        """Export one annotated four-panel waveform PNG for every component."""
        if not contexts:
            raise ValueError("contexts must contain at least one component.")
        destination = Path(output_directory)
        destination.mkdir(parents=True, exist_ok=True)
        safe_station = re.sub(r'[<>:"/\\|?*]+', "_", station_code)
        paths: list[Path] = []
        for channel, context in contexts.items():
            safe_channel = re.sub(r'[<>:"/\\|?*]+', "_", channel)
            path = destination / f"BSMA_Waveform_{safe_station}_{safe_channel}.png"
            if generate_4panel_waveform(context, channel, path, safe_station):
                paths.append(path)
        return paths

    @staticmethod
    def _sig_label(pga_gal: float) -> str:
        """Classify PGA using the BMKG instrumental-intensity scale provided by the project table."""
        gravity = 9.80665
        thresholds = (
            (0.05 * gravity, "SIG I"),
            (0.30 * gravity, "SIG II"),
            (2.8 * gravity, "SIG III"),
            (6.2 * gravity, "SIG IV"),
            (12.0 * gravity, "SIG V"),
            (22.0 * gravity, "SIG VI"),
            (40.0 * gravity, "SIG VII"),
            (75.0 * gravity, "SIG VIII"),
            (139.0 * gravity, "SIG IX"),
        )
        if not np.isfinite(pga_gal):
            return "SIG I"
        for threshold, label in thresholds:
            if pga_gal < threshold:
                return label
        return "SIG X+"

    @staticmethod
    def _render_plots(
        contexts: Mapping[str, ProcessingContext],
        directory: Path,
    ) -> dict[str, Path]:
        strongest = max(
            contexts.values(),
            key=lambda context: float(context.metrics.get("PGA", 0.0)),
        )
        acceleration = strongest.acceleration
        velocity = strongest.velocity
        displacement = strongest.displacement
        if acceleration is None or velocity is None or displacement is None:
            raise ValueError("PDF export requires completed kinematic products.")

        time = np.arange(acceleration.npts) / acceleration.sampling_rate
        time_path = directory / "time_histories.png"
        figure, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        for axis, data, label in (
            (axes[0], acceleration.data, "Acceleration (m/s2)"),
            (axes[1], velocity.data, "Velocity (m/s)"),
            (axes[2], displacement.data, "Displacement (m)"),
        ):
            axis.plot(time, data, linewidth=0.8)
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Time (s)")
        figure.tight_layout()
        figure.savefig(time_path, dpi=160)
        plt.close(figure)

        spectrum_path = directory / "response_spectrum.png"
        figure, axis = plt.subplots(figsize=(9, 4.5))
        for channel, context in contexts.items():
            # The domain contract uses the lower-case ``periods`` key.
            # Keeping this lookup aligned with the response-spectrum plugin
            # ensures the PDF never silently emits an empty spectrum panel.
            periods = context.spectral_data.get("periods")
            psa = context.spectral_data.get("PSA")
            if periods is None or psa is None:
                continue
            axis.plot(periods, np.asarray(psa) / 9.80665, label=channel)
        axis.set_xscale("log")
        axis.set_xlabel("Period (s)")
        axis.set_ylabel("PSA (g)")
        axis.grid(True, which="both", alpha=0.25)
        if len(contexts) > 1:
            axis.legend()
        figure.tight_layout()
        figure.savefig(spectrum_path, dpi=160)
        plt.close(figure)

        return {
            "Time histories": time_path,
            "5% damped response spectrum": spectrum_path,
        }
