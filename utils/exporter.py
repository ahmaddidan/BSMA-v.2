"""
BMKG Strong Motion Analyzer (BSMA)
Module: utils/exporter.py

Description
-----------
Utilities for exporting ProcessingContext results into JSON and CSV
formats.

The exporter is intentionally kept outside the processing domain and
does not modify ProcessingContext.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from core.types.context import ProcessingContext
from utils.exceptions import ErrorCode, ReportError, SeverityLevel

__all__ = ["BSMAExporter"]


class BSMAExporter:
    """
    Export BSMA processing results into JSON and CSV files.

    Parameters
    ----------
    output_dir
        Root directory used for BSMA outputs.

    Notes
    -----
    The exporter never mutates the supplied ProcessingContext.
    """

    def __init__(self, output_dir: str | Path = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "reports"

        try:
            self.reports_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
        except OSError as exc:
            raise ReportError(
                message=(
                    "Failed to create BSMA reports directory."
                ),
                error_code=ErrorCode.RP001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "exporter",
                    "output_dir": str(self.output_dir),
                    "reports_dir": str(self.reports_dir),
                },
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def export_to_json(
        self,
        context: ProcessingContext,
    ) -> Path:
        """
        Export one ProcessingContext result into a JSON file.

        Parameters
        ----------
        context
            Completed or partially completed BSMA processing context.

        Returns
        -------
        pathlib.Path
            Path to the generated JSON file.

        Raises
        ------
        ReportError
            If serialization or file writing fails.
        """

        timestamp = datetime.now(timezone.utc)
        timestamp_text = timestamp.strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )

        safe_trace_id = self._safe_filename(context.trace_id)

        filepath = (
            self.reports_dir
            / f"{safe_trace_id}_{timestamp_text}.json"
        )

        payload: dict[str, Any] = {
            "trace_id": context.trace_id,
            "export_time": timestamp.isoformat(),
            "metrics": context.metrics,
            "processing_history": list(context.history),
        }

        try:
            with filepath.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    payload,
                    file,
                    indent=4,
                    ensure_ascii=False,
                    default=self._json_serializer,
                )

        except (OSError, TypeError, ValueError) as exc:
            raise ReportError(
                message="Failed to export ProcessingContext to JSON.",
                error_code=ErrorCode.RP001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "exporter",
                    "trace_id": context.trace_id,
                    "filepath": str(filepath),
                },
                cause=exc,
            ) from exc

        return filepath

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def export_to_csv(
        self,
        context: ProcessingContext,
        filename: str = "bsma_summary.csv",
    ) -> Path:
        """
        Append processing metrics to a CSV summary file.

        The CSV schema is dynamically expanded when a new metric appears.
        Existing rows are preserved and missing values are left blank.

        Parameters
        ----------
        context
            Processing context containing the metrics to export.

        filename
            CSV filename inside the reports directory.

        Returns
        -------
        pathlib.Path
            Path to the CSV summary file.

        Raises
        ------
        ReportError
            If the CSV cannot be read, rewritten, or written.
        """

        safe_filename = self._safe_filename(filename)

        if not safe_filename.lower().endswith(".csv"):
            safe_filename += ".csv"

        filepath = self.reports_dir / safe_filename

        base_headers = [
            "timestamp",
            "trace_id",
        ]

        current_metrics = list(context.metrics.keys())

        try:
            existing_headers, existing_rows = (
                self._read_existing_csv(filepath)
            )

            if existing_headers:
                headers = self._merge_headers(
                    existing_headers,
                    current_metrics,
                )
            else:
                headers = [
                    *base_headers,
                    *current_metrics,
                ]

            timestamp = datetime.now(timezone.utc).isoformat()

            row: dict[str, Any] = {
                "timestamp": timestamp,
                "trace_id": context.trace_id,
            }

            row.update(context.metrics)

            self._write_csv(
                filepath=filepath,
                headers=headers,
                rows=[
                    *existing_rows,
                    row,
                ],
            )

        except (OSError, csv.Error, TypeError, ValueError) as exc:
            raise ReportError(
                message="Failed to export ProcessingContext to CSV.",
                error_code=ErrorCode.RP001,
                severity=SeverityLevel.ERROR,
                context={
                    "module": "exporter",
                    "trace_id": context.trace_id,
                    "filepath": str(filepath),
                },
                cause=exc,
            ) from exc

        return filepath

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_filename(value: str) -> str:
        """
        Convert an arbitrary filename or trace identifier into a
        filesystem-safe representation.
        """

        value = str(value).strip()

        if not value:
            return "bsma_output"

        invalid_chars = '<>:"/\\|?*'

        sanitized = "".join(
            "_"
            if char in invalid_chars
            else char
            for char in value
        )

        return sanitized.rstrip(". ")

    @staticmethod
    def _json_serializer(value: Any) -> Any:
        """
        Convert common NumPy scalar/array objects into JSON-compatible
        Python objects.
        """

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, np.bool_):
            return bool(value)

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        raise TypeError(
            f"Object of type {type(value).__name__} "
            "is not JSON serializable."
        )

    @staticmethod
    def _read_existing_csv(
        filepath: Path,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """
        Read an existing CSV file.

        Returns
        -------
        tuple[list[str], list[dict[str, Any]]]
            Existing header and rows.
        """

        if not filepath.is_file():
            return [], []

        with filepath.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)

            headers = list(reader.fieldnames or [])
            rows = list(reader)

        return headers, rows

    @staticmethod
    def _merge_headers(
        existing_headers: list[str],
        new_metrics: list[str],
    ) -> list[str]:
        """
        Merge existing CSV columns with newly discovered metrics.

        Existing column order is preserved. New metrics are appended.
        """

        headers = list(existing_headers)

        for metric in new_metrics:
            if metric not in headers:
                headers.append(metric)

        return headers

    @staticmethod
    def _write_csv(
        filepath: Path,
        headers: list[str],
        rows: list[dict[str, Any]],
    ) -> None:
        """
        Rewrite the complete CSV with the unified schema.
        """

        with filepath.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=headers,
                extrasaction="ignore",
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(row)