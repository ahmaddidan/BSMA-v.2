"""Bounded, failure-isolated batch orchestration for BSMA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from obspy import Inventory, Stream

from core.types.context import ProcessingContext
from services.analysis_service import AnalysisService, extract_summary_data

__all__ = ["BatchProgressCallback", "BatchResult", "BatchService"]

BatchProgressCallback = Callable[[int, int, str], None]


@dataclass(slots=True)
class BatchResult:
    """Successful outputs and explicit failures from one batch request."""

    contexts_by_station: dict[str, dict[str, ProcessingContext]] = field(
        default_factory=dict
    )
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def processed_station_count(self) -> int:
        return len(self.contexts_by_station)

    @property
    def failed_station_count(self) -> int:
        return len(self.failures)

    def summary_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for station, contexts in self.contexts_by_station.items():
            rows.extend(extract_summary_data(station, contexts))
        return rows


class BatchService:
    """Process independent station streams sequentially and transparently.

    The sequence is deliberate for the first operational release: it bounds
    memory while response correction and response-spectrum calculations are
    performed.  Fifty stations are supported through progress callbacks and
    failure isolation; a later bounded-worker implementation can retain this
    public API without changing the dashboard.
    """

    def __init__(self, analysis_service: AnalysisService) -> None:
        self.analysis_service = analysis_service

    def process_stations(
        self,
        streams_by_station: Mapping[str, Stream],
        inventories_by_station: Mapping[str, Inventory | None] | None = None,
        *,
        progress_callback: BatchProgressCallback | None = None,
    ) -> BatchResult:
        if not streams_by_station:
            raise ValueError("streams_by_station must not be empty.")

        result = BatchResult()
        inventory_lookup = inventories_by_station or {}
        total = len(streams_by_station)

        for index, (station, stream) in enumerate(
            streams_by_station.items(),
            start=1,
        ):
            if progress_callback is not None:
                progress_callback(index, total, station)
            try:
                result.contexts_by_station[station] = (
                    self.analysis_service.process_station_stream(
                        stream,
                        inventory_lookup.get(station),
                    )
                )
            except Exception as exc:
                result.failures[station] = str(exc)

        return result
