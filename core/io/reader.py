"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/io/reader.py

Scientific-grade waveform ingestion layer.

Responsibilities
----------------
1. Read MiniSEED / MiniSEED-compatible waveform files.
2. Read SAC waveform files.
3. Optionally load StationXML / XML instrument metadata.
4. Validate waveform integrity before entering the processing pipeline.
5. Detect gaps and overlaps without silently fabricating waveform samples.
6. Validate temporal sampling consistency.
7. Merge compatible waveform fragments conservatively.
8. Construct the canonical ProcessingContext.

Non-responsibilities
--------------------
This module MUST NOT:
- remove instrument response,
- filter waveform,
- detrend waveform,
- taper waveform,
- integrate acceleration,
- calculate PGA/PGV/PGD,
- interpolate unknown data gaps,
- modify physical observations for downstream convenience.

Scientific Principle
--------------------
The reader is an ingestion boundary.

Raw observational data must remain observational data. Any
transformation that changes the physical waveform must occur in an
explicit downstream processing stage and must be recorded in
processing history/provenance.

Supported waveform formats
--------------------------
- .mseed
- .miniseed
- .msd
- .sac

Supported response metadata
---------------------------
- .xml
- .stationxml

Dependencies
------------
- NumPy
- ObsPy
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import ClassVar, Iterable

import numpy as np
from obspy import Inventory, Stream, read, read_inventory

from core.types.context import ProcessingContext
from core.types.metadata import TraceMetadata
from core.types.processing_state import ProcessingState


__all__ = [
    "WaveformReader",
]


class WaveformReader:
    """
    Production-grade waveform reader for BSMA.

    The reader establishes the ingestion boundary between external
    waveform files and the internal BSMA ProcessingContext.

    Important
    ---------
    No signal-processing operation is performed here. In particular,
    data gaps are NEVER filled using interpolation.

    Parameters
    ----------
    max_workers:
        Maximum number of workers used by batch reading utilities.
    logger:
        Optional logger instance.
    """

    SUPPORTED_WAVEFORM_EXTENSIONS: ClassVar[frozenset[str]] = frozenset(
        {
            ".mseed",
            ".miniseed",
            ".msd",
            ".sac",
        }
    )

    SUPPORTED_RESPONSE_EXTENSIONS: ClassVar[frozenset[str]] = frozenset(
        {
            ".xml",
            ".stationxml",
        }
    )

    SAMPLING_RATE_RTOL: ClassVar[float] = 1.0e-5
    SAMPLING_RATE_ATOL: ClassVar[float] = 1.0e-8

    DELTA_RTOL: ClassVar[float] = 1.0e-5
    DELTA_ATOL: ClassVar[float] = 1.0e-10

    MIN_REQUIRED_SAMPLES: ClassVar[int] = 2

    def __init__(
        self,
        max_workers: int = 4,
        logger: logging.Logger | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError(
                f"max_workers must be >= 1, got {max_workers}"
            )

        self.max_workers = int(max_workers)
        self.logger = logger or logging.getLogger(__name__)

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def read_to_context(
        self,
        file_path: Path | str,
        response_path: Path | str | None = None,
    ) -> ProcessingContext:
        """
        Read a waveform and construct a ProcessingContext.

        Parameters
        ----------
        file_path:
            Path to MiniSEED or SAC waveform.

        response_path:
            Optional path to StationXML/XML instrument response.

            If omitted, the reader attempts conservative auto-discovery
            in the waveform directory. Auto-discovery is only accepted
            when exactly one compatible XML response file can be
            identified.

        Returns
        -------
        ProcessingContext
            Validated BSMA processing context.

        Notes
        -----
        Instrument response removal is intentionally NOT performed here.
        The Inventory is loaded as acquisition metadata and must be used
        by the dedicated response-correction stage.
        """
        path = Path(file_path).expanduser().resolve()
        start_time = time.perf_counter()

        self._validate_file_path(path)
        self._validate_waveform_extension(path)

        inventory: Inventory | None = None
        response_source: Path | None = None

        try:
            stream = read(str(path))

            if len(stream) == 0:
                raise ValueError(
                    f"ObsPy returned an empty Stream: {path.name}"
                )

            self._validate_raw_stream(stream, path.name)

            response_source = self._resolve_response_path(
                waveform_path=path,
                response_path=response_path,
            )

            if response_source is not None:
                inventory = self._read_inventory(response_source)

                self._validate_inventory(
                    inventory=inventory,
                    identifier=response_source.name,
                )

                self._validate_response_compatibility(
                    stream=stream,
                    inventory=inventory,
                )

            stream = self._prepare_stream(stream)

            self._validate_stream(
                stream=stream,
                identifier=path.name,
            )

            self._validate_sampling_consistency(
                stream=stream,
                identifier=path.name,
            )

            self._validate_temporal_consistency(
                stream=stream,
                identifier=path.name,
            )

            context = self._build_processing_context(
                stream=stream,
                path=path,
            )

            self._log_ingestion_summary(
                path=path,
                stream=stream,
                response_path=response_source,
                elapsed=time.perf_counter() - start_time,
            )

            return context

        except Exception:
            self.logger.exception(
                "Waveform ingestion failed.",
                extra={
                    "bsma_context": {
                        "module": "waveform_reader",
                        "file": str(path),
                        "response_file": (
                            str(response_source)
                            if response_source is not None
                            else None
                        ),
                    }
                },
            )
            raise

        finally:
            elapsed = time.perf_counter() - start_time

            self.logger.debug(
                "Waveform ingestion completed.",
                extra={
                    "bsma_context": {
                        "module": "waveform_reader",
                        "file": path.name,
                        "elapsed_seconds": elapsed,
                    }
                },
            )

    def read_batch(
        self,
        file_paths: Iterable[Path | str],
    ) -> dict[Path, ProcessingContext | Exception]:
        """
        Read multiple waveform files concurrently.

        Each file is processed independently.

        A failure in one file does not terminate the complete batch.

        Returns
        -------
        dict[Path, ProcessingContext | Exception]
            Mapping between input path and either successful context
            or the exception raised during ingestion.
        """
        paths = [
            Path(path).expanduser().resolve()
            for path in file_paths
        ]

        if not paths:
            return {}

        results: dict[Path, ProcessingContext | Exception] = {}

        with ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="bsma-reader",
        ) as executor:
            futures = {
                executor.submit(self.read_to_context, path): path
                for path in paths
            }

            for future in as_completed(futures):
                path = futures[future]

                try:
                    results[path] = future.result()
                except Exception as exc:
                    results[path] = exc

        return results

    # ==================================================================
    # RESPONSE / STATIONXML
    # ==================================================================

    def _resolve_response_path(
        self,
        waveform_path: Path,
        response_path: Path | str | None,
    ) -> Path | None:
        """
        Resolve StationXML/XML response source.

        Explicit response paths have priority.

        When no response path is supplied, the reader performs a
        conservative directory-level search. Automatic selection is
        accepted only when exactly one compatible response file exists.

        The reader deliberately avoids guessing between multiple XML
        files because applying the wrong instrument response can produce
        physically invalid acceleration amplitudes.
        """
        if response_path is not None:
            resolved = Path(response_path).expanduser().resolve()

            self._validate_file_path(resolved)
            self._validate_response_extension(resolved)

            return resolved

        candidates = sorted(
            path
            for path in waveform_path.parent.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in self.SUPPORTED_RESPONSE_EXTENSIONS
            )
        )

        if not candidates:
            self.logger.warning(
                "No StationXML/XML response file found.",
                extra={
                    "bsma_context": {
                        "module": "waveform_reader",
                        "waveform": waveform_path.name,
                    }
                },
            )
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Try exact station-name matching before refusing to guess.
        try:
            waveform_station = self._extract_station_code_from_path(
                waveform_path
            )
        except Exception:
            waveform_station = None

        if waveform_station:
            station_matches = [
                candidate
                for candidate in candidates
                if waveform_station.upper()
                in candidate.stem.upper()
            ]

            if len(station_matches) == 1:
                return station_matches[0]

        self.logger.warning(
            "Multiple XML response files found; automatic selection "
            "disabled to prevent applying the wrong instrument response.",
            extra={
                "bsma_context": {
                    "module": "waveform_reader",
                    "waveform": waveform_path.name,
                    "candidates": [p.name for p in candidates],
                }
            },
        )

        return None

    def _read_inventory(
        self,
        response_path: Path,
    ) -> Inventory:
        """Read StationXML/XML response metadata."""
        try:
            inventory = read_inventory(str(response_path))
        except Exception as exc:
            raise ValueError(
                "Failed to read StationXML/XML response: "
                f"{response_path}"
            ) from exc

        if not isinstance(inventory, Inventory):
            raise TypeError(
                "ObsPy did not return an Inventory object for "
                f"{response_path}"
            )

        return inventory

    def _validate_inventory(
        self,
        inventory: Inventory,
        identifier: str,
    ) -> None:
        """Validate basic StationXML inventory integrity."""
        if len(inventory.networks) == 0:
            raise ValueError(
                f"StationXML/XML contains no networks: {identifier}"
            )

        station_count = sum(
            len(network.stations)
            for network in inventory.networks
        )

        if station_count == 0:
            raise ValueError(
                f"StationXML/XML contains no stations: {identifier}"
            )

    def _validate_response_compatibility(
        self,
        stream: Stream,
        inventory: Inventory,
    ) -> None:
        """
        Verify that waveform traces can be associated with the Inventory.

        This validation is intentionally conservative.

        A response mismatch is not silently ignored because using an
        incorrect response can change the physical amplitude by orders
        of magnitude.
        """
        for trace in stream:
            stats = trace.stats

            network = getattr(stats, "network", "")
            station = getattr(stats, "station", "")
            location = getattr(stats, "location", "")
            channel = getattr(stats, "channel", "")

            try:
                response = inventory.get_response(
                    seed_id=trace.id,
                    datetime=stats.starttime,
                )
            except Exception as exc:
                raise ValueError(
                    "No compatible instrument response found for trace "
                    f"{trace.id} at {stats.starttime}"
                ) from exc

            if response is None:
                raise ValueError(
                    "Instrument response resolution returned None for "
                    f"{trace.id}"
                )

            self.logger.debug(
                "Instrument response matched.",
                extra={
                    "bsma_context": {
                        "network": network,
                        "station": station,
                        "location": location,
                        "channel": channel,
                        "trace_id": trace.id,
                    }
                },
            )

    # ==================================================================
    # STREAM PREPARATION
    # ==================================================================

    def _prepare_stream(
        self,
        stream: Stream,
    ) -> Stream:
        """
        Prepare waveform fragments conservatively.

        Important
        ---------
        ``fill_value='interpolate'`` is intentionally NOT used.

        A gap represents missing observations and must not be replaced
        by synthetic samples at the ingestion layer.
        """
        stream_copy = stream.copy()

        stream_copy.sort(
            keys=[
                "network",
                "station",
                "location",
                "channel",
                "starttime",
            ]
        )

        # method=1 performs a conservative merge while preserving
        # unresolved gaps rather than fabricating observations.
        stream_copy.merge(
            method=1,
            fill_value=None,
        )

        stream_copy.sort(
            keys=[
                "network",
                "station",
                "location",
                "channel",
                "starttime",
            ]
        )

        return stream_copy

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def _validate_raw_stream(
        self,
        stream: Stream,
        identifier: str,
    ) -> None:
        """Validate the raw ObsPy Stream before any merge operation."""
        if len(stream) == 0:
            raise ValueError(
                f"Stream is empty: {identifier}"
            )

        for trace in stream:
            if trace.stats.npts <= 0:
                raise ValueError(
                    f"Trace {trace.id} contains no samples."
                )

            if len(trace.data) == 0:
                raise ValueError(
                    f"Trace {trace.id} contains empty data."
                )

            sampling_rate = float(
                getattr(trace.stats, "sampling_rate", 0.0)
            )

            if (
                not np.isfinite(sampling_rate)
                or sampling_rate <= 0.0
            ):
                raise ValueError(
                    f"Trace {trace.id} has invalid sampling rate: "
                    f"{sampling_rate}"
                )

            if not np.all(np.isfinite(
                np.asarray(trace.data, dtype=np.float64)
            )):
                raise ValueError(
                    f"Trace {trace.id} contains NaN or infinite values."
                )

    def _validate_stream(
        self,
        stream: Stream,
        identifier: str,
    ) -> None:
        """
        Validate the final stream after conservative merging.
        """
        if len(stream) == 0:
            raise ValueError(
                f"Stream is empty after preparation: {identifier}"
            )

        for trace in stream:
            npts = int(trace.stats.npts)

            if npts < self.MIN_REQUIRED_SAMPLES:
                raise ValueError(
                    f"Trace {trace.id} has insufficient samples: "
                    f"{npts}"
                )

            if len(trace.data) != npts:
                raise ValueError(
                    f"Trace {trace.id} has inconsistent npts metadata: "
                    f"stats.npts={npts}, actual={len(trace.data)}"
                )

            data = np.asarray(
                trace.data,
                dtype=np.float64,
            )

            if not np.all(np.isfinite(data)):
                raise ValueError(
                    f"Trace {trace.id} contains non-finite samples."
                )

            sampling_rate = float(
                trace.stats.sampling_rate
            )

            if (
                not np.isfinite(sampling_rate)
                or sampling_rate <= 0.0
            ):
                raise ValueError(
                    f"Trace {trace.id} has invalid sampling rate."
                )

            delta = float(trace.stats.delta)

            if (
                not np.isfinite(delta)
                or delta <= 0.0
            ):
                raise ValueError(
                    f"Trace {trace.id} has invalid delta: {delta}"
                )

            expected_delta = 1.0 / sampling_rate

            if not np.isclose(
                delta,
                expected_delta,
                rtol=self.DELTA_RTOL,
                atol=self.DELTA_ATOL,
            ):
                raise ValueError(
                    f"Trace {trace.id} has inconsistent timing: "
                    f"delta={delta}, "
                    f"expected={expected_delta}"
                )

    def _validate_sampling_consistency(
        self,
        stream: Stream,
        identifier: str,
    ) -> None:
        """
        Validate sampling rates per processing-compatible station group.

        Different stations/channels may legitimately have different
        sampling rates. Therefore, validation is not performed blindly
        across the entire Stream.

        Within one station/location/component set, however, incompatible
        sampling rates are reported because synchronized three-component
        analysis requires temporal consistency.
        """
        groups: dict[
            tuple[str, str, str],
            list[float],
        ] = {}

        for trace in stream:
            key = (
                getattr(trace.stats, "network", ""),
                getattr(trace.stats, "station", ""),
                getattr(trace.stats, "location", ""),
            )

            groups.setdefault(key, []).append(
                float(trace.stats.sampling_rate)
            )

        for group_key, rates in groups.items():
            if not rates:
                continue

            reference_rate = rates[0]

            if not np.all(
                np.isclose(
                    rates,
                    reference_rate,
                    rtol=self.SAMPLING_RATE_RTOL,
                    atol=self.SAMPLING_RATE_ATOL,
                )
            ):
                raise ValueError(
                    "Inconsistent sampling rates within station group "
                    f"{group_key} in {identifier}: {rates}"
                )

    def _validate_temporal_consistency(
        self,
        stream: Stream,
        identifier: str,
    ) -> None:
        """
        Validate temporal metadata and report gaps/overlaps.

        Gaps are not treated as interpolation targets.

        They remain observable data-quality conditions and are left for
        QC / downstream policy decisions.
        """
        try:
            gaps = stream.get_gaps()
        except Exception as exc:
            raise ValueError(
                f"Failed to evaluate waveform continuity: {identifier}"
            ) from exc

        if not gaps:
            return

        gap_count = 0
        overlap_count = 0

        for gap in gaps:
            # ObsPy get_gaps() structure:
            #
            # [network, station, location, channel,
            #  starttime, endtime, delta, samples]
            #
            # Depending on ObsPy version, the final fields are handled
            # according to ObsPy's get_gaps() contract. We inspect the
            # numeric temporal difference conservatively.
            if len(gap) < 7:
                continue

            temporal_difference = float(gap[6])

            if temporal_difference > 0:
                gap_count += 1

            elif temporal_difference < 0:
                overlap_count += 1

        if gap_count > 0 or overlap_count > 0:
            self.logger.warning(
                "Waveform continuity anomalies detected.",
                extra={
                    "bsma_context": {
                        "module": "waveform_reader",
                        "identifier": identifier,
                        "gap_count": gap_count,
                        "overlap_count": overlap_count,
                    }
                },
            )

    # ==================================================================
    # PROCESSING CONTEXT
    # ==================================================================

    def _build_processing_context(
        self,
        stream: Stream,
        path: Path,
    ) -> ProcessingContext:
        """
        Construct the canonical ProcessingContext.

        Metadata is anchored to the first chronological trace, while
        the complete Stream remains available as the authoritative
        waveform collection.

        This method deliberately does not select one component as
        representative of the complete three-component waveform.
        """
        if len(stream) == 0:
            raise ValueError(
                "Cannot construct ProcessingContext from empty Stream."
            )

        stream = stream.copy()
        stream.sort(keys=["starttime"])

        base_trace = stream[0]
        stats = base_trace.stats

        starttime = stats.starttime
        endtime = stats.endtime

        metadata = TraceMetadata(
            network=getattr(stats, "network", ""),
            station=getattr(stats, "station", ""),
            location=getattr(stats, "location", ""),
            channel=getattr(stats, "channel", ""),
            starttime=starttime.datetime,
            endtime=endtime.datetime,
            sampling_rate=float(stats.sampling_rate),
            npts=int(stats.npts),
            duration=float(endtime - starttime),
            format=getattr(stats, "_format", "UNKNOWN"),
        )

        state = ProcessingState(
            is_raw=True,
            is_baseline_corrected=False,
            is_filtered=False,
            is_tapered=False,
        )

        return ProcessingContext(
            stream=stream,
            metadata=metadata,
            state=state,
            filepath=path,
        )

    # ==================================================================
    # PATH VALIDATION
    # ==================================================================

    def _validate_file_path(
        self,
        path: Path,
    ) -> None:
        """Validate filesystem existence and read permission."""
        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a regular file: {path}"
            )

        if not os.access(path, os.R_OK):
            raise PermissionError(
                f"Permission denied: {path}"
            )

    def _validate_waveform_extension(
        self,
        path: Path,
    ) -> None:
        """Validate supported waveform file extension."""
        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_WAVEFORM_EXTENSIONS:
            raise ValueError(
                f"Unsupported waveform format '{extension}' for "
                f"{path.name}. Supported formats: "
                f"{sorted(self.SUPPORTED_WAVEFORM_EXTENSIONS)}"
            )

    def _validate_response_extension(
        self,
        path: Path,
    ) -> None:
        """Validate StationXML/XML extension."""
        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_RESPONSE_EXTENSIONS:
            raise ValueError(
                f"Unsupported response format '{extension}' for "
                f"{path.name}. Supported formats: "
                f"{sorted(self.SUPPORTED_RESPONSE_EXTENSIONS)}"
            )

    # ==================================================================
    # IDENTIFIER UTILITIES
    # ==================================================================

    @staticmethod
    def _extract_station_code_from_path(
        path: Path,
    ) -> str | None:
        """
        Extract a conservative station token from a filename.

        This is used only for response auto-discovery and NEVER for
        authoritative waveform metadata.
        """
        stem = path.stem

        if not stem:
            return None

        # Common forms:
        #
        # station.mseed
        # station_HNE.mseed
        # IA.SBSSI.00.HNE.mseed
        #
        # For SEED-style names, the station is generally the second
        # token when four tokens are present.
        tokens = stem.replace("-", ".").split(".")

        if len(tokens) >= 4:
            return tokens[1]

        if "_" in stem:
            parts = stem.split("_")

            if len(parts) >= 2:
                return parts[0]

        return stem

    # ==================================================================
    # LOGGING
    # ==================================================================

    def _log_ingestion_summary(
        self,
        path: Path,
        stream: Stream,
        response_path: Path | None,
        elapsed: float,
    ) -> None:
        """
        Log a concise scientific ingestion summary.
        """
        station_groups = {
            (
                getattr(trace.stats, "network", ""),
                getattr(trace.stats, "station", ""),
                getattr(trace.stats, "location", ""),
            )
            for trace in stream
        }

        channels = sorted(
            {
                getattr(trace.stats, "channel", "")
                for trace in stream
            }
        )

        sampling_rates = sorted(
            {
                float(trace.stats.sampling_rate)
                for trace in stream
            }
        )

        try:
            gaps = stream.get_gaps()
        except Exception:
            gaps = []

        gap_count = 0
        overlap_count = 0

        for gap in gaps:
            if len(gap) < 7:
                continue

            difference = float(gap[6])

            if difference > 0:
                gap_count += 1
            elif difference < 0:
                overlap_count += 1

        self.logger.info(
            "Waveform ingestion successful.",
            extra={
                "bsma_context": {
                    "module": "waveform_reader",
                    "file": path.name,
                    "trace_count": len(stream),
                    "station_groups": len(station_groups),
                    "channels": channels,
                    "sampling_rates_hz": sampling_rates,
                    "gap_count": gap_count,
                    "overlap_count": overlap_count,
                    "response_loaded": response_path is not None,
                    "response_file": (
                        response_path.name
                        if response_path is not None
                        else None
                    ),
                    "elapsed_seconds": elapsed,
                }
            },
        )