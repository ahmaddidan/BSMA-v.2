"""
BMKG Strong Motion Analyzer (BSMA)
Module: core/io/reader.py
Description: Scientific-grade waveform ingestion. Integrates ObsPy Streams directly
into the BSMA ProcessingContext to enforce Single Source of Truth (SSOT).
"""
from __future__ import annotations

import os
import time
import logging
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import ClassVar

from obspy import Stream, read, read_inventory
from obspy.core.inventory import Inventory

# Locked Domain Models
from core.types.context import ProcessingContext
from core.types.metadata import TraceMetadata
from core.types.processing_state import ProcessingState

class WaveformReader:
    """
    Production-grade waveform reader for BSMA.
    Reads MiniSEED/SAC, performs fundamental geophysical validation, 
    and instantiates the SSOT ProcessingContext.
    """
    SUPPORTED_EXTENSIONS: ClassVar[frozenset[str]] = frozenset(
        {".mseed", ".miniseed", ".msd", ".sac"}
    )
    SAMPLING_RATE_RTOL: ClassVar[float] = 1e-5

    def __init__(self, max_workers: int = 4, logger: logging.Logger | None = None) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self.max_workers = max_workers
        self.logger = logger or logging.getLogger(__name__)

    def read_to_context(self, file_path: Path | str) -> ProcessingContext:
        """
        Reads a waveform file, validates integrity, and constructs a ProcessingContext.
        """
        path = Path(file_path).expanduser().resolve()
        start_time = time.perf_counter()
        
        self._validate_file_path(path)
        
        try:
            stream = read(str(path))
            stream = self._merge_stream(stream)
            self._validate_stream(stream, path.name)
            self._validate_sampling_rate(stream, path.name)
            
            context = self._build_processing_context(stream, path)
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to ingest waveform: {path.name} - {str(e)}")
            raise
        finally:
            elapsed = time.perf_counter() - start_time
            self.logger.debug(f"Ingestion completed in {elapsed:.4f}s for {path.name}")

    def _build_processing_context(self, stream: Stream, path: Path) -> ProcessingContext:
        """Constructs the SSOT ProcessingContext from a validated stream."""
        # Ensure chronologically first trace acts as the metadata anchor
        stream.sort(keys=["starttime"])
        base_trace = stream[0]
        stats = base_trace.stats
        
        metadata = TraceMetadata(
            network=stats.network,
            station=stats.station,
            location=stats.location,
            channel=stats.channel,
            starttime=stats.starttime.datetime,
            endtime=stats.endtime.datetime,
            sampling_rate=float(stats.sampling_rate),
            npts=int(stats.npts),
            duration=float(stats.endtime - stats.starttime),
            format=getattr(stats, "_format", "UNKNOWN")
        )
        
        state = ProcessingState(
            is_raw=True,
            is_baseline_corrected=False,
            is_filtered=False,
            is_tapered=False
        )
        
        return ProcessingContext(
            stream=stream,
            metadata=metadata,
            state=state,
            filepath=path
        )

    def _merge_stream(self, stream: Stream) -> Stream:
        """Sorts and merges fragmented traces using interpolation to prevent low-frequency drift."""
        stream_copy = stream.copy()
        stream_copy.sort(keys=["network", "station", "location", "channel", "starttime"])
        stream_copy.merge(method=1, fill_value="interpolate")
        return stream_copy

    def _validate_file_path(self, path: Path) -> None:
        """Verifies physical path existence and OS read permissions."""
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Waveform file invalid or not found: {path}")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Permission denied for waveform file: {path}")

    def _validate_stream(self, stream: Stream, identifier: str) -> None:
        """Validates trace completeness to prevent silent math errors during integration."""
        if not stream or len(stream) == 0:
            raise ValueError(f"Stream is empty after reading: {identifier}")
        for trace in stream:
            if trace.stats.npts == 0 or len(trace.data) == 0:
                raise ValueError(f"Trace {trace.id} contains zero samples.")

    def _validate_sampling_rate(self, stream: Stream, identifier: str) -> None:
        """Forces uniform time-stepping, critical for finite difference/Newmark integration."""
        if len(stream) <= 1:
            return
            
        sampling_rates = [float(tr.stats.sampling_rate) for tr in stream]
        if not np.all(np.isfinite(sampling_rates)) or any(sr <= 0 for sr in sampling_rates):
            raise ValueError(f"Invalid non-finite or <=0 sampling rates detected in {identifier}")
            
        base_sr = sampling_rates[0]
        if not np.allclose(sampling_rates, base_sr, rtol=self.SAMPLING_RATE_RTOL):
            raise ValueError(f"Inconsistent sampling rates within stream: {identifier}")