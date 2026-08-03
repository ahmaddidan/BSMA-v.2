"""
BMKG Strong Motion Analyzer (BSMA)
Domain Types: Immutable Single Source of Truth
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping
import numpy as np

from .processing_state import ProcessingState
from .cache import ProcessingCache

__all__ = ["WaveformData", "ProcessingContext"]

@dataclass(frozen=True, slots=True)
class WaveformData:
    """Kontainer fisis immutable untuk data time-series."""
    data: np.ndarray
    sampling_rate: float
    unit: str

    def __post_init__(self) -> None:
        # Validasi rigor: Paksa presisi float64 untuk stabilitas integrasi numerik
        if self.data.dtype != np.float64:
            object.__setattr__(self, 'data', self.data.astype(np.float64))


@dataclass(frozen=True, slots=True)
class ProcessingContext:
    """
    Konteks utama pipeline. Beroperasi sebagai State Machine.
    Setiap operasi modifikasi (add_history, with_state) akan mengembalikan instance baru.
    """
    trace_id: str
    metadata: dict[str, Any]
    raw_waveform: WaveformData
    
    # State Kinematika (Berasal dari IntegrationPlugin)
    acceleration: WaveformData | None = None
    velocity: WaveformData | None = None
    displacement: WaveformData | None = None
    
    # State Analisis Rekayasa (Berasal dari Parameters & Response Spectrum)
    spectral_data: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    
    # Ekosistem Status
    processing_state: ProcessingState = field(default_factory=ProcessingState)
    cache: ProcessingCache = field(default_factory=ProcessingCache)
    qc: Any = None  # Opsional, dikelola oleh QCReport
    
    # Provenance / Audit Trail
    history: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    config: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def sampling_rate(self) -> float:
        return self.raw_waveform.sampling_rate

    @property
    def npts(self) -> int:
        return self.raw_waveform.data.size

    def with_state(self, **kwargs: Any) -> ProcessingContext:
        """
        Transisi state secara fungsional murni. 
        Mengembalikan ProcessingContext baru dengan nilai yang diubah.
        """
        return replace(self, **kwargs)

    def add_history(self, step_name: str, details: dict[str, Any]) -> ProcessingContext:
        """Menambahkan log riwayat pemrosesan (Audit Trail)."""
        new_record = {"step": step_name, **details}
        return replace(self, history=self.history + (new_record,))