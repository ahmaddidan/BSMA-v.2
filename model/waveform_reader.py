"""
BMKG Strong Motion Analyzer (BSMA)
Model: Waveform Reader (Version 1.4 - Final Production Grade)

Modul ini bertanggung jawab HANYA untuk membaca (ingestion), menggabungkan,
memvalidasi struktur dasar, mengekstraksi metadata file seismik, dan inventory.
Menggunakan sintaks modern Python 3.12, optimasi memori, dan NumPy docstrings.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from obspy import Inventory, Stream, Trace, read, read_inventory

from utils.exceptions import ErrorCode, SeverityLevel, WaveformError
from utils.logger import setup_logger

__all__ = [
    "BatchReadResult",
    "WaveformReader",
]

BYTES_PER_MB: int = 1024 ** 2


@dataclass(slots=True)
class BatchReadResult:
    """
    Objek standar untuk mengembalikan hasil pembacaan batch.
    """
    successful: dict[str, Stream] = field(default_factory=dict)
    failed: dict[str, Exception] = field(default_factory=dict)
    elapsed_time_seconds: float = 0.0
    total_files: int = 0

    @property
    def success_rate(self) -> float:
        """
        Mengembalikan persentase kesuksesan pembacaan batch.

        Returns
        -------
        float
            Persentase file yang berhasil diproses (0.0 - 100.0).
        """
        if self.total_files == 0:
            return 0.0
        return (len(self.successful) / self.total_files) * 100.0


class WaveformReader:
    """
    Kelas pembaca waveform produksi untuk data seismik BSMA.
    Menerapkan Separation of Concerns (hanya Ingestion murni).
    """

    SUPPORTED_EXTENSIONS: ClassVar[frozenset[str]] = frozenset(
        {".mseed", ".miniseed", ".msd", ".sac", ".gcf", ".segy", ".sgy", ".su"}
    )
    
    SAMPLING_RATE_RTOL: ClassVar[float] = 1e-5

    def __init__(
        self,
        logger: logging.Logger | None = None,
        max_workers: int = 4,
    ) -> None:
        """
        Inisialisasi WaveformReader.

        Parameters
        ----------
        logger : logging.Logger | None, optional
            Logger kustom untuk injeksi dependensi, default None.
        max_workers : int, optional
            Jumlah maksimal thread untuk proses batch, default 4.

        Raises
        ------
        ValueError
            Jika max_workers kurang dari 1.
        """
        if max_workers < 1:
            raise ValueError(f"max_workers harus minimal 1, menerima: {max_workers}")

        self.logger = logger or setup_logger(__name__)
        self.max_workers = max_workers
        self.logger.info(
            "WaveformReader diinisialisasi.",
            extra={"bsma_context": {"max_workers": self.max_workers}}
        )

    # ------------------------------------------------------------------
    # I/O Operations
    # ------------------------------------------------------------------

    def _read_obspy(self, path: Path) -> Stream:
        """
        Membungkus fungsi baca ObsPy untuk mempermudah injeksi retry/timeout ke depannya.
        
        Parameters
        ----------
        path : Path
            Path absolut menuju file seismik.
            
        Returns
        -------
        obspy.core.stream.Stream
        """
        return read(str(path))

    def read_directory(
        self,
        directory: Path | str,
        recursive: bool = True,
        strict_extension: bool = True,
    ) -> list[Path]:
        """
        Menemukan semua file waveform di dalam direktori.

        Parameters
        ----------
        directory : Path | str
            Path menuju direktori target.
        recursive : bool, optional
            Jika True, akan memindai sub-direktori (rglob), default True.
        strict_extension : bool, optional
            Jika True, hanya memuat file dengan ekstensi terdaftar. Jika False, 
            mengambil semua file (menyerahkan identifikasi format pada fungsi baca).

        Returns
        -------
        list[Path]
            Daftar path file yang valid.

        Raises
        ------
        WaveformError
            Jika direktori tidak ditemukan atau tidak dapat diakses.
        """
        root = Path(directory).expanduser().resolve()
        
        if not root.exists() or not root.is_dir():
            raise WaveformError(
                message="Direktori tidak ditemukan atau bukan direktori valid.",
                error_code=ErrorCode.WF004,
                severity=SeverityLevel.ERROR,
                context={"module": "waveform_reader", "directory": str(root)},
            )

        iterator = root.rglob("*") if recursive else root.glob("*")
        
        if strict_extension:
            waveform_files = [
                p for p in iterator
                if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
        else:
            waveform_files = [p for p in iterator if p.is_file()]
        
        self.logger.info(
            "Pemindaian direktori selesai.",
            extra={
                "bsma_context": {
                    "directory": str(root),
                    "total_files": len(waveform_files),
                    "strict_extension": strict_extension
                }
            }
        )
        return sorted(waveform_files)

    def read_file(self, file_path: Path | str) -> Stream:
        """
        Membaca satu file seismik menjadi ObsPy Stream secara aman.

        Parameters
        ----------
        file_path : Path | str
            Path menuju file seismik.

        Returns
        -------
        obspy.core.stream.Stream
            Objek stream yang sudah di-merge dan divalidasi.

        Raises
        ------
        WaveformError
            Jika gagal membaca file atau validasi struktural gagal.
        """
        path = Path(file_path).expanduser().resolve()
        start_time = time.perf_counter()
        
        self._validate_file_path(path)
        self.logger.debug(
            "Memulai pembacaan file.",
            extra={"bsma_context": {"file_path": str(path)}}
        )

        try:
            stream = self._read_obspy(path)
            stream = self._merge_stream(stream)
            self._validate_stream(stream, str(path))
            self._validate_trace_ids(stream, str(path))
            self._validate_sampling_rate(stream, str(path))
            return stream

        except WaveformError:
            raise

        except Exception as e:
            raise WaveformError(
                message="Gagal memproses file seismik.",
                error_code=ErrorCode.WF004,
                severity=SeverityLevel.CRITICAL,
                context={"module": "waveform_reader", "file_path": str(path), "exception": str(e)},
            ) from e
            
        finally:
            elapsed = time.perf_counter() - start_time
            self.logger.debug(
                "Operasi read_file selesai.",
                extra={
                    "bsma_context": {
                        "file_path": str(path),
                        "elapsed_time_sec": round(elapsed, 4)
                    }
                }
            )

    def read_multiple(self, file_paths: list[Path | str]) -> BatchReadResult:
        """
        Membaca banyak file seismik secara paralel menggunakan ThreadPool.

        Parameters
        ----------
        file_paths : list[Path | str]
            Daftar path file yang akan dibaca.

        Returns
        -------
        BatchReadResult
            Objek dataclass berisi stream yang sukses dan error yang gagal.
        """
        start_time = time.perf_counter()
        result = BatchReadResult(total_files=len(file_paths))
        future_to_path = {}
        completed = 0

        self.logger.info(
            "Memulai pembacaan batch.",
            extra={"bsma_context": {"total_files": result.total_files}}
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for path in file_paths:
                future = executor.submit(self.read_file, file_path=path)
                future_to_path[future] = str(Path(path).expanduser().resolve())

            for future in as_completed(future_to_path):
                path_str = future_to_path[future]
                completed += 1
                
                try:
                    stream = future.result()
                    result.successful[path_str] = stream
                except WaveformError as we:
                    result.failed[path_str] = we
                    self.logger.warning(
                        "File gagal diproses dalam batch (WaveformError).",
                        extra={
                            "bsma_context": {
                                "file_path": path_str,
                                "error_type": type(we).__name__,
                                "error_message": str(we)
                            }
                        }
                    )
                except Exception as e:
                    result.failed[path_str] = e
                    self.logger.error(
                        "Terjadi kesalahan sistem tak terduga pada batch.",
                        extra={
                            "bsma_context": {
                                "file_path": path_str,
                                "error_type": type(e).__name__,
                                "error_message": str(e)
                            }
                        },
                        exc_info=True
                    )
                
                if completed % 100 == 0 or completed == result.total_files:
                    self.logger.debug(
                        "Progress pembacaan batch.",
                        extra={
                            "bsma_context": {
                                "completed": completed,
                                "total": result.total_files,
                            }
                        }
                    )

        result.elapsed_time_seconds = round(time.perf_counter() - start_time, 3)
        
        self.logger.info(
            "Pembacaan batch selesai.",
            extra={
                "bsma_context": {
                    "successful": len(result.successful),
                    "failed": len(result.failed),
                    "success_rate": round(result.success_rate, 2),
                    "elapsed_time_sec": result.elapsed_time_seconds,
                }
            }
        )
        return result
        
    def read_inventory(self, file_path: Path | str) -> Inventory:
        """
        Membaca file metadata stasiun (StationXML, RESP) untuk koreksi instrumen.

        Parameters
        ----------
        file_path : Path | str
            Path menuju file metadata inventory.

        Returns
        -------
        obspy.core.inventory.Inventory
            Objek inventory ObsPy.
            
        Raises
        ------
        WaveformError
            Jika gagal membaca atau mem-parsing metadata.
        """
        path = Path(file_path).expanduser().resolve()
        self._validate_file_path(path)
        
        try:
            inv = read_inventory(str(path))
            station_count = sum(len(net.stations) for net in inv)
            
            self.logger.info(
                "Inventory berhasil diload.",
                extra={
                    "bsma_context": {
                        "file_path": str(path),
                        "network_count": len(inv.networks),
                        "station_count": station_count
                    }
                }
            )
            return inv
            
        except Exception as e:
            raise WaveformError(
                message="Gagal membaca file inventory/metadata stasiun.",
                error_code=ErrorCode.WF004,
                severity=SeverityLevel.ERROR,
                context={"module": "waveform_reader", "file_path": str(path), "exception": str(e)},
            ) from e

    def detect_format(self, file_path: Path | str) -> str:
        """
        Mendeteksi format file seismik secara aman (headonly).

        Parameters
        ----------
        file_path : Path | str
            Path menuju file seismik.

        Returns
        -------
        str
            Nama format file (contoh: 'MSEED', 'SAC') atau 'UNKNOWN'.
        """
        path = Path(file_path).expanduser().resolve()
        try:
            self._validate_file_path(path)
            stream = read(str(path), headonly=True)
            if len(stream) > 0:
                fmt = getattr(stream[0].stats, "_format", None)
                return str(fmt) if fmt is not None else "UNKNOWN"
        except Exception as exc:
            self.logger.debug(
                "Gagal mendeteksi format file.",
                exc_info=True,
                extra={
                    "bsma_context": {
                        "file_path": str(path),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)
                    }
                }
            )
        return "UNKNOWN"

    # ------------------------------------------------------------------
    # Core Stream Manipulation
    # ------------------------------------------------------------------

    def _merge_stream(self, stream: Stream) -> Stream:
        """Menggabungkan trace yang terfragmentasi secara deterministik."""
        stream_copy = stream.copy()
        stream_copy.sort(keys=["network", "station", "location", "channel", "starttime"])
        
        try:
            stream_copy.merge(method=1, fill_value="interpolate")
        except Exception as e:
            self.logger.warning(
                "Gagal melakukan merge, mengembalikan stream asli.",
                extra={
                    "bsma_context": {
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                }
            )
            return stream

        return stream_copy

    # ------------------------------------------------------------------
    # Validation Methods (Ingestion Scope)
    # ------------------------------------------------------------------

    def _validate_file_path(self, path: Path) -> None:
        """Memastikan path file eksis, berupa file, dan memiliki izin baca."""
        if not path.exists() or not path.is_file():
            raise WaveformError(
                message="File tidak ditemukan atau bukan file valid.",
                error_code=ErrorCode.WF004,
                severity=SeverityLevel.ERROR,
                context={"module": "waveform_reader", "file_path": str(path)},
            )
        
        if not os.access(path, os.R_OK):
            raise WaveformError(
                message="File tidak memiliki izin baca (Permission Denied).",
                error_code=ErrorCode.WF004, 
                severity=SeverityLevel.CRITICAL,
                context={"module": "waveform_reader", "file_path": str(path)},
            )

    def _validate_stream(self, stream: Stream, identifier: str) -> None:
        """Memastikan stream tidak kosong dan trace memiliki sampel data."""
        if not stream or len(stream) == 0:
            raise WaveformError(
                message="Stream kosong setelah dibaca/dimerge.",
                error_code=ErrorCode.WF003,
                severity=SeverityLevel.ERROR,
                context={"module": "waveform_reader", "identifier": identifier},
            )
            
        for trace in stream:
            if trace.stats.npts == 0 or len(trace.data) == 0:
                raise WaveformError(
                    message="Trace memiliki nol sampel data.",
                    error_code=ErrorCode.WF003,
                    severity=SeverityLevel.ERROR,
                    context={"module": "waveform_reader", "identifier": identifier, "trace_id": trace.id},
                )

    def _validate_trace_ids(self, stream: Stream, identifier: str) -> None:
        """Memastikan atribut network, station, dan channel terdefinisi."""
        for trace in stream:
            if not all([trace.stats.network, trace.stats.station, trace.stats.channel]):
                raise WaveformError(
                    message="Trace kehilangan metadata identitas krusial (Network/Station/Channel).",
                    error_code=ErrorCode.WF003,
                    severity=SeverityLevel.ERROR,
                    context={"module": "waveform_reader", "identifier": identifier, "trace_id": trace.id},
                )

    def _validate_sampling_rate(self, stream: Stream, identifier: str) -> None:
        """Memastikan konsistensi numerik sampling rate pada seluruh stream."""
        if len(stream) <= 1:
            return

        sampling_rates = [float(tr.stats.sampling_rate) for tr in stream]
        
        if not np.all(np.isfinite(sampling_rates)) or any(sr <= 0 for sr in sampling_rates):
            raise WaveformError(
                message="Ditemukan sampling rate tidak valid (NaN, Inf, atau <= 0).",
                error_code=ErrorCode.WF001,
                severity=SeverityLevel.CRITICAL,
                context={"module": "waveform_reader", "identifier": identifier, "sampling_rates": sampling_rates},
            )
            
        base_sr = sampling_rates[0]
        if not np.allclose(sampling_rates, base_sr, rtol=self.SAMPLING_RATE_RTOL):
            raise WaveformError(
                message="Sampling rate tidak konsisten dalam satu stream.",
                error_code=ErrorCode.WF001,
                severity=SeverityLevel.CRITICAL,
                context={
                    "module": "waveform_reader",
                    "identifier": identifier,
                    "expected_base": base_sr,
                    "found_rates": sampling_rates,
                },
            )

    # ------------------------------------------------------------------
    # Metadata & Summaries
    # ------------------------------------------------------------------

    def extract_metadata(self, stream: Stream) -> dict[str, Any]:
        """
        Ekstrak metadata global berdasarkan trace dengan waktu paling awal.

        Parameters
        ----------
        stream : obspy.core.stream.Stream
            Stream yang akan diekstrak metadatanya.

        Returns
        -------
        dict[str, Any]
            Kompilasi metadata trace.
        """
        self._validate_stream(stream, "metadata_extraction")
        first_trace = min(stream, key=lambda tr: tr.stats.starttime)
        metadata = self._extract_trace_metadata(first_trace)
        metadata["trace_count"] = len(stream)
        return metadata

    def _extract_trace_metadata(self, trace: Trace) -> dict[str, Any]:
        """Ekstrak metadata komprehensif dari single Trace."""
        stats = trace.stats
        memory_bytes = trace.data.nbytes
        
        metadata = {
            "stream_id": trace.id,
            "network": stats.network,
            "station": stats.station,
            "location": stats.location,
            "channel": stats.channel,
            "starttime": stats.starttime.datetime,
            "endtime": stats.endtime.datetime,
            "duration": float(stats.endtime - stats.starttime),
            "sampling_rate": float(stats.sampling_rate),
            "sampling_interval_seconds": float(stats.delta),
            "npts": int(stats.npts),
            "dtype": str(trace.data.dtype),
            "dtype_itemsize": trace.data.itemsize,
            "byteorder": trace.data.dtype.byteorder,
            "is_contiguous": trace.data.flags.c_contiguous,
            "memory_size_mb": float(memory_bytes / BYTES_PER_MB),
            "format": getattr(stats, "_format", "UNKNOWN"),
            "calib": getattr(stats, "calib", None),
        }

        if hasattr(stats, "sac"):
            sac_stats = stats.sac
            metadata["latitude"] = getattr(sac_stats, "stla", None)
            metadata["longitude"] = getattr(sac_stats, "stlo", None)
            metadata["elevation"] = getattr(sac_stats, "stel", None)

        return metadata

    def stream_summary(self, stream: Stream) -> dict[str, Any]:
        """
        Kompilasi ringkasan eksekutif dan fisis seluruh stream.

        Parameters
        ----------
        stream : obspy.core.stream.Stream
            Stream yang akan dirangkum.

        Returns
        -------
        dict[str, Any]
            Dictionary berisi ringkasan statistik stream.
        """
        if len(stream) == 0:
            return {"trace_count": 0, "total_duration": 0.0, "gap_count": 0}

        start_time = min(tr.stats.starttime for tr in stream)
        end_time = max(tr.stats.endtime for tr in stream)
        duration = float(end_time - start_time)

        networks = sorted({tr.stats.network for tr in stream})
        stations = sorted({tr.stats.station for tr in stream})
        locations = sorted({tr.stats.location for tr in stream})
        channels = sorted({tr.stats.channel for tr in stream})
        
        total_samples = sum(tr.stats.npts for tr in stream)
        total_memory_bytes = sum(tr.data.nbytes for tr in stream)
        
        sampling_rates = [float(tr.stats.sampling_rate) for tr in stream]
        dominant_sr = max(set(sampling_rates), key=sampling_rates.count) if sampling_rates else None

        return {
            "trace_count": len(stream),
            "gap_count": len(stream.get_gaps()),
            "network_count": len(networks),
            "station_count": len(stations),
            "location_count": len(locations),
            "component_count": len(channels),
            "networks": networks,
            "stations": stations,
            "locations": locations,
            "channels": channels,
            "total_samples": total_samples,
            "starttime": start_time.datetime,
            "endtime": end_time.datetime,
            "total_duration": duration,
            "dominant_sampling_rate": dominant_sr,
            "total_memory_mb": float(total_memory_bytes / BYTES_PER_MB),
        }

    # ------------------------------------------------------------------
    # Quality & Utility Helpers
    # ------------------------------------------------------------------

    def has_gaps(self, stream: Stream, tolerance_seconds: float = 0.0) -> bool:
        """
        Mengecek apakah terdapat gap data pada stream.

        Parameters
        ----------
        stream : obspy.core.stream.Stream
            Stream seismik yang akan dievaluasi.
        tolerance_seconds : float, optional
            Toleransi waktu minimum (dalam detik) untuk dianggap sebagai gap. Default 0.0.

        Returns
        -------
        bool
            True jika terdeteksi gap pada stream, False sebaliknya.
        """
        return bool(stream.get_gaps(min_gap=tolerance_seconds))

    def is_counts(self, stream: Stream) -> bool:
        """
        Determine whether waveform data are still in raw instrument counts.

        Notes
        -----
        This feature has not been implemented yet because reliable detection
        requires StationXML/RESP metadata and instrument sensitivity.

        Raises
        ------
        NotImplementedError
            Fungsi ini belum diimplementasikan sepenuhnya.
        """
        raise NotImplementedError("is_counts() has not been implemented yet.")