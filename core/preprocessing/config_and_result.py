"""
BMKG Strong Motion Analyzer (BSMA)
core/preprocessing/config_and_result.py

Catatan Arsitektur: 
Dalam arsitektur BSMA v4.0, file ini dipertahankan HANYA untuk kompatibilitas mundur.
Seluruh manajemen hasil (Result) telah didelegasikan sepenuhnya kepada:
`core.types.context.ProcessingContext`
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class PreprocessingBaseConfig:
    """
    Kelas dasar untuk konfigurasi prapemrosesan.
    Seluruh konfigurasi spesifik (seperti BaselineConfig) sebaiknya diletakkan 
    langsung di dalam file plugin masing-masing (misal: baseline.py).
    """
    pass

# Hapus semua referensi ke "ProcessingResult" lama dari sini.
# Gunakan ProcessingContext dari core.types.context untuk lalu lintas data.