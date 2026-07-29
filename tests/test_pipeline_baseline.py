import pytest
import numpy as np
from core.types.context import ProcessingContext, TraceMetadata, QCReport
from core.preprocessing.baseline import BaselineRemovalPlugin, BaselineConfig
from core.pipeline import PreprocessingPipeline

class TestBaselineAndPipeline:
    
    @pytest.fixture
    def mock_context(self) -> ProcessingContext:
        """Menyiapkan data sintetis untuk pengujian."""
        # Membuat sinyal acak dengan offset/DC bias yang jelas (+10.0)
        offset = 10.0
        raw_data = np.random.randn(100) + offset
        
        metadata = TraceMetadata(
            network="IA", 
            station="BBJM", 
            location="00", 
            channel="HNE",
            sampling_rate=100.0, 
            starttime=0.0
        )
        return ProcessingContext(data=raw_data, metadata=metadata)

    def test_baseline_removal_demean(self, mock_context: ProcessingContext):
        """Memvalidasi pemotongan rata-rata (demean) dan hukum immutability."""
        
        # 1. Setup Plugin dan Pipeline
        config = BaselineConfig(method="demean")
        plugin = BaselineRemovalPlugin(config=config)
        pipeline = PreprocessingPipeline(plugins=[plugin])
        
        # 2. Eksekusi
        result_context = pipeline.run(mock_context)
        
        # 3. Asersi Fisika (Rata-rata harus sangat mendekati 0)
        assert np.isclose(np.mean(result_context.data), 0.0, atol=1e-7), "Demean gagal menormalkan sinyal."
        
        # 4. Asersi Immutability (Data pada mock_context asli tidak boleh berubah)
        assert np.mean(mock_context.data) > 5.0, "Pelanggaran Immutability: Data asli ikut berubah secara in-place!"
        assert result_context is not mock_context, "Konteks tidak di-replace dengan instance baru."
        
        # 5. Asersi Provenance (Sejarah pemrosesan harus tercatat)
        assert len(result_context.history) == 1
        assert result_context.history[0].name == "BaselineRemovalPlugin"
        assert result_context.history[0].config.method == "demean"

    def test_baseline_removal_linear(self, mock_context: ProcessingContext):
        """Memvalidasi penghilangan tren linear (detrending)."""
        
        # Menambahkan tren linear buatan (y = mx + c) pada data
        trend = np.linspace(0, 5, 100)
        context_with_trend = ProcessingContext(
            data=mock_context.data + trend,
            metadata=mock_context.metadata
        )
        
        plugin = BaselineRemovalPlugin(config=BaselineConfig(method="linear"))
        pipeline = PreprocessingPipeline(plugins=[plugin])
        
        result_context = pipeline.run(context_with_trend)
        
        # Asersi: Regresi linear harus menihilkan gradien tren
        slope, _ = np.polyfit(np.arange(100), result_context.data, 1)
        assert np.isclose(slope, 0.0, atol=1e-7), "Linear detrend gagal menghilangkan gradien kemiringan."

    def test_pipeline_nan_interception(self, mock_context: ProcessingContext):
        """Memastikan pipeline mencegat kerusakan data (NaN) dengan aman."""
        
        # Menyuntikkan NaN ke dalam data untuk mensimulasikan data rusak
        corrupted_data = mock_context.data.copy()
        corrupted_data[50] = np.nan
        corrupted_context = ProcessingContext(
            data=corrupted_data,
            metadata=mock_context.metadata
        )
        
        plugin = BaselineRemovalPlugin(config=BaselineConfig(method="demean"))
        pipeline = PreprocessingPipeline(plugins=[plugin])
        
        result_context = pipeline.run(corrupted_context)
        
        # Asersi: Pipeline harus gagal dengan gracefully dan mencatat error di QC Report
        assert result_context.qc_report.passed is False
        assert result_context.qc_report.nan_found is True