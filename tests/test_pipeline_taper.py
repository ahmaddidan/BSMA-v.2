import pytest
import numpy as np
from core.types.context import ProcessingContext, TraceMetadata
from core.preprocessing.taper import TaperingPlugin, TaperConfig

class TestTaperingPlugin:
    
    @pytest.fixture
    def mock_context(self) -> ProcessingContext:
        # Array berisi 100 angka satu. Memudahkan melihat persentase taper.
        raw_data = np.ones(100) 
        metadata = TraceMetadata(
            network="IA", station="BBJM", location="00", channel="HNE",
            sampling_rate=100.0, starttime=0.0
        )
        return ProcessingContext(data=raw_data, metadata=metadata)

    def test_cosine_taper_both_sides(self, mock_context: ProcessingContext):
        """Memastikan fungsi pelemahan bekerja di kedua ujung data."""
        
        # Setup Taper 10% (10 titik di awal dan 10 titik di akhir untuk array length 100)
        config = TaperConfig(max_percentage=0.10, side="both")
        plugin = TaperingPlugin(config)
        
        result_context = plugin.process(mock_context)
        data = result_context.data
        
        # 1. Asersi Fisika
        assert data[0] == 0.0, "Titik pertama harus persis 0.0"
        assert data[-1] == 0.0, "Titik terakhir harus persis 0.0"
        assert data[50] == 1.0, "Titik tengah tidak boleh berubah (harus 1.0)"
        
        # 2. Asersi Simetri
        assert np.allclose(data[:10], data[-10:][::-1]), "Taper kiri dan kanan harus simetris"
        
        # 3. Asersi Immutability (Data asli mock_context harus tetap 1.0)
        assert mock_context.data[0] == 1.0
        
        # 4. Asersi History
        assert result_context.history[-1].name == "TaperingPlugin"