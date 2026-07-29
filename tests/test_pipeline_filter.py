import pytest
import numpy as np
from core.types.context import ProcessingContext, TraceMetadata
from core.preprocessing.filter import ButterworthFilterPlugin, FilterConfig

class TestFilterPlugin:
    
    @pytest.fixture
    def mock_seismic_context(self) -> ProcessingContext:
        fs = 100.0  # Sampling rate 100 Hz
        t = np.arange(0, 10.0, 1/fs) # 10 detik data
        
        # Sinyal gabungan: 
        # 1. Low freq drift (0.05 Hz)
        # 2. Target signal (5.0 Hz)
        # 3. High freq noise (40.0 Hz)
        drift = 2.0 * np.sin(2 * np.pi * 0.05 * t)
        target = 1.0 * np.sin(2 * np.pi * 5.0 * t)
        noise = 0.5 * np.sin(2 * np.pi * 40.0 * t)
        
        raw_data = drift + target + noise
        
        metadata = TraceMetadata(
            network="IA", station="BBJM", location="00", channel="HNE",
            sampling_rate=fs, starttime=0.0
        )
        return ProcessingContext(data=raw_data, metadata=metadata)

    def test_bandpass_filter_attenuates_noise(self, mock_seismic_context: ProcessingContext):
        """Memastikan filter bandpass menghapus frekuensi di luar batas."""
        
        # Setup Filter Bandpass 0.1 Hz - 20.0 Hz
        config = FilterConfig(type="bandpass", freq_min=0.1, freq_max=20.0, zerophase=True)
        plugin = ButterworthFilterPlugin(config)
        
        result_context = plugin.process(mock_seismic_context)
        filtered_data = result_context.data
        
        # Ekstrak sinyal target murni untuk perbandingan
        fs = mock_seismic_context.metadata.sampling_rate
        t = np.arange(0, 10.0, 1/fs)
        pure_target = 1.0 * np.sin(2 * np.pi * 5.0 * t)
        
        # Hitung error RMS (Root Mean Square) antara sinyal terfilter dan sinyal target murni
        # Kita abaikan 1 detik pertama dan terakhir untuk menghindari efek tepi (edge effects) filter
        error = filtered_data[100:-100] - pure_target[100:-100]
        rms_error = np.sqrt(np.mean(error**2))
        
        # Asersi: Error harus sangat kecil (kurang dari 0.05) membuktikan noise terhapus
        assert rms_error < 0.05
        
        # Asersi Immutability
        assert not np.array_equal(mock_seismic_context.data, result_context.data)
        
        # Asersi History
        assert result_context.history[-1].name == "ButterworthFilterPlugin"