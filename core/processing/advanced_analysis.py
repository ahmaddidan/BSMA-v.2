"""
BMKG Strong Motion Analyzer (BSMA)
core/processing/advanced_analysis.py
"""
import numpy as np

def compute_husid_and_duration(data: np.ndarray, fs: float):
    """
    Menghitung Husid plot (energi kumulatif normalisasi) 
    dan durasi signifikan (D5-95).
    """
    dt = 1.0 / fs
    squared_acc = data ** 2
    cum_energy = np.cumsum(squared_acc) * dt
    total_energy = cum_energy[-1]
    
    if total_energy == 0:
        return np.zeros_like(data), 0.0, 0.0, 0.0
        
    husid = cum_energy / total_energy
    time = np.arange(len(data)) / fs
    
    # Durasi signifikan D5-95
    idx_5 = np.argmax(husid >= 0.05)
    idx_95 = np.argmax(husid >= 0.95)
    
    t_5 = time[idx_5]
    t_95 = time[idx_95]
    d_5_95 = max(0.0, t_95 - t_5)
    
    return husid, t_5, t_95, d_5_95

def compute_fas(data: np.ndarray, fs: float):
    """
    Menghitung Fourier Amplitude Spectra (FAS).
    """
    n = len(data)
    fft_vals = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    dt = 1.0 / fs
    fas = np.abs(fft_vals) * dt
    return freqs, fas