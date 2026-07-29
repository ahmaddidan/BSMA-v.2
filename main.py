"""
BMKG Strong Motion Analyzer (BSMA)
main.py (Batch Processing & SIG BMKG Standard)

Author: Ahmad Didane
"""
from pathlib import Path
from collections import defaultdict
import obspy
import numpy as np

from core.types.context import ProcessingContext, TraceMetadata
from core.pipeline import PipelineOrchestrator
from core.preprocessing.filter import ButterworthFilterPlugin, FilterConfig
from core.preprocessing.baseline import BaselineRemovalPlugin, BaselineConfig
from core.processing.kinematics import KinematicIntegrationPlugin
from core.processing.parameters import ParameterExtractionPlugin, ParameterConfig
from core.processing.spectra import NigamJenningsSpectraPlugin, SpectraConfig
from visualization.plots import BSMAVisualizer
from core.io.exporter import ResultExporter

def estimate_sig_bmkg(pga_gal: float) -> str:
    """
    Klasifikasi Skala Intensitas Gempabumi (SIG) resmi BMKG 
    berdasarkan nilai PGA (gal / cm/s²).
    """
    if pga_gal < 2.9:
        return "I - TIDAK DIRASAKAN (< 2.9 gal)"
    elif pga_gal < 89.0:
        return "II - DIRASAKAN (2.9 - 88 gal)"
    elif pga_gal < 167.0:
        return "III - KERUSAKAN RINGAN (89 - 167 gal)"
    elif pga_gal < 564.0:
        return "IV - KERUSAKAN SEDANG (168 - 564 gal)"
    else:
        return "V - KERUSAKAN BERAT (> 564 gal)"

def main():
    print("--- Memulai BSMA Batch Processing (Standar SIG BMKG) ---")
    
    data_dir = Path("Data/mseed")
    if not data_dir.exists():
        print(f"[ERROR] Direktori '{data_dir}' tidak ditemukan!")
        return

    mseed_files = list(data_dir.glob("*.mseed"))
    if not mseed_files:
        print(f"[WARNING] Tidak ditemukan file .mseed di {data_dir}")
        return

    output_dir = Path("output/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Konfigurasi Pipeline Runtun Lengkap menggunakan ButterworthFilterPlugin yang sudah ada
    pipeline = PipelineOrchestrator(plugins=[
        ButterworthFilterPlugin(config=FilterConfig(
            type="bandpass",
            freq_min=0.1,
            freq_max=25.0,
            corners=4,
            zerophase=True
        )),
        BaselineRemovalPlugin(config=BaselineConfig(method="linear")),
        KinematicIntegrationPlugin(),
        ParameterExtractionPlugin(config=ParameterConfig(gravity_constant=981.0)),
        NigamJenningsSpectraPlugin(config=SpectraConfig(damping=0.05))
    ])

    summary_results = []
    all_contexts = []
    station_grouped_contexts = defaultdict(list)

    for file_path in mseed_files:
        print(f"Memproses: {file_path.name}")
        try:
            st = obspy.read(str(file_path))
            tr = st[0]
            
            # Konversi data mseed ke cm/s² (Gal)
            data_raw = tr.data.astype(np.float64) * 100.0 
            
            metadata = TraceMetadata(
                network=tr.stats.network,
                station=tr.stats.station,
                location=tr.stats.location,
                channel=tr.stats.channel,
                sampling_rate=tr.stats.sampling_rate,
                starttime=tr.stats.starttime.timestamp
            )
            
            context = ProcessingContext(data=data_raw, metadata=metadata)
            final_context = pipeline.run(context)
            
            all_contexts.append(final_context)
            station_grouped_contexts[final_context.metadata.station].append(final_context)
            
            if final_context.parameters is not None:
                p = final_context.parameters
                sig_bmkg = estimate_sig_bmkg(p.pga)
                
                summary_results.append({
                    "Network": final_context.metadata.network,
                    "Station": final_context.metadata.station,
                    "Channel": final_context.metadata.channel,
                    "PGA_cm_s2": round(p.pga, 4),
                    "PGV_cm_s": round(p.pgv, 4),
                    "PGD_cm": round(p.pgd, 4),
                    "Arias_Intensity_m_s": round(p.arias_intensity, 4),
                    "SIG_BMKG": sig_bmkg
                })
                
            # Plot individual dengan anotasi parameter
            file_tag = f"{tr.stats.station}_{tr.stats.channel}"
            BSMAVisualizer.plot_time_series(final_context, save_path=str(output_dir / f"{file_tag}_timeseries.png"))
            BSMAVisualizer.plot_response_spectra(final_context, save_path=str(output_dir / f"{file_tag}_spectra.png"))

        except Exception as e:
            print(f"[ERROR] Gagal memproses file {file_path.name}: {e}")

    # Ekspor CSV Ringkasan
    if summary_results:
        ResultExporter.export_to_csv(summary_results, output_path="output/summary_parameters.csv")

    # Plot Gabungan Spektrum Respons
    if all_contexts:
        print("\n[INFO] Menghasilkan grafik perbandingan spektrum respons gabungan...")
        BSMAVisualizer.plot_combined_spectra(all_contexts, save_path=str(output_dir / "COMBINED_response_spectra.png"))

    # Plot Perbandingan Komponen Per Stasiun
    print("[INFO] Menghasilkan grafik perbandingan komponen per stasiun...")
    for stn_name, stn_contexts in station_grouped_contexts.items():
        BSMAVisualizer.plot_multi_component_comparison(
            stn_contexts, 
            station_name=stn_name, 
            save_path=str(output_dir / f"{stn_name}_component_comparison.png")
        )

    print("\n--- Seluruh Proses, Filter, & Visualisasi Selesai! ---")

if __name__ == "__main__":
    main()