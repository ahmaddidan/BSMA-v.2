"""
BMKG Strong Motion Analyzer (BSMA)
Modul: run_pipeline.py
Fungsi: GUI-Ready API, Batch Processing, dan Integrasi StationXML (Instrument Correction)
"""
import logging
from pathlib import Path
import traceback

import numpy as np
from obspy import read, read_inventory

from core.types.context import ProcessingContext, WaveformData
from core.types.processing_state import ProcessingState
from core.pipeline import PipelineBuilder
from core.preprocessing.baseline import BaselineCorrectionPlugin
from core.preprocessing.taper import TaperPlugin, TaperConfig
from core.preprocessing.filter import ButterworthFilterPlugin, FilterConfig
from core.preprocessing.integration import KinematicIntegrationPlugin, IntegrationConfig
from core.processing.parameters import ParameterExtractionPlugin, ParameterConfig
from core.processing.response_spectrum import ResponseSpectrumPlugin, ResponseSpectrumConfig
from utils.logger import setup_logger
from utils.exporter import BSMAExporter
from utils.pdf_exporter import export_station_report

def build_bsma_pipeline(logger: logging.Logger):
    """Membangun arsitektur pipeline pemrosesan standar BSMA."""
    return (
        PipelineBuilder(logger=logger, halt_on_error=True)
        .add(BaselineCorrectionPlugin(method="linear"))
        .add(TaperPlugin(config=TaperConfig(alpha=0.05)))
        .add(ButterworthFilterPlugin(
            config=FilterConfig(type="bandpass", freq_min=0.1, freq_max=25.0, corners=4, zerophase=True)
        ))
        .add(KinematicIntegrationPlugin(
            config=IntegrationConfig(remove_mean=True, remove_linear_trend=True)
        ))
        .add(ParameterExtractionPlugin(
            config=ParameterConfig(gravity=9.80665)
        ))
        .add(ResponseSpectrumPlugin(
            config=ResponseSpectrumConfig(damping=0.05, solver="nigam_jennings")
        ))
        .build()
    )

def process_earthquake_data(waveform_path: Path, xml_path: Path | None, exporter: BSMAExporter, logger: logging.Logger):
    """
    FUNGSI GUI-READY: 
    Dipanggil saat User di antarmuka (app.py) menekan tombol "Process File".
    - Otomatis mendeteksi format waveform (MSEED, SAC, SEGY, GSE2, dll).
    - Otomatis mengubah Counts ke m/s^2 jika StationXML disediakan.
    """
    logger.info(f"\n{'='*60}\nMEMPROSES FAIL: {waveform_path.name}\n{'='*60}")
    
    try:
        st = read(str(waveform_path))
        if not st:
            logger.warning(f"Fail {waveform_path.name} tidak valid. Dilewati.")
            return

        station_code = st[0].stats.station
        logger.info(f"Stream dimuat untuk stasiun {station_code} ({len(st)} channel).")

        # Muat Instrument Response jika XML tersedia
        inventory = None
        if xml_path and xml_path.exists():
            inventory = read_inventory(str(xml_path))
            logger.info(f"Metadata StationXML dimuat: {xml_path.name}")

        pipeline = build_bsma_pipeline(logger)
        processed_contexts = {}

        for trace in st:
            channel = trace.stats.channel
            logger.info(f"-> Analisis komponen: {channel} (SR: {trace.stats.sampling_rate} Hz)")
            
            # INSTRUMENT CORRECTION: Mengubah Counts ke Percepatan m/s^2
            if inventory:
                logger.info(f"   [!] Menerapkan koreksi instrumen (Counts -> m/s²).")
                
                # === ALGORITMA SINKRONISASI LOKASI DINAMIS ===
                try:
                    # Mengambil kode lokasi asli dari XML dan menerapkannya ke trace MSEED
                    xml_loc_code = inventory[0][0][0].location_code
                    trace.stats.location = xml_loc_code
                except Exception:
                    pass
                # =============================================

                try:
                    # Menghilangkan respon alat dengan water level dan pre-filter standar seismologi
                    trace.remove_response(
                        inventory=inventory, 
                        output="ACC", 
                        water_level=60,
                        pre_filt=[0.05, 0.1, 30.0, 35.0] 
                    )
                except Exception as xml_e:
                    logger.warning(f"   [!] Gagal koreksi instrumen untuk {channel}: {xml_e}. Memakai data mentah.")

            raw_wave = WaveformData(
                data=trace.data,
                sampling_rate=trace.stats.sampling_rate,
                unit="m/s^2"
            )
            
            initial_context = ProcessingContext(
                trace_id=trace.id,
                metadata=dict(trace.stats),
                raw_waveform=raw_wave,
                acceleration=raw_wave, 
                processing_state=ProcessingState(),
                history=()
            )
            
            # Eksekusi Pipeline Matematika
            final_context = pipeline.run(initial_context)
            processed_contexts[channel] = final_context
            
            # Ekspor JSON/CSV Ringkasan
            exporter.export_to_json(final_context)
            exporter.export_to_csv(final_context)
            
            pga_gal = final_context.metrics.get('PGA', 0.0) * 100.0
            logger.info(f"   [+] Selesai | PGA: {pga_gal:.2f} Gal")

        # Bangun Laporan PDF Gabungan
        logger.info("-> Membuat Laporan PDF Gabungan...")
        pdf_path = export_station_report(
            station_code=station_code, 
            contexts=processed_contexts, 
            output_dir="outputs/reports"
        )
        logger.info(f"   [+] Laporan PDF Tersimpan: {pdf_path}")

    except Exception as e:
        logger.error(f"Gagal memproses fail {waveform_path.name}: {str(e)}")
        logger.error(traceback.format_exc())


def main():
    logger = setup_logger("bsma_backend")
    logger.info("Memulai BSMA Batch Processor...")

    # Folder Utama Data
    data_dir = Path("Data/mseed")
    xml_dir = Path("Data/stationXML") # Direktori opsional untuk file kalibrasi .xml
    
    if not data_dir.exists():
        logger.error(f"Direktori {data_dir} tidak ditemukan!")
        return

    # Mendukung segala ekstensi seismologi standar
    supported_extensions = ["*.mseed", "*.miniseed", "*.sac", "*.SAC", "*.seed"]
    waveform_files = []
    for ext in supported_extensions:
        waveform_files.extend(list(data_dir.glob(ext)))
    
    if not waveform_files:
        logger.warning(f"Tidak ada fail waveform gempa di dalam folder {data_dir}")
        return

    exporter = BSMAExporter()

    # Eksekusi Batch
    for wf_file in waveform_files:
        # Coba cari file XML dengan nama stasiun yang sama (contoh: BBJM.xml)
        # Pada skenario real GUI, ini dipilih manual oleh analis.
        station_name_guess = wf_file.stem.split("_")[1] if "_" in wf_file.stem else wf_file.stem
        possible_xml = xml_dir / f"{station_name_guess}.xml"
        
        xml_path = possible_xml if possible_xml.exists() else None
        
        process_earthquake_data(wf_file, xml_path, exporter, logger)
        
    logger.info(f"\n{'='*60}\nBATCH PROCESSING SELESAI.\n{'='*60}")

if __name__ == "__main__":
    main()