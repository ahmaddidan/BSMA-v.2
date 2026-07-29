"""
BMKG Strong Motion Analyzer (BSMA)
core/io/exporter.py
"""
import csv
from pathlib import Path
from typing import List, Dict, Any

class ResultExporter:
    """
    Bertugas mengekspor hasil ringkasan parameter kuat gerak tanah 
    dari proses batch ke format standar seperti CSV.
    """
    @staticmethod
    def export_to_csv(results: List[Dict[str, Any]], output_path: str = "output/summary_parameters.csv"):
        """Menyimpan daftar hasil analisis parameter ke dalam file CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if not results:
            print("[WARNING] Tidak ada data parameter untuk diekspor.")
            return

        # Ambil kunci dari dictionary pertama sebagai header kolom CSV
        fieldnames = list(results[0].keys())
        
        try:
            with open(path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    writer.writerow(row)
            print(f"[INFO] Rekapitulasi parameter berhasil disimpan ke: {path}")
        except Exception as e:
            print(f"[ERROR] Gagal menulis file CSV: {e}")