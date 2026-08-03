"""
BMKG Strong Motion Analyzer (BSMA)
Module: utils/exporter.py
Description: Utilities to export ProcessingContext metrics into JSON and CSV formats.
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path

from core.types.context import ProcessingContext

class BSMAExporter:
    """
    Kelas utilitas untuk mengekspor hasil analisis (metrics dan metadata)
    ke dalam format standar (JSON dan CSV).
    """
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "reports"
        
        # Buat direktori otomatis jika belum ada
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def export_to_json(self, context: ProcessingContext) -> Path:
        """Menyimpan seluruh detail satu rekaman gempa ke file JSON tunggal."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_trace_id = context.trace_id.replace(".", "_")
        filename = f"{safe_trace_id}_{timestamp}.json"
        filepath = self.reports_dir / filename
        
        payload = {
            "trace_id": context.trace_id,
            "export_time": timestamp,
            "metrics": context.metrics,
            "processing_history": [step for step in context.history]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
            
        return filepath

    def export_to_csv(self, context: ProcessingContext, filename: str = "bsma_summary.csv") -> Path:
        """
        Menambahkan hasil metrik ke file CSV ringkasan utama (append mode).
        Sangat cocok untuk melihat tren banyak stasiun di Excel.
        """
        filepath = self.reports_dir / filename
        file_exists = filepath.is_file()
        
        # Siapkan kolom: timestamp + trace_id + seluruh kunci metrik yang ada
        headers = ["timestamp", "trace_id"] + list(context.metrics.keys())
        
        row_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trace_id": context.trace_id,
        }
        row_data.update(context.metrics)
        
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            
            # Tulis header tabel jika file baru pertama kali dibuat
            if not file_exists:
                writer.writeheader()
                
            # Tambahkan baris data
            writer.writerow(row_data)
            
        return filepath