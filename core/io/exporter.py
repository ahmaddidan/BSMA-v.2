"""
BMKG Strong Motion Analyzer (BSMA)
core/io/exporter.py

Ekspor hasil analisis BSMA ke format tabular.

Author: Ahmad Didane
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence


class ResultExporter:
    """
    Menangani ekspor hasil analisis BSMA ke format CSV.

    Modul ini tidak melakukan pemrosesan sinyal.
    Tanggung jawabnya hanya melakukan serialisasi hasil analisis.
    """

    @staticmethod
    def export_to_csv(
        results: Sequence[Mapping[str, Any]],
        output_path: str | Path = "outputs/summary_parameters.csv",
    ) -> Path:
        """
        Mengekspor hasil analisis ke file CSV.

        Parameters
        ----------
        results
            Sequence mapping/dictionary yang berisi hasil analisis.
        output_path
            Lokasi file CSV keluaran.

        Returns
        -------
        pathlib.Path
            Path file CSV yang berhasil dibuat.

        Raises
        ------
        ValueError
            Jika results kosong atau tidak memiliki field.
        OSError
            Jika file tidak dapat ditulis.
        """

        if not results:
            raise ValueError(
                "Tidak ada data parameter untuk diekspor."
            )

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Mengumpulkan seluruh field dari seluruh hasil.
        #
        # Urutan field mengikuti kemunculan pertama masing-masing
        # key sehingga struktur CSV tetap deterministic.
        fieldnames: list[str] = []

        for row in results:
            for key in row.keys():
                key_str = str(key)

                if key_str not in fieldnames:
                    fieldnames.append(key_str)

        if not fieldnames:
            raise ValueError(
                "Tidak ditemukan field pada hasil analisis."
            )

        try:
            with path.open(
                mode="w",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                    extrasaction="ignore",
                )

                writer.writeheader()

                for row in results:
                    normalized_row = {
                        str(key): value
                        for key, value in row.items()
                    }

                    writer.writerow(normalized_row)

        except OSError as exc:
            raise OSError(
                f"Gagal menulis file CSV: {path}"
            ) from exc

        return path