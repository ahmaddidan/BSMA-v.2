try:
    import fitz
except ImportError:
    fitz = None

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from services import AnalysisConfiguration, AnalysisService, ExportService


def _pdf_text(path):
    if fitz is None:
        return ""
    doc = fitz.open(path)
    return "\n".join(page.get_text("text") for page in doc)


def _contexts():
    sampling_rate = 100.0
    time = np.arange(0.0, 20.0, 1.0 / sampling_rate)
    trace = Trace(
        data=(0.5 * np.sin(2.0 * np.pi * time)).astype(np.float64),
        header={
            "network": "XX",
            "station": "TEST",
            "location": "00",
            "channel": "HNE",
            "sampling_rate": sampling_rate,
            "starttime": UTCDateTime(2024, 1, 1),
        },
    )
    service = AnalysisService(
        AnalysisConfiguration(
            input_unit="m/s^2",
            freq_max_hz=20.0,
            response_periods=(0.1, 0.5, 1.0),
        )
    )
    return service.process_station_stream(Stream([trace]))


def test_export_service_generates_csv_and_personally_neutral_pdf(tmp_path):
    contexts = _contexts()
    exporter = ExportService()

    csv_path = exporter.export_batch_csv({"TEST": contexts}, tmp_path / "summary.csv")
    pdf_path = exporter.export_station_pdf(
        "TEST",
        contexts,
        tmp_path / "report.pdf",
        event_info={"magnitude": "M5.0", "depth_km": "10"},
    )

    assert csv_path.read_text(encoding="utf-8-sig").startswith("station,channel")
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_export_station_pdf_includes_event_information_when_present(tmp_path):
    contexts = _contexts()
    exporter = ExportService()

    pdf_path = exporter.export_station_pdf(
        "TEST",
        contexts,
        tmp_path / "event_report.pdf",
        event_info={
            "time": "2024-01-01 00:00:00",
            "latitude": "-7.2",
            "longitude": "110.4",
            "magnitude": "M5.2",
            "depth_km": "15",
            "epicentral_distance_km": "30",
        },
    )

    pdf_text = _pdf_text(pdf_path)
    assert "INFORMASI EVENT" in pdf_text.upper()
    assert ". INFORMASI EVENT" not in pdf_text.upper()
    assert "MAGNITUDO" in pdf_text.upper()
    assert "M5.2" in pdf_text


def test_sig_bmkg_thresholds_are_complete_and_ordered():
    exporter = ExportService()

    assert exporter._sig_label(0.0) == "SIG I"
    assert exporter._sig_label(2.8) == "SIG I"
    assert exporter._sig_label(2.9) == "SIG II"
    assert exporter._sig_label(88.9) == "SIG II"
    assert exporter._sig_label(89.0) == "SIG III"
    assert exporter._sig_label(167.9) == "SIG III"
    assert exporter._sig_label(168.0) == "SIG IV"
    assert exporter._sig_label(564.9) == "SIG IV"
    assert exporter._sig_label(565.0) == "SIG V"
