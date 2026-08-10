from __future__ import annotations

import numpy as np
from obspy import Stream, Trace

from core.preprocessing.qc import QCAnalyzer, QCSeverity


def test_quiet_window_is_warning_not_a_rejection():
    sampling_rate = 100.0
    quiet = np.zeros(500, dtype=np.float64)
    motion = np.random.default_rng(42).normal(0.0, 0.01, 5000)
    stream = Stream([Trace(data=np.concatenate((quiet, motion)), header={"sampling_rate": sampling_rate})])

    report = QCAnalyzer().analyze_stream(stream, identifier="quiet-window")
    metrics = next(iter(report.trace_metrics.values()))

    flatline = next(issue for issue in metrics.issues if issue.name == "FLATLINE")
    assert flatline.severity is QCSeverity.WARNING
    assert metrics.is_valid
