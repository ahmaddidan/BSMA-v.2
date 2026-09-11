import pytest

from core.types.qc import (
    QCReport,
    QCResult,
    QCSeverity,
    QCStatus,
)


def make_result(name, status):

    return QCResult(
        status=status,
        severity=QCSeverity.LOW,
        validator_name=name,
        message="test",
        metrics={"value": 10},
    )


def test_empty_report():

    report = QCReport()

    assert report.overall_status == QCStatus.PASS
    assert len(report) == 0


def test_add_result():

    report = QCReport()

    result = make_result("SpikeValidator", QCStatus.PASS)

    report.add_result(result)

    assert len(report) == 1

    assert report.get("SpikeValidator") is result


def test_overall_pass():

    report = QCReport()

    report.add_result(make_result("A", QCStatus.PASS))
    report.add_result(make_result("B", QCStatus.PASS))

    assert report.overall_status == QCStatus.PASS


def test_overall_warning():

    report = QCReport()

    report.add_result(make_result("A", QCStatus.PASS))
    report.add_result(make_result("B", QCStatus.WARNING))

    assert report.overall_status == QCStatus.WARNING


def test_overall_fail():

    report = QCReport()

    report.add_result(make_result("A", QCStatus.PASS))
    report.add_result(make_result("B", QCStatus.FAIL))

    assert report.overall_status == QCStatus.FAIL


def test_summary():

    report = QCReport()

    report.add_result(make_result("A", QCStatus.PASS))
    report.add_result(make_result("B", QCStatus.WARNING))
    report.add_result(make_result("C", QCStatus.FAIL))

    assert "PASS=1 | WARNING=1 | FAIL=1" in report.summary()


def test_to_dict():

    report = QCReport()

    report.add_result(make_result("SpikeValidator", QCStatus.PASS))

    data = report.to_dict()

    assert data["overall_status"] == "PASS"

    assert "SpikeValidator" in data["results"]


def test_metrics_are_immutable():

    result = QCResult(
        status=QCStatus.PASS,
        severity=QCSeverity.LOW,
        validator_name="SpikeValidator",
        message="OK",
        metrics={"threshold": 5},
    )

    with pytest.raises(TypeError):
        result.metrics["threshold"] = 10


def test_failed_warning_pass_helpers():

    report = QCReport()

    report.add_result(make_result("A", QCStatus.PASS))
    report.add_result(make_result("B", QCStatus.WARNING))
    report.add_result(make_result("C", QCStatus.FAIL))

    assert len(report.passed()) == 1
    assert len(report.warnings()) == 1
    assert len(report.failed()) == 1


def test_trace_qc_metrics_harmonized_thresholds():
    """Verify harmonized scientific QC thresholds (PASS >= 70, WARNING 50-69, FAIL < 50)."""
    from core.preprocessing.qc import TraceQCMetrics, QCIssue, QCSeverity as PreprocQCSeverity

    # 1. Clean trace: 100 -> PASS
    m_clean = TraceQCMetrics(trace_id="TEST.1")
    m_clean.calculate_quality_score()
    assert m_clean.quality_score == 100
    assert m_clean.is_valid is True
    assert m_clean.status == "PASS"

    # 2. Warning trace: 1 warning (-15) -> 85 -> PASS
    m_warn1 = TraceQCMetrics(trace_id="TEST.2")
    m_warn1.issues.append(QCIssue("MINOR", PreprocQCSeverity.WARNING))
    m_warn1.calculate_quality_score()
    assert m_warn1.quality_score == 85
    assert m_warn1.status == "PASS"

    # 3. Warning trace: 3 warnings (-45) -> 55 -> WARNING (50-69)
    m_warn3 = TraceQCMetrics(trace_id="TEST.3")
    for _ in range(3):
        m_warn3.issues.append(QCIssue("WARN", PreprocQCSeverity.WARNING))
    m_warn3.calculate_quality_score()
    assert m_warn3.quality_score == 55
    assert m_warn3.is_valid is True
    assert m_warn3.status == "WARNING"

    # 4. Severe trace: 2 errors (-80) -> 20 -> FAIL (<50)
    m_err = TraceQCMetrics(trace_id="TEST.4")
    m_err.issues.append(QCIssue("ERR1", PreprocQCSeverity.ERROR))
    m_err.issues.append(QCIssue("ERR2", PreprocQCSeverity.ERROR))
    m_err.calculate_quality_score()
    assert m_err.quality_score == 20
    assert m_err.is_valid is False
    assert m_err.status == "FAIL"

    # 5. Clipping fatal anomaly -> always FAIL
    m_clip = TraceQCMetrics(trace_id="TEST.5", has_clipping=True)
    m_clip.calculate_quality_score()
    assert m_clip.is_valid is False
    assert m_clip.status == "FAIL"