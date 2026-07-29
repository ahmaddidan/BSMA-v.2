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

    assert report.summary() == "PASS=1 | WARNING=1 | FAIL=1"


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