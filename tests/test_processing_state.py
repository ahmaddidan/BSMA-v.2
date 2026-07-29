import pytest

from core.types.processing_state import (
    ProcessingState,
    StageStatus,
)


def test_default_state():

    state = ProcessingState()

    assert state.baseline is StageStatus.NOT_RUN
    assert state.filter is StageStatus.NOT_RUN
    assert state.completed is False
    assert state.failed is False
    assert state.running is False


def test_completed():

    state = ProcessingState(
        baseline=StageStatus.SUCCESS,
        detrend=StageStatus.SUCCESS,
        taper=StageStatus.SUCCESS,
        filter=StageStatus.SUCCESS,
        integration=StageStatus.SUCCESS,
        parameters=StageStatus.SUCCESS,
        spectrum=StageStatus.SUCCESS,
        export=StageStatus.SUCCESS,
    )

    assert state.completed
    assert not state.failed
    assert not state.running


def test_failed():

    state = ProcessingState()

    state.filter = StageStatus.FAILED

    assert state.failed
    assert not state.completed


def test_running():

    state = ProcessingState()

    state.integration = StageStatus.RUNNING

    assert state.running
    assert not state.completed


def test_reset():

    state = ProcessingState()

    state.filter = StageStatus.SUCCESS
    state.integration = StageStatus.FAILED

    state.reset()

    assert state.filter is StageStatus.NOT_RUN
    assert state.integration is StageStatus.NOT_RUN


def test_to_dict():

    state = ProcessingState()

    d = state.to_dict()

    assert d["filter"] == "NOT_RUN"
    assert d["baseline"] == "NOT_RUN"


def test_stage_status_values():

    assert StageStatus.SUCCESS.value == "SUCCESS"
    assert StageStatus.FAILED.value == "FAILED"