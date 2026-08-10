from dataclasses import replace

from core.types.processing_state import ProcessingState, StageStatus


def test_processing_state_readiness_follows_dependencies():
    state = ProcessingState()
    assert state.is_raw
    assert not state.ready_for_integration
    ready = replace(
        state,
        baseline=StageStatus.SUCCESS,
        detrend=StageStatus.SUCCESS,
        taper=StageStatus.SUCCESS,
        filter=StageStatus.SUCCESS,
    )
    assert ready.ready_for_integration


def test_processing_state_serializes_current_stage_values():
    state = replace(ProcessingState(), integration=StageStatus.SUCCESS)
    exported = state.to_dict()
    assert exported["integration"] == StageStatus.SUCCESS.value
