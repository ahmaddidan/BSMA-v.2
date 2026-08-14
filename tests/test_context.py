from pathlib import Path

import numpy as np
from dataclasses import replace

from app import _clear_uploaded_data, WAVEFORM_DIRECTORY, INVENTORY_DIRECTORY
from core.types.processing_state import StageStatus
from tests.helpers import make_context


def test_clear_uploaded_data_removes_stale_input_files(tmp_path, monkeypatch):
    waveform_dir = tmp_path / "mseed"
    inventory_dir = tmp_path / "stationXML"
    waveform_dir.mkdir()
    inventory_dir.mkdir()
    stale_waveform = waveform_dir / "old.mseed"
    stale_inventory = inventory_dir / "old.xml"
    stale_waveform.write_bytes(b"stale")
    stale_inventory.write_bytes(b"stale")

    monkeypatch.setattr("app.WAVEFORM_DIRECTORY", waveform_dir)
    monkeypatch.setattr("app.INVENTORY_DIRECTORY", inventory_dir)

    _clear_uploaded_data()

    assert not stale_waveform.exists()
    assert not stale_inventory.exists()


def test_context_transitions_are_immutable():
    source = make_context(np.array([0.0, 1.0, 0.0]))
    updated = source.with_metrics(PGA=1.0).with_state(
        processing_state=replace(source.processing_state, baseline=StageStatus.SUCCESS)
    )
    assert source.metrics == {}
    assert updated.metrics["PGA"] == 1.0
    assert updated.processing_state.baseline is StageStatus.SUCCESS


def test_context_preserves_raw_waveform_when_acceleration_changes():
    source = make_context(np.array([0.0, 1.0, 0.0]))
    updated = source.with_acceleration(source.acceleration.copy_with(data=np.array([1.0, 2.0, 1.0])))
    assert np.array_equal(source.raw_waveform.data, [0.0, 1.0, 0.0])
    assert np.array_equal(updated.acceleration.data, [1.0, 2.0, 1.0])
