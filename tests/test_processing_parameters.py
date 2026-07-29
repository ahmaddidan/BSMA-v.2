"""
Tests for ParameterExtractionPlugin
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from core.processing.parameters import (
    ParameterExtractionPlugin,
)
from core.types.context import ProcessingContext
from core.types.metadata import TraceMetadata


class TestParameterExtraction:
    @pytest.fixture
    def mock_integrated_context(self) -> ProcessingContext:
        """
        Context yang sudah memiliki velocity dan displacement.
        """

        fs = 100.0

        t = np.arange(0.0, 10.0, 1.0 / fs)

        acc = 2.0 * np.sin(2 * np.pi * t)

        velocity = 5.0 * np.cos(2 * np.pi * t)

        displacement = 10.0 * np.sin(2 * np.pi * t)

        start = datetime(2024, 1, 1)

        metadata = TraceMetadata(
            network="IA",
            station="BBJM",
            location="00",
            channel="HNE",
            sampling_rate=fs,
            starttime=start,
            endtime=start + timedelta(seconds=len(acc) / fs),
            npts=len(acc),
        )

        context = ProcessingContext(
            waveform=acc,
            metadata=metadata,
        )

        context.cache.velocity = velocity
        context.cache.displacement = displacement

        return context

    def test_peak_values_extraction(
        self,
        mock_integrated_context,
    ):

        plugin = ParameterExtractionPlugin()

        result = plugin.process(mock_integrated_context)

        assert result.cache.pga == pytest.approx(2.0)

        assert result.cache.pgv == pytest.approx(5.0)

        assert result.cache.pgd == pytest.approx(10.0)

        assert result.cache.arias_intensity > 0.0

    def test_missing_kinematics_returns_error(self):

        fs = 100.0

        start = datetime(2024, 1, 1)

        metadata = TraceMetadata(
            network="IA",
            station="BBJM",
            location="00",
            channel="HNE",
            sampling_rate=fs,
            starttime=start,
            endtime=start + timedelta(seconds=10),
            npts=1000,
        )

        acc = np.zeros(1000)

        context = ProcessingContext(
            waveform=acc,
            metadata=metadata,
        )

        plugin = ParameterExtractionPlugin()

        result = plugin.process(context)

        assert result.qc.has_error

        assert result.cache.velocity is None

        assert result.cache.displacement is None