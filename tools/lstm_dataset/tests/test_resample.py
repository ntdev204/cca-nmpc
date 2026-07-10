"""Tests for resample.py (DS-02)."""

from __future__ import annotations

import numpy as np
import pytest

from tools.lstm_dataset.schema import TrajectoryRecord
from tools.lstm_dataset.resample import resample_trajectory


def _make_records(times, xs, ys, vxs, vys, track_id=0, session_id="test_session", sequence_id=0):
    """Helper to create TrajectoryRecord list."""
    return [
        TrajectoryRecord(
            session_id=session_id,
            sequence_id=sequence_id,
            timestamp=float(t),
            track_id=track_id,
            x=float(x),
            y=float(y),
            vx=float(vxx),
            vy=float(vyy),
            c=1.0,
        )
        for t, x, y, vxx, vyy in zip(times, xs, ys, vxs, vys)
    ]


def test_resample_uniform_motion():
    """Test resampling of uniform linear motion recovers expected positions."""
    # Trajectory at 1 m/s in x-direction, sampled at irregular times
    times = np.array([0.0, 0.1, 0.3, 0.5])
    x = times * 1.0  # uniform motion
    y = np.zeros_like(times)
    vx = np.ones_like(times)
    vy = np.zeros_like(times)

    records = _make_records(times, x, y, vx, vy)
    resampled, gaps = resample_trajectory(records, dt=0.1)

    # Expect 6 timesteps: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
    expected_x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    assert len(resampled) == 6
    np.testing.assert_allclose(resampled[:, 1], expected_x, atol=1e-6)
    assert len(gaps) == 0  # no large gaps (0.2s < default 0.2s threshold — use strict)


def test_resample_with_large_gap():
    """Test that large gaps are flagged."""
    # Gap between t=1.0 and t=3.0 (2 seconds, > 0.5s threshold)
    times = np.array([0.0, 0.5, 1.0, 3.0, 3.5])
    x = times * 0.5
    y = np.zeros_like(times)
    vx = np.full_like(times, 0.5)
    vy = np.zeros_like(times)

    records = _make_records(times, x, y, vx, vy)
    resampled, gaps = resample_trajectory(records, dt=0.5, velocity_rederive_threshold=0.6)

    # Should have flagged gap at t=1.0 to t=3.0 (2s gap > 0.6s threshold)
    assert len(gaps) > 0


def test_resample_empty_records():
    """Test that empty records raises error."""
    with pytest.raises(ValueError, match="empty"):
        resample_trajectory([], dt=0.1)


def test_resample_invalid_dt():
    """Test that dt <= 0 raises error."""
    records = _make_records([0.0, 0.1], [0.0, 0.1], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0])

    with pytest.raises(ValueError, match="positive"):
        resample_trajectory(records, dt=0.0)

    with pytest.raises(ValueError, match="positive"):
        resample_trajectory(records, dt=-0.1)


def test_resample_output_shape():
    """Test output array has correct column count."""
    times = np.linspace(0, 1, 10)
    x = np.zeros(10)
    y = np.zeros(10)
    vx = np.zeros(10)
    vy = np.zeros(10)

    records = _make_records(times, x, y, vx, vy)
    resampled, gaps = resample_trajectory(records, dt=0.1)

    # Output should have 5 columns: [timestamp, x, y, vx, vy]
    assert resampled.ndim == 2
    assert resampled.shape[1] == 5


def test_resample_monotonic_timestamps():
    """Test that output timestamps are monotonically increasing."""
    times = np.array([0.0, 0.1, 0.25, 0.4, 0.5])
    x = times * 2.0
    y = np.zeros(5)
    vx = np.full(5, 2.0)
    vy = np.zeros(5)

    records = _make_records(times, x, y, vx, vy)
    resampled, _ = resample_trajectory(records, dt=0.1)

    diffs = np.diff(resampled[:, 0])
    assert np.all(diffs > 0), "Timestamps should be strictly increasing"
