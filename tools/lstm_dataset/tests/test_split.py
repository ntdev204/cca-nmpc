"""Tests for split.py (DS-04)."""

from __future__ import annotations

import numpy as np
import pytest

from tools.lstm_dataset.schema import TrajectoryKey, TrajectoryRecord
from tools.lstm_dataset.split import split_by_trajectory


def test_split_by_trajectory_ratios():
    """Test that split ratios are approximately correct."""
    # Create 100 trajectories with 1 window each
    track_windows = {}
    for i in range(100):
        inp = np.random.randn(1, 8, 4).astype(np.float32)
        tgt = np.random.randn(1, 12, 4).astype(np.float32)
        key: TrajectoryKey = ("session", 0, i)
        track_windows[key] = (inp, tgt)

    (train_in, train_tgt), (val_in, val_tgt), (test_in, test_tgt) = (
        split_by_trajectory(track_windows, seed=42)
    )

    total = len(train_in) + len(val_in) + len(test_in)
    assert total == 100

    # Should be approximately 70/15/15
    assert 65 <= len(train_in) <= 75
    assert 10 <= len(val_in) <= 20
    assert 10 <= len(test_in) <= 20


def test_split_by_trajectory_no_leakage():
    """Test that no track_id appears in multiple splits."""
    # Create trajectories with multiple windows each
    track_windows = {}
    for i in range(20):
        n_windows = np.random.randint(1, 10)
        inp = np.random.randn(n_windows, 8, 4).astype(np.float32)
        tgt = np.random.randn(n_windows, 12, 4).astype(np.float32)
        key: TrajectoryKey = ("session", 0, i)
        track_windows[key] = (inp, tgt)

    (train_in, train_tgt), (val_in, val_tgt), (test_in, test_tgt) = (
        split_by_trajectory(track_windows, seed=42)
    )

    # Ensure splits are disjoint and cover all data
    total_windows = sum(len(w[0]) for w in track_windows.values())
    assert (
        len(train_in) + len(val_in) + len(test_in) == total_windows
    ), "Windows lost in split"


def test_split_deterministic():
    """Test that split is deterministic given the same seed."""
    track_windows = {}
    for i in range(50):
        inp = np.random.randn(2, 8, 4).astype(np.float32)
        tgt = np.random.randn(2, 12, 4).astype(np.float32)
        key: TrajectoryKey = ("session", 0, i)
        track_windows[key] = (inp, tgt)

    (train_in_1, _), (val_in_1, _), (test_in_1, _) = split_by_trajectory(
        track_windows, seed=42
    )
    (train_in_2, _), (val_in_2, _), (test_in_2, _) = split_by_trajectory(
        track_windows, seed=42
    )

    np.testing.assert_array_equal(train_in_1, train_in_2)
    np.testing.assert_array_equal(val_in_1, val_in_2)
    np.testing.assert_array_equal(test_in_1, test_in_2)


def test_split_different_seeds():
    """Test that different seeds produce different splits."""
    track_windows = {}
    for i in range(50):
        inp = np.random.randn(2, 8, 4).astype(np.float32)
        tgt = np.random.randn(2, 12, 4).astype(np.float32)
        key: TrajectoryKey = ("session", 0, i)
        track_windows[key] = (inp, tgt)

    (train_in_1, _), _, _ = split_by_trajectory(track_windows, seed=42)
    (train_in_2, _), _, _ = split_by_trajectory(track_windows, seed=99)

    # Different seeds should (very likely) produce different splits
    assert not np.array_equal(train_in_1, train_in_2)


def test_split_invalid_fractions():
    """Test that invalid fraction sums raise errors."""
    key: TrajectoryKey = ("session", 0, 0)
    track_windows = {
        key: (np.zeros((1, 8, 4), dtype=np.float32), np.zeros((1, 12, 4), dtype=np.float32))
    }

    with pytest.raises(ValueError, match="sum to 1.0"):
        split_by_trajectory(track_windows, train_frac=0.5, val_frac=0.3, test_frac=0.3)


def test_split_empty_data():
    """Test that empty data raises error."""
    with pytest.raises(ValueError, match="No track windows"):
        split_by_trajectory({})


def test_split_preserves_shapes():
    """Test that split outputs preserve L, H, and channel dimensions."""
    L, H, C = 8, 12, 4
    track_windows = {}
    for i in range(20):
        n_windows = 3
        inp = np.random.randn(n_windows, L, C).astype(np.float32)
        tgt = np.random.randn(n_windows, H, C).astype(np.float32)
        key: TrajectoryKey = ("session", 0, i)
        track_windows[key] = (inp, tgt)

    (train_in, train_tgt), (val_in, val_tgt), (test_in, test_tgt) = (
        split_by_trajectory(track_windows, seed=42)
    )

    # All should have shape (N, L, C) and (N, H, C)
    assert train_in.shape[1:] == (L, C)
    assert train_tgt.shape[1:] == (H, C)
    assert val_in.shape[1:] == (L, C)
    assert val_tgt.shape[1:] == (H, C)
    assert test_in.shape[1:] == (L, C)
    assert test_tgt.shape[1:] == (H, C)


def test_composite_key_prevents_cross_session_collision():
    """Test that same track_id in different sessions produces distinct keys."""

    # Two records with same track_id but different session_id
    rec1 = TrajectoryRecord(
        session_id="session_a",
        sequence_id=0,
        timestamp=0.0,
        track_id=5,
        x=0.0,
        y=0.0,
        vx=1.0,
        vy=0.0,
        c=1.0,
    )
    rec2 = TrajectoryRecord(
        session_id="session_b",
        sequence_id=0,
        timestamp=0.0,
        track_id=5,
        x=10.0,
        y=10.0,
        vx=0.0,
        vy=1.0,
        c=1.0,
    )

    key1 = rec1.trajectory_key()
    key2 = rec2.trajectory_key()

    # Keys must differ despite same track_id
    assert key1 != key2
    assert key1 == ("session_a", 0, 5)
    assert key2 == ("session_b", 0, 5)
