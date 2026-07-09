"""Tests for windowing.py (DS-03)."""

from __future__ import annotations

import numpy as np
import pytest

from tools.lstm_dataset.windowing import extract_windows


def test_extract_windows_shapes():
    """Test that output shapes are correct."""
    L, H = 8, 12
    n_timesteps = 50
    times = np.arange(n_timesteps, dtype=np.float64) * 0.1
    x = times
    y = np.zeros(n_timesteps)
    vx = np.ones(n_timesteps)
    vy = np.zeros(n_timesteps)

    resampled = np.column_stack([times, x, y, vx, vy])
    gap_segments = []

    inputs, targets = extract_windows(resampled, gap_segments, L, H)

    # Expect N = n_timesteps - L - H + 1 windows
    expected_n = n_timesteps - L - H + 1
    assert inputs.shape == (expected_n, L, 4)
    assert targets.shape == (expected_n, H, 4)
    assert inputs.dtype == np.float32
    assert targets.dtype == np.float32


def test_extract_windows_channel_order():
    """Test that channel order is [x, y, vx, vy]."""
    L, H = 2, 2
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    vx = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    vy = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

    resampled = np.column_stack([times, x, y, vx, vy])
    gap_segments = []

    inputs, targets = extract_windows(resampled, gap_segments, L, H)

    # First window input should be timesteps 0,1
    np.testing.assert_allclose(inputs[0, 0], [1.0, 10.0, 0.1, 0.01])
    np.testing.assert_allclose(inputs[0, 1], [2.0, 20.0, 0.2, 0.02])

    # First window target should be timesteps 2,3
    np.testing.assert_allclose(targets[0, 0], [3.0, 30.0, 0.3, 0.03])
    np.testing.assert_allclose(targets[0, 1], [4.0, 40.0, 0.4, 0.04])


def test_extract_windows_discard_gap_windows():
    """Test that windows spanning large gaps are discarded."""
    L, H = 3, 2
    n_timesteps = 10
    times = np.arange(n_timesteps, dtype=np.float64) * 0.1
    x = times
    y = np.zeros(n_timesteps)
    vx = np.ones(n_timesteps)
    vy = np.zeros(n_timesteps)

    resampled = np.column_stack([times, x, y, vx, vy])

    # Mark indices 4-6 as a gap segment
    gap_segments = [(4, 6)]

    inputs, targets = extract_windows(
        resampled, gap_segments, L, H, max_gap_fraction=0.2
    )

    # Windows that overlap significantly with the gap should be discarded
    # Window starting at i=2: covers [2,3,4,5,6], 3/5 overlap → discard
    # Window starting at i=3: covers [3,4,5,6,7], 3/5 overlap → discard
    # Should have fewer windows than without gap
    expected_without_gap = n_timesteps - L - H + 1
    assert len(inputs) < expected_without_gap


def test_extract_windows_invalid_params():
    """Test that invalid L/H raises errors."""
    resampled = np.zeros((10, 5))
    gap_segments = []

    with pytest.raises(ValueError, match="L must be"):
        extract_windows(resampled, gap_segments, L=0, H=5)

    with pytest.raises(ValueError, match="H must be"):
        extract_windows(resampled, gap_segments, L=5, H=0)

    with pytest.raises(ValueError, match="Insufficient data"):
        extract_windows(resampled, gap_segments, L=8, H=12)


def test_extract_windows_empty_trajectory():
    """Test handling of trajectory with all windows discarded."""
    L, H = 3, 2
    n_timesteps = 10
    times = np.arange(n_timesteps, dtype=np.float64) * 0.1
    x = times
    y = np.zeros(n_timesteps)
    vx = np.ones(n_timesteps)
    vy = np.zeros(n_timesteps)

    resampled = np.column_stack([times, x, y, vx, vy])

    # Mark entire trajectory as gap
    gap_segments = [(0, n_timesteps - 1)]

    inputs, targets = extract_windows(
        resampled, gap_segments, L, H, max_gap_fraction=0.0
    )

    assert inputs.shape == (0, L, 4)
    assert targets.shape == (0, H, 4)
