"""Tests for normalize.py (DS-05)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tools.lstm_dataset.normalize import (
    apply_normalization,
    compute_normalization_stats,
    inverse_normalization,
    load_normalization_stats,
    save_normalization_stats,
)


def test_compute_normalization_stats():
    """Test that mean and std are computed correctly per channel."""
    # Create simple data with known statistics
    # Channel 0: mean=1.0, std=1.0
    # Channel 1: mean=10.0, std=2.0
    data = np.array(
        [
            [[0.0, 8.0, 0.0, 0.0], [1.0, 10.0, 0.0, 0.0], [2.0, 12.0, 0.0, 0.0]],
            [[0.0, 8.0, 0.0, 0.0], [1.0, 10.0, 0.0, 0.0], [2.0, 12.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )  # (2, 3, 4)

    stats = compute_normalization_stats(data)

    assert stats["channels"] == ["x", "y", "vx", "vy"]
    assert len(stats["mean"]) == 4
    assert len(stats["std"]) == 4

    # Channel 0: values are [0,1,2,0,1,2] → mean=1.0
    np.testing.assert_allclose(stats["mean"][0], 1.0, atol=1e-6)
    # Channel 1: values are [8,10,12,8,10,12] → mean=10.0
    np.testing.assert_allclose(stats["mean"][1], 10.0, atol=1e-6)


def test_apply_normalization():
    """Test that normalization is applied correctly."""
    data = np.array(
        [[[1.0, 10.0, 0.5, -0.5], [2.0, 20.0, 1.0, -1.0]]], dtype=np.float32
    )  # (1, 2, 4)

    stats = {
        "mean": [1.0, 10.0, 0.5, -0.5],
        "std": [1.0, 5.0, 0.5, 0.5],
        "channels": ["x", "y", "vx", "vy"],
    }

    normalized = apply_normalization(data, stats)

    # First timestep should be all zeros (mean)
    np.testing.assert_allclose(normalized[0, 0], [0.0, 0.0, 0.0, 0.0], atol=1e-6)

    # Second timestep: (2-1)/1=1, (20-10)/5=2, (1-0.5)/0.5=1, (-1-(-0.5))/0.5=-1
    np.testing.assert_allclose(normalized[0, 1], [1.0, 2.0, 1.0, -1.0], atol=1e-6)


def test_inverse_normalization():
    """Test that inverse normalization recovers original data."""
    original = np.array(
        [[[1.0, 10.0, 0.5, -0.5], [2.0, 20.0, 1.0, -1.0]]], dtype=np.float32
    )

    stats = {
        "mean": [1.0, 10.0, 0.5, -0.5],
        "std": [1.0, 5.0, 0.5, 0.5],
        "channels": ["x", "y", "vx", "vy"],
    }

    normalized = apply_normalization(original, stats)
    recovered = inverse_normalization(normalized, stats)

    np.testing.assert_allclose(recovered, original, atol=1e-6)


def test_save_load_normalization_stats():
    """Test round-trip save/load of normalization stats."""
    stats = {
        "mean": [1.0, 2.0, 3.0, 4.0],
        "std": [0.5, 1.0, 1.5, 2.0],
        "channels": ["x", "y", "vx", "vy"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "stats.json"
        save_normalization_stats(stats, path)

        loaded = load_normalization_stats(path)

        assert loaded["channels"] == stats["channels"]
        np.testing.assert_allclose(loaded["mean"], stats["mean"])
        np.testing.assert_allclose(loaded["std"], stats["std"])


def test_normalization_stats_file_format():
    """Test that saved stats file has the correct JSON schema."""
    stats = {
        "mean": [1.0, 2.0, 3.0, 4.0],
        "std": [0.5, 1.0, 1.5, 2.0],
        "channels": ["x", "y", "vx", "vy"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "stats.json"
        save_normalization_stats(stats, path)

        with open(path) as f:
            loaded_json = json.load(f)

        assert "mean" in loaded_json
        assert "std" in loaded_json
        assert "channels" in loaded_json
        assert len(loaded_json["mean"]) == 4
        assert len(loaded_json["std"]) == 4
        assert loaded_json["channels"] == ["x", "y", "vx", "vy"]


def test_apply_normalization_immutable():
    """Test that apply_normalization doesn't mutate input."""
    original = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
    original_copy = original.copy()

    stats = {
        "mean": [0.0, 0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0, 1.0],
        "channels": ["x", "y", "vx", "vy"],
    }

    _ = apply_normalization(original, stats)

    np.testing.assert_array_equal(original, original_copy)
