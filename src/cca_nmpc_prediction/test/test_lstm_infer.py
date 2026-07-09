"""Tests for TensorRT LSTM inference wrapper + normalization (Section 3.2).

Runtime inference uses a TensorRT .engine (GPU-only), so the GPU path is not
exercised on Windows. These tests cover the ROS-free pieces: frozen-stat
normalization round-trip, the deterministic MockLSTMPredictor, and that the
TensorRT adapter fails clearly without a valid engine.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from cca_nmpc_prediction.lstm_infer import (
    load_normalization,
    normalize_window,
    denormalize,
    MockLSTMPredictor,
    TensorRtLSTMPredictor,
)


def _write_stats(tmp_path, mean, std):
    p = tmp_path / "normalization_stats.json"
    p.write_text(json.dumps({"mean": mean, "std": std,
                             "channels": ["x", "y", "vx", "vy"]}))
    return p


def test_normalization_roundtrip(tmp_path):
    stats = _write_stats(tmp_path, [1.0, 2.0, 0.0, 0.0], [2.0, 3.0, 1.0, 1.0])
    mean, std = load_normalization(stats)
    w = np.random.randn(8, 4).astype(np.float32)
    norm = normalize_window(w, mean, std)
    recovered = denormalize(norm, mean, std)
    assert np.allclose(recovered, w, atol=1e-5)


def test_load_stats_rejects_wrong_channels(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"mean": [0, 0], "std": [1, 1], "channels": []}))
    with pytest.raises(ValueError):
        load_normalization(p)


def test_zero_std_guarded(tmp_path):
    stats = _write_stats(tmp_path, [0.0] * 4, [0.0, 1.0, 1.0, 1.0])
    _mean, std = load_normalization(stats)
    assert std[0] == 1.0  # zero std replaced by 1.0


def test_mock_constant_velocity_extrapolation():
    pred = MockLSTMPredictor(horizon=12, dt=0.1)
    # last state x=1, y=2, vx=0.5, vy=-0.5
    window = np.zeros((8, 4), dtype=np.float32)
    window[-1] = [1.0, 2.0, 0.5, -0.5]
    out = pred.predict(window)
    assert out.shape == (12, 4)
    # step 1: x = 1 + 0.5*0.1 = 1.05, y = 2 - 0.5*0.1 = 1.95
    assert abs(out[0, 0] - 1.05) < 1e-6
    assert abs(out[0, 1] - 1.95) < 1e-6
    # velocity is held constant
    assert abs(out[-1, 2] - 0.5) < 1e-6


def test_mock_rejects_bad_window():
    pred = MockLSTMPredictor()
    with pytest.raises(ValueError):
        pred.predict(np.zeros((8, 3), dtype=np.float32))


def test_normalize_rejects_bad_window(tmp_path):
    stats = _write_stats(tmp_path, [0.0] * 4, [1.0] * 4)
    mean, std = load_normalization(stats)
    with pytest.raises(ValueError):
        normalize_window(np.zeros((8, 3)), mean, std)


def test_tensorrt_missing_engine_raises(tmp_path):
    stats = _write_stats(tmp_path, [0.0] * 4, [1.0] * 4)
    with pytest.raises(FileNotFoundError):
        TensorRtLSTMPredictor(tmp_path / "nope.engine", stats)
