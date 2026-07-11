import json

import numpy as np
import pytest

from cca_nmpc_prediction.lstm_infer import (
    load_normalization,
    normalize_window,
    denormalize,
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
    assert std[0] == 1.0


def test_normalize_rejects_bad_window(tmp_path):
    stats = _write_stats(tmp_path, [0.0] * 4, [1.0] * 4)
    mean, std = load_normalization(stats)
    with pytest.raises(ValueError):
        normalize_window(np.zeros((8, 3)), mean, std)


def test_tensorrt_missing_engine_raises(tmp_path):
    stats = _write_stats(tmp_path, [0.0] * 4, [1.0] * 4)
    with pytest.raises(FileNotFoundError):
        TensorRtLSTMPredictor(tmp_path / "nope.engine", stats)
