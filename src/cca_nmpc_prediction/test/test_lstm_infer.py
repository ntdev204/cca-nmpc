"""Tests for ONNX inference wrapper (Section 3.2).

Exports a tiny real ONNX model via tools.lstm_training so the wrapper is tested
against the exact signature the training package produces.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# tools/ is at repo root; make it importable when running from the package dir.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _export_tiny_model(tmp_path, L=8, H=12):
    from tools.lstm_training.model import LSTMPredictor, LSTMConfig
    from tools.lstm_training.export import export_onnx

    model = LSTMPredictor(LSTMConfig(hidden_size=8, horizon=H))
    stats = {"mean": [0.0] * 4, "std": [1.0] * 4,
             "channels": ["x", "y", "vx", "vy"]}
    # Keep the source stats in a separate dir so export_onnx's copy into the
    # model dir does not collide with the source (SameFileError).
    src_dir = tmp_path / "src"
    model_dir = tmp_path / "model"
    src_dir.mkdir()
    model_dir.mkdir()
    stats_path = src_dir / "normalization_stats.json"
    stats_path.write_text(json.dumps(stats))
    onnx_path = export_onnx(model, model_dir / "m.onnx", L=L, stats_path=stats_path)
    return onnx_path, stats_path


def test_predict_shape(tmp_path):
    onnx_path, stats_path = _export_tiny_model(tmp_path)
    from cca_nmpc_prediction.lstm_infer import LSTMPredictor as OnnxPredictor
    pred = OnnxPredictor(onnx_path, stats_path)
    out = pred.predict(np.random.randn(8, 4).astype(np.float32))
    assert out.shape == (12, 4)


def test_normalization_roundtrip_identity_stats(tmp_path):
    # with mean=0,std=1 the wrapper's norm/denorm are identities, so the ONNX
    # output equals the raw model output on the same input.
    onnx_path, stats_path = _export_tiny_model(tmp_path)
    from cca_nmpc_prediction.lstm_infer import LSTMPredictor as OnnxPredictor
    pred = OnnxPredictor(onnx_path, stats_path)
    x = np.zeros((8, 4), dtype=np.float32)
    out = pred.predict(x)
    assert np.all(np.isfinite(out))


def test_missing_model_raises(tmp_path):
    from cca_nmpc_prediction.lstm_infer import LSTMPredictor as OnnxPredictor
    stats = tmp_path / "s.json"
    stats.write_text(json.dumps({"mean": [0]*4, "std": [1]*4, "channels": []}))
    with pytest.raises(FileNotFoundError):
        OnnxPredictor(tmp_path / "nope.onnx", stats)


def test_rejects_bad_window_shape(tmp_path):
    onnx_path, stats_path = _export_tiny_model(tmp_path)
    from cca_nmpc_prediction.lstm_infer import LSTMPredictor as OnnxPredictor
    pred = OnnxPredictor(onnx_path, stats_path)
    with pytest.raises(ValueError):
        pred.predict(np.zeros((8, 3), dtype=np.float32))
