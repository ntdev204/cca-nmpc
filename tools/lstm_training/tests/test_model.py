"""Tests for the LSTM model + dataset shapes."""
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from tools.lstm_training.model import LSTMPredictor, LSTMConfig
from tools.lstm_training.dataset import TrajectoryDataset


def test_forward_shape():
    model = LSTMPredictor(LSTMConfig(horizon=12))
    x = torch.randn(5, 8, 4)
    y = model(x)
    assert y.shape == (5, 12, 4)


def test_default_config_no_required_args():
    model = LSTMPredictor()
    assert model.config.input_size == 4
    assert model.config.horizon == 12


def test_forward_rejects_wrong_channels():
    model = LSTMPredictor()
    try:
        model(torch.randn(2, 8, 3))
        assert False
    except ValueError:
        pass


def test_dataset_shapes_and_norm():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        inputs = np.random.randn(10, 8, 4).astype(np.float32)
        targets = np.random.randn(10, 12, 4).astype(np.float32)
        np.savez(d / "train.npz", inputs=inputs, targets=targets)
        stats = {"mean": [0.0] * 4, "std": [1.0] * 4,
                 "channels": ["x", "y", "vx", "vy"]}
        (d / "stats.json").write_text(json.dumps(stats))

        ds = TrajectoryDataset(d / "train.npz", d / "stats.json")
        assert len(ds) == 10
        x, y = ds[0]
        assert x.shape == (8, 4) and y.shape == (12, 4)


def test_dataset_denormalize_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        inputs = np.random.randn(4, 8, 4).astype(np.float32)
        targets = np.random.randn(4, 12, 4).astype(np.float32)
        np.savez(d / "train.npz", inputs=inputs, targets=targets)
        stats = {"mean": [1.0, 2.0, 0.0, 0.0], "std": [2.0, 3.0, 1.0, 1.0],
                 "channels": ["x", "y", "vx", "vy"]}
        (d / "stats.json").write_text(json.dumps(stats))
        ds = TrajectoryDataset(d / "train.npz", d / "stats.json")
        x, _ = ds[0]
        recovered = ds.denormalize(x).numpy()
        assert np.allclose(recovered, inputs[0], atol=1e-4)
