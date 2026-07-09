"""Shared fixtures: a tiny synthetic .npz dataset + frozen stats on disk."""
import json

import numpy as np
import pytest


def _make_split(path, n, L=8, H=12, seed=0):
    rng = np.random.default_rng(seed)
    # simple learnable pattern: target continues the input's last velocity
    inputs = rng.standard_normal((n, L, 4)).astype(np.float32)
    targets = rng.standard_normal((n, H, 4)).astype(np.float32)
    np.savez(path, inputs=inputs, targets=targets)


@pytest.fixture
def dataset_dir(tmp_path):
    _make_split(tmp_path / "train.npz", 40, seed=1)
    _make_split(tmp_path / "val.npz", 12, seed=2)
    _make_split(tmp_path / "test.npz", 12, seed=3)
    stats = {"mean": [0.0] * 4, "std": [1.0] * 4,
             "channels": ["x", "y", "vx", "vy"]}
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    return tmp_path
