"""Evaluation metrics: ADE / FDE and per-channel MSE in denormalized units.

ADE (Average Displacement Error) = mean over all horizon steps of the position
error; FDE (Final Displacement Error) = position error at the last horizon step.
Both are reported in physical metres by denormalizing with the frozen stats.
"""
from __future__ import annotations

import numpy as np
import torch

from .dataset import TrajectoryDataset
from .model import LSTMPredictor


def _position_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-sample, per-step Euclidean position error over (N, H, 4)."""
    dxy = pred[..., 0:2] - target[..., 0:2]
    return np.sqrt(np.sum(dxy ** 2, axis=-1))       # (N, H)


def evaluate(model: LSTMPredictor, dataset: TrajectoryDataset) -> dict:
    """Compute ADE/FDE (metres) and per-channel MSE in physical units."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            p = model(x.unsqueeze(0)).squeeze(0)
            preds.append(dataset.denormalize(p).numpy())
            targets.append(dataset.denormalize(y).numpy())
    if not preds:
        return {"ade": 0.0, "fde": 0.0, "mse_per_channel": [0.0] * 4, "n": 0}

    pred = np.stack(preds)          # (N, H, 4)
    target = np.stack(targets)
    err = _position_error(pred, target)
    mse_ch = np.mean((pred - target) ** 2, axis=(0, 1))  # (4,)
    return {
        "ade": float(np.mean(err)),
        "fde": float(np.mean(err[:, -1])),
        "mse_per_channel": [float(v) for v in mse_ch],
        "n": len(dataset),
    }
