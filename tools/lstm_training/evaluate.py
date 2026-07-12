"""Evaluation metrics: ADE / FDE and per-channel MSE in denormalized units.

ADE (Average Displacement Error) = mean over all horizon steps of the position
error; FDE (Final Displacement Error) = position error at the last horizon step.
Both are reported in physical metres by denormalizing with the frozen stats.

IMPORTANT — label provenance: Dataset 01 targets are tracker observations
(YOLO/depth/Kalman), i.e. *pseudo-ground-truth*, not an independent reference.
ADE/FDE against these targets measures how well the LSTM reproduces the tracker
output, and can UNDERSTATE true human-position error because predictions and
targets share the same perception chain. Every result therefore carries a
``label_source`` field so the paper never presents tracker pseudo-label ADE/FDE
as absolute accuracy. Use ``label_source="independent_reference"`` only for a
mocap / overhead-camera / AprilTag subset (docs/09_roadmap.md Section 4,
docs/04_dataset_specification.md Section 2.6).
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


def evaluate(
    model: LSTMPredictor,
    dataset: TrajectoryDataset,
    label_source: str = "tracker_pseudo_gt",
) -> dict:
    """Compute ADE/FDE (metres) and per-channel MSE in physical units.

    Args:
        model: trained predictor.
        dataset: evaluation dataset.
        label_source: provenance of the targets. Defaults to
            ``"tracker_pseudo_gt"`` because Dataset 01 targets are tracker
            pseudo-labels; pass ``"independent_reference"`` for a mocap / marker
            subset. Echoed into the result so downstream reporting cannot
            silently treat pseudo-label error as absolute accuracy.
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            p = model(x.unsqueeze(0)).squeeze(0)
            preds.append(dataset.denormalize(p).numpy())
            targets.append(dataset.denormalize(y).numpy())
    if not preds:
        return {"ade": 0.0, "fde": 0.0, "mse_per_channel": [0.0] * 4, "n": 0,
                "label_source": label_source}

    pred = np.stack(preds)          # (N, H, 4)
    target = np.stack(targets)
    err = _position_error(pred, target)
    mse_ch = np.mean((pred - target) ** 2, axis=(0, 1))  # (4,)
    return {
        "ade": float(np.mean(err)),
        "fde": float(np.mean(err[:, -1])),
        "mse_per_channel": [float(v) for v in mse_ch],
        "n": len(dataset),
        "label_source": label_source,
    }
