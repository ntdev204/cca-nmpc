"""Torch Dataset over the .npz splits + normalization_stats.json.

Reads the dataset-package contract: .npz with keys "inputs" (N,L,4) and
"targets" (N,H,4), and normalization_stats.json = {"mean":[4],"std":[4],
"channels":[...]}. Inputs and targets are z-score normalized with the SAME
frozen train-split stats (never recomputed) and denormalization is exposed for
evaluation in physical units.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_stats(stats_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load per-channel (mean, std) as float32 arrays of length 4."""
    with open(stats_path) as f:
        stats = json.load(f)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if mean.shape != (4,) or std.shape != (4,):
        raise ValueError("normalization stats must have 4 channels")
    return mean, std


class TrajectoryDataset(Dataset):
    """Normalized windowed trajectory dataset for the LSTM predictor."""

    def __init__(self, npz_path: str | Path, stats_path: str | Path):
        with np.load(npz_path) as data:
            if "inputs" not in data or "targets" not in data:
                raise KeyError("npz must contain 'inputs' and 'targets'")
            self._inputs = data["inputs"].astype(np.float32).copy()
            self._targets = data["targets"].astype(np.float32).copy()
        self._mean, self._std = load_stats(stats_path)
        self._std_safe = np.where(self._std == 0.0, 1.0, self._std)

    def __len__(self) -> int:
        return self._inputs.shape[0]

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self._mean) / self._std_safe

    def denormalize(self, arr: np.ndarray | torch.Tensor):
        """Inverse z-score back to physical units ([x, y, vx, vy])."""
        if isinstance(arr, torch.Tensor):
            mean = torch.tensor(self._mean, device=arr.device)
            std = torch.tensor(self._std_safe, device=arr.device)
            return arr * std + mean
        return arr * self._std_safe + self._mean

    def __getitem__(self, idx: int):
        x = self._normalize(self._inputs[idx])
        y = self._normalize(self._targets[idx])
        return torch.from_numpy(x), torch.from_numpy(y)
