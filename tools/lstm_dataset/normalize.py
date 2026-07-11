"""Z-score normalization computed on train split only (DS-05).

Normalization stats (mean, std per channel) are computed from the training set
and applied to all splits. Stats are saved/loaded as JSON for inference reuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def compute_normalization_stats(
    train_inputs: np.ndarray,
) -> Dict[str, List[float]]:
    """Compute per-channel mean and std from training inputs.

    Args:
        train_inputs: Shape (N, L, 4), channels [x, y, vx, vy].

    Returns:
        Dict with keys "mean", "std", "channels".
        Each mean/std is a list of 4 floats.

    Raises:
        ValueError: If train_inputs is empty or has wrong shape.
    """
    if train_inputs.size == 0:
        raise ValueError("Cannot compute stats from empty training data.")
    if train_inputs.ndim != 3 or train_inputs.shape[2] != 4:
        raise ValueError(
            f"Expected shape (N, L, 4), got {train_inputs.shape}"
        )

    # Flatten across batch and time dimensions: (N*L, 4)
    flat = train_inputs.reshape(-1, 4)
    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)

    # Prevent division by zero for constant channels
    std = np.where(std < 1e-8, 1.0, std)

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "channels": ["x", "y", "vx", "vy"],
    }


def apply_normalization(
    data: np.ndarray,
    stats: Dict[str, List[float]],
) -> np.ndarray:
    """Apply z-score normalization: (data - mean) / std.

    Args:
        data: Shape (N, L, 4) or (N, H, 4).
        stats: Dict with "mean" and "std" keys.

    Returns:
        Normalized array of same shape.
    """
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    return (data - mean) / std


def inverse_normalization(
    normalized: np.ndarray,
    stats: Dict[str, List[float]],
) -> np.ndarray:
    """Inverse z-score normalization: data * std + mean.

    Args:
        normalized: Shape (N, L, 4) or (N, H, 4).
        stats: Dict with "mean" and "std" keys.

    Returns:
        Denormalized array of same shape.
    """
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    return normalized * std + mean


def save_normalization_stats(
    stats: Dict[str, List[float]],
    path: Path,
) -> None:
    """Save normalization stats to JSON.

    Args:
        stats: Dict with "mean", "std", "channels".
        path: Output JSON file path.
    """
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_normalization_stats(path: Path) -> Dict[str, List[float]]:
    """Load normalization stats from JSON.

    Args:
        path: JSON file path.

    Returns:
        Dict with "mean", "std", "channels".

    Raises:
        FileNotFoundError: If stats file doesn't exist.
        ValueError: If JSON is malformed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")

    with open(path, "r") as f:
        stats = json.load(f)

    required = {"mean", "std", "channels"}
    if not required.issubset(stats.keys()):
        raise ValueError(f"Stats JSON must contain keys: {required}")

    return stats
