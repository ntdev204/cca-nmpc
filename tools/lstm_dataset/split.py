"""Trajectory-level train/val/test split (DS-04).

Splits by track_id so no single trajectory spans multiple splits.
Deterministic given a seed.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np


def split_by_trajectory(
    track_windows: Dict[int, Tuple[np.ndarray, np.ndarray]],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Split windowed data by trajectory (track_id).

    Each track_id appears in exactly one split. Windows from the same track
    always go to the same split, preventing data leakage.

    Args:
        track_windows: Dict mapping track_id -> (inputs, targets), where
            inputs is (N, L, 4) and targets is (N, H, 4).
        train_frac: Fraction for training (default 0.70).
        val_frac: Fraction for validation (default 0.15).
        test_frac: Fraction for testing (default 0.15).
        seed: Random seed for deterministic shuffling.

    Returns:
        Tuple of ((train_in, train_tgt), (val_in, val_tgt), (test_in, test_tgt)).

    Raises:
        ValueError: If fractions don't sum to ~1.0 or no data provided.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"Fractions must sum to 1.0, got {train_frac + val_frac + test_frac}"
        )
    if not track_windows:
        raise ValueError("No track windows provided for splitting.")

    # Deterministically shuffle track IDs
    track_ids = sorted(track_windows.keys())
    rng = random.Random(seed)
    shuffled_ids = track_ids[:]
    rng.shuffle(shuffled_ids)

    n = len(shuffled_ids)
    n_train = math.ceil(n * train_frac)
    n_val = math.ceil(n * val_frac)
    # test gets the remainder
    n_test = n - n_train - n_val
    if n_test < 0:
        n_val += n_test
        n_test = 0

    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train : n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val :]

    return (
        _gather_windows(track_windows, train_ids),
        _gather_windows(track_windows, val_ids),
        _gather_windows(track_windows, test_ids),
    )


def _gather_windows(
    track_windows: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ids: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate windows from a list of track IDs."""
    if not ids:
        # Return empty with compatible shapes
        sample_in, sample_tgt = next(iter(track_windows.values()))
        L, C = sample_in.shape[1], sample_in.shape[2]
        H = sample_tgt.shape[1]
        return (
            np.empty((0, L, C), dtype=np.float32),
            np.empty((0, H, C), dtype=np.float32),
        )

    inputs_parts = [track_windows[tid][0] for tid in ids]
    targets_parts = [track_windows[tid][1] for tid in ids]

    return (
        np.concatenate(inputs_parts, axis=0),
        np.concatenate(targets_parts, axis=0),
    )
