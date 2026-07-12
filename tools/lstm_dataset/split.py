"""Trajectory-level train/val/test split (DS-04).

Splits by composite trajectory key (session_id, run_id, sequence_id, track_id)
so no single trajectory spans multiple splits and no cross-session/cross-run ID
collision occurs (P1 dataset identity). Deterministic given a seed.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np

from .schema import TrajectoryKey


def split_by_trajectory(
    track_windows: Dict[TrajectoryKey, Tuple[np.ndarray, np.ndarray]],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Split windowed data by composite trajectory key.

    Each (session_id, run_id, sequence_id, track_id) appears in exactly one
    split. Windows from the same composite key always go to the same split,
    preventing data leakage across sessions and runs.

    Args:
        track_windows: Dict mapping TrajectoryKey -> (inputs, targets), where
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

    # Sort for determinism, then shuffle
    track_keys = sorted(track_windows.keys())
    rng = random.Random(seed)
    shuffled_keys = track_keys[:]
    rng.shuffle(shuffled_keys)

    n = len(shuffled_keys)
    n_train = math.ceil(n * train_frac)
    n_val = math.ceil(n * val_frac)
    n_test = n - n_train - n_val
    if n_test < 0:
        n_val += n_test
        n_test = 0

    train_keys = shuffled_keys[:n_train]
    val_keys = shuffled_keys[n_train : n_train + n_val]
    test_keys = shuffled_keys[n_train + n_val :]

    return (
        _gather_windows(track_windows, train_keys),
        _gather_windows(track_windows, val_keys),
        _gather_windows(track_windows, test_keys),
    )


def split_by_group(
    track_windows: Dict[TrajectoryKey, Tuple[np.ndarray, np.ndarray]],
    group_of: Dict[TrajectoryKey, str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Split so every trajectory of a group (e.g. a subject) stays in one split.

    Trajectory-level split (``split_by_trajectory``) prevents window leakage but
    the same person can still appear in train/val/test, which only supports
    "seen-subject trajectory prediction". Grouping by ``subject_id`` and holding
    whole subjects out enables the stronger "generalizes to new walking styles"
    claim (docs/09_roadmap.md Section 5). Fractions apply to the number of
    groups, not windows, so per-split window counts vary with group sizes.

    Args:
        track_windows: TrajectoryKey -> (inputs, targets).
        group_of: TrajectoryKey -> group label (e.g. subject_id). Every key in
            ``track_windows`` must be present.
        train_frac, val_frac, test_frac: group-count fractions (sum to 1.0).
        seed: deterministic group shuffle.

    Returns:
        ((train_in, train_tgt), (val_in, val_tgt), (test_in, test_tgt)).

    Raises:
        ValueError: fractions don't sum to ~1.0, no data, or a key lacks a group.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"Fractions must sum to 1.0, got {train_frac + val_frac + test_frac}"
        )
    # Every split must be positive: a held-out split needs non-empty train, val
    # AND test. Zero val/test also makes the val/(val+test) ratio a 0/0 division.
    if train_frac <= 0 or val_frac <= 0 or test_frac <= 0:
        raise ValueError(
            f"train_frac, val_frac and test_frac must all be > 0 for a "
            f"three-way held-out split, got train={train_frac}, "
            f"val={val_frac}, test={test_frac}."
        )
    if not track_windows:
        raise ValueError("No track windows provided for splitting.")
    missing = [k for k in track_windows if k not in group_of]
    if missing:
        raise ValueError(f"{len(missing)} trajectory keys have no group label.")

    # Group keys by label, then split the set of labels (not the trajectories).
    keys_by_group: Dict[str, List[TrajectoryKey]] = {}
    for key in track_windows:
        keys_by_group.setdefault(group_of[key], []).append(key)

    if len(keys_by_group) == 1:
        only_group = next(iter(keys_by_group.keys()))
        raise ValueError(
            f"Cannot perform held-out split with only one group ('{only_group}'). "
            f"All {len(track_windows)} trajectories belong to the same group."
        )

    groups = sorted(keys_by_group.keys())
    rng = random.Random(seed)
    shuffled = groups[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    # Guarantee non-empty splits: floor for train, allocate remainder to val/test.
    # Ceiled train+val can exhaust all groups when n < 10, leaving test empty.
    n_train = math.floor(n * train_frac)
    remainder = n - n_train
    n_val = math.floor(remainder * (val_frac / (val_frac + test_frac)))
    n_test = remainder - n_val

    # Guard: require at least 1 group per split for held-out claim to hold.
    # Minimum n that yields non-empty splits depends on fractions; for default
    # 70/15/15 the floor formula gives 2/1/1 at n=4 (first valid count).
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"Cannot split {n} groups into 3 non-empty sets with "
            f"train={train_frac}, val={val_frac}, test={test_frac}. "
            f"Need more subject groups (current: {n}). "
            f"Use --split-mode trajectory if fewer subjects available."
        )

    train_groups = shuffled[:n_train]
    val_groups = shuffled[n_train : n_train + n_val]
    test_groups = shuffled[n_train + n_val :]

    def _keys(labels: List[str]) -> List[TrajectoryKey]:
        return [k for label in labels for k in keys_by_group[label]]

    return (
        _gather_windows(track_windows, _keys(train_groups)),
        _gather_windows(track_windows, _keys(val_groups)),
        _gather_windows(track_windows, _keys(test_groups)),
    )


def _gather_windows(
    track_windows: Dict[TrajectoryKey, Tuple[np.ndarray, np.ndarray]],
    keys: List[TrajectoryKey],
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate windows from a list of composite trajectory keys."""
    if not keys:
        sample_in, sample_tgt = next(iter(track_windows.values()))
        L, C = sample_in.shape[1], sample_in.shape[2]
        H = sample_tgt.shape[1]
        return (
            np.empty((0, L, C), dtype=np.float32),
            np.empty((0, H, C), dtype=np.float32),
        )

    inputs_parts = [track_windows[k][0] for k in keys]
    targets_parts = [track_windows[k][1] for k in keys]

    return (
        np.concatenate(inputs_parts, axis=0),
        np.concatenate(targets_parts, axis=0),
    )
