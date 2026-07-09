"""Sliding-window extraction for LSTM input/target pairs (DS-03).

Produces (N, L, 4) inputs and (N, H, 4) targets, discarding windows
spanning track-loss gaps.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def extract_windows(
    resampled: np.ndarray,
    gap_segments: List[Tuple[int, int]],
    L: int,
    H: int,
    max_gap_fraction: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract sliding windows from a resampled trajectory.

    Args:
        resampled: (N, 5) array [timestamp, x, y, vx, vy].
        gap_segments: List of (start_idx, end_idx) marking large-gap regions
            from resample_trajectory.
        L: Input sequence length.
        H: Prediction horizon length.
        max_gap_fraction: Discard windows if more than this fraction of
            the window overlaps a gap segment.

    Returns:
        inputs: (N_windows, L, 4) array with channels [x, y, vx, vy].
        targets: (N_windows, H, 4) array with channels [x, y, vx, vy].

    Raises:
        ValueError: If L < 1, H < 1, or resampled has insufficient length.
    """
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    if H < 1:
        raise ValueError(f"H must be >= 1, got {H}")
    if len(resampled) < L + H:
        raise ValueError(
            f"Insufficient data: need {L + H} timesteps, got {len(resampled)}"
        )

    # Extract position and velocity columns (drop timestamp)
    data = resampled[:, 1:]  # [x, y, vx, vy]

    # Build gap mask (1 = in gap, 0 = valid)
    gap_mask = np.zeros(len(resampled), dtype=bool)
    for start, end in gap_segments:
        gap_mask[start : end + 1] = True

    inputs_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []

    for i in range(len(resampled) - L - H + 1):
        window_slice = slice(i, i + L + H)
        window_gap = gap_mask[window_slice]
        gap_fraction = np.mean(window_gap)

        if gap_fraction > max_gap_fraction:
            continue

        inp = data[i : i + L]
        tgt = data[i + L : i + L + H]
        inputs_list.append(inp)
        targets_list.append(tgt)

    if not inputs_list:
        # Return empty arrays with correct shape
        return (
            np.empty((0, L, 4), dtype=np.float32),
            np.empty((0, H, 4), dtype=np.float32),
        )

    inputs = np.stack(inputs_list, axis=0).astype(np.float32)
    targets = np.stack(targets_list, axis=0).astype(np.float32)

    return inputs, targets
