"""Resample trajectory records to a fixed time step.

Linear interpolation on position; finite-difference re-derivation of velocity
where gaps exceed a threshold (DS-02).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .schema import TrajectoryRecord


def resample_trajectory(
    records: List[TrajectoryRecord],
    dt: float,
    velocity_rederive_threshold: float = 0.2,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Resample a single trajectory to a fixed timestep.

    Args:
        records: List of TrajectoryRecord for one track_id, sorted by timestamp.
        dt: Target timestep in seconds (e.g., 0.125 for 8 Hz).
        velocity_rederive_threshold: If gap > this many seconds, re-derive
            velocity via finite difference instead of interpolating.

    Returns:
        resampled: (N, 5) array with columns [timestamp, x, y, vx, vy].
        gap_segments: List of (start_idx, end_idx) pairs marking segments
            where gaps exceeded the threshold. Windows spanning these should
            be discarded.

    Raises:
        ValueError: If records is empty or dt <= 0.
    """
    if not records:
        raise ValueError("Cannot resample empty trajectory.")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    # Extract arrays
    t = np.array([r.timestamp for r in records], dtype=np.float64)
    x = np.array([r.x for r in records], dtype=np.float64)
    y = np.array([r.y for r in records], dtype=np.float64)
    vx = np.array([r.vx for r in records], dtype=np.float64)
    vy = np.array([r.vy for r in records], dtype=np.float64)

    # Build uniform grid
    t_min, t_max = t[0], t[-1]
    n_steps = int(np.ceil((t_max - t_min) / dt)) + 1
    t_uniform = np.linspace(t_min, t_max, n_steps)

    # Linearly interpolate position
    x_interp = np.interp(t_uniform, t, x)
    y_interp = np.interp(t_uniform, t, y)

    # Detect large gaps in the original timeline
    dt_orig = np.diff(t)
    gap_mask = dt_orig > velocity_rederive_threshold
    gap_indices = np.where(gap_mask)[0]

    # For segments with large gaps, re-derive velocity via finite difference
    vx_interp = np.interp(t_uniform, t, vx)
    vy_interp = np.interp(t_uniform, t, vy)

    gap_segments: List[Tuple[int, int]] = []
    for gap_idx in gap_indices:
        # Find indices in t_uniform that fall between t[gap_idx] and t[gap_idx+1]
        start_t = t[gap_idx]
        end_t = t[gap_idx + 1]
        affected = (t_uniform >= start_t) & (t_uniform <= end_t)
        affected_indices = np.where(affected)[0]
        if len(affected_indices) > 0:
            gap_segments.append((affected_indices[0], affected_indices[-1]))

        # Re-derive velocity via central difference for affected region
        if len(affected_indices) > 1:
            idx_start = affected_indices[0]
            idx_end = affected_indices[-1] + 1
            # Central difference
            vx_fd = np.gradient(x_interp[idx_start:idx_end], t_uniform[idx_start:idx_end])
            vy_fd = np.gradient(y_interp[idx_start:idx_end], t_uniform[idx_start:idx_end])
            vx_interp[idx_start:idx_end] = vx_fd
            vy_interp[idx_start:idx_end] = vy_fd

    # Stack into output array
    resampled = np.column_stack([t_uniform, x_interp, y_interp, vx_interp, vy_interp])
    return resampled, gap_segments
