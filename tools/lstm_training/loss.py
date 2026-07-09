"""Weighted prediction loss (Eq. 13.2): L = MSE_position + lambda * MSE_velocity.

Channels 0,1 are position (x, y); channels 2,3 are velocity (vx, vy). The
lambda weight scales the velocity term only, letting position accuracy dominate
while still shaping the velocity prediction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_trajectory_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_vel: float = 0.5,
) -> torch.Tensor:
    """Position MSE + lambda * velocity MSE over (B, H, 4) tensors."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.size(-1) != 4:
        raise ValueError("last dim must be 4 ([x, y, vx, vy])")

    mse_pos = F.mse_loss(pred[..., 0:2], target[..., 0:2])
    mse_vel = F.mse_loss(pred[..., 2:4], target[..., 2:4])
    return mse_pos + lambda_vel * mse_vel
