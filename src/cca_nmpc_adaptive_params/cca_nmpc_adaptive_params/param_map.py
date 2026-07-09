#!/usr/bin/env python3
"""Closed-form adaptive-parameter maps (Eqs. 10.1-10.3).

Pure Python, ROS-free. Maps the (gated) context index phi to NMPC runtime
parameters: safety distance, saturated velocity limits with minimum-motion
floors (the "frozen robot" fix), and the tracking-weight diagonal Q(phi).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveConfig:
    """Adaptive-parameter constants (docs/07 adaptive_param_node block)."""
    d_safe0: float
    k_d: float
    v_x0: float
    v_y0: float
    omega_0: float
    k_x: float
    k_y: float
    k_omega: float
    v_x_min: float
    v_y_min: float
    omega_min: float
    q0_diag: tuple[float, float, float]
    qh_diag: tuple[float, float, float]


def d_safe(phi: float, cfg: AdaptiveConfig) -> float:
    """Safety distance d_safe(phi) = d_safe0 + k_d * phi (Eq. 10.1)."""
    return cfg.d_safe0 + cfg.k_d * phi


def saturated_velocity_limits(
    phi: float, cfg: AdaptiveConfig
) -> tuple[float, float, float]:
    """Saturated (vx_max, vy_max, omega_max) with floors (Eq. 10.2).

    v = max(v_min, v0 - k*phi) so the robot always keeps a non-zero escape
    velocity even at phi = 1.
    """
    vx = max(cfg.v_x_min, cfg.v_x0 - cfg.k_x * phi)
    vy = max(cfg.v_y_min, cfg.v_y0 - cfg.k_y * phi)
    omega = max(cfg.omega_min, cfg.omega_0 - cfg.k_omega * phi)
    return vx, vy, omega


def q_diag(phi: float, cfg: AdaptiveConfig) -> tuple[float, float, float]:
    """Tracking-weight diagonal Q(phi) = Q0 + phi * Qh (Eq. 10.3).

    Length-3 diagonal over [x_r, y_r, theta_r] — NOT a flattened 3x3.
    """
    return tuple(
        q0 + phi * qh for q0, qh in zip(cfg.q0_diag, cfg.qh_diag)
    )
