#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveConfig:
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
    return cfg.d_safe0 + cfg.k_d * phi


def saturated_velocity_limits(
    phi: float, cfg: AdaptiveConfig
) -> tuple[float, float, float]:
    vx = max(cfg.v_x_min, cfg.v_x0 - cfg.k_x * phi)
    vy = max(cfg.v_y_min, cfg.v_y0 - cfg.k_y * phi)
    omega = max(cfg.omega_min, cfg.omega_0 - cfg.k_omega * phi)
    return vx, vy, omega


def q_diag(phi: float, cfg: AdaptiveConfig) -> tuple[float, float, float]:
    return tuple(
        q0 + phi * qh for q0, qh in zip(cfg.q0_diag, cfg.qh_diag)
    )
