#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RelativeMotion:
    d_h: float
    cos_dtheta: float
    v_h_speed: float


def robot_world_velocity(
    vx: float, vy: float, theta: float
) -> tuple[float, float]:
    c, s = math.cos(theta), math.sin(theta)
    return (vx * c - vy * s, vx * s + vy * c)


def compute_relative_motion(
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    robot_vx: float,
    robot_vy: float,
    human_x: float,
    human_y: float,
    human_vx: float,
    human_vy: float,
    epsilon: float = 1e-3,
) -> RelativeMotion:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    dx = robot_x - human_x
    dy = robot_y - human_y
    d_h = math.hypot(dx, dy)

    denom_d = max(d_h, epsilon)
    ex, ey = dx / denom_d, dy / denom_d

    rvx, rvy = robot_world_velocity(robot_vx, robot_vy, robot_theta)
    vrx, vry = human_vx - rvx, human_vy - rvy
    v_rel_norm = math.hypot(vrx, vry)

    cos_dtheta = (vrx * ex + vry * ey) / max(v_rel_norm, epsilon)
    cos_dtheta = max(-1.0, min(1.0, cos_dtheta))

    v_h_speed = math.hypot(human_vx, human_vy)
    return RelativeMotion(d_h=d_h, cos_dtheta=cos_dtheta, v_h_speed=v_h_speed)
