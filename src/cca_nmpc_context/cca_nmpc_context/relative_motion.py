#!/usr/bin/env python3
"""Relative-motion geometry between robot and a human (Eqs. 7.1-7.5).

Pure NumPy, ROS-free — unit-testable on Windows. All quantities are in the 2D
map frame. The robot velocity is expressed in the world frame from body-frame
commands using the heading ``theta`` (Eq. 7.3), matching the Mecanum dynamics
(Eq. 4.1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RelativeMotion:
    """Result of the relative-motion computation for one human."""
    d_h: float            # Eq. 7.2 distance robot<->human [m]
    cos_dtheta: float     # Eq. 7.5 relative-approach cosine in [-1, 1]
    v_h_speed: float      # |v_h| human speed [m/s]


def robot_world_velocity(
    vx: float, vy: float, theta: float
) -> tuple[float, float]:
    """Robot body-frame velocity -> world frame (Eq. 7.3)."""
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
    """Compute distance, approach cosine, and human speed (Eqs. 7.1-7.5).

    ``cos_dtheta`` follows Eq. 7.5 with the unit direction ``e`` from human to
    robot (Eq. 7.4): it is positive when the relative velocity ``v_rel = v_h -
    v_r`` points from the human toward the robot (closing / approaching) and
    negative when receding.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    dx = robot_x - human_x
    dy = robot_y - human_y
    d_h = math.hypot(dx, dy)                          # Eq. 7.2

    # Unit direction from human to robot (Eq. 7.4), epsilon-guarded.
    denom_d = max(d_h, epsilon)
    ex, ey = dx / denom_d, dy / denom_d

    # Relative velocity v_rel = v_h - v_r (Eq. 7.3), robot vel in world frame.
    rvx, rvy = robot_world_velocity(robot_vx, robot_vy, robot_theta)
    vrx, vry = human_vx - rvx, human_vy - rvy
    v_rel_norm = math.hypot(vrx, vry)

    # Eq. 7.5, epsilon-guarded against a stationary relative motion.
    cos_dtheta = (vrx * ex + vry * ey) / max(v_rel_norm, epsilon)
    cos_dtheta = max(-1.0, min(1.0, cos_dtheta))

    v_h_speed = math.hypot(human_vx, human_vy)
    return RelativeMotion(d_h=d_h, cos_dtheta=cos_dtheta, v_h_speed=v_h_speed)
