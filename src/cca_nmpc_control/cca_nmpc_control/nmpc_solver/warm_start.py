"""
Warm-start shift-and-append and invalidation triggers.

The shift-and-append scheme (docs/08_solver_design.md Section 5.1) initializes
the next solve from the previous solution shifted one step, with the state
trajectory re-propagated from the shifted controls (kept dynamically consistent
rather than shifted directly). Section 5.2 enumerates the situations where the
previous solution is NOT a reasonable guess and a reset() must happen instead:
goal changed, map relocalization, robot lifted, odom jump.

Pure Python / NumPy, ROS-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def shift_and_append_controls(u_seq: np.ndarray) -> np.ndarray:
    """Shift controls left one step, duplicating the last (Section 5.1).

    Args:
        u_seq: (N, nu) applied-then-planned control sequence.
    Returns:
        (N, nu) warm-start control sequence: u1..u_{N-1}, u_{N-1}.
    """
    u_seq = np.asarray(u_seq, float)
    if u_seq.ndim != 2 or u_seq.shape[0] < 1:
        raise ValueError("u_seq must be (N, nu) with N >= 1")
    return np.vstack([u_seq[1:], u_seq[-1:]])


def repropagate_states(
    x0: np.ndarray, u_seq: np.ndarray, integrate
) -> np.ndarray:
    """Re-propagate the state trajectory from x0 through shifted controls.

    Args:
        x0: (nx,) current state.
        u_seq: (N, nu) control sequence.
        integrate: callable (x, u) -> x_next (e.g. OCPSpec.integrate_numpy).
    Returns:
        (N+1, nx) state trajectory.
    """
    x0 = np.asarray(x0, float).flatten()
    xs = [x0]
    for k in range(u_seq.shape[0]):
        xs.append(np.asarray(integrate(xs[-1], u_seq[k]), float).flatten())
    return np.vstack(xs)


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class InvalidationThresholds:
    odom_jump_pos_thresh_m: float = 0.30
    odom_jump_yaw_thresh_rad: float = 0.35


def detect_odom_jump(
    prev_pose: tuple[float, float, float],
    curr_pose: tuple[float, float, float],
    thresh: InvalidationThresholds,
) -> bool:
    """True if consecutive /odom differ beyond thresholds (Section 5.2)."""
    dx = curr_pose[0] - prev_pose[0]
    dy = curr_pose[1] - prev_pose[1]
    dpos = math.hypot(dx, dy)
    dyaw = abs(wrap_angle(curr_pose[2] - prev_pose[2]))
    return (
        dpos > thresh.odom_jump_pos_thresh_m
        or dyaw > thresh.odom_jump_yaw_thresh_rad
    )


def detect_goal_change(
    prev_goal: tuple[float, float, float] | None,
    curr_goal: tuple[float, float, float] | None,
    tol: float = 1e-6,
) -> bool:
    """True if the reference goal changed discontinuously (Section 5.2)."""
    if prev_goal is None or curr_goal is None:
        return prev_goal is not curr_goal
    return any(abs(a - b) > tol for a, b in zip(prev_goal, curr_goal))


def should_reset(
    *,
    goal_changed: bool = False,
    map_relocalized: bool = False,
    robot_lifted: bool = False,
    odom_jump: bool = False,
) -> bool:
    """Aggregate the Section 5.2 invalidation triggers into a reset decision."""
    return bool(goal_changed or map_relocalized or robot_lifted or odom_jump)
