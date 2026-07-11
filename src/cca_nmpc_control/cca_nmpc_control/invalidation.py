#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

from cca_nmpc_control.nmpc_solver.warm_start import (
    InvalidationThresholds,
    detect_odom_jump,
    detect_goal_change,
    should_reset,
)


@dataclass
class InvalidationEvents:
    goal_changed: bool = False
    odom_jump: bool = False
    robot_lifted: bool = False
    map_relocalized: bool = False

    @property
    def any(self) -> bool:
        return should_reset(
            goal_changed=self.goal_changed,
            odom_jump=self.odom_jump,
            robot_lifted=self.robot_lifted,
            map_relocalized=self.map_relocalized,
        )


class InvalidationDetector:

    def __init__(
        self,
        pos_thresh_m: float = 0.30,
        yaw_thresh_rad: float = 0.35,
    ):
        self._thresh = InvalidationThresholds(pos_thresh_m, yaw_thresh_rad)
        self._prev_pose: tuple[float, float, float] | None = None
        self._prev_goal: tuple[float, float, float] | None = None

    def check(
        self,
        pose: tuple[float, float, float],
        goal: tuple[float, float, float] | None,
        *,
        robot_lifted: bool = False,
        map_relocalized: bool = False,
    ) -> InvalidationEvents:
        odom_jump = False
        if self._prev_pose is not None:
            odom_jump = detect_odom_jump(self._prev_pose, pose, self._thresh)

        goal_changed = False
        if self._prev_goal is not None or goal is not None:
            goal_changed = detect_goal_change(self._prev_goal, goal)

        self._prev_pose = pose
        self._prev_goal = goal
        return InvalidationEvents(
            goal_changed=goal_changed,
            odom_jump=odom_jump,
            robot_lifted=robot_lifted,
            map_relocalized=map_relocalized,
        )

    def reset_history(self) -> None:
        self._prev_pose = None
        self._prev_goal = None
