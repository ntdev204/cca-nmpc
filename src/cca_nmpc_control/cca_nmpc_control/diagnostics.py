#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

LEVEL_OK = 0
LEVEL_WARN = 1
LEVEL_ERROR = 2
LEVEL_STALE = 3


@dataclass
class DiagnosticsData:
    diagnostic_level: int
    solver_success: bool
    solve_time_ms: float
    slacks: list[tuple[int, float]]
    num_humans_active: int
    cost_total: float
    cost_track: float
    cost_control: float
    cost_smooth: float
    cost_human: float
    cost_obstacle: float
    cost_terminal: float
    fallback_triggered: bool
    fallback_code: int
    fallback_reason: str


def _level_from(success: bool, fallback_triggered: bool) -> int:
    if not success and fallback_triggered:
        return LEVEL_ERROR
    if fallback_triggered:
        return LEVEL_WARN
    return LEVEL_OK


def build_diagnostics(
    *,
    solver_success: bool,
    solve_time_ms: float,
    cost_breakdown: dict[str, float],
    slacks: dict[int, float],
    fallback_triggered: bool,
    fallback_code: int,
    fallback_reason: str = "",
) -> DiagnosticsData:
    cb = cost_breakdown
    track = cb.get("tracking_cost", 0.0)
    control = cb.get("control_cost", 0.0)
    smooth = cb.get("smooth_cost", 0.0)
    human = cb.get("human_cost", 0.0)
    obstacle = cb.get("obstacle_cost", 0.0)
    terminal = cb.get("terminal_cost", 0.0)
    total = track + control + smooth + human + obstacle + terminal

    slack_list = sorted((int(tid), float(s)) for tid, s in slacks.items())
    return DiagnosticsData(
        diagnostic_level=_level_from(solver_success, fallback_triggered),
        solver_success=solver_success,
        solve_time_ms=solve_time_ms,
        slacks=slack_list,
        num_humans_active=len(slack_list),
        cost_total=total,
        cost_track=track,
        cost_control=control,
        cost_smooth=smooth,
        cost_human=human,
        cost_obstacle=obstacle,
        cost_terminal=terminal,
        fallback_triggered=fallback_triggered,
        fallback_code=fallback_code,
        fallback_reason=fallback_reason,
    )
