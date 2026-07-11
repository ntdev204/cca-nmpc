#!/usr/bin/env python3
"""NmpcDiagnostics assembly helpers (Solver Design Sections 9, 11, 12).

Pure Python, ROS-free. Produces a plain dataclass mirroring the NmpcDiagnostics
message field-for-field; the node maps it onto the real message. slacks is a
list of (track_id, slack) — NOT a positional float array — matching SlackValue[].
"""
from __future__ import annotations

from dataclasses import dataclass

# Mirror NmpcDiagnostics LEVEL_* constants (ROS-free).
LEVEL_OK = 0
LEVEL_WARN = 1
LEVEL_ERROR = 2
LEVEL_STALE = 3


@dataclass
class DiagnosticsData:
    diagnostic_level: int
    solver_success: bool
    solve_time_ms: float
    slacks: list[tuple[int, float]]        # (track_id, slack) per human
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
    """Assemble diagnostics from a SolveResult-like set of inputs.

    ``cost_breakdown`` uses the SolveResult keys (tracking_cost, control_cost,
    smooth_cost, obstacle_cost, human_cost, terminal_cost); cost_total is their
    sum (Eq. 11.1).
    """
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
