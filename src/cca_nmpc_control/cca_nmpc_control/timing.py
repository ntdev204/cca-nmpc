#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CycleTiming:
    parameter_update_ms: float = 0.0
    solver_ms: float = 0.0
    publish_ms: float = 0.0
    total_cycle_ms: float = 0.0
    over_budget: bool = False

    @property
    def accounted_ms(self) -> float:
        return self.parameter_update_ms + self.solver_ms + self.publish_ms


def build_cycle_timing(
    parameter_update_ms: float,
    solver_ms: float,
    publish_ms: float,
    dt_s: float,
    total_cycle_ms: float | None = None,
) -> CycleTiming:
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    total = (
        total_cycle_ms
        if total_cycle_ms is not None
        else parameter_update_ms + solver_ms + publish_ms
    )
    budget_ms = dt_s * 1e3
    return CycleTiming(
        parameter_update_ms=parameter_update_ms,
        solver_ms=solver_ms,
        publish_ms=publish_ms,
        total_cycle_ms=total,
        over_budget=total > budget_ms,
    )
