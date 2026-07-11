#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FALLBACK_NONE = 0
FALLBACK_SOLVER_FAILED = 1
FALLBACK_TIMEOUT = 2
FALLBACK_HELD_PREVIOUS = 3
FALLBACK_SAFE_STOP = 4


@dataclass
class FallbackDecision:
    u: np.ndarray
    fallback_triggered: bool
    fallback_code: int
    fallback_reason: str
    needs_reset: bool = False


def safe_stop_ramp(
    u_prev: np.ndarray, dt: float, decel_limit: float
) -> np.ndarray:
    u_prev = np.asarray(u_prev, float).flatten()
    max_step = max(0.0, decel_limit * dt)
    out = np.zeros_like(u_prev)
    for i, v in enumerate(u_prev):
        if abs(v) <= max_step:
            out[i] = 0.0
        else:
            out[i] = v - np.sign(v) * max_step
    return out


class FallbackController:

    def __init__(self, timeout_hold_cycles: int, dt: float, decel_limit: float):
        if timeout_hold_cycles < 0:
            raise ValueError("timeout_hold_cycles must be >= 0")
        self._max_hold = timeout_hold_cycles
        self._dt = dt
        self._decel = decel_limit
        self._consecutive_fail = 0
        self._last_good_u = np.zeros(3)
        self._last_cmd = np.zeros(3)

    def on_success(self, u0: np.ndarray) -> FallbackDecision:
        self._consecutive_fail = 0
        self._last_good_u = np.asarray(u0, float).flatten()
        self._last_cmd = self._last_good_u.copy()
        return FallbackDecision(
            u=self._last_good_u.copy(),
            fallback_triggered=False,
            fallback_code=FALLBACK_NONE,
            fallback_reason="",
        )

    def on_failure(self, *, timed_out: bool) -> FallbackDecision:
        self._consecutive_fail += 1
        base_code = FALLBACK_TIMEOUT if timed_out else FALLBACK_SOLVER_FAILED

        if self._consecutive_fail <= self._max_hold:
            self._last_cmd = self._last_good_u.copy()
            return FallbackDecision(
                u=self._last_good_u.copy(),
                fallback_triggered=True,
                fallback_code=FALLBACK_HELD_PREVIOUS,
                fallback_reason=(
                    f"held previous control ({self._consecutive_fail}"
                    f"/{self._max_hold}); trigger={base_code}"
                ),
            )

        ramp = safe_stop_ramp(self._last_cmd, self._dt, self._decel)
        self._last_cmd = ramp
        return FallbackDecision(
            u=ramp,
            fallback_triggered=True,
            fallback_code=FALLBACK_SAFE_STOP,
            fallback_reason="safe-stop ramp after exhausting holds",
            needs_reset=True,
        )

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_fail
