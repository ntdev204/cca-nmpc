#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _GateState:
    phi_filtered: float
    phi_used: float
    cycles_since_update: int
    initialized: bool = False


class PhiSmoother:

    def __init__(self, alpha: float, t_dwell_cycles: int):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if t_dwell_cycles < 1:
            raise ValueError("t_dwell_cycles must be >= 1")
        self._alpha = alpha
        self._t_dwell = t_dwell_cycles
        self._states: dict[int, _GateState] = {}

    def update(self, track_id: int, phi_raw: float) -> tuple[float, float]:
        st = self._states.get(track_id)
        if st is None or not st.initialized:
            st = _GateState(
                phi_filtered=phi_raw,
                phi_used=phi_raw,
                cycles_since_update=0,
                initialized=True,
            )
            self._states[track_id] = st
            return st.phi_filtered, st.phi_used

        st.phi_filtered = (
            self._alpha * st.phi_filtered + (1.0 - self._alpha) * phi_raw
        )

        st.cycles_since_update += 1
        if st.cycles_since_update >= self._t_dwell:
            st.phi_used = st.phi_filtered
            st.cycles_since_update = 0

        return st.phi_filtered, st.phi_used

    def drop(self, track_id: int) -> None:
        self._states.pop(track_id, None)

    def active_tracks(self) -> list[int]:
        return list(self._states.keys())
