#!/usr/bin/env python3
"""Two-stage phi smoothing: EMA then dwell-time gate (Eqs. 12.6-12.7).

Pure Python, ROS-free, per-track stateful. The two stages are DISTINCT and must
not be merged (Math Model Section 12.1):
  * EMA (12.6) bounds *how noisy* phi_j is (continuous, every cycle).
  * Dwell gate (12.7) bounds *how often* the value the solver sees may change
    (event-based, at most once per T_dwell cycles).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _GateState:
    """Per-track smoothing state."""
    phi_filtered: float           # last EMA output (Eq. 12.6)
    phi_used: float               # last gated value (Eq. 12.7)
    cycles_since_update: int      # k - k_last
    initialized: bool = False


class PhiSmoother:
    """Per-track EMA + dwell-time gate for the context index.

    ``alpha`` is the EMA factor (Eq. 12.6): phi_filtered(k) =
    alpha*phi_filtered(k-1) + (1-alpha)*phi(k). ``t_dwell_cycles`` is the
    minimum number of cycles between gate updates (Eq. 12.7).
    """

    def __init__(self, alpha: float, t_dwell_cycles: int):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if t_dwell_cycles < 1:
            raise ValueError("t_dwell_cycles must be >= 1")
        self._alpha = alpha
        self._t_dwell = t_dwell_cycles
        self._states: dict[int, _GateState] = {}

    def update(self, track_id: int, phi_raw: float) -> tuple[float, float]:
        """Advance one cycle for ``track_id``; return (phi_filtered, phi_used).

        On the first sample for a track, EMA and gate both seed to ``phi_raw``
        (no spurious lag/hold on entry) and the gate update clock starts.
        """
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

        # (a) EMA smoothing — continuous, every cycle (Eq. 12.6).
        st.phi_filtered = (
            self._alpha * st.phi_filtered + (1.0 - self._alpha) * phi_raw
        )

        # (b) Dwell-time gate — updates at most once per T_dwell (Eq. 12.7).
        st.cycles_since_update += 1
        if st.cycles_since_update >= self._t_dwell:
            st.phi_used = st.phi_filtered
            st.cycles_since_update = 0

        return st.phi_filtered, st.phi_used

    def drop(self, track_id: int) -> None:
        """Forget a track that is no longer present."""
        self._states.pop(track_id, None)

    def active_tracks(self) -> list[int]:
        return list(self._states.keys())
