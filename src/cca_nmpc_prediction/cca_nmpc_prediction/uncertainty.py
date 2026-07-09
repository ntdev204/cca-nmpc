#!/usr/bin/env python3
"""Predictive uncertainty sigma_h (Eqs. 6.3, 8.3, 13.3).

Pure Python/NumPy, ROS-free, per-track stateful.

  * Eq. 6.3: sigma_h = variance of (predicted - realized) state over the last W
    LSTM cycles, computed at each LSTM refresh.
  * Eq. 13.3: BETWEEN refreshes the held prediction ages, so sigma_h grows
    monotonically: sigma_h(k) = min(sigma_max, sigma_base + beta*dt), never
    decreasing within a hold interval, reset to the fresh Eq. 6.3 value on each
    new refresh.
  * Eq. 8.3: sigma_h_tilde = clip(sigma_h / sigma_max, 0, 1) feeds the context
    uncertainty term u_h.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class _TrackUncertainty:
    errors: deque = field(default_factory=lambda: deque())
    sigma_base: float = 0.0     # last Eq. 6.3 value at most recent refresh
    sigma_h: float = 0.0        # current value (may have grown, Eq. 13.3)


class UncertaintyEstimator:
    """Rolling prediction-error variance + stale growth, per track."""

    def __init__(self, window_W: int, beta: float, sigma_max: float):
        if window_W < 1:
            raise ValueError("window_W must be >= 1")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if sigma_max <= 0.0:
            raise ValueError("sigma_max must be positive")
        self._W = window_W
        self._beta = beta
        self._sigma_max = sigma_max
        self._tracks: dict[int, _TrackUncertainty] = {}

    def refresh(self, track_id: int, prediction_error: float) -> float:
        """Record a realized-vs-predicted error and recompute sigma_h (Eq. 6.3).

        ``prediction_error`` is a scalar per cycle (e.g. Euclidean position
        error between the one-step-ahead prediction made last cycle and the
        state realized this cycle). Returns the refreshed sigma_h.
        """
        tr = self._tracks.setdefault(track_id, _TrackUncertainty(deque(maxlen=self._W)))
        # deque created without maxlen if setdefault hit default factory
        if tr.errors.maxlen != self._W:
            tr.errors = deque(tr.errors, maxlen=self._W)
        tr.errors.append(float(prediction_error))

        n = len(tr.errors)
        if n >= 2:
            mean = sum(tr.errors) / n
            var = sum((e - mean) ** 2 for e in tr.errors) / n
        else:
            var = 0.0
        tr.sigma_base = min(self._sigma_max, var)
        tr.sigma_h = tr.sigma_base
        return tr.sigma_h

    def age(self, track_id: int, dt_since_refresh: float) -> float:
        """Grow sigma_h monotonically while the prediction is held (Eq. 13.3)."""
        tr = self._tracks.get(track_id)
        if tr is None:
            return 0.0
        grown = tr.sigma_base + self._beta * max(0.0, dt_since_refresh)
        tr.sigma_h = min(self._sigma_max, max(tr.sigma_h, grown))
        return tr.sigma_h

    def sigma_tilde(self, track_id: int) -> float:
        """Clipped normalized uncertainty for u_h (Eq. 8.3)."""
        tr = self._tracks.get(track_id)
        if tr is None:
            return 0.0
        return min(1.0, max(0.0, tr.sigma_h / self._sigma_max))

    def sigma_h(self, track_id: int) -> float:
        tr = self._tracks.get(track_id)
        return tr.sigma_h if tr else 0.0

    def drop(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)
