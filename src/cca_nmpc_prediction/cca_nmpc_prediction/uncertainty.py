#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class _TrackUncertainty:
    errors: deque = field(default_factory=lambda: deque())
    sigma_base: float = 0.0
    sigma_h: float = 0.0


class UncertaintyEstimator:

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
        tr = self._tracks.setdefault(track_id, _TrackUncertainty(deque(maxlen=self._W)))
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
        tr = self._tracks.get(track_id)
        if tr is None:
            return self._sigma_max
        grown = tr.sigma_base + self._beta * max(0.0, dt_since_refresh)
        tr.sigma_h = min(self._sigma_max, max(tr.sigma_h, grown))
        return tr.sigma_h

    def sigma_tilde(self, track_id: int) -> float:
        tr = self._tracks.get(track_id)
        if tr is None:
            return 1.0
        return min(1.0, max(0.0, tr.sigma_h / self._sigma_max))

    def sigma_h(self, track_id: int) -> float:
        tr = self._tracks.get(track_id)
        return tr.sigma_h if tr else self._sigma_max

    def drop(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)
