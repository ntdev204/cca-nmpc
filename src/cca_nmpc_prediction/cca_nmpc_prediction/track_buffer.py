#!/usr/bin/env python3
"""Per-track rolling buffers of length L (Section 3.2, Eq. 6.1).

Pure Python/NumPy, ROS-free. Matches tracks by track_id (never array index),
evicts tracks unseen beyond a timeout, and exposes which tracks are "ready"
(have a full length-L window for LSTM inference).
"""
from __future__ import annotations

from collections import deque

import numpy as np


class TrackBufferManager:
    """Maintains length-L state buffers per track_id."""

    def __init__(self, L: int, max_age_sec: float):
        if L < 1:
            raise ValueError("L must be >= 1")
        self._L = L
        self._max_age = max_age_sec
        self._buffers: dict[int, deque] = {}
        self._last_seen: dict[int, float] = {}

    def update(self, track_id: int, state: np.ndarray, t: float) -> None:
        """Append a [x, y, vx, vy] sample for track_id at time t."""
        state = np.asarray(state, float).flatten()
        if state.shape != (4,):
            raise ValueError("state must be length-4 [x, y, vx, vy]")
        buf = self._buffers.get(track_id)
        if buf is None:
            buf = deque(maxlen=self._L)
            self._buffers[track_id] = buf
        buf.append(state)
        self._last_seen[track_id] = t

    def prune(self, current_time: float) -> None:
        """Evict tracks not updated within max_age_sec."""
        stale = [
            tid for tid, ts in self._last_seen.items()
            if current_time - ts > self._max_age
        ]
        for tid in stale:
            self._buffers.pop(tid, None)
            self._last_seen.pop(tid, None)

    def is_ready(self, track_id: int) -> bool:
        """True if the track has a full length-L window."""
        buf = self._buffers.get(track_id)
        return buf is not None and len(buf) == self._L

    def ready_tracks(self) -> list[int]:
        return [tid for tid in self._buffers if self.is_ready(tid)]

    def get_window(self, track_id: int) -> np.ndarray:
        """Return the (L, 4) window for a ready track."""
        if not self.is_ready(track_id):
            raise KeyError(f"track {track_id} is not ready")
        return np.stack(list(self._buffers[track_id]))

    def active_tracks(self) -> list[int]:
        return list(self._buffers.keys())
