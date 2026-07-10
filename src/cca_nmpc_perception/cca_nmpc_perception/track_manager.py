#!/usr/bin/env python3
"""Multi-target track manager with nearest-neighbor association."""
import numpy as np
from dataclasses import dataclass
from typing import Sequence
from .kalman_filter import KalmanTrack, predict, update, make_kalman_matrices


@dataclass
class Track:
    """Track with persistent ID and Kalman state."""
    track_id: int
    kalman: KalmanTrack
    confidence: float = 1.0

    @property
    def position(self) -> tuple[float, float]:
        """Current (x, y) estimate."""
        return float(self.kalman.state[0]), float(self.kalman.state[1])

    @property
    def velocity(self) -> tuple[float, float]:
        """Current (vx, vy) estimate."""
        return float(self.kalman.state[2]), float(self.kalman.state[3])


class TrackManager:
    """Multi-object tracker with nearest-neighbor association."""

    def __init__(
        self,
        process_noise_std: float,
        measurement_noise_std: float,
        association_distance_gate: float,
        max_track_age_sec: float
    ):
        self._process_noise_std = process_noise_std
        self._measurement_noise_std = measurement_noise_std
        self._association_distance_gate = association_distance_gate
        self._max_track_age_sec = max_track_age_sec

        self._Q, self._R, self._H = make_kalman_matrices(
            process_noise_std, measurement_noise_std
        )

        self._tracks: list[Track] = []
        self._next_track_id: int = 0

    def update(
        self,
        measurements: Sequence[tuple[float, ...]],
        current_time: float
    ) -> list[Track]:
        """Update tracks with new measurements.

        Args:
            measurements: List of ``(x, y)`` or ``(x, y, confidence)`` tuples in
                the map frame. A missing confidence defaults to 1.0.
            current_time: Current timestamp in seconds

        Returns:
            List of all active tracks after update
        """
        positions = [(float(m[0]), float(m[1])) for m in measurements]
        confidences = [
            float(m[2]) if len(m) > 2 else 1.0 for m in measurements
        ]

        # Predict all tracks to current time
        for track in self._tracks:
            dt = current_time - track.kalman.last_update_time
            if dt > 0:
                track.kalman = predict(track.kalman, dt, self._Q, current_time)

        # Associate measurements to tracks
        if positions and self._tracks:
            associations = self._nearest_neighbor_associate(positions)
        else:
            associations = {}

        # Update associated tracks
        for meas_idx, track_idx in associations.items():
            track = self._tracks[track_idx]
            measurement = np.array(positions[meas_idx], dtype=float)
            track.kalman = update(
                track.kalman, measurement, self._H, self._R, current_time
            )
            track.confidence = confidences[meas_idx]

        # Create new tracks for unassociated measurements
        associated_meas = set(associations.keys())
        for meas_idx, (x, y) in enumerate(positions):
            if meas_idx not in associated_meas:
                new_track = self._create_track(
                    x, y, current_time, confidences[meas_idx]
                )
                self._tracks.append(new_track)

        # Prune stale tracks
        self._prune_stale_tracks(current_time)

        return self._tracks.copy()

    def _nearest_neighbor_associate(
        self, measurements: Sequence[tuple[float, float]]
    ) -> dict[int, int]:
        """Nearest-neighbor association with distance gating.

        Returns:
            Dict mapping measurement_index -> track_index
        """
        associations: dict[int, int] = {}
        used_tracks = set()

        # Build distance matrix
        distances = np.zeros((len(measurements), len(self._tracks)))
        for i, (mx, my) in enumerate(measurements):
            for j, track in enumerate(self._tracks):
                tx, ty = track.position
                distances[i, j] = np.sqrt((mx - tx)**2 + (my - ty)**2)

        # Greedy nearest-neighbor assignment
        flat_indices = np.argsort(distances.ravel())
        for flat_idx in flat_indices:
            meas_idx, track_idx = np.unravel_index(flat_idx, distances.shape)
            if meas_idx in associations or track_idx in used_tracks:
                continue
            if distances[meas_idx, track_idx] > self._association_distance_gate:
                continue
            associations[int(meas_idx)] = int(track_idx)
            used_tracks.add(int(track_idx))

        return associations

    def _create_track(
        self, x: float, y: float, timestamp: float, confidence: float = 1.0
    ) -> Track:
        """Initialize new track at measurement position with zero velocity."""
        kalman = KalmanTrack(
            state=np.array([x, y, 0.0, 0.0], dtype=float),
            covariance=np.eye(4),
            last_update_time=timestamp
        )
        track = Track(
            track_id=self._next_track_id, kalman=kalman, confidence=confidence
        )
        self._next_track_id += 1
        return track

    def _prune_stale_tracks(self, current_time: float) -> None:
        """Remove tracks not updated within max_track_age_sec."""
        self._tracks = [
            track for track in self._tracks
            if (current_time - track.kalman.last_update_time) <= self._max_track_age_sec
        ]

    def get_tracks(self) -> list[Track]:
        """Return current active tracks."""
        return self._tracks.copy()
