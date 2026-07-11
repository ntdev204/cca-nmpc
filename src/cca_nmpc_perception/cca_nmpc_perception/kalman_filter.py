#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass, field


@dataclass
class KalmanTrack:
    state: np.ndarray = field(default_factory=lambda: np.zeros(4))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4))
    last_update_time: float = 0.0
    last_measurement_time: float | None = None

    def __post_init__(self) -> None:
        if self.last_measurement_time is None:
            self.last_measurement_time = self.last_update_time


def make_kalman_matrices(
    process_noise_std: float,
    measurement_noise_std: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)
    Q = np.eye(4) * (process_noise_std ** 2)
    R = np.eye(2) * (measurement_noise_std ** 2)
    return Q, R, H


def predict(
    track: KalmanTrack,
    dt: float,
    Q: np.ndarray,
    target_time: float
) -> KalmanTrack:
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]], dtype=float)

    predicted_state = F @ track.state
    predicted_cov = F @ track.covariance @ F.T + Q

    return KalmanTrack(
        state=predicted_state,
        covariance=predicted_cov,
        last_update_time=target_time,
        last_measurement_time=track.last_measurement_time,
    )


def update(
    track: KalmanTrack,
    measurement: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    timestamp: float
) -> KalmanTrack:
    y_innov = measurement - H @ track.state
    S = H @ track.covariance @ H.T + R
    K = track.covariance @ H.T @ np.linalg.inv(S)

    updated_state = track.state + K @ y_innov
    I_KH = np.eye(4) - K @ H
    updated_cov = I_KH @ track.covariance

    return KalmanTrack(
        state=updated_state,
        covariance=updated_cov,
        last_update_time=timestamp,
        last_measurement_time=timestamp,
    )
