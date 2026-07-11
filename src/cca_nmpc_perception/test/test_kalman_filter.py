#!/usr/bin/env python3
"""Tests for kalman_filter.py (HP-06)."""
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cca_nmpc_perception'))

from cca_nmpc_perception.kalman_filter import (
    KalmanTrack,
    make_kalman_matrices,
    predict,
    update,
)


class TestMakeKalmanMatrices:

    def test_H_shape(self):
        Q, R, H = make_kalman_matrices(0.1, 0.15)
        assert H.shape == (2, 4)

    def test_H_extracts_position(self):
        _, _, H = make_kalman_matrices(0.1, 0.15)
        state = np.array([3.0, 4.0, 1.0, 2.0])
        meas = H @ state
        np.testing.assert_array_almost_equal(meas, [3.0, 4.0])

    def test_Q_shape_and_scale(self):
        Q, _, _ = make_kalman_matrices(0.2, 0.15)
        assert Q.shape == (4, 4)
        assert abs(Q[0, 0] - 0.04) < 1e-9

    def test_R_shape_and_scale(self):
        _, R, _ = make_kalman_matrices(0.1, 0.3)
        assert R.shape == (2, 2)
        assert abs(R[0, 0] - 0.09) < 1e-9


class TestPredict:

    def test_position_advances_with_velocity(self):
        Q, _, _ = make_kalman_matrices(0.1, 0.15)
        track = KalmanTrack(
            state=np.array([1.0, 2.0, 0.5, -0.5]),
            covariance=np.eye(4),
            last_update_time=0.0
        )
        predicted = predict(track, dt=1.0, Q=Q, target_time=1.0)
        # x = 1.0 + 0.5*1.0 = 1.5; y = 2.0 + (-0.5)*1.0 = 1.5
        assert abs(predicted.state[0] - 1.5) < 1e-9
        assert abs(predicted.state[1] - 1.5) < 1e-9

    def test_velocity_unchanged_in_constant_velocity_model(self):
        Q, _, _ = make_kalman_matrices(0.0, 0.15)  # zero process noise for clean test
        track = KalmanTrack(
            state=np.array([0.0, 0.0, 2.0, -1.0]),
            covariance=np.eye(4),
            last_update_time=0.0
        )
        predicted = predict(track, dt=0.5, Q=Q, target_time=0.5)
        assert abs(predicted.state[2] - 2.0) < 1e-9
        assert abs(predicted.state[3] - (-1.0)) < 1e-9

    def test_predict_returns_new_track_immutable(self):
        Q, _, _ = make_kalman_matrices(0.1, 0.15)
        original_state = np.array([1.0, 2.0, 0.1, 0.1])
        track = KalmanTrack(state=original_state.copy(), covariance=np.eye(4), last_update_time=0.0)
        predicted = predict(track, dt=0.1, Q=Q, target_time=0.1)
        # Original unchanged
        np.testing.assert_array_equal(track.state, original_state)
        assert predicted is not track

    def test_covariance_grows_with_nonzero_Q(self):
        Q, _, _ = make_kalman_matrices(0.5, 0.15)
        track = KalmanTrack(state=np.zeros(4), covariance=np.eye(4), last_update_time=0.0)
        predicted = predict(track, dt=1.0, Q=Q, target_time=1.0)
        # All diagonal elements should increase
        for i in range(4):
            assert predicted.covariance[i, i] > track.covariance[i, i]

    def test_predict_advances_last_update_time_to_target_time(self):
        Q, _, _ = make_kalman_matrices(0.1, 0.15)
        track = KalmanTrack(state=np.zeros(4), covariance=np.eye(4), last_update_time=5.0)
        predicted = predict(track, dt=1.0, Q=Q, target_time=6.0)
        assert predicted.last_update_time == 6.0


class TestUpdate:

    def test_update_moves_state_toward_measurement(self):
        Q, R, H = make_kalman_matrices(0.1, 0.15)
        track = KalmanTrack(
            state=np.array([0.0, 0.0, 0.0, 0.0]),
            covariance=np.eye(4) * 10,  # large initial uncertainty
            last_update_time=0.0
        )
        measurement = np.array([3.0, 4.0])
        updated = update(track, measurement, H, R, timestamp=1.0)

        # State x,y should move toward measurement
        assert updated.state[0] > 0.0
        assert updated.state[1] > 0.0
        assert updated.state[0] < 3.0  # not overshoot
        assert updated.state[1] < 4.0

    def test_update_sets_timestamp(self):
        Q, R, H = make_kalman_matrices(0.1, 0.15)
        track = KalmanTrack(state=np.zeros(4), covariance=np.eye(4), last_update_time=0.0)
        updated = update(track, np.array([1.0, 1.0]), H, R, timestamp=42.5)
        assert updated.last_update_time == 42.5

    def test_update_returns_new_track_immutable(self):
        Q, R, H = make_kalman_matrices(0.1, 0.15)
        original_state = np.array([0.0, 0.0, 0.0, 0.0])
        track = KalmanTrack(state=original_state.copy(), covariance=np.eye(4), last_update_time=0.0)
        updated = update(track, np.array([1.0, 1.0]), H, R, timestamp=1.0)
        np.testing.assert_array_equal(track.state, original_state)
        assert updated is not track

    def test_covariance_shrinks_after_update(self):
        Q, R, H = make_kalman_matrices(0.1, 0.15)
        track = KalmanTrack(state=np.zeros(4), covariance=np.eye(4) * 5, last_update_time=0.0)
        updated = update(track, np.array([0.0, 0.0]), H, R, timestamp=1.0)
        # Position uncertainty (top-left 2x2) should shrink
        assert updated.covariance[0, 0] < track.covariance[0, 0]
        assert updated.covariance[1, 1] < track.covariance[1, 1]

    def test_perfect_measurement_at_state(self):
        """Update with measurement exactly at predicted position: state should be ~unchanged."""
        Q, R, H = make_kalman_matrices(0.01, 0.01)  # tight noises
        track = KalmanTrack(
            state=np.array([5.0, 3.0, 0.0, 0.0]),
            covariance=np.eye(4) * 0.01,
            last_update_time=0.0
        )
        updated = update(track, np.array([5.0, 3.0]), H, R, timestamp=1.0)
        assert abs(updated.state[0] - 5.0) < 0.01
        assert abs(updated.state[1] - 3.0) < 0.01
