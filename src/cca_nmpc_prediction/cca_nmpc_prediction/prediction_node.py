#!/usr/bin/env python3
"""prediction_node: LSTM human-motion prediction (Architecture Section 3.2).

Thin ROS glue over the pure cores (track_buffer, lstm_infer, uncertainty).
Subscribes /human_states; on a timer at f_lstm_hz runs the ONNX LSTM for each
ready track and publishes /human_predictions (HumanPredictionArray) and
/human_pred_uncertainty (HumanUncertaintyArray), matched by track_id.

sigma_h is refreshed from realized-vs-predicted error at each LSTM cycle
(Eq. 6.3) and would grow between refreshes (Eq. 13.3); here the node runs
prediction every timer tick so the refresh path is the primary one.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node

from cca_nmpc_msgs.msg import (
    HumanStateArray,
    HumanPrediction,
    HumanPredictionArray,
    HumanUncertainty,
    HumanUncertaintyArray,
)

from .track_buffer import TrackBufferManager
from .lstm_infer import TensorRtLSTMPredictor, MockLSTMPredictor
from .uncertainty import UncertaintyEstimator


class PredictionNode(Node):
    def __init__(self) -> None:
        super().__init__('prediction_node')
        self._load_params()

        self._buffers = TrackBufferManager(self._L, self._max_age)
        self._uncertainty = UncertaintyEstimator(
            self._W, self._beta, self._sigma_max)
        self._predictor = self._build_predictor()
        # last one-step-ahead prediction per track, for Eq. 6.3 error
        self._last_pred1: dict[int, np.ndarray] = {}

        self._sub = self.create_subscription(
            HumanStateArray, '/human_states', self._on_states, 10)
        self._pub_pred = self.create_publisher(
            HumanPredictionArray, '/human_predictions', 10)
        self._pub_unc = self.create_publisher(
            HumanUncertaintyArray, '/human_pred_uncertainty', 10)
        self._timer = self.create_timer(1.0 / self._f_lstm, self._on_cycle)
        self.get_logger().info('PredictionNode started')

    def _load_params(self) -> None:
        d = self.declare_parameter
        # TensorRT serialized engine (built per target GPU from the ONNX export).
        d('engine_path', 'models/lstm_predictor_v1_on_lstm_dataset_v1.engine')
        d('normalization_stats_path', 'models/normalization_stats.json')
        d('L', 8); d('H', 12); d('f_lstm_hz', 8.0)
        d('uncertainty_window_W', 20)
        d('sigma_growth_rate_beta', 0.05)
        d('sigma_max', 0.5)
        d('max_track_age_sec', 1.0)
        # Mock predictor for non-GPU dev / smoke tests (skips engine load).
        d('use_mock_predictor', False)
        g = self.get_parameter
        self._engine_path = g('engine_path').value
        self._stats_path = g('normalization_stats_path').value
        self._L = int(g('L').value)
        self._H = int(g('H').value)
        self._f_lstm = float(g('f_lstm_hz').value)
        self._W = int(g('uncertainty_window_W').value)
        self._beta = float(g('sigma_growth_rate_beta').value)
        self._sigma_max = float(g('sigma_max').value)
        self._max_age = float(g('max_track_age_sec').value)
        self._use_mock = bool(g('use_mock_predictor').value)

    def _build_predictor(self):
        """TensorRT predictor with MockLSTMPredictor fallback (non-GPU dev)."""
        dt = 1.0 / self._f_lstm
        if self._use_mock:
            self.get_logger().warn('use_mock_predictor=true, using MockLSTMPredictor')
            return MockLSTMPredictor(horizon=self._H, dt=dt)
        try:
            predictor = TensorRtLSTMPredictor(
                self._engine_path, self._stats_path, horizon=self._H)
            self.get_logger().info('Using TensorRtLSTMPredictor')
            return predictor
        except (ImportError, FileNotFoundError, RuntimeError) as e:
            self.get_logger().warn(
                f'TRT predictor unavailable ({e}), using MockLSTMPredictor')
            return MockLSTMPredictor(horizon=self._H, dt=dt)

    def _on_states(self, msg: HumanStateArray) -> None:
        t = self.get_clock().now().nanoseconds / 1e9
        for h in msg.humans:
            state = np.array([h.x, h.y, h.vx, h.vy], float)
            # Eq. 6.3 error: compare last cycle's 1-step prediction to now.
            if h.track_id in self._last_pred1:
                p = self._last_pred1[h.track_id]
                err = float(np.hypot(p[0] - h.x, p[1] - h.y))
                self._uncertainty.refresh(h.track_id, err)
            self._buffers.update(h.track_id, state, t)
        self._buffers.prune(t)

    def _on_cycle(self) -> None:
        now = self.get_clock().now()
        pred_arr = HumanPredictionArray()
        pred_arr.header.stamp = now.to_msg()
        pred_arr.header.frame_id = 'map'
        unc_arr = HumanUncertaintyArray()
        unc_arr.header.stamp = now.to_msg()
        unc_arr.header.frame_id = 'map'

        for tid in self._buffers.ready_tracks():
            window = self._buffers.get_window(tid)
            pred = self._predictor.predict(window)     # (H, 4) physical
            self._last_pred1[tid] = pred[0].copy()

            hp = HumanPrediction()
            hp.track_id = tid
            hp.horizon_length = self._H
            hp.x_hat = [float(v) for v in pred[:, 0]]
            hp.y_hat = [float(v) for v in pred[:, 1]]
            hp.vx_hat = [float(v) for v in pred[:, 2]]
            hp.vy_hat = [float(v) for v in pred[:, 3]]
            hp.prediction_stamp = now.to_msg()
            pred_arr.predictions.append(hp)

            hu = HumanUncertainty()
            hu.track_id = tid
            hu.sigma_h = self._uncertainty.sigma_h(tid)
            hu.sigma_h_clipped = self._uncertainty.sigma_tilde(tid)
            unc_arr.uncertainties.append(hu)

        self._pub_pred.publish(pred_arr)
        self._pub_unc.publish(unc_arr)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(PredictionNode())
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
