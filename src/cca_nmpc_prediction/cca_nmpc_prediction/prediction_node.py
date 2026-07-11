#!/usr/bin/env python3
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
from .lstm_infer import TensorRtLSTMPredictor
from .uncertainty import UncertaintyEstimator


class PredictionNode(Node):
    def __init__(self) -> None:
        super().__init__('prediction_node')
        self._load_params()

        self._buffers = TrackBufferManager(self._L, self._max_age)
        self._uncertainty = UncertaintyEstimator(
            self._W, self._beta, self._sigma_max)
        self._predictor = self._build_predictor()
        self._last_pred1: dict[int, np.ndarray] = {}
        self._last_refresh_time: dict[int, float] = {}
        self._refreshed_this_cycle: set[int] = set()

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
        d('engine_path', 'models/lstm_predictor_v1_on_lstm_dataset_v1.engine')
        d('normalization_stats_path', 'models/normalization_stats.json')
        d('L', 8)
        d('H', 12)
        d('f_lstm_hz', 8.0)
        d('uncertainty_window_W', 20)
        d('sigma_growth_rate_beta', 0.05)
        d('sigma_max', 0.5)
        d('max_track_age_sec', 1.0)
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

    def _build_predictor(self):
        predictor = TensorRtLSTMPredictor(
            self._engine_path, self._stats_path, horizon=self._H)
        self.get_logger().info('Using TensorRtLSTMPredictor')
        return predictor

    def _on_states(self, msg: HumanStateArray) -> None:
        t = self.get_clock().now().nanoseconds / 1e9
        for h in msg.humans:
            state = np.array([h.x, h.y, h.vx, h.vy], float)
            tid = h.track_id
            if tid in self._last_pred1 and tid not in self._refreshed_this_cycle:
                p = self._last_pred1[tid]
                err = float(np.hypot(p[0] - h.x, p[1] - h.y))
                self._uncertainty.refresh(tid, err)
                self._last_refresh_time[tid] = t
                self._refreshed_this_cycle.add(tid)
            self._buffers.update(tid, state, t)
        self._buffers.prune(t)

    def _on_cycle(self) -> None:
        now = self.get_clock().now()
        now_sec = now.nanoseconds / 1e9
        pred_arr = HumanPredictionArray()
        pred_arr.header.stamp = now.to_msg()
        pred_arr.header.frame_id = 'map'
        unc_arr = HumanUncertaintyArray()
        unc_arr.header.stamp = now.to_msg()
        unc_arr.header.frame_id = 'map'

        for tid in self._buffers.ready_tracks():
            window = self._buffers.get_window(tid)
            pred = self._predictor.predict(window)
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

            t_ref = self._last_refresh_time.get(tid)
            if t_ref is not None:
                self._uncertainty.age(tid, now_sec - t_ref)

            hu = HumanUncertainty()
            hu.track_id = tid
            hu.sigma_h = self._uncertainty.sigma_h(tid)
            hu.sigma_h_clipped = self._uncertainty.sigma_tilde(tid)
            unc_arr.uncertainties.append(hu)

        self._refreshed_this_cycle.clear()

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
