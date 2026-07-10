#!/usr/bin/env python3
"""context_node: continuous context estimation (Architecture Section 3.3).

Thin ROS glue over the pure modules (relative_motion, context_score, smoothing,
aggregate). Subscribes /human_states, /human_predictions, /human_pred_uncertainty,
/odom; publishes /context_index (ContextIndexArray).

Geometric terms (Eqs. 7.1-7.5) are recomputed every cycle from the LATEST
Kalman-filtered current state; only the predicted future trajectory is held
between LSTM refreshes (Section 13.1). On perception/prediction dropout the used
aggregate falls back to phi=1 (Architecture Section 4).
"""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

from cca_nmpc_msgs.msg import (
    HumanStateArray,
    HumanPredictionArray,
    HumanUncertaintyArray,
    ContextIndex,
    ContextIndexArray,
)

from .relative_motion import compute_relative_motion
from .context_score import ContextWeights, context_score, sigmoid
from .smoothing import PhiSmoother
from .aggregate import aggregate_context


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (theta_r) from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class ContextNode(Node):
    def __init__(self) -> None:
        super().__init__('context_node')
        self._declare_and_load_params()

        self._smoother = PhiSmoother(self._alpha, self._t_dwell)
        self._weights = ContextWeights(**self._weight_dict)

        # latest inputs
        self._latest_states = None       # HumanStateArray
        self._latest_uncert: dict[int, float] = {}   # track_id -> sigma_h_clipped
        self._robot = (0.0, 0.0, 0.0, 0.0, 0.0)  # x,y,theta,vx,vy
        self._last_states_time = None
        self._last_pred_time = None

        self._sub_states = self.create_subscription(
            HumanStateArray, '/human_states', self._on_states, 10)
        self._sub_pred = self.create_subscription(
            HumanPredictionArray, '/human_predictions', self._on_predictions, 10)
        self._sub_unc = self.create_subscription(
            HumanUncertaintyArray, '/human_pred_uncertainty', self._on_uncertainty, 10)
        self._sub_odom = self.create_subscription(
            Odometry, '/odom', self._on_odom, 10)

        self._pub = self.create_publisher(ContextIndexArray, '/context_index', 10)
        self._timer = self.create_timer(1.0 / self._rate_hz, self._on_cycle)
        self.get_logger().info('ContextNode started')

    def _declare_and_load_params(self) -> None:
        self.declare_parameter('d0', 3.0)
        self.declare_parameter('v_max_ref', 1.5)
        self.declare_parameter('epsilon', 1.0e-3)
        self.declare_parameter('sigma_max', 0.5)
        self.declare_parameter('weights.w_d', 1.2)
        self.declare_parameter('weights.w_v', 0.8)
        self.declare_parameter('weights.w_theta', 0.6)
        self.declare_parameter('weights.w_u', 1.0)
        self.declare_parameter('weights.b', -0.5)
        self.declare_parameter('ema_filter.alpha', 0.7)
        self.declare_parameter('dwell_time.T_dwell_cycles', 4)
        self.declare_parameter('fallback_phi_on_dropout', 1.0)
        self.declare_parameter('input_timeout_sec', 0.5)
        self.declare_parameter('rate_hz', 20.0)

        gp = self.get_parameter
        self._d0 = gp('d0').value
        self._v_max_ref = gp('v_max_ref').value
        self._epsilon = gp('epsilon').value
        self._sigma_max = gp('sigma_max').value
        self._weight_dict = {
            'w_d': gp('weights.w_d').value,
            'w_v': gp('weights.w_v').value,
            'w_theta': gp('weights.w_theta').value,
            'w_u': gp('weights.w_u').value,
            'b': gp('weights.b').value,
        }
        self._alpha = gp('ema_filter.alpha').value
        self._t_dwell = int(gp('dwell_time.T_dwell_cycles').value)
        self._fallback_phi = gp('fallback_phi_on_dropout').value
        self._timeout = gp('input_timeout_sec').value
        self._rate_hz = gp('rate_hz').value

    # -------------------------------------------------------------- callbacks
    def _on_states(self, msg: HumanStateArray) -> None:
        self._latest_states = msg
        self._last_states_time = self.get_clock().now()

    def _on_predictions(self, msg: HumanPredictionArray) -> None:
        # Predicted future trajectory is held between LSTM refreshes; the node
        # keeps the freshness stamp to detect dropout, geometric terms are
        # recomputed from current states each cycle (Section 13.1).
        self._last_pred_time = self.get_clock().now()

    def _on_uncertainty(self, msg: HumanUncertaintyArray) -> None:
        self._latest_uncert = {
            u.track_id: u.sigma_h_clipped for u in msg.uncertainties
        }

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        theta = _yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self._robot = (p.x, p.y, theta, v.x, v.y)

    def _is_dropout(self) -> bool:
        if self._last_states_time is None:
            return True
        age = (self.get_clock().now() - self._last_states_time).nanoseconds / 1e9
        return age > self._timeout

    # ------------------------------------------------------------------ cycle
    def _on_cycle(self) -> None:
        out = ContextIndexArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'map'

        dropout = self._is_dropout()
        humans = [] if (dropout or self._latest_states is None) \
            else self._latest_states.humans

        phi_raw, phi_used, d_list, present = [], [], [], []
        rx, ry, rtheta, rvx, rvy = self._robot

        for h in humans:
            rm = compute_relative_motion(
                rx, ry, rtheta, rvx, rvy,
                h.x, h.y, h.vx, h.vy, epsilon=self._epsilon,
            )
            # Missing uncertainty → fully uncertain (fail-safe; Eq. 8.3, sigma_tilde=1.0)
            sigma_tilde = self._latest_uncert.get(h.track_id, 1.0)
            z = context_score(
                rm.d_h, rm.v_h_speed, rm.cos_dtheta,
                h.confidence, sigma_tilde, self._weights,
                self._d0, self._v_max_ref,
            )
            phi_j = sigmoid(z)
            filt, used = self._smoother.update(h.track_id, phi_j)

            ci = ContextIndex()
            ci.track_id = h.track_id
            ci.z = z
            ci.phi_j = phi_j
            ci.phi_j_filtered = filt
            ci.phi_j_used = used
            ci.d_j = rm.d_h
            out.contexts.append(ci)

            phi_raw.append(phi_j)
            phi_used.append(used)
            d_list.append(rm.d_h)
            present.append(h.track_id)

        # forget tracks no longer present
        for tid in self._smoother.active_tracks():
            if tid not in present:
                self._smoother.drop(tid)

        agg = aggregate_context(
            phi_raw, phi_used, d_list,
            dropout=dropout, fallback_phi=self._fallback_phi,
        )
        out.phi_aggregate = agg.phi_aggregate
        out.phi_aggregate_used = agg.phi_aggregate_used
        out.d_h_aggregate = (
            0.0 if math.isinf(agg.d_h_aggregate) else agg.d_h_aggregate
        )
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ContextNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
