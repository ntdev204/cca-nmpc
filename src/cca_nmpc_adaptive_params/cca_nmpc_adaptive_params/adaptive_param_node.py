#!/usr/bin/env python3
from __future__ import annotations

import rclpy
from rclpy.node import Node

from cca_nmpc_msgs.msg import (
    ContextIndexArray,
    AdaptiveParams,
    HumanSafetyDistance,
)

from .param_map import (
    AdaptiveConfig, d_safe, saturated_velocity_limits, q_diag,
)


class AdaptiveParamNode(Node):
    def __init__(self) -> None:
        super().__init__('adaptive_param_node')
        self._cfg = self._load_config()
        self._sub = self.create_subscription(
            ContextIndexArray, '/context_index', self._on_context, 10)
        self._pub = self.create_publisher(AdaptiveParams, '/adaptive_params', 10)
        self.get_logger().info('AdaptiveParamNode started')

    def _load_config(self) -> AdaptiveConfig:
        d = self.declare_parameter
        d('d_safe0', 0.6)
        d('k_d', 0.8)
        d('v_x0', 1.0)
        d('v_y0', 0.8)
        d('omega_0', 1.2)
        d('k_x', 0.6)
        d('k_y', 0.5)
        d('k_omega', 0.7)
        d('v_x_min', 0.08)
        d('v_y_min', 0.06)
        d('omega_min', 0.1)
        d('Q0_diag', [5.0, 5.0, 2.0])
        d('Qh_diag', [8.0, 8.0, 3.0])
        g = self.get_parameter
        return AdaptiveConfig(
            d_safe0=g('d_safe0').value, k_d=g('k_d').value,
            v_x0=g('v_x0').value, v_y0=g('v_y0').value, omega_0=g('omega_0').value,
            k_x=g('k_x').value, k_y=g('k_y').value, k_omega=g('k_omega').value,
            v_x_min=g('v_x_min').value, v_y_min=g('v_y_min').value,
            omega_min=g('omega_min').value,
            q0_diag=tuple(g('Q0_diag').value), qh_diag=tuple(g('Qh_diag').value),
        )

    def _on_context(self, msg: ContextIndexArray) -> None:
        phi_used = msg.phi_aggregate_used
        vx, vy, omega = saturated_velocity_limits(phi_used, self._cfg)

        out = AdaptiveParams()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'map'
        out.d_safe_aggregate = d_safe(phi_used, self._cfg)
        out.vx_max = vx
        out.vy_max = vy
        out.omega_max = omega
        out.q_diag = list(q_diag(phi_used, self._cfg))
        out.phi_aggregate_used = phi_used

        for ci in msg.contexts:
            hsd = HumanSafetyDistance()
            hsd.track_id = ci.track_id
            hsd.d_safe = d_safe(ci.phi_j_used, self._cfg)
            out.d_safe_per_human.append(hsd)

        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(AdaptiveParamNode())
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
