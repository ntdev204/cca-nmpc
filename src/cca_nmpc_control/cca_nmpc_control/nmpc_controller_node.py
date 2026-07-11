#!/usr/bin/env python3
"""nmpc_controller_node: CCA-NMPC control loop (Architecture Section 3.5).

Thin ROS glue over the (ROS-free) SolverInterface backend. Subscribes
/adaptive_params, /human_predictions, /odom, /goal_pose; publishes /cmd_vel and
/nmpc_diagnostics. Runs the solve loop with a wall-clock deadline and the
fallback hierarchy (Solver Design Section 9), calling reset() on any warm-start
invalidation trigger (Section 5.2). The node depends ONLY on SolverInterface —
no backend branching here.
"""
from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from nav2_msgs.msg import Costmap

from cca_nmpc_msgs.msg import (
    AdaptiveParams,
    HumanPredictionArray,
    HumanStateArray,
    ContextIndexArray,
    NmpcDiagnostics,
    SlackValue,
)

from .nmpc_solver import CasadiSolver
from .nmpc_solver.interface import AdaptiveParamsInput, HumanSafetyDistance
from .fallback import FallbackController
from .timing import build_cycle_timing
from .invalidation import InvalidationDetector
from .diagnostics import build_diagnostics


def _yaw(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _resample_reference(path: Path, count: int) -> dict[str, np.ndarray]:
    if not path.poses:
        raise ValueError("reference path is empty")
    points = np.array([[p.pose.position.x, p.pose.position.y] for p in path.poses])
    headings = np.unwrap(np.array([_yaw(p.pose.orientation) for p in path.poses]))
    if len(points) == 1:
        points = np.vstack([points, points])
        headings = np.repeat(headings, 2)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    if arc[-1] <= 1e-9:
        target = np.zeros(count)
    else:
        target = np.linspace(0.0, arc[-1], count)
    return {
        "x": np.interp(target, arc, points[:, 0]),
        "y": np.interp(target, arc, points[:, 1]),
        "theta": np.interp(target, arc, headings),
    }


def _costmap_obstacles(msg: Costmap, robot_xy: tuple[float, float], limit: int) -> np.ndarray:
    data = np.asarray(msg.data, dtype=np.uint8)
    occupied = np.flatnonzero(data >= 253)
    if occupied.size == 0:
        return np.empty((0, 2))
    width = int(msg.metadata.size_x)
    resolution = float(msg.metadata.resolution)
    ox = msg.metadata.origin.position.x
    oy = msg.metadata.origin.position.y
    points = np.column_stack((
        ox + (occupied % width + 0.5) * resolution,
        oy + (occupied // width + 0.5) * resolution,
    ))
    distances = np.sum((points - np.asarray(robot_xy)) ** 2, axis=1)
    return points[np.argsort(distances)[:limit]]


class NmpcControllerNode(Node):
    def __init__(self) -> None:
        super().__init__('nmpc_controller_node')
        self._load_params()
        self._solver = self._build_solver()
        self._fallback = FallbackController(
            self._timeout_hold_cycles, self._dt, self._safe_stop_decel)
        self._invalidation = InvalidationDetector(
            self._odom_jump_pos, self._odom_jump_yaw)

        self._latest_params: AdaptiveParams | None = None
        self._latest_pred: HumanPredictionArray | None = None
        self._latest_humans: dict[int, tuple[float, float]] = {}  # track_id -> (x, y)
        self._phi_by_track: dict[int, float] = {}   # per-human phi_j_used
        self._pose = (0.0, 0.0, 0.0)
        self._goal = None
        self._reference_path: Path | None = None
        self._costmap: Costmap | None = None
        self._received_at: dict[str, float] = {}

        self.create_subscription(AdaptiveParams, '/adaptive_params',
                                 self._on_params, 10)
        self.create_subscription(HumanPredictionArray, '/human_predictions',
                                 self._on_pred, 10)
        self.create_subscription(HumanStateArray, '/human_states',
                                 self._on_human_states, 10)
        self.create_subscription(ContextIndexArray, '/context_index',
                                 self._on_context, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self._on_goal, 10)
        self.create_subscription(Path, '/reference_path', self._on_reference, 10)
        self.create_subscription(Costmap, '/local_costmap/costmap', self._on_costmap, 10)
        self._pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self._pub_diag = self.create_publisher(
            NmpcDiagnostics, '/nmpc_diagnostics', 10)
        self._timer = self.create_timer(self._dt, self._on_cycle)
        self.get_logger().info('NmpcControllerNode started')

    def _load_params(self) -> None:
        d = self.declare_parameter
        d('solver_backend', 'casadi')
        d('horizon_N', 20)
        d('dt', 0.05)
        d('f_lstm_hz', 8.0)
        d('R_diag', [0.1, 0.1, 0.05])
        d('Rd_diag', [0.05, 0.05, 0.02])
        d('P_diag', [10.0, 10.0, 4.0])
        d('w_h', 3.0)
        d('w_obstacle', 2.0)
        d('P_diag_terminal', 0.0)
        d('w_slack', 50.0)
        d('C_collision', 0.9)
        d('d0', 3.0)
        d('max_humans_in_solver', 6)
        d('timeout_hold_cycles', 3)
        d('odom_jump_pos_thresh_m', 0.30)
        d('odom_jump_yaw_thresh_rad', 0.35)
        d('safe_stop_decel_limit', 1.0)
        d('input_timeout_sec', 0.5)
        d('max_obstacle_samples', 64)
        d('obstacle_sigma', 0.35)
        d('solver_max_cpu_time_sec', 0.045)
        d('strict_runtime_mode', True)
        d('require_costmap', True)
        d('require_reference_path', True)
        d('allow_empty_humans', False)
        g = self.get_parameter
        self._backend = g('solver_backend').value
        self._N = int(g('horizon_N').value)
        self._dt = float(g('dt').value)
        self._solver_params = {
            'horizon_N': self._N, 'dt': self._dt,
            'f_lstm_hz': float(g('f_lstm_hz').value),
            'max_humans_in_solver': int(g('max_humans_in_solver').value),
            'R_diag': list(g('R_diag').value), 'Rd_diag': list(g('Rd_diag').value),
            'P_diag': list(g('P_diag').value), 'w_h': float(g('w_h').value),
            'w_slack': float(g('w_slack').value), 'd0': float(g('d0').value),
            'w_obstacle': float(g('w_obstacle').value),
            'max_obstacle_samples': int(g('max_obstacle_samples').value),
            'obstacle_sigma': float(g('obstacle_sigma').value),
            'solver_max_cpu_time_sec': float(g('solver_max_cpu_time_sec').value),
        }
        self._max_humans = int(g('max_humans_in_solver').value)
        self._timeout_hold_cycles = int(g('timeout_hold_cycles').value)
        self._odom_jump_pos = float(g('odom_jump_pos_thresh_m').value)
        self._odom_jump_yaw = float(g('odom_jump_yaw_thresh_rad').value)
        self._safe_stop_decel = float(g('safe_stop_decel_limit').value)
        self._input_timeout = float(g('input_timeout_sec').value)
        self._max_obstacles = int(g('max_obstacle_samples').value)
        self._strict_runtime_mode = bool(g('strict_runtime_mode').value)
        self._require_costmap = bool(g('require_costmap').value)
        self._require_reference_path = bool(g('require_reference_path').value)
        self._allow_empty_humans = bool(g('allow_empty_humans').value)

    def _build_solver(self):
        """Construct the backend from YAML — node stays backend-agnostic."""
        if self._backend == 'acados':
            from .nmpc_solver.acados_solver import AcadosSolver
            solver = AcadosSolver()
        else:
            solver = CasadiSolver()
        solver.initialize(self._solver_params)
        return solver

    # -------------------------------------------------------------- callbacks
    def _on_params(self, msg: AdaptiveParams) -> None:
        self._latest_params = msg
        self._mark_received('params')

    def _on_pred(self, msg: HumanPredictionArray) -> None:
        self._latest_pred = msg
        self._mark_received('predictions')

    def _on_human_states(self, msg: HumanStateArray) -> None:
        self._latest_humans = {
            h.track_id: (h.x, h.y) for h in msg.humans
        }
        self._mark_received('humans')

    def _on_context(self, msg: ContextIndexArray) -> None:
        # Per-human gated phi_j feeds J_human weighting in the solver. Sourced
        # from /context_index (the correct origin) — plan 08's subscription list
        # omitted it; added here for J_human consistency (docs/plan gap).
        self._phi_by_track = {c.track_id: c.phi_j_used for c in msg.contexts}
        self._mark_received('context')

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self._pose = (p.x, p.y, _yaw(msg.pose.pose.orientation))
        self._mark_received('odom')

    def _on_goal(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self._goal = (p.x, p.y, _yaw(msg.pose.orientation))

    def _on_reference(self, msg: Path) -> None:
        if msg.header.frame_id and msg.header.frame_id != 'map':
            self.get_logger().error('reference_path must be in map frame')
            return
        self._reference_path = msg
        self._mark_received('reference')

    def _on_costmap(self, msg: Costmap) -> None:
        if msg.header.frame_id and msg.header.frame_id != 'map':
            self.get_logger().error('costmap must be in map frame')
            return
        self._costmap = msg
        self._mark_received('costmap')

    def _mark_received(self, name: str) -> None:
        self._received_at[name] = time.monotonic()

    def _inputs_fresh(self) -> bool:
        """Check required input freshness.

        strict_runtime_mode=True (default): all topics required (full stack).
        strict_runtime_mode=False: base topics always required; costmap /
        reference_path / human-related topics gated by their require_* flags.
        allow_empty_humans=True: drop predictions/humans/context freshness so
        empty-scene / partial pipelines can still solve.
        """
        now = time.monotonic()
        required: list[str] = ['params', 'odom']
        if self._strict_runtime_mode:
            required.extend(
                ['predictions', 'humans', 'context', 'reference', 'costmap']
            )
        else:
            if not self._allow_empty_humans:
                required.extend(['predictions', 'humans', 'context'])
            if self._require_reference_path:
                required.append('reference')
            if self._require_costmap:
                required.append('costmap')
        return all(
            name in self._received_at
            and now - self._received_at[name] <= self._input_timeout
            for name in required
        )

    # ------------------------------------------------------------------ cycle
    def _on_cycle(self) -> None:
        # Warm-start invalidation (Section 5.2) -> reset() before solving.
        events = self._invalidation.check(self._pose, self._goal)
        if events.any:
            self._solver.reset()
            self.get_logger().warn('warm-start invalidated -> solver.reset()')

        if self._latest_params is None or self._goal is None or not self._inputs_fresh():
            self._publish_stop()          # no inputs yet -> safe default
            return

        t_param0 = time.perf_counter()
        self._push_inputs()
        t_param = (time.perf_counter() - t_param0) * 1e3

        x0 = np.array(self._pose, float)
        deadline = self._dt
        t_solve0 = time.perf_counter()
        try:
            result = self._solver.solve(x0)
            solve_ms = (time.perf_counter() - t_solve0) * 1e3
            timed_out = solve_ms > deadline * 1e3
            ok = result.success and not timed_out
        except Exception as exc:                     # noqa: BLE001
            self.get_logger().error(f'solve() raised: {exc}')
            result, solve_ms, timed_out, ok = None, 0.0, False, False

        if ok:
            decision = self._fallback.on_success(result.u0)
        else:
            decision = self._fallback.on_failure(timed_out=timed_out)
            if decision.needs_reset:
                self._solver.reset()

        t_pub0 = time.perf_counter()
        self._publish_cmd(decision.u)
        self._publish_diag(result, decision, solve_ms)
        t_pub = (time.perf_counter() - t_pub0) * 1e3

        timing = build_cycle_timing(t_param, solve_ms, t_pub, self._dt)
        if timing.over_budget:
            self.get_logger().warn(
                f'cycle over budget: {timing.total_cycle_ms:.1f}ms > '
                f'{self._dt*1e3:.1f}ms')

    def _push_inputs(self) -> None:
        if self._reference_path is not None:
            self._solver.set_reference(
                _resample_reference(self._reference_path, self._N + 1)
            )
        else:
            # No reference path: hold current pose as flat reference (smoke-test).
            x, y, th = self._pose
            n = self._N + 1
            self._solver.set_reference({
                'x': np.full(n, x, float),
                'y': np.full(n, y, float),
                'theta': np.full(n, th, float),
            })
        if self._costmap is not None:
            obstacles = _costmap_obstacles(
                self._costmap, self._pose[:2], self._max_obstacles
            )
        else:
            obstacles = np.empty((0, 2))
        self._solver.set_obstacles(obstacles)

        preds, uncs = [], []
        if self._latest_pred is not None:
            candidates = []
            for hp in self._latest_pred.predictions:
                # Conservative default phi_j=1.0 if context not yet seen for a
                # tracked human (be MORE cautious under missing context, matching
                # the dropout philosophy in Architecture Section 4).
                phi_j = self._phi_by_track.get(hp.track_id, 1.0)
                x_hat = np.asarray(hp.x_hat, float)
                y_hat = np.asarray(hp.y_hat, float)
                current = self._latest_humans.get(hp.track_id)
                if current is None or x_hat.size == 0 or y_hat.size == 0:
                    continue
                risk = (phi_j, -float(np.min(np.hypot(x_hat - self._pose[0], y_hat - self._pose[1]))))
                candidates.append((risk, hp, phi_j, current))
            candidates.sort(key=lambda item: item[0], reverse=True)
            for _risk, hp, phi_j, current in candidates[:self._max_humans]:
                # Prepend current human position to LSTM future predictions.
                # Solver expects horizon length N+1 with current sample at index 0.
                human_position_x_horizon = np.concatenate([[current[0]], np.asarray(hp.x_hat, float)])
                human_position_y_horizon = np.concatenate([[current[1]], np.asarray(hp.y_hat, float)])
                preds.append((hp.track_id, human_position_x_horizon, human_position_y_horizon, phi_j))
                uncs.append(0.0)
        self._solver.set_human_predictions(preds, uncs)

        ap = self._latest_params
        d_safe_list = [
            HumanSafetyDistance(track_id=h.track_id, d_safe=h.d_safe)
            for h in ap.d_safe_per_human
        ]
        self._solver.set_adaptive_params(AdaptiveParamsInput(
            vx_max=ap.vx_max, vy_max=ap.vy_max, omega_max=ap.omega_max,
            q_diag=np.array(ap.q_diag), d_safe_aggregate=ap.d_safe_aggregate,
            d_safe_per_human=d_safe_list,
        ))

    def _publish_cmd(self, u) -> None:
        tw = Twist()
        tw.linear.x = float(u[0])
        tw.linear.y = float(u[1])
        tw.angular.z = float(u[2])
        self._pub_cmd.publish(tw)

    def _publish_stop(self) -> None:
        self._pub_cmd.publish(Twist())   # all-zero safe default

    def _publish_diag(self, result, decision, solve_ms) -> None:
        cost = result.cost_breakdown if result else {}
        slacks = result.slacks if result else {}
        data = build_diagnostics(
            solver_success=bool(result.success) if result else False,
            solve_time_ms=solve_ms,
            cost_breakdown=cost,
            slacks=slacks,
            fallback_triggered=decision.fallback_triggered,
            fallback_code=decision.fallback_code,
            fallback_reason=decision.fallback_reason,
        )
        msg = NmpcDiagnostics()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.diagnostic_level = data.diagnostic_level
        msg.solver_success = data.solver_success
        msg.solve_time_ms = data.solve_time_ms
        msg.slacks = [SlackValue(track_id=t, slack=s) for t, s in data.slacks]
        msg.num_humans_active = data.num_humans_active
        msg.cost_total = data.cost_total
        msg.cost_track = data.cost_track
        msg.cost_control = data.cost_control
        msg.cost_smooth = data.cost_smooth
        msg.cost_human = data.cost_human
        msg.cost_obstacle = data.cost_obstacle
        msg.cost_terminal = data.cost_terminal
        msg.fallback_triggered = data.fallback_triggered
        msg.fallback_code = data.fallback_code
        msg.fallback_reason = data.fallback_reason
        self._pub_diag.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(NmpcControllerNode())
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
