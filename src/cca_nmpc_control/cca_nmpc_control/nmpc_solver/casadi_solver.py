"""
CasADi + IPOPT NMPC backend (reference implementation).

Multiple-shooting transcription of the OCP in docs/01_mathematical_model.md
(Eqs. 11.1-12.7), mapped per docs/08_solver_design.md. Structure (number of
decision variables and constraints) is FIXED at initialize() time — the number
of human constraint slots is max_humans_in_solver, unused slots filled with the
dummy-human convention — so only runtime *values* change per solve. This mirrors
what a code-generated acados solver requires and is the anti-drift guard.

Decision variables z = [X_0..X_N, U_0..U_{N-1}, s_0..s_{M-1}]
  X_k in R^3 (states, multiple shooting), U_k in R^3 (controls),
  s_j >= 0 one slack per human slot (Eq. 12.3 soft constraint).

Runtime parameters p = [x0, reference, per-human (x_hat,y_hat,phi_j), d_safe].
Velocity box bounds (Eq. 12.2) enter as lbx/ubx on U at solve time.

Reference: docs/08_solver_design.md Sections 3, 4, 6, 7, 8, 10, 11.
"""
from __future__ import annotations

import time

import numpy as np
import casadi as ca

from .interface import (
    SolverInterface,
    SolveResult,
    SolverDiagnostics,
    AdaptiveParamsInput,
)
from . import ocp_spec
from . import dummy_humans
from .warm_start import shift_and_append_controls, repropagate_states


class CasadiSolver(SolverInterface):
    """CasADi + IPOPT implementation of SolverInterface."""

    def __init__(self) -> None:
        self._built = False
        self._spec: ocp_spec.OCPSpec | None = None
        self._solver: ca.Function | None = None
        # config
        self._N = 0
        self._dt = 0.0
        self._M = 0  # max_humans_in_solver
        self._f_lstm_hz = 8.0  # LSTM prediction frequency (Hz)
        self._r_diag = np.array([0.1, 0.1, 0.05])
        self._rd_diag = np.array([0.05, 0.05, 0.02])
        self._p_diag = np.array([10.0, 10.0, 4.0])
        self._w_h = 3.0
        self._w_slack = 50.0
        self._d0 = 3.0
        self._w_obstacle = 2.0
        self._obstacle_sigma = 0.35
        self._max_obstacles = 64
        self._max_cpu_time = 0.0
        # runtime inputs
        self._ref: dict | None = None
        self._human_slots: list | None = None
        self._d_safe_slots: list[float] | None = None
        self._active_track_ids: list[int] = []
        self._q_diag = np.array([5.0, 5.0, 2.0])
        self._vx_max = 1.0
        self._vy_max = 0.8
        self._omega_max = 1.2
        self._obstacles = np.empty((0, 2), dtype=float)
        # warm start + diagnostics
        self._z_warm: np.ndarray | None = None
        self._last_diag = SolverDiagnostics(0, 0.0, "n/a", 0.0, 0.0)

    # ------------------------------------------------------------------ build
    def initialize(self, params: dict) -> None:
        """Build the fixed symbolic NLP once (Section 6.1 lifecycle)."""
        self._N = int(params["horizon_N"])
        self._dt = float(params["dt"])
        self._M = int(params["max_humans_in_solver"])
        self._f_lstm_hz = float(params.get("f_lstm_hz", 8.0))
        self._r_diag = np.asarray(params.get("R_diag", self._r_diag), float)
        self._rd_diag = np.asarray(params.get("Rd_diag", self._rd_diag), float)
        self._p_diag = np.asarray(params.get("P_diag", self._p_diag), float)
        self._w_h = float(params.get("w_h", self._w_h))
        self._w_slack = float(params.get("w_slack", self._w_slack))
        self._d0 = float(params.get("d0", self._d0))
        self._w_obstacle = float(params.get("w_obstacle", self._w_obstacle))
        self._obstacle_sigma = float(params.get("obstacle_sigma", self._obstacle_sigma))
        self._max_obstacles = int(params.get("max_obstacle_samples", self._max_obstacles))
        self._max_cpu_time = float(params.get("solver_max_cpu_time_sec", 0.0))
        if self._N < 1 or self._dt <= 0 or self._M < 1:
            raise ValueError("horizon_N, dt, max_humans_in_solver must be positive")

        self._spec = ocp_spec.OCPSpec(self._dt)
        self._build_nlp()
        self._built = True

    def _build_nlp(self) -> None:
        N, M, nx, nu = self._N, self._M, ocp_spec.NX, ocp_spec.NU
        f = self._spec.f_discrete

        X = ca.MX.sym("X", nx, N + 1)
        U = ca.MX.sym("U", nu, N)
        S = ca.MX.sym("S", M)  # per-human slack

        # Parameters: x0(3), ref(3*(N+1)), q_diag(3), per-human
        # [x_hat(N+1), y_hat(N+1), phi_j(1), d_safe(1)] * M
        P_x0 = ca.MX.sym("P_x0", nx)
        P_ref = ca.MX.sym("P_ref", nx, N + 1)
        P_q = ca.MX.sym("P_q", nx)
        P_hx = ca.MX.sym("P_hx", N + 1, M)
        P_hy = ca.MX.sym("P_hy", N + 1, M)
        P_phi = ca.MX.sym("P_phi", M)
        P_dsafe = ca.MX.sym("P_dsafe", M)
        P_obs = ca.MX.sym("P_obs", 3, self._max_obstacles)  # x, y, active

        g = [X[:, 0] - P_x0]           # initial-state equality
        J = ca.MX(0)

        for k in range(N):
            e = X[:, k] - P_ref[:, k]
            J += ocp_spec.build_tracking_cost_stage(e, P_q)
            J += ocp_spec.build_control_cost_stage(U[:, k], self._r_diag)
            if k > 0:
                J += ocp_spec.build_smooth_cost_stage(
                    U[:, k] - U[:, k - 1], self._rd_diag
                )
            for j in range(M):
                J += ocp_spec.build_human_hinge_cost_stage(
                    X[:, k], P_hx[k, j], P_hy[k, j], P_phi[j], self._d0, self._w_h
                )
            for j in range(self._max_obstacles):
                squared_distance = (
                    (X[0, k] - P_obs[0, j]) ** 2
                    + (X[1, k] - P_obs[1, j]) ** 2
                )
                J += (
                    self._w_obstacle * P_obs[2, j]
                    * ca.exp(-squared_distance / (2.0 * self._obstacle_sigma ** 2))
                )
            g.append(f(x=X[:, k], u=U[:, k])["x_next"] - X[:, k + 1])

        # terminal cost
        J += ocp_spec.build_terminal_cost(X[:, N] - P_ref[:, N], self._p_diag)

        # slack penalty: one per human slot (not per stage)
        for j in range(M):
            J += ocp_spec.build_soft_constraint_slack_cost(S[j], self._w_slack)

        # safety constraint at ALL stages k=0..N (Eq. 12.3, per-horizon)
        for k in range(N + 1):
            for j in range(M):
                g.append(
                    ocp_spec.build_soft_human_constraint(
                        X[:, k], P_hx[k, j], P_hy[k, j], S[j], P_dsafe[j]
                    )
                )

        z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1), S)
        p = ca.vertcat(
            P_x0, ca.reshape(P_ref, -1, 1), P_q,
            ca.reshape(P_hx, -1, 1), ca.reshape(P_hy, -1, 1), P_phi, P_dsafe,
            ca.reshape(P_obs, -1, 1),
        )
        nlp = {"x": z, "p": p, "f": J, "g": ca.vertcat(*g)}
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
        if self._max_cpu_time > 0.0:
            opts["ipopt.max_cpu_time"] = self._max_cpu_time
        self._solver = ca.nlpsol("nmpc", "ipopt", nlp, opts)
        self._n_eq = nx * (N + 1)          # equality constraints
        self._n_ineq = M * (N + 1)         # soft-human inequalities (all stages)

    # --------------------------------------------------------------- setters
    def set_reference(self, ref_trajectory: dict) -> None:
        self._require_built()
        N = self._N
        ref = np.zeros((ocp_spec.NX, N + 1))
        ref[0, :] = np.asarray(ref_trajectory["x"], float)[: N + 1]
        ref[1, :] = np.asarray(ref_trajectory["y"], float)[: N + 1]
        ref[2, :] = np.asarray(ref_trajectory["theta"], float)[: N + 1]
        self._ref = {"array": ref}

    def set_human_predictions(self, predictions: list, uncertainties: list) -> None:
        self._require_built()
        slots = dummy_humans.fill_dummy_human_slots(
            list(predictions), self._M, self._N
        )
        self._human_slots = slots
        self._active_track_ids = [tid for (tid, *_rest) in predictions]

    def set_adaptive_params(self, adaptive_params: AdaptiveParamsInput) -> None:
        self._require_built()
        self._vx_max = float(adaptive_params.vx_max)
        self._vy_max = float(adaptive_params.vy_max)
        self._omega_max = float(adaptive_params.omega_max)
        self._q_diag = np.asarray(adaptive_params.q_diag, float).flatten()[:3]
        if self._human_slots is None:
            # d_safe cannot be bound before predictions define the slot order.
            self._pending_d_safe = {
                h.track_id: h.d_safe for h in adaptive_params.d_safe_per_human
            }
            self._d_safe_slots = None
            return
        active = {h.track_id: h.d_safe for h in adaptive_params.d_safe_per_human}
        self._d_safe_slots = dummy_humans.build_d_safe_slots(
            active, self._human_slots
        )
        self._pending_d_safe = active

    def set_obstacles(self, obstacle_points: np.ndarray) -> None:
        points = np.asarray(obstacle_points, dtype=float)
        if points.size == 0:
            self._obstacles = np.empty((0, 2), dtype=float)
            return
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("obstacle_points must have shape (K, 2)")
        self._obstacles = points[: self._max_obstacles].copy()

    # ----------------------------------------------------------------- solve
    def solve(self, x0: np.ndarray) -> SolveResult:
        self._require_built()
        if self._ref is None or self._human_slots is None:
            raise RuntimeError("set_reference and set_human_predictions first")
        if self._d_safe_slots is None:
            active = getattr(self, "_pending_d_safe", {})
            self._d_safe_slots = dummy_humans.build_d_safe_slots(
                active, self._human_slots
            )

        x0 = np.asarray(x0, float).flatten()

        p = self._assemble_params(x0)
        lbx, ubx = self._var_bounds()
        z0 = self._warm_start_vector(x0)

        # equality g == 0 for dynamics; inequality g >= 0 for soft-human
        lbg = np.concatenate([np.zeros(self._n_eq), np.zeros(self._n_ineq)])
        ubg = np.concatenate([np.zeros(self._n_eq), np.full(self._n_ineq, ca.inf)])

        t0 = time.perf_counter()
        sol = self._solver(x0=z0, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        solve_ms = (time.perf_counter() - t0) * 1e3

        stats = self._solver.stats()
        success = bool(stats.get("success", False))
        z = np.array(sol["x"]).flatten()
        self._z_warm = z

        return self._build_result(z, sol, stats, success, solve_ms, x0)

    # ------------------------------------------------------------- diagnostics
    def get_diagnostics(self) -> SolverDiagnostics:
        return self._last_diag

    def reset(self) -> None:
        """Clear warm start (Section 5.2 triggers), keep the built NLP."""
        self._z_warm = None

    def shutdown(self) -> None:
        self._solver = None
        self._built = False

    # --------------------------------------------------------------- helpers
    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError("initialize() must be called before use")

    def _n_states(self) -> int:
        return ocp_spec.NX * (self._N + 1)

    def _n_controls(self) -> int:
        return ocp_spec.NU * self._N

    def _var_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Box bounds on z: states free, controls |u|<=cap (Eq. 12.2), slack>=0."""
        ns, M = self._n_states(), self._M
        lbx = np.concatenate([
            np.full(ns, -ca.inf),
            np.tile([-self._vx_max, -self._vy_max, -self._omega_max], self._N),
            np.zeros(M),
        ])
        ubx = np.concatenate([
            np.full(ns, ca.inf),
            np.tile([self._vx_max, self._vy_max, self._omega_max], self._N),
            np.full(M, ca.inf),
        ])
        return lbx, ubx

    def _assemble_params(self, x0: np.ndarray) -> np.ndarray:
        N, M = self._N, self._M
        ref = self._ref["array"]
        hx = np.zeros((N + 1, M))
        hy = np.zeros((N + 1, M))
        phi = np.zeros(M)
        for j, (_tid, x_hat, y_hat, phi_j) in enumerate(self._human_slots):
            xa = np.asarray(x_hat, float)
            ya = np.asarray(y_hat, float)
            # Time-interpolate LSTM predictions onto NMPC grid
            hx[:, j] = _interpolate_lstm_to_nmpc(xa, self._f_lstm_hz, self._dt, N + 1)
            hy[:, j] = _interpolate_lstm_to_nmpc(ya, self._f_lstm_hz, self._dt, N + 1)
            phi[j] = float(phi_j)
        dsafe = np.asarray(self._d_safe_slots, float)
        obstacles = np.zeros((3, self._max_obstacles), dtype=float)
        count = min(len(self._obstacles), self._max_obstacles)
        if count:
            obstacles[:2, :count] = self._obstacles[:count].T
            obstacles[2, :count] = 1.0
        return np.concatenate([
            x0,
            ref.flatten(order="F"),
            self._q_diag,
            hx.flatten(order="F"),
            hy.flatten(order="F"),
            phi,
            dsafe,
            obstacles.flatten(order="F"),
        ])

    def _warm_start_vector(self, x0: np.ndarray) -> np.ndarray:
        if self._z_warm is not None and self._z_warm.size == (
            self._n_states() + self._n_controls() + self._M
        ):
            ns = self._n_states()
            controls = self._z_warm[ns:ns + self._n_controls()].reshape(
                self._N, ocp_spec.NU
            )
            shifted = shift_and_append_controls(controls)
            states = repropagate_states(x0, shifted, self._spec.integrate_numpy)
            slacks = self._z_warm[ns + self._n_controls():]
            return np.concatenate([states.reshape(-1), shifted.reshape(-1), slacks])
        # cold start: replicate x0 across states, zero controls/slack
        states = np.tile(x0, self._N + 1)
        return np.concatenate([
            states, np.zeros(self._n_controls()), np.zeros(self._M)
        ])

    def _build_result(self, z, sol, stats, success, solve_ms, x0) -> SolveResult:
        N, nx, nu = self._N, ocp_spec.NX, ocp_spec.NU
        ns = self._n_states()
        X = z[:ns].reshape(N + 1, nx)
        U = z[ns:ns + self._n_controls()].reshape(N, nu)
        S = z[ns + self._n_controls():]
        u0 = U[0].copy() if N > 0 else np.zeros(nu)

        slacks = {
            tid: float(S[j])
            for j, (tid, *_r) in enumerate(self._human_slots)
            if not dummy_humans.is_dummy_slot(tid)
        }
        cost = self._cost_breakdown(X, U, S)
        obj = float(sol["f"])
        g = np.array(sol["g"]).flatten()
        viol = float(np.max(np.abs(g[: self._n_eq]))) if self._n_eq else 0.0
        self._last_diag = SolverDiagnostics(
            sqp_iterations=int(stats.get("iter_count", 0)),
            kkt_residual=float(stats.get("iterations", {}).get("inf_pr", [0.0])[-1])
            if isinstance(stats.get("iterations"), dict) else 0.0,
            qp_status=str(stats.get("return_status", "unknown")),
            constraint_violation=viol,
            objective_value=obj,
        )
        return SolveResult(
            u0=u0,
            trajectory={"x": X[:, 0], "y": X[:, 1], "theta": X[:, 2], "u": U},
            success=success,
            solve_time_ms=solve_ms,
            cost_breakdown=cost,
            slacks=slacks,
        )

    def _cost_breakdown(self, X, U, S) -> dict[str, float]:
        N = self._N
        ref = self._ref["array"]
        # Interpolate each human's prediction onto NMPC time grid
        fitted = [
            (np.asarray(_interpolate_lstm_to_nmpc(np.asarray(x_hat, float), self._f_lstm_hz, self._dt, N + 1)),
             np.asarray(_interpolate_lstm_to_nmpc(np.asarray(y_hat, float), self._f_lstm_hz, self._dt, N + 1)),
             float(phi_j))
            for (_tid, x_hat, y_hat, phi_j) in self._human_slots
        ]
        track = ctrl = smooth = human = obstacle = 0.0
        for k in range(N):
            e = X[k] - ref[:, k]
            track += float(e @ (self._q_diag * e))
            ctrl += float(U[k] @ (self._r_diag * U[k]))
            if k > 0:
                du = U[k] - U[k - 1]
                smooth += float(du @ (self._rd_diag * du))
            for x_hat, y_hat, phi_j in fitted:
                d = np.hypot(X[k, 0] - x_hat[k], X[k, 1] - y_hat[k])
                hinge = max(0.0, self._d0 - d)
                human += self._w_h * phi_j * hinge ** 2
            if len(self._obstacles):
                d2 = np.sum((self._obstacles - X[k, :2]) ** 2, axis=1)
                obstacle += float(np.sum(
                    self._w_obstacle
                    * np.exp(-d2 / (2.0 * self._obstacle_sigma ** 2))
                ))
        eN = X[N] - ref[:, N]
        terminal = float(eN @ (self._p_diag * eN))
        slack_pen = float(self._w_slack * np.sum(np.square(S)))
        return {
            "tracking_cost": track,
            "control_cost": ctrl,
            "smooth_cost": smooth,
            "obstacle_cost": obstacle,
            "human_cost": human + slack_pen,
            "terminal_cost": terminal,
        }


def _interpolate_lstm_to_nmpc(
    lstm_prediction: np.ndarray, f_lstm_hz: float, dt_nmpc: float, n_nmpc: int
) -> np.ndarray:
    """
    Interpolate LSTM predictions (8Hz grid) onto NMPC time grid (20Hz).

    Args:
        lstm_prediction: H-length array from LSTM (e.g., x_hat or y_hat)
        f_lstm_hz: LSTM prediction frequency (Hz), e.g., 8.0
        dt_nmpc: NMPC timestep (s), e.g., 0.05
        n_nmpc: Number of NMPC stages (N+1)

    Returns:
        n_nmpc-length array interpolated onto NMPC time grid.
        If lstm_prediction is empty, returns dummy values.
    """
    if lstm_prediction.size == 0:
        return np.full(n_nmpc, dummy_humans.DUMMY_D_J)

    # LSTM time grid: [0, 1/f_lstm, 2/f_lstm, ..., (H-1)/f_lstm]
    H = lstm_prediction.size
    t_lstm = np.arange(H) / f_lstm_hz

    # NMPC time grid: [0, dt_nmpc, 2*dt_nmpc, ..., (N)*dt_nmpc]
    t_nmpc = np.arange(n_nmpc) * dt_nmpc

    # Interpolate (extrapolate with edge values if t_nmpc extends beyond t_lstm)
    return np.interp(t_nmpc, t_lstm, lstm_prediction)
