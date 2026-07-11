"""Offline end-to-end pipeline harness (BR-02).

Wires the ROS-free cores of plans 01/04/05/06/07 into a single control step
WITHOUT ROS: scenario states -> analytic prediction -> context (phi_j, EMA+gate) ->
adaptive params (caps, Q, per-human d_safe) -> CasADi solver -> cmd_vel. This is
the key Windows deliverable proving the math pipeline end-to-end (plan 10).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Make the ROS-free cores of the packages importable without ROS.
_SRC = Path(__file__).resolve().parents[2] / "src"
for _pkg in ("cca_nmpc_context", "cca_nmpc_prediction",
             "cca_nmpc_adaptive_params", "cca_nmpc_control"):
    p = str(_SRC / _pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

from cca_nmpc_context.relative_motion import compute_relative_motion  # noqa: E402
from cca_nmpc_context.context_score import (  # noqa: E402
    ContextWeights, context_score, sigmoid,
)
from cca_nmpc_context.smoothing import PhiSmoother  # noqa: E402
from cca_nmpc_context.aggregate import aggregate_context  # noqa: E402
from cca_nmpc_adaptive_params.param_map import (  # noqa: E402
    AdaptiveConfig, d_safe, saturated_velocity_limits, q_diag,
)
from cca_nmpc_control.nmpc_solver import (  # noqa: E402
    CasadiSolver, AdaptiveParamsInput, HumanSafetyDistance,
)


@dataclass
class HumanObs:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    confidence: float = 0.9


def _default_adaptive() -> AdaptiveConfig:
    return AdaptiveConfig(
        d_safe0=0.6, k_d=0.8, v_x0=1.0, v_y0=0.8, omega_0=1.2,
        k_x=0.6, k_y=0.5, k_omega=0.7,
        v_x_min=0.08, v_y_min=0.06, omega_min=0.1,
        q0_diag=(5.0, 5.0, 2.0), qh_diag=(8.0, 8.0, 3.0),
    )


@dataclass
class PipelineConfig:
    L: int = 8
    H: int = 12
    horizon_N: int = 15
    dt: float = 0.1
    d0: float = 3.0
    v_max_ref: float = 1.5
    max_humans: int = 4
    weights: ContextWeights = field(
        default_factory=lambda: ContextWeights(1.2, 0.8, 0.6, 1.0, -0.5))
    adaptive: AdaptiveConfig = field(default_factory=_default_adaptive)


@dataclass
class StepResult:
    cmd_vel: np.ndarray            # [vx, vy, omega]
    solver_success: bool
    phi_aggregate_used: float
    slacks: dict
    per_human_phi: dict


class Pipeline:
    """Wires the ROS-free cores into a single offline control step."""

    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        c = self.cfg
        self._smoother = PhiSmoother(alpha=0.7, t_dwell_cycles=4)
        self._solver = CasadiSolver()
        self._solver.initialize({
            "horizon_N": c.horizon_N, "dt": c.dt,
            "max_humans_in_solver": c.max_humans, "d0": c.d0,
        })

    def step(
        self,
        robot: tuple[float, float, float, float, float],
        humans: list[HumanObs],
        goal: tuple[float, float, float],
    ) -> StepResult:
        """Run one full pipeline step -> cmd_vel."""
        c = self.cfg
        rx, ry, rtheta, rvx, rvy = robot

        # Analytic scenario prediction; runtime prediction remains TensorRT-only.
        predictions = []
        for h in humans:
            t = np.arange(1, c.H + 1, dtype=np.float32) * c.dt
            pred = np.column_stack((
                h.x + h.vx * t,
                h.y + h.vy * t,
                np.full(c.H, h.vx),
                np.full(c.H, h.vy),
            )).astype(np.float32)
            predictions.append((h.track_id, pred))

        # context: phi_j per human, EMA + dwell gate, then aggregate
        phi_raw, phi_used_list, d_list, per_human_phi, active_ids = [], [], [], {}, []
        for h in humans:
            rm = compute_relative_motion(
                rx, ry, rtheta, rvx, rvy, h.x, h.y, h.vx, h.vy)
            z = context_score(
                rm.d_h, rm.v_h_speed, rm.cos_dtheta, h.confidence, 0.0,
                c.weights, c.d0, c.v_max_ref)
            _filt, used = self._smoother.update(h.track_id, sigmoid(z))
            phi_raw.append(sigmoid(z))
            phi_used_list.append(used)
            d_list.append(rm.d_h)
            per_human_phi[h.track_id] = used
            active_ids.append(h.track_id)
        for tid in self._smoother.active_tracks():
            if tid not in active_ids:
                self._smoother.drop(tid)
        agg = aggregate_context(phi_raw, phi_used_list, d_list)

        # adaptive params
        vx, vy, omega = saturated_velocity_limits(agg.phi_aggregate_used, c.adaptive)
        d_safe_per_human = [
            HumanSafetyDistance(
                track_id=tid, d_safe=d_safe(per_human_phi[tid], c.adaptive))
            for tid in active_ids
        ]

        # solver
        N = c.horizon_N
        self._solver.set_reference({
            "x": np.linspace(rx, goal[0], N + 1),
            "y": np.linspace(ry, goal[1], N + 1),
            "theta": np.full(N + 1, goal[2]),
        })
        solver_preds = []
        for (tid, pred) in predictions[: c.max_humans]:
            xh = np.concatenate([pred[:, 0], [pred[-1, 0]]])[: N + 1]
            yh = np.concatenate([pred[:, 1], [pred[-1, 1]]])[: N + 1]
            solver_preds.append((tid, xh, yh, per_human_phi.get(tid, 1.0)))
        self._solver.set_human_predictions(solver_preds, [0.0] * len(solver_preds))
        self._solver.set_adaptive_params(AdaptiveParamsInput(
            vx_max=vx, vy_max=vy, omega_max=omega,
            q_diag=np.array(q_diag(agg.phi_aggregate_used, c.adaptive)),
            d_safe_aggregate=d_safe(agg.phi_aggregate_used, c.adaptive),
            d_safe_per_human=d_safe_per_human))

        result = self._solver.solve(np.array([rx, ry, rtheta]))
        return StepResult(
            cmd_vel=result.u0,
            solver_success=result.success,
            phi_aggregate_used=agg.phi_aggregate_used,
            slacks=result.slacks,
            per_human_phi=per_human_phi,
        )
