"""Stage-2 sensitivity sweep vs solve success (CB-04).

Perturbs each weight by +/-perturbation_pct independently, measures the change
in d_safe, v_x_max, and NMPC solve-success rate, and re-scales weights whose
perturbation degrades solve success beyond the threshold (Math Model 8.1). Uses
the plan-07 CasADi solver so "solve success" is measured, not assumed.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
_CTX = _SRC / "cca_nmpc_context"
_CTRL = _SRC / "cca_nmpc_control"
for p in (str(_CTX), str(_CTRL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cca_nmpc_context.context_score import ContextWeights  # noqa: E402
from cca_nmpc_control.nmpc_solver import (  # noqa: E402
    CasadiSolver, AdaptiveParamsInput, HumanSafetyDistance,
)

from .features import FeatureVector, phi_from_weights  # noqa: E402


@dataclass
class WeightSensitivity:
    weight_name: str
    d_safe_change_pct: float
    vx_max_change_pct: float
    solve_success_change_pct: float
    rescaled: bool


_WEIGHT_FIELDS = ("w_d", "w_v", "w_theta", "w_u", "b")


def _phi_stats(features, weights):
    phis = [phi_from_weights(f, weights) for f in features]
    return float(np.mean(phis)) if phis else 0.0


def _solve_success_rate(phis: list[float], solver: CasadiSolver, N: int) -> float:
    """Evaluate diverse crossing distances instead of one mean-phi scenario."""
    outcomes = []
    for index, phi in enumerate(phis):
        xs = np.linspace(0.0, 3.0, N + 1)
        solver.set_reference({"x": xs, "y": np.zeros(N + 1), "theta": np.zeros(N + 1)})
        hx = np.full(N + 1, 0.8 + 0.35 * (index % 5))
        hy = np.linspace(-0.8, 0.8, N + 1)
        solver.set_human_predictions([(1, hx, hy, phi)], [0.0])
        d_safe = 0.6 + 0.8 * phi
        solver.set_adaptive_params(AdaptiveParamsInput(
            vx_max=max(0.08, 1.0 - 0.6 * phi), vy_max=0.8, omega_max=1.2,
            q_diag=np.array([5.0, 5.0, 2.0]), d_safe_aggregate=d_safe,
            d_safe_per_human=[HumanSafetyDistance(track_id=1, d_safe=d_safe)],
        ))
        result = solver.solve(np.zeros(3))
        outcomes.append(result.success and max(result.slacks.values(), default=0.0) < 0.1)
    return float(np.mean(outcomes))


def run_sensitivity(
    features: list[FeatureVector],
    weights: ContextWeights,
    perturbation_pct: float = 0.20,
    max_degradation_pct: float = 10.0,
    horizon_N: int = 15,
    dt: float = 0.1,
) -> tuple[ContextWeights, list[WeightSensitivity]]:
    """Sweep each weight +/-pct; rescale weights that dominate solve success."""
    solver = CasadiSolver()
    solver.initialize({"horizon_N": horizon_N, "dt": dt,
                       "max_humans_in_solver": 4})

    base_phi = _phi_stats(features, weights)
    base_dsafe = 0.6 + 0.8 * base_phi
    base_vx = max(0.08, 1.0 - 0.6 * base_phi)
    base_phis = [phi_from_weights(f, weights) for f in features]
    base_success = _solve_success_rate(base_phis, solver, horizon_N)

    report: list[WeightSensitivity] = []
    adjusted = weights
    for name in _WEIGHT_FIELDS:
        val = getattr(weights, name)
        worst_success_change = 0.0
        d_change = v_change = 0.0
        for sign in (1.0, -1.0):
            pert = replace(weights, **{name: val * (1.0 + sign * perturbation_pct)})
            phi = _phi_stats(features, pert)
            dsafe = 0.6 + 0.8 * phi
            vx = max(0.08, 1.0 - 0.6 * phi)
            success = _solve_success_rate(
                [phi_from_weights(f, pert) for f in features], solver, horizon_N)
            d_change = max(d_change, _pct(dsafe, base_dsafe))
            v_change = max(v_change, _pct(vx, base_vx))
            worst_success_change = max(
                worst_success_change, _pct(success, base_success))

        rescaled = worst_success_change > max_degradation_pct
        if rescaled:
            adjusted = replace(adjusted, **{name: getattr(adjusted, name) * 0.5})
        report.append(WeightSensitivity(
            weight_name=name,
            d_safe_change_pct=d_change,
            vx_max_change_pct=v_change,
            solve_success_change_pct=worst_success_change,
            rescaled=rescaled,
        ))
    return adjusted, report


def _pct(new: float, base: float) -> float:
    if base == 0.0:
        return 0.0 if new == 0.0 else 100.0
    return abs(new - base) / abs(base) * 100.0
