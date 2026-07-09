"""Tests that the interface contract is satisfiable (SV-01)."""
import numpy as np

from cca_nmpc_control.nmpc_solver import (
    SolverInterface,
    SolveResult,
    SolverDiagnostics,
    AdaptiveParamsInput,
    HumanSafetyDistance,
)


def test_solve_result_fields():
    r = SolveResult(
        u0=np.zeros(3), trajectory={}, success=True, solve_time_ms=1.0,
        cost_breakdown={"tracking_cost": 0.0}, slacks={1: 0.0},
    )
    assert r.success is True
    assert r.slacks[1] == 0.0


def test_diagnostics_fields():
    d = SolverDiagnostics(1, 0.0, "ok", 0.0, 0.0)
    assert d.qp_status == "ok"


def test_adaptive_params_input_field_names_match_msg():
    ap = AdaptiveParamsInput(
        vx_max=1.0, vy_max=0.8, omega_max=1.2,
        q_diag=np.array([5.0, 5.0, 2.0]), d_safe_aggregate=0.6,
        d_safe_per_human=[HumanSafetyDistance(track_id=1, d_safe=0.9)],
    )
    assert ap.q_diag.shape == (3,)
    assert ap.d_safe_per_human[0].track_id == 1


def test_stub_can_implement_abc():
    class _Stub(SolverInterface):
        def initialize(self, params): ...
        def set_reference(self, ref_trajectory): ...
        def set_human_predictions(self, predictions, uncertainties): ...
        def set_adaptive_params(self, adaptive_params): ...
        def solve(self, x0): return None
        def get_diagnostics(self): return None
        def reset(self): ...
        def shutdown(self): ...

    s = _Stub()
    assert isinstance(s, SolverInterface)
