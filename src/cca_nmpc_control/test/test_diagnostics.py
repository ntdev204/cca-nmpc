"""Tests for NmpcDiagnostics assembly (Sections 9, 11, 12)."""
from cca_nmpc_control.diagnostics import (
    build_diagnostics, LEVEL_OK, LEVEL_WARN, LEVEL_ERROR,
)
from cca_nmpc_control.fallback import FALLBACK_NONE, FALLBACK_SAFE_STOP


COST = {
    "tracking_cost": 1.0, "control_cost": 0.5, "smooth_cost": 0.2,
    "human_cost": 0.3, "obstacle_cost": 0.0, "terminal_cost": 2.0,
}


def test_cost_total_is_sum():
    d = build_diagnostics(
        solver_success=True, solve_time_ms=8.0, cost_breakdown=COST,
        slacks={}, fallback_triggered=False, fallback_code=FALLBACK_NONE,
    )
    assert abs(d.cost_total - 4.0) < 1e-9


def test_slacks_are_track_id_pairs_sorted():
    d = build_diagnostics(
        solver_success=True, solve_time_ms=8.0, cost_breakdown=COST,
        slacks={5: 0.2, 1: 0.0}, fallback_triggered=False,
        fallback_code=FALLBACK_NONE,
    )
    assert d.slacks == [(1, 0.0), (5, 0.2)]
    assert d.num_humans_active == 2


def test_level_ok_on_success():
    d = build_diagnostics(
        solver_success=True, solve_time_ms=8.0, cost_breakdown=COST,
        slacks={}, fallback_triggered=False, fallback_code=FALLBACK_NONE,
    )
    assert d.diagnostic_level == LEVEL_OK


def test_level_warn_on_fallback_hold():
    d = build_diagnostics(
        solver_success=True, solve_time_ms=8.0, cost_breakdown=COST,
        slacks={}, fallback_triggered=True, fallback_code=3,
    )
    assert d.diagnostic_level == LEVEL_WARN


def test_level_error_on_failed_and_fallback():
    d = build_diagnostics(
        solver_success=False, solve_time_ms=8.0, cost_breakdown=COST,
        slacks={}, fallback_triggered=True, fallback_code=FALLBACK_SAFE_STOP,
        fallback_reason="safe stop",
    )
    assert d.diagnostic_level == LEVEL_ERROR
    assert d.fallback_reason == "safe stop"


def test_missing_cost_keys_default_zero():
    d = build_diagnostics(
        solver_success=True, solve_time_ms=1.0, cost_breakdown={},
        slacks={}, fallback_triggered=False, fallback_code=FALLBACK_NONE,
    )
    assert d.cost_total == 0.0
