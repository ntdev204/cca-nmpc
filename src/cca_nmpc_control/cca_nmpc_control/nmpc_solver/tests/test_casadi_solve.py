import numpy as np

from cca_nmpc_control.nmpc_solver import (
    CasadiSolver,
    AdaptiveParamsInput,
    HumanSafetyDistance,
)


def _make_solver(N=15, dt=0.1, M=4):
    s = CasadiSolver()
    s.initialize({
        "horizon_N": N, "dt": dt, "max_humans_in_solver": M,
        "R_diag": [0.1, 0.1, 0.05], "Rd_diag": [0.05, 0.05, 0.02],
        "P_diag": [10.0, 10.0, 4.0], "w_h": 3.0, "w_slack": 50.0, "d0": 3.0,
    })
    return s, N, M


def _straight_ref(N, goal_x, goal_y):
    xs = np.linspace(0.0, goal_x, N + 1)
    ys = np.linspace(0.0, goal_y, N + 1)
    return {"x": xs, "y": ys, "theta": np.zeros(N + 1)}


def _adaptive(M_humans):
    return AdaptiveParamsInput(
        vx_max=1.0, vy_max=0.8, omega_max=1.2,
        q_diag=np.array([5.0, 5.0, 2.0]), d_safe_aggregate=0.6,
        d_safe_per_human=M_humans,
    )


def test_solves_goal_ahead_no_human():
    s, N, M = _make_solver()
    s.set_reference(_straight_ref(N, 2.0, 0.0))
    s.set_human_predictions([], [])
    s.set_adaptive_params(_adaptive([]))
    r = s.solve(np.array([0.0, 0.0, 0.0]))
    assert r.success
    assert r.u0[0] > 0.0
    assert abs(r.u0[0]) <= 1.0 + 1e-6
    assert abs(r.u0[1]) <= 0.8 + 1e-6
    assert abs(r.u0[2]) <= 1.2 + 1e-6
    assert set(r.cost_breakdown) >= {
        "tracking_cost", "control_cost", "smooth_cost",
        "human_cost", "terminal_cost", "obstacle_cost",
    }


def test_velocity_bounds_respected():
    s, N, M = _make_solver()
    s.set_reference(_straight_ref(N, 50.0, 0.0))
    s.set_human_predictions([], [])
    s.set_adaptive_params(_adaptive([]))
    r = s.solve(np.array([0.0, 0.0, 0.0]))
    assert r.success
    assert r.u0[0] <= 1.0 + 1e-6


def test_human_in_path_activates_avoidance_or_slack():
    s, N, M = _make_solver()
    s.set_reference(_straight_ref(N, 3.0, 0.0))
    hx = np.full(N + 1, 1.5)
    hy = np.full(N + 1, 0.0)
    s.set_human_predictions([(1, hx, hy, 0.9)], [0.1])
    s.set_adaptive_params(_adaptive([HumanSafetyDistance(track_id=1, d_safe=1.0)]))
    r = s.solve(np.array([0.0, 0.0, 0.0]))
    assert r.success
    assert 1 in r.slacks
    assert r.cost_breakdown["human_cost"] >= 0.0


def test_no_slack_when_human_far():
    s, N, M = _make_solver()
    s.set_reference(_straight_ref(N, 2.0, 0.0))
    hx = np.full(N + 1, 0.0)
    hy = np.full(N + 1, 20.0)
    s.set_human_predictions([(2, hx, hy, 0.5)], [0.1])
    s.set_adaptive_params(_adaptive([HumanSafetyDistance(track_id=2, d_safe=0.8)]))
    r = s.solve(np.array([0.0, 0.0, 0.0]))
    assert r.success
    assert r.slacks[2] < 1e-3


def test_reset_clears_warm_start():
    s, N, M = _make_solver()
    s.set_reference(_straight_ref(N, 2.0, 0.0))
    s.set_human_predictions([], [])
    s.set_adaptive_params(_adaptive([]))
    s.solve(np.array([0.0, 0.0, 0.0]))
    assert s._z_warm is not None
    s.reset()
    assert s._z_warm is None


def test_diagnostics_populated():
    s, N, M = _make_solver()
    s.set_reference(_straight_ref(N, 2.0, 0.0))
    s.set_human_predictions([], [])
    s.set_adaptive_params(_adaptive([]))
    s.solve(np.array([0.0, 0.0, 0.0]))
    d = s.get_diagnostics()
    assert d.objective_value >= 0.0
    assert d.constraint_violation < 1e-4


def test_obstacle_on_reference_path_raises_cost_and_deviates_trajectory():
    N = 15
    ref = _straight_ref(N, 3.0, 0.0)
    x0 = np.array([0.0, 0.0, 0.0])

    s_base, _, _ = _make_solver(N=N)
    s_base.set_reference(ref)
    s_base.set_human_predictions([], [])
    s_base.set_adaptive_params(_adaptive([]))
    r_base = s_base.solve(x0)
    assert r_base.success

    s_obs, _, _ = _make_solver(N=N)
    s_obs.set_reference(ref)
    s_obs.set_obstacles(np.array([[1.5, 0.0]]))
    s_obs.set_human_predictions([], [])
    s_obs.set_adaptive_params(_adaptive([]))
    r_obs = s_obs.solve(x0)
    assert r_obs.success

    assert r_obs.cost_breakdown["obstacle_cost"] > r_base.cost_breakdown["obstacle_cost"]

    max_y_base = float(np.max(np.abs(r_base.trajectory["y"])))
    max_y_obs = float(np.max(np.abs(r_obs.trajectory["y"])))
    lateral_deflected = (max_y_obs > max_y_base + 1e-4) or (abs(r_obs.u0[1]) > 1e-4)
    assert lateral_deflected, (
        f"Expected lateral deflection from obstacle but got "
        f"max|y|_base={max_y_base:.6f}, max|y|_obs={max_y_obs:.6f}, "
        f"u0[1]={r_obs.u0[1]:.6f}"
    )
