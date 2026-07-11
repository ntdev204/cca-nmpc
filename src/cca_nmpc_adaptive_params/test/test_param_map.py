from cca_nmpc_adaptive_params.param_map import (
    AdaptiveConfig, d_safe, saturated_velocity_limits, q_diag,
)

CFG = AdaptiveConfig(
    d_safe0=0.6, k_d=0.8,
    v_x0=1.0, v_y0=0.8, omega_0=1.2,
    k_x=0.6, k_y=0.5, k_omega=0.7,
    v_x_min=0.08, v_y_min=0.06, omega_min=0.1,
    q0_diag=(5.0, 5.0, 2.0), qh_diag=(8.0, 8.0, 3.0),
)


def test_d_safe_linear():
    assert abs(d_safe(0.0, CFG) - 0.6) < 1e-9
    assert abs(d_safe(1.0, CFG) - 1.4) < 1e-9
    assert d_safe(0.5, CFG) > d_safe(0.1, CFG)


def test_velocity_nominal_at_phi_zero():
    vx, vy, w = saturated_velocity_limits(0.0, CFG)
    assert abs(vx - 1.0) < 1e-9
    assert abs(vy - 0.8) < 1e-9
    assert abs(w - 1.2) < 1e-9


def test_velocity_respects_floor_at_phi_one():
    vx, vy, w = saturated_velocity_limits(1.0, CFG)
    assert abs(vx - 0.4) < 1e-9 and abs(vy - 0.3) < 1e-9 and abs(w - 0.5) < 1e-9
    assert vx >= CFG.v_x_min and vy >= CFG.v_y_min and w >= CFG.omega_min


def test_velocity_floor_clamps_extreme():
    cfg = AdaptiveConfig(**{**CFG.__dict__, "k_x": 10.0})
    vx, _vy, _w = saturated_velocity_limits(1.0, cfg)
    assert vx == cfg.v_x_min


def test_velocity_never_zero():
    for phi in (0.0, 0.5, 1.0):
        vx, vy, w = saturated_velocity_limits(phi, CFG)
        assert vx > 0 and vy > 0 and w > 0


def test_q_diag_linear_and_length_3():
    q0 = q_diag(0.0, CFG)
    q1 = q_diag(1.0, CFG)
    assert q0 == (5.0, 5.0, 2.0)
    assert q1 == (13.0, 13.0, 5.0)
    assert len(q_diag(0.5, CFG)) == 3


def test_d_safe_per_human_monotone():
    vals = [d_safe(p, CFG) for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
