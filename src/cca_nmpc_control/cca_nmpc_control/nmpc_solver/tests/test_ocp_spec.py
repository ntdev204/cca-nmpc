import math

import numpy as np

from cca_nmpc_control.nmpc_solver import ocp_spec


def test_dynamics_match_eq_4_1_at_sample():
    spec = ocp_spec.OCPSpec(dt=0.05)
    x = np.array([0.0, 0.0, 0.0])
    u = np.array([1.0, 0.5, 0.2])
    x_next = spec.integrate_numpy(x, u)
    assert x_next[0] > 0 and x_next[1] > 0 and x_next[2] > 0
    assert x_next[0] == np.clip(x_next[0], 0.049, 0.051)


def test_dynamics_rotation_at_theta_90():
    spec = ocp_spec.OCPSpec(dt=0.05)
    x = np.array([0.0, 0.0, math.pi / 2])
    u = np.array([1.0, 0.0, 0.0])
    x_next = spec.integrate_numpy(x, u)
    assert x_next[1] > 0
    assert abs(x_next[0]) < 1e-6


def test_dimensions():
    assert ocp_spec.NX == 3
    assert ocp_spec.NU == 3


def test_integrator_builds():
    spec = ocp_spec.OCPSpec(dt=0.1)
    assert spec.f_discrete is not None


def _hinge_cost(x_robot, x_h, y_h, phi, d0, w_h, beta):
    import casadi as ca
    xr = ca.MX.sym("xr", 3)
    expr = ocp_spec.build_human_hinge_cost_stage(
        xr, x_h, y_h, phi, d0, w_h, hinge_beta=beta
    )
    fn = ca.Function("h", [xr], [expr])
    return float(fn(np.array(x_robot)))


def test_hinge_default_is_exact_math_model():
    # Default (beta<=0) must equal w_h*phi*max(0, d0-d_j)^2 exactly.
    # Robot at origin, human at (1,0): d_j=1, d0=3 -> hinge=2 -> 3*1*4=12.
    cost = _hinge_cost([0.0, 0.0, 0.0], 1.0, 0.0, phi=1.0, d0=3.0, w_h=3.0, beta=0.0)
    assert abs(cost - 12.0) < 1e-4


def test_hinge_zero_when_beyond_d0():
    # d_j=5 > d0=3 -> exact hinge is exactly zero.
    cost = _hinge_cost([5.0, 0.0, 0.0], 0.0, 0.0, phi=1.0, d0=3.0, w_h=3.0, beta=0.0)
    assert abs(cost) < 1e-6


def test_softplus_hinge_approximates_exact_and_stays_positive():
    # Large beta -> softplus ~ exact hinge inside the active region.
    exact = _hinge_cost([0.0, 0.0, 0.0], 1.0, 0.0, phi=1.0, d0=3.0, w_h=3.0, beta=0.0)
    smooth = _hinge_cost([0.0, 0.0, 0.0], 1.0, 0.0, phi=1.0, d0=3.0, w_h=3.0, beta=30.0)
    assert abs(smooth - exact) < 0.2
    # Softplus is strictly positive even beyond d0 (smooth tail), unlike exact 0.
    tail = _hinge_cost([5.0, 0.0, 0.0], 0.0, 0.0, phi=1.0, d0=3.0, w_h=3.0, beta=30.0)
    assert tail >= 0.0
