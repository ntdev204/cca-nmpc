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
