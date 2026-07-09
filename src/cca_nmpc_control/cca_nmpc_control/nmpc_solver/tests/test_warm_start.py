"""Tests for warm-start shift-and-append and invalidation (SV-05)."""
import math

import numpy as np

from cca_nmpc_control.nmpc_solver import warm_start as ws
from cca_nmpc_control.nmpc_solver import ocp_spec


def test_shift_and_append_drops_first_dupes_last():
    u = np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
    shifted = ws.shift_and_append_controls(u)
    assert np.allclose(shifted[0], [2.0, 0, 0])
    assert np.allclose(shifted[-1], [3.0, 0, 0])
    assert shifted.shape == u.shape


def test_repropagate_states_dynamically_consistent():
    spec = ocp_spec.OCPSpec(dt=0.05)
    x0 = np.array([0.0, 0.0, 0.0])
    u = np.tile([1.0, 0.0, 0.0], (4, 1))
    traj = ws.repropagate_states(x0, u, spec.integrate_numpy)
    assert traj.shape == (5, 3)
    assert traj[-1, 0] > traj[0, 0]  # moved forward


def test_odom_jump_detection():
    th = ws.InvalidationThresholds(0.30, 0.35)
    assert ws.detect_odom_jump((0, 0, 0), (0.5, 0, 0), th) is True
    assert ws.detect_odom_jump((0, 0, 0), (0.1, 0, 0), th) is False
    assert ws.detect_odom_jump((0, 0, 0), (0, 0, 0.5), th) is True


def test_goal_change_detection():
    assert ws.detect_goal_change((1, 1, 0), (1, 1, 0)) is False
    assert ws.detect_goal_change((1, 1, 0), (2, 1, 0)) is True
    assert ws.detect_goal_change(None, (1, 1, 0)) is True


def test_should_reset_aggregates_triggers():
    assert ws.should_reset() is False
    assert ws.should_reset(odom_jump=True) is True
    assert ws.should_reset(goal_changed=True) is True


def test_wrap_angle():
    assert abs(ws.wrap_angle(3 * math.pi)) - math.pi < 1e-9
