import math

from cca_nmpc_context.relative_motion import (
    compute_relative_motion,
    robot_world_velocity,
)


def test_distance():
    rm = compute_relative_motion(0, 0, 0, 0, 0, 3, 4, 0, 0)
    assert abs(rm.d_h - 5.0) < 1e-9


def test_head_on_approach_positive_cos():
    rm = compute_relative_motion(
        robot_x=0, robot_y=0, robot_theta=0, robot_vx=0, robot_vy=0,
        human_x=2.0, human_y=0.0, human_vx=-1.0, human_vy=0.0,
    )
    assert rm.cos_dtheta > 0.99


def test_receding_negative_cos():
    rm = compute_relative_motion(
        robot_x=0, robot_y=0, robot_theta=0, robot_vx=0, robot_vy=0,
        human_x=2.0, human_y=0.0, human_vx=1.0, human_vy=0.0,
    )
    assert rm.cos_dtheta < -0.99


def test_crossing_near_zero_cos():
    rm = compute_relative_motion(
        robot_x=0, robot_y=0, robot_theta=0, robot_vx=0, robot_vy=0,
        human_x=2.0, human_y=0.0, human_vx=0.0, human_vy=1.0,
    )
    assert abs(rm.cos_dtheta) < 1e-6


def test_human_speed():
    rm = compute_relative_motion(0, 0, 0, 0, 0, 1, 1, 3.0, 4.0)
    assert abs(rm.v_h_speed - 5.0) < 1e-9


def test_robot_world_velocity_rotation():
    vx, vy = robot_world_velocity(1.0, 0.0, math.pi / 2)
    assert abs(vx) < 1e-9 and abs(vy - 1.0) < 1e-9


def test_epsilon_guards_zero_distance():
    rm = compute_relative_motion(1, 1, 0, 0, 0, 1, 1, 0, 0)
    assert rm.d_h == 0.0
    assert -1.0 <= rm.cos_dtheta <= 1.0


def test_epsilon_must_be_positive():
    try:
        compute_relative_motion(0, 0, 0, 0, 0, 1, 1, 0, 0, epsilon=0.0)
        assert False
    except ValueError:
        pass
