"""Tests for warm-start invalidation detection (Section 5.2)."""
from cca_nmpc_control.invalidation import InvalidationDetector


def test_no_reset_on_normal_motion():
    det = InvalidationDetector(pos_thresh_m=0.30, yaw_thresh_rad=0.35)
    det.check((0.0, 0.0, 0.0), goal=(5.0, 0.0, 0.0))       # first cycle: seed
    ev = det.check((0.1, 0.0, 0.0), goal=(5.0, 0.0, 0.0))  # small move
    assert ev.any is False


def test_odom_position_jump_triggers_reset():
    det = InvalidationDetector(pos_thresh_m=0.30, yaw_thresh_rad=0.35)
    det.check((0.0, 0.0, 0.0), goal=(5.0, 0.0, 0.0))
    ev = det.check((1.0, 0.0, 0.0), goal=(5.0, 0.0, 0.0))  # 1m jump
    assert ev.odom_jump is True
    assert ev.any is True


def test_yaw_jump_triggers_reset():
    det = InvalidationDetector(pos_thresh_m=0.30, yaw_thresh_rad=0.35)
    det.check((0.0, 0.0, 0.0), goal=(5.0, 0.0, 0.0))
    ev = det.check((0.0, 0.0, 1.0), goal=(5.0, 0.0, 0.0))  # 1rad yaw jump
    assert ev.odom_jump is True


def test_goal_change_triggers_reset():
    det = InvalidationDetector()
    det.check((0.0, 0.0, 0.0), goal=(5.0, 0.0, 0.0))
    ev = det.check((0.0, 0.0, 0.0), goal=(9.0, 0.0, 0.0))
    assert ev.goal_changed is True
    assert ev.any is True


def test_lift_and_relocalize_flags():
    det = InvalidationDetector()
    det.check((0.0, 0.0, 0.0), goal=None)
    ev = det.check((0.0, 0.0, 0.0), goal=None, robot_lifted=True)
    assert ev.robot_lifted is True and ev.any is True
    ev2 = det.check((0.0, 0.0, 0.0), goal=None, map_relocalized=True)
    assert ev2.map_relocalized is True and ev2.any is True


def test_reset_history_clears_state():
    det = InvalidationDetector()
    det.check((0.0, 0.0, 0.0), goal=(1.0, 0.0, 0.0))
    det.reset_history()
    # after reset, first check seeds again -> no odom jump even if far
    ev = det.check((10.0, 0.0, 0.0), goal=(1.0, 0.0, 0.0))
    assert ev.odom_jump is False
