import numpy as np

from cca_nmpc_control.nmpc_solver import dummy_humans as dh


def _human(tid, n=6):
    return (tid, np.zeros(n + 1), np.zeros(n + 1), 0.5)


def test_fills_to_max_slots():
    slots = dh.fill_dummy_human_slots([_human(3)], max_slots=4, horizon_N=6)
    assert len(slots) == 4
    assert slots[0][0] == 3
    assert all(slots[i][0] == -1 for i in range(1, 4))


def test_dummy_slot_is_far_and_zero_phi():
    slots = dh.fill_dummy_human_slots([], max_slots=2, horizon_N=6)
    for tid, x_hat, y_hat, phi in slots:
        assert tid == -1
        assert np.all(x_hat >= dh.DUMMY_D_J)
        assert phi == dh.DUMMY_PHI_J


def test_too_many_humans_raises():
    try:
        dh.fill_dummy_human_slots([_human(1), _human(2)], max_slots=1, horizon_N=6)
        assert False, "should raise"
    except ValueError:
        pass


def test_build_d_safe_slots_matches_by_track_id():
    slots = dh.fill_dummy_human_slots([_human(7), _human(9)], max_slots=3, horizon_N=6)
    d_safe = dh.build_d_safe_slots({7: 1.2, 9: 0.8}, slots)
    assert d_safe[0] == 1.2
    assert d_safe[1] == 0.8
    assert d_safe[2] == dh.DUMMY_D_SAFE


def test_dummy_never_logged_as_near_miss():
    assert dh.should_log_as_near_miss(-1, 0.01) is False
    assert dh.should_log_as_near_miss(5, 0.5) is True
    assert dh.should_log_as_near_miss(5, 5.0) is False
