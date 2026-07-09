"""Tests for EMA + dwell-time gate (Eqs. 12.6-12.7)."""
from cca_nmpc_context.smoothing import PhiSmoother


def test_first_sample_seeds_both():
    sm = PhiSmoother(alpha=0.7, t_dwell_cycles=4)
    filt, used = sm.update(track_id=1, phi_raw=0.8)
    assert filt == 0.8 and used == 0.8


def test_ema_converges():
    sm = PhiSmoother(alpha=0.5, t_dwell_cycles=1)
    sm.update(1, 0.0)
    filt = 0.0
    for _ in range(30):
        filt, _used = sm.update(1, 1.0)
    assert filt > 0.99  # converges toward the constant input


def test_gate_holds_for_t_dwell_cycles():
    # small alpha -> filtered tracks raw quickly; gate still holds used value
    sm = PhiSmoother(alpha=0.01, t_dwell_cycles=3)
    sm.update(1, 0.2)          # seed -> used=0.2
    _f, u1 = sm.update(1, 0.9)  # cycle 1: held
    _f, u2 = sm.update(1, 0.9)  # cycle 2: held
    _f, u3 = sm.update(1, 0.9)  # cycle 3: gate fires -> update
    assert u1 == 0.2
    assert u2 == 0.2
    assert u3 > 0.2  # updated after T_dwell cycles


def test_state_is_per_track():
    sm = PhiSmoother(alpha=0.5, t_dwell_cycles=2)
    sm.update(1, 0.1)
    sm.update(2, 0.9)
    f1, _ = sm.update(1, 0.1)
    f2, _ = sm.update(2, 0.9)
    assert f1 < f2  # independent state


def test_drop_forgets_track():
    sm = PhiSmoother(alpha=0.5, t_dwell_cycles=2)
    sm.update(1, 0.5)
    sm.drop(1)
    assert 1 not in sm.active_tracks()


def test_invalid_params():
    for bad_alpha in (0.0, 1.0, -0.1):
        try:
            PhiSmoother(alpha=bad_alpha, t_dwell_cycles=3)
            assert False
        except ValueError:
            pass
    try:
        PhiSmoother(alpha=0.5, t_dwell_cycles=0)
        assert False
    except ValueError:
        pass
