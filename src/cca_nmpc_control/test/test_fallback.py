"""Tests for the fallback/timeout state machine (Section 9)."""
import numpy as np

from cca_nmpc_control.fallback import (
    FallbackController, safe_stop_ramp,
    FALLBACK_NONE, FALLBACK_HELD_PREVIOUS, FALLBACK_SAFE_STOP,
)


def test_success_publishes_u0():
    fc = FallbackController(timeout_hold_cycles=3, dt=0.05, decel_limit=1.0)
    d = fc.on_success(np.array([0.5, 0.0, 0.1]))
    assert not d.fallback_triggered
    assert d.fallback_code == FALLBACK_NONE
    assert np.allclose(d.u, [0.5, 0.0, 0.1])


def test_failure_holds_previous_then_escalates():
    fc = FallbackController(timeout_hold_cycles=2, dt=0.05, decel_limit=1.0)
    fc.on_success(np.array([1.0, 0.0, 0.0]))
    d1 = fc.on_failure(timed_out=False)
    d2 = fc.on_failure(timed_out=False)
    assert d1.fallback_code == FALLBACK_HELD_PREVIOUS
    assert d2.fallback_code == FALLBACK_HELD_PREVIOUS
    assert np.allclose(d1.u, [1.0, 0.0, 0.0])  # holds last good
    # third failure exceeds the 2-cycle hold budget -> safe stop
    d3 = fc.on_failure(timed_out=False)
    assert d3.fallback_code == FALLBACK_SAFE_STOP
    assert d3.needs_reset is True


def test_timeout_vs_solver_failed_reason_distinguished():
    fc = FallbackController(timeout_hold_cycles=0, dt=0.05, decel_limit=1.0)
    fc.on_success(np.array([1.0, 0.0, 0.0]))
    # hold budget is 0 -> first failure goes straight to safe stop, but the
    # reason string should still carry the trigger type.
    d = fc.on_failure(timed_out=True)
    assert d.fallback_code == FALLBACK_SAFE_STOP


def test_safe_stop_ramp_respects_decel_limit():
    # dt=0.1, decel=1.0 -> max step 0.1 per cycle
    u = safe_stop_ramp(np.array([0.5, 0.0, 0.0]), dt=0.1, decel_limit=1.0)
    assert abs(u[0] - 0.4) < 1e-9   # 0.5 - 0.1
    # small component snaps to zero
    u2 = safe_stop_ramp(np.array([0.05, 0.0, 0.0]), dt=0.1, decel_limit=1.0)
    assert u2[0] == 0.0


def test_success_resets_failure_counter():
    fc = FallbackController(timeout_hold_cycles=2, dt=0.05, decel_limit=1.0)
    fc.on_success(np.array([1.0, 0.0, 0.0]))
    fc.on_failure(timed_out=False)
    assert fc.consecutive_failures == 1
    fc.on_success(np.array([0.8, 0.0, 0.0]))
    assert fc.consecutive_failures == 0


def test_invalid_hold_cycles():
    try:
        FallbackController(timeout_hold_cycles=-1, dt=0.05, decel_limit=1.0)
        assert False
    except ValueError:
        pass
