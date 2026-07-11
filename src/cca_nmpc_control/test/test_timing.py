from cca_nmpc_control.timing import build_cycle_timing


def test_total_defaults_to_sum():
    t = build_cycle_timing(0.5, 8.0, 0.2, dt_s=0.05)
    assert abs(t.total_cycle_ms - 8.7) < 1e-9
    assert abs(t.accounted_ms - 8.7) < 1e-9


def test_under_budget_not_flagged():
    t = build_cycle_timing(0.5, 8.0, 0.2, dt_s=0.05)
    assert t.over_budget is False


def test_over_budget_flagged():
    t = build_cycle_timing(10.0, 50.0, 5.0, dt_s=0.05)
    assert t.over_budget is True


def test_explicit_total_used():
    t = build_cycle_timing(0.5, 8.0, 0.2, dt_s=0.05, total_cycle_ms=12.0)
    assert t.total_cycle_ms == 12.0
    assert abs(t.accounted_ms - 8.7) < 1e-9


def test_invalid_dt():
    try:
        build_cycle_timing(1, 1, 1, dt_s=0.0)
        assert False
    except ValueError:
        pass
