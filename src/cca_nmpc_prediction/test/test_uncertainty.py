from cca_nmpc_prediction.uncertainty import UncertaintyEstimator


def test_variance_matches_reference():
    est = UncertaintyEstimator(window_W=10, beta=0.05, sigma_max=100.0)
    est.refresh(1, 0.0)
    sigma = est.refresh(1, 2.0)
    assert abs(sigma - 1.0) < 1e-9


def test_single_sample_zero_variance():
    est = UncertaintyEstimator(window_W=5, beta=0.05, sigma_max=100.0)
    assert est.refresh(1, 3.0) == 0.0


def test_growth_is_monotone_and_capped():
    est = UncertaintyEstimator(window_W=5, beta=0.5, sigma_max=1.0)
    est.refresh(1, 0.0)
    est.refresh(1, 0.4)
    base = est.sigma_h(1)
    s1 = est.age(1, dt_since_refresh=0.1)
    s2 = est.age(1, dt_since_refresh=0.5)
    s3 = est.age(1, dt_since_refresh=100.0)
    assert s1 >= base
    assert s2 >= s1
    assert s3 == 1.0


def test_refresh_resets_growth():
    est = UncertaintyEstimator(window_W=5, beta=1.0, sigma_max=10.0)
    est.refresh(1, 0.0)
    est.refresh(1, 1.0)
    est.age(1, dt_since_refresh=5.0)
    grown = est.sigma_h(1)
    fresh = est.refresh(1, 1.0)
    assert fresh <= grown


def test_sigma_tilde_clipped():
    est = UncertaintyEstimator(window_W=5, beta=0.05, sigma_max=2.0)
    est.refresh(1, 0.0)
    est.refresh(1, 4.0)
    assert est.sigma_tilde(1) == 1.0


def test_invalid_params():
    for bad in (dict(window_W=0, beta=0.1, sigma_max=1.0),
                dict(window_W=5, beta=0.0, sigma_max=1.0),
                dict(window_W=5, beta=0.1, sigma_max=0.0)):
        try:
            UncertaintyEstimator(**bad)
            assert False
        except ValueError:
            pass


def test_unknown_track_fail_safe():
    est = UncertaintyEstimator(window_W=5, beta=0.05, sigma_max=10.0)
    assert est.sigma_tilde(999) == 1.0
    assert est.sigma_h(999) == 10.0
    assert est.age(999, 1.0) == 10.0
