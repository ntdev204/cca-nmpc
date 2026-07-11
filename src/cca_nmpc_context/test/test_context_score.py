from cca_nmpc_context.context_score import (
    ContextWeights,
    sigmoid,
    uncertainty_term,
    context_index,
    context_score,
)

W = ContextWeights(w_d=1.2, w_v=0.8, w_theta=0.6, w_u=1.0, b=-0.5)


def test_sigmoid_bounds():
    assert abs(sigmoid(0.0) - 0.5) < 1e-9
    assert 0.0 <= sigmoid(-50) < 1e-9
    assert 0.999 < sigmoid(50) <= 1.0


def test_uncertainty_high_when_confidence_low():
    assert uncertainty_term(0.0, 0.0) == 1.0
    assert uncertainty_term(1.0, 0.0) == 0.0
    assert uncertainty_term(1.0, 1.0) == 1.0


def test_low_confidence_raises_phi():
    base = dict(d_h=2.0, v_h_speed=0.5, cos_dtheta=0.0, weights=W, d0=3.0, v_max_ref=1.5)
    phi_confident = context_index(confidence=1.0, sigma_h_tilde=0.0, **base)
    phi_uncertain = context_index(confidence=0.1, sigma_h_tilde=0.0, **base)
    assert phi_uncertain > phi_confident


def test_high_uncertainty_raises_phi():
    base = dict(d_h=2.0, v_h_speed=0.5, cos_dtheta=0.0, weights=W, d0=3.0, v_max_ref=1.5)
    phi_low_sigma = context_index(confidence=0.9, sigma_h_tilde=0.0, **base)
    phi_high_sigma = context_index(confidence=0.9, sigma_h_tilde=1.0, **base)
    assert phi_high_sigma > phi_low_sigma


def test_closer_human_raises_phi():
    base = dict(v_h_speed=0.5, cos_dtheta=0.0, confidence=0.9, sigma_h_tilde=0.0,
                weights=W, d0=3.0, v_max_ref=1.5)
    phi_far = context_index(d_h=2.9, **base)
    phi_near = context_index(d_h=0.2, **base)
    assert phi_near > phi_far


def test_phi_in_unit_interval():
    for d in (0.0, 1.0, 3.0, 10.0):
        phi = context_index(d, 2.0, 1.0, 0.5, 0.5, W, 3.0, 1.5)
        assert 0.0 <= phi <= 1.0


def test_invalid_d0_raises():
    try:
        context_score(1.0, 0.5, 0.0, 0.9, 0.0, W, d0=0.0, v_max_ref=1.5)
        assert False
    except ValueError:
        pass
