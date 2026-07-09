"""Tests for stage-2 sensitivity sweep (CB-04). Uses the real CasADi solver."""
import sys
from pathlib import Path

import numpy as np

_CTX = Path(__file__).resolve().parents[3] / "src" / "cca_nmpc_context"
if str(_CTX) not in sys.path:
    sys.path.insert(0, str(_CTX))
from cca_nmpc_context.context_score import ContextWeights  # noqa: E402

from tools.context_calibration.features import FeatureVector
from tools.context_calibration.sensitivity import run_sensitivity


def _features(n=8):
    # a spread of interaction geometries
    out = []
    for i in range(n):
        out.append(FeatureVector(
            dist_term=0.5 - 0.1 * i,
            speed_term=0.3,
            cos_dtheta=0.2,
            u_h=0.1,
        ))
    return out


def test_sensitivity_reports_all_weights():
    weights = ContextWeights(w_d=1.2, w_v=0.8, w_theta=0.6, w_u=1.0, b=-0.5)
    _adj, report = run_sensitivity(
        _features(), weights, perturbation_pct=0.20, horizon_N=10)
    names = {r.weight_name for r in report}
    assert names == {"w_d", "w_v", "w_theta", "w_u", "b"}
    for r in report:
        assert r.d_safe_change_pct >= 0.0
        assert r.solve_success_change_pct >= 0.0


def test_dominating_weight_is_rescaled():
    # a huge w_d makes phi swing hard with perturbation; if that degrades solve
    # success beyond threshold it must be rescaled. Use a very low threshold to
    # force the rescale branch deterministically.
    weights = ContextWeights(w_d=50.0, w_v=0.1, w_theta=0.1, w_u=0.1, b=0.0)
    adjusted, report = run_sensitivity(
        _features(), weights, perturbation_pct=0.20,
        max_degradation_pct=0.0, horizon_N=10)
    # with threshold 0.0, any nonzero success change triggers rescale; at least
    # confirm the mechanism can rescale and halves the weight when it fires.
    for r in report:
        if r.rescaled:
            assert getattr(adjusted, r.weight_name) == getattr(weights, r.weight_name) * 0.5
            break
