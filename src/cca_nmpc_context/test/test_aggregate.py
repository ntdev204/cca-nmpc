"""Tests for multi-human aggregation + dropout fallback (Eqs. 9.1-9.2)."""
import math

from cca_nmpc_context.aggregate import aggregate_context


def test_max_min_aggregation():
    agg = aggregate_context(
        phi_j=[0.2, 0.9, 0.5],
        phi_j_used=[0.1, 0.8, 0.4],
        d_j=[3.0, 1.0, 2.0],
    )
    assert agg.phi_aggregate == 0.9          # max raw (Eq. 9.1)
    assert agg.phi_aggregate_used == 0.8      # max gated
    assert agg.d_h_aggregate == 1.0           # min distance (Eq. 9.2)


def test_raw_and_used_independent():
    # used aggregate must come from phi_j_used, not phi_j
    agg = aggregate_context(phi_j=[0.9], phi_j_used=[0.3], d_j=[2.0])
    assert agg.phi_aggregate == 0.9
    assert agg.phi_aggregate_used == 0.3


def test_no_humans_is_not_dropout():
    agg = aggregate_context(phi_j=[], phi_j_used=[], d_j=[])
    assert agg.phi_aggregate_used == 0.0
    assert math.isinf(agg.d_h_aggregate)


def test_dropout_forces_conservative_phi():
    # even with a stale low value present, dropout -> used = 1.0
    agg = aggregate_context(
        phi_j=[0.1], phi_j_used=[0.1], d_j=[5.0], dropout=True,
    )
    assert agg.phi_aggregate_used == 1.0


def test_dropout_custom_fallback():
    agg = aggregate_context(
        phi_j=[], phi_j_used=[], d_j=[], dropout=True, fallback_phi=0.95,
    )
    assert agg.phi_aggregate_used == 0.95
