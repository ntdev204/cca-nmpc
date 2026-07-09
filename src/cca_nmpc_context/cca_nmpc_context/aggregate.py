#!/usr/bin/env python3
"""Multi-human aggregation and conservative dropout fallback (Eqs. 9.1-9.2).

Pure Python, ROS-free. The raw aggregate ``phi = max_j phi_j`` (Eq. 9.1) is for
diagnostics; the *gated* aggregate ``max_j phi_j_used`` is what adaptive_param_node
consumes for global caps / Q. ``d_h = min_j d_j`` (Eq. 9.2) is monitoring only.

On perception/prediction dropout the used aggregate falls back to a conservative
``phi = 1`` (maximally cautious) rather than holding a stale low value
(Architecture Section 4).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Aggregate:
    phi_aggregate: float          # max_j phi_j (raw, diagnostics) — Eq. 9.1
    phi_aggregate_used: float     # max_j phi_j_used (gated, consumed downstream)
    d_h_aggregate: float          # min_j d_j (monitoring) — Eq. 9.2


def aggregate_context(
    phi_j: list[float],
    phi_j_used: list[float],
    d_j: list[float],
    dropout: bool = False,
    fallback_phi: float = 1.0,
    no_human_d: float = float("inf"),
) -> Aggregate:
    """Aggregate per-human context indices.

    Args:
        phi_j: raw per-human phi (diagnostics aggregate).
        phi_j_used: gated per-human phi (downstream aggregate).
        d_j: per-human distances.
        dropout: if True, a required input source is missing/stale -> the used
            aggregate is forced to ``fallback_phi`` (conservative, Eq.-9 fix).
        fallback_phi: conservative phi on dropout (default 1.0).
        no_human_d: d_h_aggregate when no humans are tracked (default +inf).

    Empty (no humans) is NOT a dropout: with nobody tracked the situation is
    unconstrained, so the used aggregate is 0.0 unless ``dropout`` is set.
    """
    if dropout:
        # Distance is meaningless on dropout; report the freshest available min
        # if any, else the no-human sentinel.
        d_agg = min(d_j) if d_j else no_human_d
        return Aggregate(
            phi_aggregate=max(phi_j) if phi_j else fallback_phi,
            phi_aggregate_used=fallback_phi,
            d_h_aggregate=d_agg,
        )

    if not phi_j:
        return Aggregate(
            phi_aggregate=0.0,
            phi_aggregate_used=0.0,
            d_h_aggregate=no_human_d,
        )

    return Aggregate(
        phi_aggregate=max(phi_j),
        phi_aggregate_used=max(phi_j_used) if phi_j_used else 0.0,
        d_h_aggregate=min(d_j) if d_j else no_human_d,
    )
