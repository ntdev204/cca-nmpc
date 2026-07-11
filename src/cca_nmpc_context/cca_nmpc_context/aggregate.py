#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Aggregate:
    phi_aggregate: float
    phi_aggregate_used: float
    d_h_aggregate: float


def aggregate_context(
    phi_j: list[float],
    phi_j_used: list[float],
    d_j: list[float],
    dropout: bool = False,
    fallback_phi: float = 1.0,
    no_human_d: float = float("inf"),
) -> Aggregate:
    if dropout:
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
