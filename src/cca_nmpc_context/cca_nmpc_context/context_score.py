#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextWeights:
    w_d: float
    w_v: float
    w_theta: float
    w_u: float
    b: float


def sigmoid(z: float) -> float:
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def uncertainty_term(confidence: float, sigma_h_tilde: float) -> float:
    c = max(0.0, min(1.0, confidence))
    s = max(0.0, min(1.0, sigma_h_tilde))
    return 1.0 - c * (1.0 - s)


def context_score(
    d_h: float,
    v_h_speed: float,
    cos_dtheta: float,
    confidence: float,
    sigma_h_tilde: float,
    weights: ContextWeights,
    d0: float,
    v_max_ref: float,
) -> float:
    if d0 <= 0.0:
        raise ValueError("d0 must be positive")
    if v_max_ref <= 0.0:
        raise ValueError("v_max_ref must be positive")

    u_h = uncertainty_term(confidence, sigma_h_tilde)
    z = (
        weights.w_d * (d0 - d_h) / d0
        + weights.w_v * v_h_speed / v_max_ref
        + weights.w_theta * cos_dtheta
        + weights.w_u * u_h
        + weights.b
    )
    return z


def context_index(
    d_h: float,
    v_h_speed: float,
    cos_dtheta: float,
    confidence: float,
    sigma_h_tilde: float,
    weights: ContextWeights,
    d0: float,
    v_max_ref: float,
) -> float:
    z = context_score(
        d_h, v_h_speed, cos_dtheta, confidence, sigma_h_tilde,
        weights, d0, v_max_ref,
    )
    return sigmoid(z)
