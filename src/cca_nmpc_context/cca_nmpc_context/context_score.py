#!/usr/bin/env python3
"""Continuous context score and index (Eqs. 8.1-8.3).

Pure Python, ROS-free. Implements the *corrected* uncertainty-aware context
score: low detection confidence or high predictive uncertainty must make the
robot MORE cautious (raise phi), not less — see Math Model Section 8.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextWeights:
    """Calibrated weights for the context score (Eq. 8.2), Section 8.1."""
    w_d: float
    w_v: float
    w_theta: float
    w_u: float
    b: float


def sigmoid(z: float) -> float:
    """Numerically stable logistic sigma(z) (Eq. 8.1)."""
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def uncertainty_term(confidence: float, sigma_h_tilde: float) -> float:
    """u_h = 1 - c*(1 - sigma_tilde) (Eq. 8.3).

    High when confidence ``c`` is low OR clipped predictive uncertainty
    ``sigma_h_tilde`` is high; low when both detection and prediction are
    reliable. ``sigma_h_tilde`` must already be clipped to [0, 1] (Eq. 8.3).
    """
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
    """Context score z (Eq. 8.2).

    z = w_d*(d0 - d_h)/d0 + w_v*|v_h|/v_max + w_theta*cos(dtheta)
        + w_u*u_h + b
    """
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
    """phi_j = sigma(z) in [0, 1] (Eqs. 8.1-8.2)."""
    z = context_score(
        d_h, v_h_speed, cos_dtheta, confidence, sigma_h_tilde,
        weights, d0, v_max_ref,
    )
    return sigmoid(z)
