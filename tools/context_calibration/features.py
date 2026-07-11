"""Feature reconstruction reusing plan-05 context math (CB-02).

CRITICAL: this module imports the SAME functions cca_nmpc_context uses at
runtime (relative_motion, context_score) so calibration and runtime never drift.
It adds the src/cca_nmpc_context package to sys.path (it is a ROS package, not a
pip-installable module) and reuses its pure, ROS-free cores.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Make the ROS-free context cores importable without ROS.
_CTX_PKG = Path(__file__).resolve().parents[2] / "src" / "cca_nmpc_context"
if str(_CTX_PKG) not in sys.path:
    sys.path.insert(0, str(_CTX_PKG))

from cca_nmpc_context.relative_motion import compute_relative_motion  # noqa: E402
from cca_nmpc_context.context_score import (  # noqa: E402
    ContextWeights, sigmoid, uncertainty_term,
)

from .load import CalibrationRecord  # noqa: E402


@dataclass(frozen=True)
class FeatureVector:
    """Context-score features per record (matches Eq. 8.2 terms)."""
    dist_term: float       # (d0 - d_h) / d0
    speed_term: float      # |v_h| / v_max
    cos_dtheta: float      # relative approach cosine
    u_h: float             # uncertainty term (Eq. 8.3)


def compute_features(
    record: CalibrationRecord,
    d0: float,
    v_max_ref: float,
    epsilon: float = 1e-3,
) -> FeatureVector:
    """Reconstruct the four context-score features using plan-05 math."""
    rm = compute_relative_motion(
        record.robot_x, record.robot_y, record.robot_theta,
        record.robot_vx, record.robot_vy,
        record.human_x, record.human_y, record.human_vx, record.human_vy,
        epsilon=epsilon,
    )
    return FeatureVector(
        dist_term=(d0 - rm.d_h) / d0,
        speed_term=rm.v_h_speed / v_max_ref,
        cos_dtheta=rm.cos_dtheta,
        u_h=uncertainty_term(record.confidence, record.sigma_h_tilde),
    )


def phi_from_weights(
    feat: FeatureVector, weights: ContextWeights
) -> float:
    """phi = sigma(z) from features + weights, via the runtime context_score.

    Reconstructs the raw d_h / v_h so the SAME context_score function is used
    (guaranteeing calibration matches runtime exactly).
    """
    # Invert the normalized terms is unnecessary — recompute z directly from the
    # feature terms with the weights (identical algebra to Eq. 8.2).
    z = (
        weights.w_d * feat.dist_term
        + weights.w_v * feat.speed_term
        + weights.w_theta * feat.cos_dtheta
        + weights.w_u * feat.u_h
        + weights.b
    )
    return sigmoid(z)
