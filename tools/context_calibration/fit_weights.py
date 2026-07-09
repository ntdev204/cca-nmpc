"""Stage-1 weight fit: logistic regression of phi to the danger proxy (CB-03).

Fits w_d, w_v, w_theta, w_u, b so that phi = sigma(z) matches the proxy danger
label. Pure NumPy gradient descent (no sklearn dependency) on the standard
logistic loss — the model IS the context score (Eq. 8.2), so the fitted
coefficients are exactly the runtime weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CTX_PKG = Path(__file__).resolve().parents[2] / "src" / "cca_nmpc_context"
if str(_CTX_PKG) not in sys.path:
    sys.path.insert(0, str(_CTX_PKG))
from cca_nmpc_context.context_score import ContextWeights  # noqa: E402

from .features import FeatureVector


def _design_matrix(features: list[FeatureVector]) -> np.ndarray:
    """(N, 5) matrix [dist, speed, cos, u_h, 1] (bias column last)."""
    return np.array([
        [f.dist_term, f.speed_term, f.cos_dtheta, f.u_h, 1.0]
        for f in features
    ], dtype=float)


def fit_weights(
    features: list[FeatureVector],
    labels: list[float],
    lr: float = 0.1,
    epochs: int = 2000,
    l2: float = 1e-4,
    seed: int = 0,
) -> ContextWeights:
    """Logistic-regression fit of the context weights to the danger proxy."""
    if len(features) != len(labels):
        raise ValueError("features and labels must align")
    if not features:
        raise ValueError("no features to fit")

    X = _design_matrix(features)
    y = np.asarray(labels, float)
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=X.shape[1])

    n = X.shape[0]
    for _ in range(epochs):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (p - y) / n + l2 * w
        w -= lr * grad

    return ContextWeights(
        w_d=float(w[0]), w_v=float(w[1]), w_theta=float(w[2]),
        w_u=float(w[3]), b=float(w[4]),
    )


def classification_accuracy(
    features: list[FeatureVector],
    labels: list[float],
    weights: ContextWeights,
    threshold: float = 0.5,
) -> float:
    """Fraction of records whose phi crosses ``threshold`` matching the label."""
    from .features import phi_from_weights

    correct = 0
    for f, y in zip(features, labels):
        pred = 1.0 if phi_from_weights(f, weights) >= threshold else 0.0
        correct += int(pred == y)
    return correct / len(labels)
