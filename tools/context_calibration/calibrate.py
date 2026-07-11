"""Calibration CLI: stage1 fit -> stage2 sensitivity -> report (CB-05).

    python -m tools.context_calibration.calibrate --csv <path> [--d0 3.0 ...]
"""
from __future__ import annotations

import argparse

from .load import load_calibration_csv, danger_label
from .features import compute_features
from .fit_weights import fit_weights, classification_accuracy
from .sensitivity import run_sensitivity
from .report import weights_yaml_block, sensitivity_table


def run_calibration(
    csv_path: str,
    d0: float = 3.0,
    v_max_ref: float = 1.5,
    danger_distance: float = 1.0,
    perturbation_pct: float = 0.20,
    max_degradation_pct: float = 10.0,
) -> dict:
    """Full two-stage calibration; returns fitted weights + reports."""
    records = load_calibration_csv(csv_path)
    features = [compute_features(r, d0, v_max_ref) for r in records]
    labels = [danger_label(r, danger_distance) for r in records]

    if len(features) < 5:
        raise ValueError("calibration requires at least 5 samples for held-out validation")
    indices = list(range(len(features)))
    split = max(1, int(0.8 * len(indices)))
    train_idx, val_idx = indices[:split], indices[split:]
    if not val_idx:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    weights = fit_weights([features[i] for i in train_idx], [labels[i] for i in train_idx])
    train_acc = classification_accuracy(
        [features[i] for i in train_idx], [labels[i] for i in train_idx], weights)
    val_acc = classification_accuracy(
        [features[i] for i in val_idx], [labels[i] for i in val_idx], weights)
    adjusted, sens = run_sensitivity(
        features, weights, perturbation_pct, max_degradation_pct)

    return {
        "weights": weights,
        "adjusted_weights": adjusted,
        "train_accuracy": train_acc,
        "validation_accuracy": val_acc,
        "sensitivity": sens,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CCA-NMPC context weight calibration")
    ap.add_argument("--csv", required=True, help="calibration dataset CSV")
    ap.add_argument("--d0", type=float, default=3.0)
    ap.add_argument("--v-max-ref", type=float, default=1.5)
    ap.add_argument("--danger-distance", type=float, default=1.0)
    ap.add_argument("--perturbation-pct", type=float, default=0.20)
    ap.add_argument("--max-degradation-pct", type=float, default=10.0)
    args = ap.parse_args(argv)

    out = run_calibration(
        args.csv, args.d0, args.v_max_ref, args.danger_distance,
        args.perturbation_pct, args.max_degradation_pct,
    )
    print(f"# train accuracy: {out['train_accuracy']:.3f}\n")
    print(f"# validation accuracy: {out['validation_accuracy']:.3f}\n")
    print(weights_yaml_block(out["adjusted_weights"]))
    print("\n# sensitivity report\n")
    print(sensitivity_table(out["sensitivity"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
