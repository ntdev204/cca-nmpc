"""Calibration report: weights YAML block + sensitivity table (CB-05)."""
from __future__ import annotations

import sys
from pathlib import Path

_CTX = Path(__file__).resolve().parents[2] / "src" / "cca_nmpc_context"
if str(_CTX) not in sys.path:
    sys.path.insert(0, str(_CTX))
from cca_nmpc_context.context_score import ContextWeights  # noqa: E402

from .sensitivity import WeightSensitivity


def weights_yaml_block(weights: ContextWeights) -> str:
    """Emit a ready-to-paste context_node.weights YAML block."""
    return (
        "context_node:\n"
        "  ros__parameters:\n"
        "    weights:  # calibrated (Section 8.1) — do not hand-edit\n"
        f"      w_d: {weights.w_d:.6f}\n"
        f"      w_v: {weights.w_v:.6f}\n"
        f"      w_theta: {weights.w_theta:.6f}\n"
        f"      w_u: {weights.w_u:.6f}\n"
        f"      b: {weights.b:.6f}\n"
    )


def sensitivity_table(report: list[WeightSensitivity]) -> str:
    """Human-readable sensitivity table."""
    lines = [
        "weight   d_safe%   vx_max%   solve_success%   rescaled",
        "-------  --------  --------  ---------------  --------",
    ]
    for r in report:
        lines.append(
            f"{r.weight_name:<7}  {r.d_safe_change_pct:7.2f}  "
            f"{r.vx_max_change_pct:7.2f}  {r.solve_success_change_pct:14.2f}  "
            f"{'yes' if r.rescaled else 'no'}"
        )
    return "\n".join(lines)
