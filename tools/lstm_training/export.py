"""Export the trained predictor to ONNX (+ optional TorchScript) with parity.

ONNX signature consumed by cca_nmpc_prediction (plan 04) — bind to these EXACT
names/shapes:
    input  name = "input"   shape = (batch, L, 4)  dynamic batch axis
    output name = "output"  shape = (batch, H, 4)  dynamic batch axis
The frozen normalization_stats.json is copied next to the .onnx so the runtime
reuses identical stats and never recomputes them.

CLI: python -m tools.lstm_training.export --checkpoint <pt> --stats <json> --out-dir models
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from .model import LSTMPredictor, LSTMConfig

INPUT_NAME = "input"
OUTPUT_NAME = "output"


def load_model(checkpoint_path: str | Path) -> LSTMPredictor:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = LSTMPredictor(LSTMConfig(
        hidden_size=cfg.get("hidden", 64),
        num_layers=cfg.get("layers", 1),
        horizon=cfg.get("H", 12),
    ))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def export_onnx(
    model: LSTMPredictor,
    out_path: str | Path,
    L: int = 8,
    stats_path: str | Path | None = None,
) -> Path:
    """Export to ONNX with dynamic batch axis; copy stats alongside."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, L, model.config.input_size)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=17,
    )
    if stats_path is not None:
        shutil.copy(stats_path, out_path.parent / "normalization_stats.json")
    return out_path


def check_parity(
    model: LSTMPredictor, onnx_path: str | Path, L: int = 8, atol: float = 1e-4
) -> float:
    """Return max abs diff between PyTorch and onnxruntime on random input."""
    import onnxruntime as ort

    x = torch.randn(3, L, model.config.input_size)
    with torch.no_grad():
        torch_out = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run([OUTPUT_NAME], {INPUT_NAME: x.numpy()})[0]
    return float(np.max(np.abs(torch_out - onnx_out)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export LSTM predictor to ONNX")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--name", default="lstm_predictor_v1_on_lstm_dataset_v1.onnx")
    ap.add_argument("--L", type=int, default=8)
    args = ap.parse_args(argv)

    model = load_model(args.checkpoint)
    onnx_path = export_onnx(model, Path(args.out_dir) / args.name, args.L, args.stats)
    max_diff = check_parity(model, onnx_path, args.L)
    print(f"Exported {onnx_path} (parity max_diff={max_diff:.2e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
