#!/usr/bin/env python3
"""ONNX LSTM inference wrapper with frozen normalization (Section 3.2).

Pure Python (onnxruntime + numpy), ROS-free. Normalizes the (L,4) input window
with the FROZEN train-split stats, runs the ONNX model, and denormalizes the
(H,4) output back to physical units. Binds to the exact ONNX signature exported
by tools.lstm_training.export: input "input" (batch,L,4), output "output"
(batch,H,4). Stats are never recomputed at runtime.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

INPUT_NAME = "input"
OUTPUT_NAME = "output"


def load_normalization(stats_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load frozen (mean, std) float32 arrays of length 4."""
    with open(stats_path) as f:
        stats = json.load(f)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if mean.shape != (4,) or std.shape != (4,):
        raise ValueError("normalization stats must have 4 channels")
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


class LSTMPredictor:
    """ONNX-backed trajectory predictor with frozen normalization."""

    def __init__(self, model_path: str | Path, stats_path: str | Path):
        import onnxruntime as ort

        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self._mean, self._std = load_normalization(stats_path)
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Run inference on a single (L,4) window -> (H,4) physical units."""
        window = np.asarray(window, np.float32)
        if window.ndim != 2 or window.shape[1] != 4:
            raise ValueError("window must be (L, 4)")
        norm = (window - self._mean) / self._std
        batched = norm[None, :, :].astype(np.float32)   # (1, L, 4)
        out = self._session.run([OUTPUT_NAME], {INPUT_NAME: batched})[0]
        pred = out[0]                                    # (H, 4)
        return pred * self._std + self._mean             # denormalize
