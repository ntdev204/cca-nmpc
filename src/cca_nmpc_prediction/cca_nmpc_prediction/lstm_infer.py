#!/usr/bin/env python3
"""TensorRT LSTM inference with frozen normalization (Section 3.2).

Runtime inference uses a YOLO-style TensorRT ``.engine`` (built on the target
GPU from the ONNX exported by tools.lstm_training — ONNX is only the build-time
intermediate, never loaded at runtime). TensorRT is imported lazily so this
module imports on machines without CUDA/TensorRT; unit tests use MockLSTMPredictor.

The engine's I/O signature mirrors the ONNX export: input (batch, L, 4), output
(batch, H, 4), channel order [x, y, vx, vy]. Normalization uses the FROZEN
train-split stats (never recomputed at runtime).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np


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


def normalize_window(window: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """z-score a single (L, 4) window with frozen stats."""
    window = np.asarray(window, np.float32)
    if window.ndim != 2 or window.shape[1] != 4:
        raise ValueError("window must be (L, 4)")
    return (window - mean) / std


def denormalize(pred: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Inverse z-score an (H, 4) prediction back to physical units."""
    return pred * std + mean


class LSTMPredictorProtocol(Protocol):
    """Predictor interface: (L,4) window -> (H,4) prediction in physical units."""

    def predict(self, window: np.ndarray) -> np.ndarray:
        ...


class MockLSTMPredictor:
    """Deterministic constant-velocity predictor — no TensorRT dependency.

    Extrapolates the last observed (x, y) using the last (vx, vy) over H steps
    at ``dt``. Used for Windows unit tests and non-GPU smoke runs, exactly like
    MockDetector in the perception package.
    """

    def __init__(self, horizon: int = 12, dt: float = 0.125) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self._H = horizon
        self._dt = dt

    def predict(self, window: np.ndarray) -> np.ndarray:
        window = np.asarray(window, np.float32)
        if window.ndim != 2 or window.shape[1] != 4:
            raise ValueError("window must be (L, 4)")
        x, y, vx, vy = window[-1]
        out = np.zeros((self._H, 4), dtype=np.float32)
        for k in range(1, self._H + 1):
            out[k - 1] = [x + vx * self._dt * k, y + vy * self._dt * k, vx, vy]
        return out


class TensorRtLSTMPredictor:
    """TensorRT ``.engine`` LSTM adapter with frozen normalization.

    TensorRT/pycuda are imported lazily so this class only requires a GPU at
    construction time. Raises a clear error on a missing/invalid engine before
    the node spins, mirroring TensorRtYoloDetector.
    """

    def __init__(
        self,
        engine_path: str | Path,
        stats_path: str | Path,
        horizon: int = 12,
    ) -> None:
        import os

        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        self._mean, self._std = load_normalization(stats_path)
        self._H = horizon
        self._engine, self._context = self._load_engine(str(engine_path))

    @staticmethod
    def _load_engine(engine_path: str):
        """Deserialize TensorRT engine and create an execution context."""
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                "tensorrt is not installed. Use MockLSTMPredictor on machines "
                "without GPU/TensorRT."
            ) from exc

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError(f"Failed to create execution context: {engine_path}")
        return engine, context

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Normalize -> TensorRT inference -> denormalize to (H, 4)."""
        import pycuda.driver as cuda

        norm = normalize_window(window, self._mean, self._std)
        blob = np.ascontiguousarray(norm[np.newaxis, :, :].astype(np.float32))

        input_name = self._engine.get_tensor_name(0)
        output_name = self._engine.get_tensor_name(1)
        self._context.set_input_shape(input_name, blob.shape)

        out_shape = self._context.get_tensor_shape(output_name)
        host_out = np.empty(out_shape, dtype=np.float32)

        d_in = cuda.mem_alloc(blob.nbytes)
        d_out = cuda.mem_alloc(host_out.nbytes)
        cuda.memcpy_htod(d_in, blob)
        self._context.set_tensor_address(input_name, int(d_in))
        self._context.set_tensor_address(output_name, int(d_out))
        self._context.execute_async_v3(stream_handle=cuda.Stream().handle)
        cuda.memcpy_dtoh(host_out, d_out)

        pred = host_out[0]                       # (H, 4)
        return denormalize(pred, self._mean, self._std)
