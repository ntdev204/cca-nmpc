#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np


def load_normalization(stats_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(stats_path) as f:
        stats = json.load(f)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if mean.shape != (4,) or std.shape != (4,):
        raise ValueError("normalization stats must have 4 channels")
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def normalize_window(window: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    window = np.asarray(window, np.float32)
    if window.ndim != 2 or window.shape[1] != 4:
        raise ValueError("window must be (L, 4)")
    return (window - mean) / std


def denormalize(pred: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return pred * std + mean


class LSTMPredictorProtocol(Protocol):

    def predict(self, window: np.ndarray) -> np.ndarray:
        ...


class TensorRtLSTMPredictor:

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
        import pycuda.driver as cuda
        self._stream = cuda.Stream()

    @staticmethod
    def _load_engine(engine_path: str):
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                "tensorrt is not installed on this deployment device."
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
        import pycuda.driver as cuda

        window = np.asarray(window, dtype=np.float32)
        if window.ndim != 2 or window.shape[1] != 4:
            raise ValueError("window must be (L, 4)")
        origin = window[-1, :2].copy()
        local_window = window.copy()
        local_window[:, :2] -= origin
        norm = normalize_window(local_window, self._mean, self._std)
        blob = np.ascontiguousarray(norm[np.newaxis, :, :].astype(np.float32))

        input_name = self._engine.get_tensor_name(0)
        output_name = self._engine.get_tensor_name(1)
        self._context.set_input_shape(input_name, blob.shape)

        out_shape = self._context.get_tensor_shape(output_name)
        host_out = np.empty(out_shape, dtype=np.float32)

        d_in = cuda.mem_alloc(blob.nbytes)
        d_out = cuda.mem_alloc(host_out.nbytes)
        cuda.memcpy_htod_async(d_in, blob, self._stream)
        self._context.set_tensor_address(input_name, int(d_in))
        self._context.set_tensor_address(output_name, int(d_out))
        self._context.execute_async_v3(stream_handle=self._stream.handle)
        cuda.memcpy_dtoh_async(host_out, d_out, self._stream)
        self._stream.synchronize()

        pred = denormalize(host_out[0], self._mean, self._std)
        pred[:, :2] += origin
        return pred
