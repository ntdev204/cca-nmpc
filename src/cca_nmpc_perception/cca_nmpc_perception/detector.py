from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


class HumanDetector(Protocol):
    def detect(self, image: np.ndarray) -> Sequence[Detection]:
        ...


class TensorRtYoloDetector:
    _PERSON_CLASS_ID: int = 0

    def __init__(self, engine_path: str, confidence_threshold: float) -> None:
        import os

        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f'TensorRT engine not found: {engine_path}')

        self._confidence_threshold = confidence_threshold
        self._engine, self._context = self._load_engine(engine_path)
        import pycuda.driver as cuda
        self._stream = cuda.Stream()

    @staticmethod
    def _load_engine(engine_path: str):
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                'tensorrt package is not installed. '
                'Install the target TensorRT runtime on the deployment device.'
            ) from exc

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError(f'Failed to deserialize TensorRT engine: {engine_path}')

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError(f'Failed to create TensorRT execution context for: {engine_path}')

        return engine, context

    def detect(self, image: np.ndarray) -> list[Detection]:
        raw_outputs = self._run_inference(image)
        return self._parse_outputs(raw_outputs, image.shape)

    def _run_inference(self, image: np.ndarray) -> np.ndarray:
        import pycuda.driver as cuda
        import numpy as np

        input_h, input_w = self._get_input_shape()

        import cv2
        resized = cv2.resize(image, (input_w, input_h))
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.ascontiguousarray(blob[np.newaxis, :, :, :])

        input_binding = self._engine.get_tensor_name(0)
        output_binding = self._engine.get_tensor_name(1)

        self._context.set_input_shape(input_binding, blob.shape)

        output_shape = self._context.get_tensor_shape(output_binding)
        host_output = np.empty(output_shape, dtype=np.float32)

        d_input = cuda.mem_alloc(blob.nbytes)
        d_output = cuda.mem_alloc(host_output.nbytes)

        cuda.memcpy_htod_async(d_input, blob, self._stream)

        self._context.set_tensor_address(input_binding, int(d_input))
        self._context.set_tensor_address(output_binding, int(d_output))
        self._context.execute_async_v3(stream_handle=self._stream.handle)
        cuda.memcpy_dtoh_async(host_output, d_output, self._stream)
        self._stream.synchronize()
        return host_output

    def _get_input_shape(self) -> tuple[int, int]:
        name = self._engine.get_tensor_name(0)
        shape = self._engine.get_tensor_shape(name)
        return int(shape[2]), int(shape[3])

    def _parse_outputs(self, raw: np.ndarray, image_shape: tuple) -> list[Detection]:
        if raw.ndim == 3:
            raw = raw.squeeze(0)

        h, w = self._get_input_shape()
        detections: list[Detection] = []

        for row in raw:
            x1, y1, x2, y2, conf, cls = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
            if int(cls) != self._PERSON_CLASS_ID:
                continue
            if conf < self._confidence_threshold:
                continue
            detections.append(Detection(
                x1=x1 / w,
                y1=y1 / h,
                x2=x2 / w,
                y2=y2 / h,
                confidence=conf,
            ))

        return detections
