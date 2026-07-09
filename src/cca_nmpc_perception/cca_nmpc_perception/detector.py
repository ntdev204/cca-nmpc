"""Detector interface, mock, and TensorRT adapter for YOLO human detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    """Single human detection from a YOLO model.

    Bounding box coordinates are normalized [0, 1] relative to image dimensions.
    """

    x1: float  # left (normalized)
    y1: float  # top (normalized)
    x2: float  # right (normalized)
    y2: float  # bottom (normalized)
    confidence: float  # detection confidence in [0, 1]

    @property
    def cx(self) -> float:
        """Horizontal center (normalized)."""
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        """Vertical center (normalized)."""
        return (self.y1 + self.y2) / 2.0


class HumanDetector(Protocol):
    """Detector interface: takes an RGB image array, returns human detections."""

    def detect(self, image: np.ndarray) -> Sequence[Detection]:
        """Detect humans in an BGR/RGB image.

        Args:
            image: HxWxC uint8 numpy array.

        Returns:
            Sequence of Detection objects (may be empty).
        """
        ...


class MockDetector:
    """Deterministic mock detector for unit tests — no TensorRT dependency.

    Returns a fixed list of detections configured at construction time.
    """

    def __init__(self, detections: Sequence[Detection] | None = None) -> None:
        self._detections: Sequence[Detection] = detections if detections is not None else []

    def detect(self, image: np.ndarray) -> Sequence[Detection]:  # noqa: ARG002
        return self._detections


class TensorRtYoloDetector:
    """YOLO26m TensorRT .engine adapter.

    TensorRT is imported lazily so the module can be imported in environments
    without CUDA/TensorRT installed (tests, CI without GPU).
    """

    _PERSON_CLASS_ID: int = 0  # COCO: person = class 0

    def __init__(self, engine_path: str, confidence_threshold: float) -> None:
        """Load TensorRT engine.

        Raises:
            FileNotFoundError: if engine_path does not exist.
            RuntimeError: if engine fails to deserialize or bind I/O tensors.
            ImportError: if tensorrt/pycuda packages are not installed.
        """
        import os

        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f'TensorRT engine not found: {engine_path}')

        self._confidence_threshold = confidence_threshold
        self._engine, self._context = self._load_engine(engine_path)

    @staticmethod
    def _load_engine(engine_path: str):
        """Deserialize TensorRT engine and create execution context."""
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                'tensorrt package is not installed. '
                'Use MockDetector for environments without GPU/TensorRT.'
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
        """Run YOLO inference and return human detections above confidence threshold."""
        raw_outputs = self._run_inference(image)
        return self._parse_outputs(raw_outputs, image.shape)

    def _run_inference(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image and run TensorRT inference.

        Returns raw YOLO output tensor (N, 6) in [x1, y1, x2, y2, conf, cls] format.
        """
        import pycuda.driver as cuda
        import numpy as np

        input_h, input_w = self._get_input_shape()

        # Preprocess: resize + normalize
        import cv2
        resized = cv2.resize(image, (input_w, input_h))
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC to CHW
        blob = np.ascontiguousarray(blob[np.newaxis, :, :, :])

        # Allocate host/device buffers and run inference
        # NOTE: full binding logic depends on engine I/O tensor names; this is
        # a representative implementation for YOLO26m with a single output tensor.
        input_binding = self._engine.get_tensor_name(0)
        output_binding = self._engine.get_tensor_name(1)

        self._context.set_input_shape(input_binding, blob.shape)

        output_shape = self._context.get_tensor_shape(output_binding)
        host_output = np.empty(output_shape, dtype=np.float32)

        d_input = cuda.mem_alloc(blob.nbytes)
        d_output = cuda.mem_alloc(host_output.nbytes)

        cuda.memcpy_htod(d_input, blob)

        self._context.set_tensor_address(input_binding, int(d_input))
        self._context.set_tensor_address(output_binding, int(d_output))
        self._context.execute_async_v3(stream_handle=cuda.Stream().handle)

        cuda.memcpy_dtoh(host_output, d_output)
        return host_output

    def _get_input_shape(self) -> tuple[int, int]:
        """Return (height, width) expected by engine's first input tensor."""
        name = self._engine.get_tensor_name(0)
        shape = self._engine.get_tensor_shape(name)
        # shape is (N, C, H, W)
        return int(shape[2]), int(shape[3])

    def _parse_outputs(self, raw: np.ndarray, image_shape: tuple) -> list[Detection]:
        """Filter by person class and confidence, normalize coordinates.

        raw format: (N, 6) where columns are [x1, y1, x2, y2, confidence, class_id].
        Coordinates assumed pixel-absolute relative to engine input size.
        """
        if raw.ndim == 3:
            # Some YOLO variants output (1, N, 6); squeeze batch dim
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
