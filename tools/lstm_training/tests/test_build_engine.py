"""Tests for the ONNX->TensorRT engine builder (import + guards only).

The actual engine build is GPU/TensorRT-only and runs on the target device, so
here we only verify the module imports cleanly without TensorRT and that its
input guards fire (missing ONNX -> FileNotFoundError; no TensorRT -> ImportError).
"""
import importlib

import pytest


def test_module_imports_without_tensorrt():
    mod = importlib.import_module("tools.lstm_training.build_engine")
    assert hasattr(mod, "build_engine")


def test_missing_onnx_raises(tmp_path):
    from tools.lstm_training.build_engine import build_engine
    with pytest.raises(FileNotFoundError):
        build_engine(tmp_path / "nope.onnx", tmp_path / "out.engine")


def test_build_raises_without_tensorrt(tmp_path):
    # A present (dummy) ONNX file but no TensorRT installed -> ImportError.
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"not a real onnx, but the file exists")
    from tools.lstm_training.build_engine import build_engine
    try:
        import tensorrt  # noqa: F401
        pytest.skip("tensorrt is installed; skipping the no-TRT guard test")
    except ImportError:
        with pytest.raises(ImportError):
            build_engine(onnx, tmp_path / "out.engine")
