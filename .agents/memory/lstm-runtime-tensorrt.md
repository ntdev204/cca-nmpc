---
name: lstm-runtime-tensorrt
description: LSTM prediction runtime uses TensorRT .engine, not ONNX (user directive)
metadata:
  type: feedback
---

The LSTM predictor in cca_nmpc_prediction MUST load a TensorRT `.engine` at runtime, NOT onnxruntime/ONNX. User directive (goal): "sử dụng tensorRT .engine chứ không phải ONNX".

**Why:** Consistency with the YOLO perception detector (also TensorRT `.engine`), and TensorRT is the target-GPU deployment runtime. ONNX is only the portable build-time intermediate.

**How to apply:**
- Pipeline: PyTorch train → `export.py` ONNX → `build_engine.py` (ONNX→`.engine`, GPU/target-only, lazy `import tensorrt`) → `.engine` is the runtime artifact.
- `lstm_infer.py` provides `TensorRtLSTMPredictor` (lazy TRT+pycuda import, mirrors `TensorRtYoloDetector`), `MockLSTMPredictor` (constant-velocity, no TRT — Windows tests / non-GPU), and pure `normalize_window`/`denormalize`/`load_normalization`.
- `prediction_node`: param `engine_path` (was `model_path`), `use_mock_predictor` flag, `_build_predictor()` falls back to mock on ImportError/FileNotFoundError/RuntimeError (like perception `_build_detector`).
- Engine is NOT portable across GPU/TensorRT versions — never commit `.engine`; build on target. (.gitignore already excludes *.engine, *.onnx.)
- Windows tests never touch the GPU path; they cover Mock + normalization + missing-engine guard. See [[env-toolchain]], [[implementation-plan-state]].
