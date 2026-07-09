"""Build a TensorRT ``.engine`` from the exported ONNX (target GPU only).

Runtime inference in cca_nmpc_prediction consumes a TensorRT ``.engine``, not
ONNX. ONNX (from export.py) is the portable build-time intermediate; this tool
turns it into a GPU/TensorRT-version-specific engine that must be built ON the
target device (the engine is NOT portable across GPUs/TensorRT versions — same
constraint as the YOLO detector engine).

TensorRT is imported lazily so this module imports on machines without CUDA.

CLI:
    python -m tools.lstm_training.build_engine \
        --onnx models/lstm_predictor_v1_on_lstm_dataset_v1.onnx \
        --engine models/lstm_predictor_v1_on_lstm_dataset_v1.engine
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    fp16: bool = False,
    max_workspace_gb: float = 1.0,
) -> Path:
    """Compile an ONNX model into a serialized TensorRT engine.

    Raises ImportError if TensorRT is unavailable (i.e. not on the target GPU).
    """
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    try:
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover - GPU-only path
        raise ImportError(
            "tensorrt is not installed. Build the engine on the target GPU "
            "device where TensorRT is available."
        ) from exc

    logger = trt.Logger(trt.Logger.WARNING)  # pragma: no cover - GPU-only
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("Failed to parse ONNX:\n" + "\n".join(errs))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(max_workspace_gb * (1 << 30))
    )
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return engine_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--workspace-gb", type=float, default=1.0)
    args = ap.parse_args(argv)
    out = build_engine(args.onnx, args.engine, args.fp16, args.workspace_gb)
    print(f"Built TensorRT engine: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
