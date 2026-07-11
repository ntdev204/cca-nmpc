#!/usr/bin/env python3
"""Build TensorRT .engine for human detection from YOLO weights.

Usage:
    python3 build_engine.py --weights path/to/weights.pt --output models/yolo26m_human.engine

The output engine path defaults to models/yolo26m_human.engine to match
docs/config (yolo_engine_path). Pass the actual weights via --weights;
do not hard-code a model name here.

Requirements:
    - torch, ultralytics, tensorrt (pip install torch ultralytics)
    - CUDA-capable GPU + matching TensorRT runtime
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build YOLO TensorRT engine")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to YOLO weights file (.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/yolo26m_human.engine"),
        help="Output path for .engine file",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision (default: FP32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device (default: 0)",
    )
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"ERROR: weights file not found: {args.weights}", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics torch", file=sys.stderr)
        return 1

    try:
        import tensorrt as trt
        print(f"TensorRT version: {trt.__version__}")
    except ImportError:
        print("ERROR: tensorrt not installed. Install TensorRT runtime for your CUDA version.", file=sys.stderr)
        return 1

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO weights: {args.weights}")
    model = YOLO(str(args.weights))

    print(f"Exporting to TensorRT engine: {args.output}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  FP16: {args.fp16}")
    print(f"  Device: {args.device}")

    # Export to TensorRT
    # ultralytics export() API: https://docs.ultralytics.com/modes/export/
    model.export(
        format="engine",
        imgsz=args.imgsz,
        batch=args.batch,
        half=args.fp16,
        device=args.device,
        simplify=True,
        workspace=4,  # GB
    )

    # ultralytics saves engine next to the weights file with .engine suffix
    expected_output = args.weights.with_suffix(".engine")
    if expected_output.exists() and expected_output != args.output:
        import shutil
        shutil.move(str(expected_output), str(args.output))
        print(f"Moved engine to: {args.output}")
    elif args.output.exists():
        print(f"Engine created at: {args.output}")
    else:
        print(f"WARNING: Expected engine not found at {args.output}", file=sys.stderr)
        return 1

    print("\nEngine build complete.")
    print("Use this path in cca_nmpc_params.yaml:")
    print(f"  yolo_engine_path: \"{args.output}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
