#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    """Return the SHA256 hex digest of a file for provenance tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


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
        default=None,
        help=(
            "Output path for .engine file. Default: models/<weights-stem>.engine, "
            "derived from the weights filename so the engine name never implies a "
            "detector version that does not match the supplied weights."
        ),
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

    # Derive the engine name from the weights filename when not given explicitly,
    # so a YOLOv8/YOLO11 checkpoint never lands in a file named like YOLO26m.
    if args.output is None:
        args.output = Path("models") / f"{args.weights.stem}.engine"

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

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO weights: {args.weights}")
    model = YOLO(str(args.weights))

    print(f"Exporting to TensorRT engine: {args.output}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  FP16: {args.fp16}")
    print(f"  Device: {args.device}")

    model.export(
        format="engine",
        imgsz=args.imgsz,
        batch=args.batch,
        half=args.fp16,
        device=args.device,
        simplify=True,
        workspace=4,
    )

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

    _write_provenance(args)

    print("\nEngine build complete.")
    print("Use this path in cca_nmpc_params.yaml:")
    print(f"  yolo_engine_path: \"{args.output}\"")
    return 0


def _write_provenance(args) -> None:
    """Write a <engine>.metadata.json sidecar recording detector provenance.

    Reproducibility: the engine file itself carries no reliable record of which
    weights/version produced it, so the paper cannot state the detector version
    from the engine alone. This sidecar pins weights path + SHA256, Ultralytics
    version, imgsz, precision, batch, device, and build time next to the engine.
    """
    try:
        import ultralytics
        ultralytics_version = ultralytics.__version__
    except Exception:
        ultralytics_version = "unknown"
    try:
        import tensorrt as trt
        trt_version = trt.__version__
    except Exception:
        trt_version = "unknown"

    metadata = {
        "engine": str(args.output),
        "weights": str(args.weights),
        "weights_sha256": _sha256(args.weights),
        "ultralytics_version": ultralytics_version,
        "tensorrt_version": trt_version,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "precision": "fp16" if args.fp16 else "fp32",
        "device": args.device,
        "built_utc": datetime.now(timezone.utc).isoformat(),
    }
    sidecar = args.output.with_suffix(".metadata.json")
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote provenance metadata: {sidecar}")


if __name__ == "__main__":
    sys.exit(main())
