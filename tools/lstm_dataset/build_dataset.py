"""CLI tool to build train/val/test datasets from raw CSV (DS-06).

Usage:
    python -m tools.lstm_dataset.build_dataset --input data/raw.csv --output data/processed
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .loaders import load_csv
from .normalize import (
    compute_normalization_stats,
    save_normalization_stats,
)
from .resample import resample_trajectory
from .schema import TrajectoryKey
from .split import split_by_trajectory, split_by_group
from .windowing import extract_windows

DATASET_VERSION = "lstm_dataset_v1"


def compute_file_checksum(path: Path) -> str:
    """Compute SHA256 checksum of file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def build_dataset(
    input_csv: Path,
    output_dir: Path,
    L: int = 8,
    H: int = 12,
    dt: float = 0.125,
    seed: int = 42,
    velocity_rederive_threshold: float = 0.5,
    split_mode: str = "trajectory",
) -> Dict:
    """Build train/val/test datasets from raw CSV.

    Args:
        input_csv: Path to raw trajectory CSV.
        output_dir: Directory for output files.
        L: Input sequence length (timesteps).
        H: Prediction horizon (timesteps).
        dt: Target resampling period (seconds).
        seed: Random seed for deterministic splitting.
        velocity_rederive_threshold: Gap threshold for velocity re-derivation (s).
        split_mode: "trajectory" (default) or "subject". "trajectory" prevents
            window leakage; "subject" holds out entire subjects for unseen-human
            generalization claims.

    Returns:
        Manifest dict with file paths and metadata.
    """
    if L < 1 or H < 1:
        raise ValueError(f"L and H must be >= 1, got L={L}, H={H}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load trajectories
    print(f"[1/6] Loading trajectories from {input_csv}...")
    trajectories = load_csv(input_csv)
    print(f"      Loaded {len(trajectories)} trajectories.")

    # 2. Resample each track to fixed dt
    print(f"[2/6] Resampling to dt={dt}s...")
    resampled_tracks: Dict[TrajectoryKey, Tuple] = {}
    for key, records in trajectories.items():
        resampled, gaps = resample_trajectory(
            records, dt=dt,
            velocity_rederive_threshold=velocity_rederive_threshold,
        )
        resampled_tracks[key] = (resampled, gaps)
    print(f"      Resampled {len(resampled_tracks)} trajectories.")

    # 3. Extract sliding windows per track
    print(f"[3/6] Creating sliding windows (L={L}, H={H})...")
    track_windows: Dict[TrajectoryKey, Tuple[np.ndarray, np.ndarray]] = {}
    for key, (resampled, gaps) in resampled_tracks.items():
        if len(resampled) < L + H:
            continue
        inputs, targets = extract_windows(resampled, gaps, L=L, H=H)
        if len(inputs) > 0:
            track_windows[key] = (inputs, targets)

    if not track_windows:
        raise ValueError("No valid windows created. Check input data and L/H values.")

    total_windows = sum(len(v[0]) for v in track_windows.values())
    print(f"      Created {total_windows} windows from {len(track_windows)} trajectories.")

    # 4. Split by trajectory or subject
    if split_mode == "trajectory":
        print(f"[4/6] Splitting by trajectory (70/15/15, seed={seed})...")
        (train_in, train_tgt), (val_in, val_tgt), (test_in, test_tgt) = split_by_trajectory(
            track_windows, seed=seed
        )
    elif split_mode == "subject":
        print(f"[4/6] Splitting by subject (70/15/15, seed={seed})...")
        group_of = {
            key: records[0].subject_id
            for key, records in trajectories.items()
            if key in track_windows
        }
        # Reject unidentified trajectories: the synthetic "unknown" group merges
        # multiple distinct people into one pseudo-subject that can land in
        # train AND test, silently breaking the subject-held-out guarantee.
        unknown_keys = [k for k, s in group_of.items() if s == "unknown"]
        if unknown_keys:
            raise ValueError(
                f"--split-mode subject requires every trajectory to have a known "
                f"subject_id; found {len(unknown_keys)} trajectory(ies) with "
                f"subject_id='unknown'. Unidentified subjects cannot be held out "
                "and would invalidate the split. Set subject_id in CSV for all "
                "trajectories or use --split-mode trajectory."
            )
        known = set(group_of.values())
        if len(known) < 4:
            raise ValueError(
                f"--split-mode subject requires at least 4 distinct subject_id "
                f"values for non-empty train/val/test (70/15/15 floor allocation), "
                f"found: {sorted(known)}. "
                "Set subject_id in CSV or use --split-mode trajectory."
            )
        (train_in, train_tgt), (val_in, val_tgt), (test_in, test_tgt) = split_by_group(
            track_windows, group_of, seed=seed
        )
    else:
        raise ValueError(f"split_mode must be 'trajectory' or 'subject', got '{split_mode}'")
    print(f"      Train: {len(train_in)} windows")
    print(f"      Val:   {len(val_in)} windows")
    print(f"      Test:  {len(test_in)} windows")

    # 5. Compute normalization stats from train split only
    print("[5/6] Computing normalization stats from training split...")
    stats = compute_normalization_stats(train_in)
    stats_path = output_dir / "normalization_stats.json"
    save_normalization_stats(stats, stats_path)
    print(f"      Saved stats to {stats_path}")

    # 6. Save .npz files in RAW physical units (NOT normalized).
    # Normalization is applied exactly once, at read time, by
    # TrajectoryDataset using these frozen train-split stats. Storing raw
    # windows here keeps the stats the single source of truth and avoids the
    # silent double-normalization bug (builder normalizing + Dataset
    # normalizing again would train the LSTM on a (z-mean)/std scale and make
    # denormalize() wrong). See tools/lstm_training/dataset.py.
    print("[6/6] Writing .npz files (raw units; normalized once at read time)...")
    splits = {
        "train": (train_in, train_tgt),
        "val": (val_in, val_tgt),
        "test": (test_in, test_tgt),
    }

    manifest = {
        "version": DATASET_VERSION,
        "L": L,
        "H": H,
        "dt": dt,
        "seed": seed,
        "split_mode": split_mode,
        # Records are grouped and split by this composite identity. run_id is
        # part of the key so tracks from different physical runs that reuse a
        # reset tracker id + sequence_id are never merged (P1 dataset identity).
        "trajectory_key": ["session_id", "run_id", "sequence_id", "track_id"],
        "splits": {},
    }

    for split_name, (inp, tgt) in splits.items():
        npz_path = output_dir / f"{split_name}.npz"
        np.savez_compressed(npz_path, inputs=inp, targets=tgt)
        checksum = compute_file_checksum(npz_path)
        manifest["splits"][split_name] = {
            "path": str(npz_path),
            "num_windows": len(inp),
            "checksum": checksum,
        }
        print(f"      {split_name}.npz: shape inputs={inp.shape}, targets={tgt.shape} (checksum: {checksum})")

    manifest["normalization_stats"] = str(stats_path)
    return manifest


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build LSTM trajectory prediction dataset from raw CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to raw trajectory CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for processed datasets.",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=8,
        help="Input sequence length (timesteps).",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=12,
        help="Prediction horizon (timesteps).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.125,
        help="Target resampling period (seconds).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    parser.add_argument(
        "--velocity-rederive-threshold",
        type=float,
        default=0.5,
        help="Gap threshold for velocity re-derivation (seconds).",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="trajectory",
        choices=["trajectory", "subject"],
        help="Split mode: 'trajectory' (default, no window leakage) or 'subject' (hold out entire subjects).",
    )

    args = parser.parse_args()

    try:
        manifest = build_dataset(
            args.input,
            args.output,
            L=args.L,
            H=args.H,
            dt=args.dt,
            seed=args.seed,
            velocity_rederive_threshold=args.velocity_rederive_threshold,
            split_mode=args.split_mode,
        )
        print("\n=== Build Complete ===")
        print(f"Version: {manifest['version']}")
        print(f"Configuration: L={manifest['L']}, H={manifest['H']}, dt={manifest['dt']}s")
        print(f"Normalization stats: {manifest['normalization_stats']}")
        print("\nSplits:")
        for split_name, info in manifest["splits"].items():
            print(f"  {split_name}: {info['num_windows']} windows, checksum {info['checksum']}")
    except Exception as e:
        print(f"ERROR: {e}")
        raise

    return 0


if __name__ == "__main__":
    exit(main())
