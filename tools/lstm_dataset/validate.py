"""Validate raw trajectory CSV before it is accepted for dataset building."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    row: int
    detail: str


def validate_csv(path: str | Path, teleport_speed_mps: float = 4.0) -> list[ValidationIssue]:
    df = pd.read_csv(path)
    required = {
        "dataset_version", "session_id", "run_id", "sequence_id", "track_id",
        "scenario_id", "environment_id", "timestamp", "frame_id", "x", "y",
        "vx", "vy", "confidence", "is_observed",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        return [ValidationIssue("missing_columns", -1, ",".join(missing))]
    issues: list[ValidationIssue] = []
    bad_conf = df.index[~df["confidence"].between(0.0, 1.0)]
    issues.extend(ValidationIssue("invalid_confidence", int(i), str(df.at[i, "confidence"])) for i in bad_conf)
    if (~df["is_observed"].astype(bool)).any():
        for i in df.index[~df["is_observed"].astype(bool)]:
            issues.append(ValidationIssue("predicted_sample", int(i), "training rows must be observed"))
    keys = ["session_id", "run_id", "sequence_id", "track_id"]
    for _key, group in df.groupby(keys, sort=False):
        group = group.sort_values("timestamp")
        duplicated = group["timestamp"].duplicated(keep=False)
        for i in group.index[duplicated]:
            issues.append(ValidationIssue("duplicate_timestamp", int(i), str(group.at[i, "timestamp"])))
        dt = np.diff(group["timestamp"].to_numpy(float))
        distance = np.linalg.norm(np.diff(group[["x", "y"]].to_numpy(float), axis=0), axis=1)
        speed = np.divide(distance, dt, out=np.full_like(distance, np.inf), where=dt > 0)
        for offset in np.flatnonzero(speed > teleport_speed_mps):
            issues.append(ValidationIssue("teleport", int(group.index[offset + 1]), f"{speed[offset]:.3f} m/s"))
        if group["frame_id"].nunique() != 1:
            issues.append(ValidationIssue("frame_switch", int(group.index[0]), "frame_id changes within trajectory"))
    return issues


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--teleport-speed-mps", type=float, default=4.0)
    args = parser.parse_args(argv)
    issues = validate_csv(args.csv, args.teleport_speed_mps)
    for issue in issues:
        print(f"{issue.code}: row={issue.row}: {issue.detail}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
