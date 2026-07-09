"""Context-calibration dataset loader (CB-01).

Reads the Section 3.2 schema: robot pose, human states, min_future_distance,
evasive_reaction_flag. CSV-first (one row per robot-human pair per timestep),
pure Python + numpy. The proxy danger label combines a close min-future-distance
with an observed evasive reaction (standard social-nav proxy, Math Model 8.1).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CalibrationRecord:
    """One robot-human interaction sample with a post-hoc danger proxy."""
    robot_x: float
    robot_y: float
    robot_theta: float
    robot_vx: float
    robot_vy: float
    human_x: float
    human_y: float
    human_vx: float
    human_vy: float
    confidence: float
    sigma_h_tilde: float
    min_future_distance: float
    evasive_reaction_flag: bool


_REQUIRED = [
    "robot_x", "robot_y", "robot_theta", "robot_vx", "robot_vy",
    "human_x", "human_y", "human_vx", "human_vy", "confidence",
    "sigma_h_tilde", "min_future_distance", "evasive_reaction_flag",
]


def load_calibration_csv(path: str | Path) -> list[CalibrationRecord]:
    """Load calibration records from a CSV with the Section 3.2 columns."""
    records: list[CalibrationRecord] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in _REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        for row in reader:
            records.append(CalibrationRecord(
                robot_x=float(row["robot_x"]),
                robot_y=float(row["robot_y"]),
                robot_theta=float(row["robot_theta"]),
                robot_vx=float(row["robot_vx"]),
                robot_vy=float(row["robot_vy"]),
                human_x=float(row["human_x"]),
                human_y=float(row["human_y"]),
                human_vx=float(row["human_vx"]),
                human_vy=float(row["human_vy"]),
                confidence=float(row["confidence"]),
                sigma_h_tilde=float(row["sigma_h_tilde"]),
                min_future_distance=float(row["min_future_distance"]),
                evasive_reaction_flag=_parse_bool(row["evasive_reaction_flag"]),
            ))
    if not records:
        raise ValueError("no records loaded from calibration CSV")
    return records


def danger_label(
    record: CalibrationRecord, danger_distance: float = 1.0
) -> float:
    """Proxy danger label in {0, 1} (Math Model Section 8.1).

    Dangerous if the human came within ``danger_distance`` in the near future OR
    an evasive reaction was observed.
    """
    close = record.min_future_distance < danger_distance
    return 1.0 if (close or record.evasive_reaction_flag) else 0.0


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "t")
