"""CSV loader and optional rosbag2 loader for trajectory records.

Rosbag2 support is behind an import guard (DS-07): if the rosbags
library is not installed this module still imports cleanly.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .schema import REQUIRED_COLUMNS, TrajectoryRecord


def load_csv(path: str | Path) -> Dict[int, List[TrajectoryRecord]]:
    """Load trajectory records from a CSV file.

    The CSV must have at minimum the columns listed in REQUIRED_COLUMNS.
    Records are grouped by track_id and sorted by timestamp.

    Args:
        path: Path to the CSV file.

    Returns:
        Dict mapping track_id -> list of TrajectoryRecord sorted by timestamp.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If required columns are missing or data is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    _validate_csv_columns(df, path)

    tracks: Dict[int, List[TrajectoryRecord]] = {}
    for _, row in df.iterrows():
        record = TrajectoryRecord(
            timestamp=float(row["timestamp"]),
            track_id=int(row["track_id"]),
            x=float(row["x"]),
            y=float(row["y"]),
            vx=float(row["vx"]),
            vy=float(row["vy"]),
            c=float(row["c"]),
        )
        tracks.setdefault(record.track_id, []).append(record)

    # Sort each track by timestamp
    return {tid: sorted(recs, key=lambda r: r.timestamp) for tid, recs in tracks.items()}


def _validate_csv_columns(df: pd.DataFrame, path: Path) -> None:
    """Raise ValueError if required columns are missing."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {path} missing required columns: {missing}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )
    if len(df) == 0:
        raise ValueError(f"CSV {path} contains no data rows.")


# ---------------------------------------------------------------------------
# Optional rosbag2 loader (DS-07) — degrades gracefully if dep absent
# ---------------------------------------------------------------------------

def _rosbag_available() -> bool:
    """Check whether the rosbags library is importable."""
    try:
        import rosbags  # noqa: F401
        return True
    except ImportError:
        return False


def load_rosbag(path: str | Path, topic: str = "/human_states") -> Dict[int, List[TrajectoryRecord]]:
    """Load trajectory records from a rosbag2 (.db3) file.

    Requires the `rosbags` library to be installed. Raises ImportError
    with a helpful message if it is not available.

    Args:
        path: Path to rosbag2 directory or .db3 file.
        topic: ROS2 topic name containing human state messages.

    Returns:
        Dict mapping track_id -> list of TrajectoryRecord sorted by timestamp.

    Raises:
        ImportError: If the rosbags library is not installed.
    """
    if not _rosbag_available():
        raise ImportError(
            "rosbag2 loading requires the 'rosbags' library. "
            "Install with: pip install rosbags\n"
            "This dependency is optional; use load_csv() on Windows without ROS2."
        )

    from rosbags.rosbag2 import Reader  # type: ignore
    from rosbags.serde import deserialize_cdr  # type: ignore

    path = Path(path)
    tracks: Dict[int, List[TrajectoryRecord]] = {}

    with Reader(path) as reader:
        for connection, timestamp_ns, rawdata in reader.messages():
            if connection.topic != topic:
                continue
            msg = deserialize_cdr(rawdata, connection.msgtype)
            timestamp_s = timestamp_ns / 1e9
            # Assumes HumanStates message with a 'states' array
            for state in msg.states:
                record = TrajectoryRecord(
                    timestamp=timestamp_s,
                    track_id=int(state.track_id),
                    x=float(state.x),
                    y=float(state.y),
                    vx=float(state.vx),
                    vy=float(state.vy),
                    c=float(state.confidence),
                )
                tracks.setdefault(record.track_id, []).append(record)

    return {tid: sorted(recs, key=lambda r: r.timestamp) for tid, recs in tracks.items()}
