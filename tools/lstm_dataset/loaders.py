"""CSV loader and optional rosbag2 loader for trajectory records.

Rosbag2 support is behind an import guard (DS-07): if the rosbags
library is not installed this module still imports cleanly.
Trajectory key is composite (session_id, sequence_id, track_id) to
prevent cross-session ID collision (P1 #7).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .schema import REQUIRED_COLUMNS, TrajectoryKey, TrajectoryRecord


def load_csv(path: str | Path) -> Dict[TrajectoryKey, List[TrajectoryRecord]]:
    """Load trajectory records from a CSV file.

    The CSV must have at minimum the columns listed in REQUIRED_COLUMNS,
    including session_id and sequence_id. Records are grouped by composite
    key (session_id, sequence_id, track_id) and sorted by timestamp.

    Args:
        path: Path to the CSV file.

    Returns:
        Dict mapping (session_id, sequence_id, track_id) -> sorted records.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If required columns are missing or data is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    _validate_csv_columns(df, path)

    tracks: Dict[TrajectoryKey, List[TrajectoryRecord]] = {}
    for _, row in df.iterrows():
        record = TrajectoryRecord(
            session_id=str(row["session_id"]),
            sequence_id=int(row["sequence_id"]),
            timestamp=float(row["timestamp"]),
            track_id=int(row["track_id"]),
            x=float(row["x"]),
            y=float(row["y"]),
            vx=float(row["vx"]),
            vy=float(row["vy"]),
            c=float(row["confidence"] if "confidence" in row else row["c"]),
        )
        key = record.trajectory_key()
        tracks.setdefault(key, []).append(record)

    # Sort each trajectory by timestamp
    return {key: sorted(recs, key=lambda r: r.timestamp) for key, recs in tracks.items()}


def _validate_csv_columns(df: pd.DataFrame, path: Path) -> None:
    """Raise ValueError if required columns are missing."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if "c" in missing and "confidence" in df.columns:
        missing.remove("c")
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


def load_rosbag(
    path: str | Path,
    topic: str = "/human_states",
    session_id: str = "default",
    sequence_id: int = 0,
) -> Dict[TrajectoryKey, List[TrajectoryRecord]]:
    """Load trajectory records from a rosbag2 (.db3) file.

    Requires the `rosbags` library to be installed. Raises ImportError
    with a helpful message if it is not available.

    Args:
        path: Path to rosbag2 directory or .db3 file.
        topic: ROS2 topic name containing human state messages.
        session_id: Recording session identifier (default "default").
        sequence_id: Sequence within session (default 0).

    Returns:
        Dict mapping (session_id, sequence_id, track_id) -> sorted records.

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
    tracks: Dict[TrajectoryKey, List[TrajectoryRecord]] = {}

    with Reader(path) as reader:
        for connection, timestamp_ns, rawdata in reader.messages():
            if connection.topic != topic:
                continue
            msg = deserialize_cdr(rawdata, connection.msgtype)
            # P1 #7: use msg.header.stamp, not bag write time
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                timestamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            else:
                timestamp_s = timestamp_ns / 1e9
            # P1 #9: msg.humans not msg.states
            if not hasattr(msg, "humans"):
                raise ValueError(
                    f"{topic} must contain cca_nmpc_msgs/HumanStateArray.humans"
                )
            humans = msg.humans
            for state in humans:
                record = TrajectoryRecord(
                    session_id=session_id,
                    sequence_id=sequence_id,
                    timestamp=timestamp_s,
                    track_id=int(state.track_id),
                    x=float(state.x),
                    y=float(state.y),
                    vx=float(state.vx),
                    vy=float(state.vy),
                    c=float(state.confidence),
                )
                key = record.trajectory_key()
                tracks.setdefault(key, []).append(record)

    return {key: sorted(recs, key=lambda r: r.timestamp) for key, recs in tracks.items()}
