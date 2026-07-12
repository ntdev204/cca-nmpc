"""Record schema and column contract for trajectory data.

Per Section 2.2 of docs/04_dataset_specification.md.
Composite trajectory key (session_id, run_id, sequence_id, track_id) prevents
cross-session AND cross-run leakage. Without run_id, two physically unrelated
human tracks recorded in different runs but sharing sequence_id=0 and a reset
tracker ID would be merged into one impossible trajectory (P1 dataset identity).
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


# Column names in the expected order
REQUIRED_COLUMNS = [
    "session_id", "run_id", "sequence_id", "timestamp", "track_id",
    "x", "y", "vx", "vy", "confidence",
]

# Channel order for windowed tensors [x, y, vx, vy]
TENSOR_CHANNELS = ["x", "y", "vx", "vy"]


@dataclass
class TrajectoryRecord:
    """Single timestep observation of one tracked human.

    Fields match Section 2.2 schema. Composite key (session_id, run_id,
    sequence_id, track_id) uniquely identifies a trajectory across multiple
    recording sessions and runs, and avoids ID collision when track_id resets
    between runs that share a sequence_id.
    """
    session_id: str       # Recording session identifier
    run_id: str           # Recording run within session (tracker resets per run)
    sequence_id: int      # Sequence within run (rosbag split index)
    timestamp: float      # ROS time (s)
    track_id: int         # Tracker-assigned ID (resets per run)
    x: float              # Position x (m) in map frame
    y: float              # Position y (m) in map frame
    vx: float             # Velocity x (m/s) in map frame
    vy: float             # Velocity y (m/s) in map frame
    confidence: float     # Detection confidence [0, 1]
    # Optional metadata retained for subject-held-out / stratified splits
    # (docs/09_roadmap.md Section 5) and provenance. Not part of the identity key.
    subject_id: str = "unknown"       # Person identity across trajectories
    scenario_id: str = "unknown"      # e.g. passing, crossing, following, static
    environment_id: str = "unknown"   # Recording environment
    frame_id: str = ""                # TF frame of x/y (map); used to reject frame switches
    is_observed: bool = True          # True = tracker observation, not a predicted fill

    def __post_init__(self):
        """Validate field types and ranges."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.track_id < 0:
            raise ValueError(f"track_id={self.track_id} must be non-negative")
        if self.sequence_id < 0:
            raise ValueError(f"sequence_id={self.sequence_id} must be non-negative")

    def trajectory_key(self) -> "TrajectoryKey":
        """Return composite trajectory identifier."""
        return (self.session_id, self.run_id, self.sequence_id, self.track_id)


# (session_id, run_id, sequence_id, track_id)
TrajectoryKey = Tuple[str, str, int, int]


def records_to_array(records: List[TrajectoryRecord], channels: List[str] = None) -> np.ndarray:
    """Convert list of records to numpy array with specified channels.

    Args:
        records: List of TrajectoryRecord objects.
        channels: List of field names to extract (default: TENSOR_CHANNELS).

    Returns:
        Array of shape (N, len(channels)) where N = len(records).
    """
    if not records:
        return np.array([]).reshape(0, len(channels or TENSOR_CHANNELS))

    if channels is None:
        channels = TENSOR_CHANNELS

    data = [[getattr(r, ch) for ch in channels] for r in records]
    return np.array(data, dtype=np.float64)
