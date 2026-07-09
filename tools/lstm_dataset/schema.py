"""Record schema and column contract for trajectory data.

Per Section 2.2 of docs/04_dataset_specification.md.
"""

from dataclasses import dataclass
from typing import List
import numpy as np


# Column names in the expected order
REQUIRED_COLUMNS = ["timestamp", "track_id", "x", "y", "vx", "vy", "c"]

# Channel order for windowed tensors [x, y, vx, vy]
TENSOR_CHANNELS = ["x", "y", "vx", "vy"]


@dataclass
class TrajectoryRecord:
    """Single timestep observation of one tracked human.

    Fields match Section 2.2 schema.
    """
    timestamp: float      # ROS time (s)
    track_id: int         # Persistent per-human ID
    x: float              # Position x (m) in map frame
    y: float              # Position y (m) in map frame
    vx: float             # Velocity x (m/s) in map frame
    vy: float             # Velocity y (m/s) in map frame
    c: float              # Detection confidence [0, 1]

    def __post_init__(self):
        """Validate field types and ranges."""
        if not 0.0 <= self.c <= 1.0:
            raise ValueError(f"Confidence c={self.c} must be in [0, 1]")
        if self.track_id < 0:
            raise ValueError(f"track_id={self.track_id} must be non-negative")


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
