from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.node import Node
from cca_nmpc_msgs.msg import HumanStateArray

FIELDS = (
    "dataset_version", "session_id", "run_id", "sequence_id", "track_id",
    "subject_id", "scenario_id", "environment_id", "timestamp", "frame_id",
    "x", "y", "vx", "vy", "confidence", "is_observed",
)


class TrajectoryLogger(Node):
    def __init__(self) -> None:
        super().__init__("human_trajectory_logger")
        d = self.declare_parameter
        d("output_csv", "datasets/human_trajectories.csv")
        defaults = {
            "dataset_version": "1.0", "session_id": "required",
            "run_id": "required", "sequence_id": "0", "subject_id": "unknown",
            "scenario_id": "required", "environment_id": "required",
            "camera_model": "required", "operator": "required",
        }
        for name, default in defaults.items():
            d(name, default)
        self._values = {name: str(self.get_parameter(name).value) for name in defaults}
        required = ("session_id", "run_id", "scenario_id", "environment_id")
        missing = [name for name in required if self._values[name] == "required"]
        if missing:
            raise ValueError(f"Required dataset metadata not configured: {missing}")
        self._path = Path(str(self.get_parameter("output_csv").value))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self._path.exists()
        self._file = self._path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        if is_new:
            self._writer.writeheader()
        self._write_metadata()
        self.create_subscription(HumanStateArray, "/human_states", self._on_states, 10)

    def _write_metadata(self) -> None:
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, timeout=2
            ).strip()
        except (OSError, subprocess.SubprocessError):
            sha = "unknown"
        metadata = self._values | {
            "git_sha": sha, "created_utc": datetime.now(timezone.utc).isoformat()
        }
        self._path.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    def _on_states(self, msg: HumanStateArray) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        base = {
            "dataset_version": self._values["dataset_version"],
            "session_id": self._values["session_id"], "run_id": self._values["run_id"],
            "sequence_id": self._values["sequence_id"],
            "subject_id": self._values["subject_id"],
            "scenario_id": self._values["scenario_id"],
            "environment_id": self._values["environment_id"],
            "timestamp": stamp, "frame_id": msg.header.frame_id, "is_observed": True,
        }
        for human in msg.humans:
            self._writer.writerow(base | {
                "track_id": human.track_id, "x": human.x, "y": human.y,
                "vx": human.vx, "vy": human.vy, "confidence": human.confidence,
            })
        self._file.flush()

    def destroy_node(self):
        self._file.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
