"""Integration test: full pipeline from CSV to .npz files."""

import tempfile
from pathlib import Path

import numpy as np

from tools.lstm_dataset.build_dataset import build_dataset


def test_full_pipeline_synthetic():
    """Test the full pipeline with synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic CSV with enough trajectories that a 70/15/15
        # trajectory-level split yields non-empty val/test (needs >= ~7 tracks).
        csv_path = tmpdir / "synthetic.csv"
        with open(csv_path, "w") as f:
            f.write("timestamp,track_id,x,y,vx,vy,c\n")
            for track_id in range(1, 13):  # 12 trajectories
                vx = 0.5 * ((track_id % 5) - 2)  # spread of directions
                vy = 0.4 * ((track_id % 3) - 1)
                for i in range(50):
                    t = i * 0.1
                    x = track_id + vx * t * 10.0
                    y = vy * t * 10.0
                    f.write(f"{t},{track_id},{x},{y},{vx*10.0},{vy*10.0},0.9\n")

        output_dir = tmpdir / "output"

        # Build dataset
        manifest = build_dataset(
            input_csv=csv_path,
            output_dir=output_dir,
            L=8,
            H=12,
            dt=0.125,
            seed=42,
        )

        # Verify manifest structure
        assert manifest["version"] == "lstm_dataset_v1"
        assert manifest["L"] == 8
        assert manifest["H"] == 12
        assert manifest["dt"] == 0.125
        assert manifest["seed"] == 42

        # Verify all splits exist
        assert set(manifest["splits"].keys()) == {"train", "val", "test"}

        # Verify files exist
        for split_name in ["train", "val", "test"]:
            npz_path = Path(manifest["splits"][split_name]["path"])
            assert npz_path.exists()

            # Load and verify structure. Use a context manager so the NpzFile
            # handle is closed before TemporaryDirectory cleanup (Windows locks
            # open files -> WinError 32 on rmtree otherwise).
            with np.load(npz_path) as data:
                assert "inputs" in data
                assert "targets" in data
                inputs = data["inputs"].copy()
                targets = data["targets"].copy()

            # Check shapes
            assert inputs.ndim == 3
            assert targets.ndim == 3
            assert inputs.shape[1] == 8  # L
            assert inputs.shape[2] == 4  # channels
            assert targets.shape[1] == 12  # H
            assert targets.shape[2] == 4  # channels
            assert inputs.shape[0] == targets.shape[0]  # same N

            # Check dtype
            assert inputs.dtype == np.float32
            assert targets.dtype == np.float32

            print(f"{split_name}: {inputs.shape[0]} windows")

        # Verify normalization stats exist and have correct structure
        stats_path = Path(manifest["normalization_stats"])
        assert stats_path.exists()

        import json
        with open(stats_path) as f:
            stats = json.load(f)

        assert "mean" in stats
        assert "std" in stats
        assert "channels" in stats
        assert len(stats["mean"]) == 4
        assert len(stats["std"]) == 4
        assert stats["channels"] == ["x", "y", "vx", "vy"]

        # Verify splits sum to approximately all windows
        total_windows = sum(
            manifest["splits"][split]["num_windows"]
            for split in ["train", "val", "test"]
        )
        assert total_windows > 0
        # With 12 tracks the split must populate every partition.
        assert manifest["splits"]["train"]["num_windows"] > 0
        assert manifest["splits"]["val"]["num_windows"] > 0
        assert manifest["splits"]["test"]["num_windows"] > 0

        print(f"\nTotal windows: {total_windows}")
        print(f"Train: {manifest['splits']['train']['num_windows']}")
        print(f"Val: {manifest['splits']['val']['num_windows']}")
        print(f"Test: {manifest['splits']['test']['num_windows']}")


if __name__ == "__main__":
    test_full_pipeline_synthetic()
    print("\n✓ Integration test passed!")
