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
            f.write("session_id,sequence_id,timestamp,track_id,x,y,vx,vy,c\n")
            for track_id in range(1, 13):  # 12 trajectories
                vx = 0.5 * ((track_id % 5) - 2)  # spread of directions
                vy = 0.4 * ((track_id % 3) - 1)
                for i in range(50):
                    t = i * 0.1
                    x = track_id + vx * t * 10.0
                    y = vy * t * 10.0
                    f.write(f"test_session,0,{t},{track_id},{x},{y},{vx*10.0},{vy*10.0},0.9\n")

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


def _write_synthetic_csv(csv_path: Path) -> None:
    """Write a 12-trajectory synthetic CSV (shared by tests below)."""
    with open(csv_path, "w") as f:
        f.write("session_id,sequence_id,timestamp,track_id,x,y,vx,vy,c\n")
        for track_id in range(1, 13):
            vx = 0.5 * ((track_id % 5) - 2)
            vy = 0.4 * ((track_id % 3) - 1)
            for i in range(50):
                t = i * 0.1
                x = track_id + vx * t * 10.0
                y = vy * t * 10.0
                f.write(f"test_session,0,{t},{track_id},{x},{y},{vx*10.0},{vy*10.0},0.9\n")


def test_npz_stores_raw_units_not_normalized():
    """Regression: the .npz must hold RAW physical units, not normalized values.

    The builder saves windows to disk; TrajectoryDataset (training package)
    z-score normalizes on read. If the builder also normalized before saving,
    the LSTM would train on doubly-normalized data (wrong scale) and
    denormalize() would be wrong. Guard that silent bug here.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        csv_path = tmpdir / "synthetic.csv"
        _write_synthetic_csv(csv_path)
        output_dir = tmpdir / "output"

        manifest = build_dataset(
            input_csv=csv_path, output_dir=output_dir, L=8, H=12, dt=0.125, seed=42
        )

        import json
        with open(manifest["normalization_stats"]) as f:
            stats = json.load(f)
        mean = np.asarray(stats["mean"], dtype=np.float64)

        with np.load(manifest["splits"]["train"]["path"]) as data:
            train_in = data["inputs"].astype(np.float64).copy()

        # Positions are stored in raw local-frame metres (relative to the final
        # observation), not z-scores. The final observed position is therefore
        # exactly the local origin for every input window.
        on_disk_mean = train_in.reshape(-1, 4).mean(axis=0)
        np.testing.assert_allclose(train_in[:, -1, :2], 0.0, atol=1e-7)
        np.testing.assert_allclose(on_disk_mean, mean, atol=1e-7)
        # And it must match the frozen stats' mean (stats computed from same raw train split).
        np.testing.assert_allclose(on_disk_mean[0], mean[0], rtol=0.05)


def test_trajectory_dataset_normalizes_once_and_roundtrips():
    """TrajectoryDataset applies normalization exactly once; denormalize inverts it."""
    __import__("torch")
    from tools.lstm_training.dataset import TrajectoryDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        csv_path = tmpdir / "synthetic.csv"
        _write_synthetic_csv(csv_path)
        output_dir = tmpdir / "output"

        manifest = build_dataset(
            input_csv=csv_path, output_dir=output_dir, L=8, H=12, dt=0.125, seed=42
        )
        stats_path = manifest["normalization_stats"]
        train_path = manifest["splits"]["train"]["path"]

        ds = TrajectoryDataset(train_path, stats_path)
        x_norm, y_norm = ds[0]

        # Normalized inputs should be roughly standardized (|value| typically < ~5),
        # i.e. NOT the raw x ~ track_id scale.
        assert float(x_norm.abs().max()) < 20.0

        # denormalize(normalize(raw)) == raw for the same window read from disk.
        with np.load(train_path) as data:
            raw_x0 = data["inputs"][0].astype(np.float32).copy()
        recovered = ds.denormalize(x_norm).numpy()
        np.testing.assert_allclose(recovered, raw_x0, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_full_pipeline_synthetic()
    test_npz_stores_raw_units_not_normalized()
    test_trajectory_dataset_normalizes_once_and_roundtrips()
    print("\n✓ Integration tests passed!")
