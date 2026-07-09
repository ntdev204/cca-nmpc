"""Smoke test: a short training run decreases loss and writes a checkpoint."""
from pathlib import Path

from tools.lstm_training.train import train, TrainConfig


def test_smoke_train_writes_checkpoint_and_decreases(dataset_dir):
    cfg = TrainConfig(epochs=3, patience=3, hidden=16, batch_size=8, seed=0)
    summary = train(
        train_npz=dataset_dir / "train.npz",
        val_npz=dataset_dir / "val.npz",
        stats_path=dataset_dir / "stats.json",
        out_dir=dataset_dir / "models",
        cfg=cfg,
    )
    assert Path(summary["checkpoint"]).exists()
    history = summary["history"]
    assert len(history) >= 1
    # training loss at the end should be <= the first epoch (learning happened)
    assert history[-1]["train_loss"] <= history[0]["train_loss"] + 1e-6


def test_seed_is_deterministic(dataset_dir):
    cfg = TrainConfig(epochs=2, patience=2, hidden=16, batch_size=8, seed=7)
    s1 = train(dataset_dir / "train.npz", dataset_dir / "val.npz",
               dataset_dir / "stats.json", dataset_dir / "m1", cfg)
    s2 = train(dataset_dir / "train.npz", dataset_dir / "val.npz",
               dataset_dir / "stats.json", dataset_dir / "m2", cfg)
    assert abs(s1["best_val_loss"] - s2["best_val_loss"]) < 1e-6
