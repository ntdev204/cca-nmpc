"""Training loop with early stopping, checkpointing, deterministic seed.

CLI: python -m tools.lstm_training.train --config <yaml/json>
Pure PyTorch CPU-friendly. Reads the .npz splits produced by tools.lstm_dataset.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import TrajectoryDataset
from .model import LSTMPredictor, LSTMConfig
from .loss import weighted_trajectory_loss


@dataclass
class TrainConfig:
    L: int = 8
    H: int = 12
    lambda_vel: float = 0.5
    hidden: int = 64
    layers: int = 1
    lr: float = 1e-3
    epochs: int = 20
    patience: int = 5
    batch_size: int = 32
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _epoch(model, loader, loss_fn, optim=None) -> float:
    train = optim is not None
    model.train(train)
    total, n = 0.0, 0
    for x, y in loader:
        pred = model(x)
        loss = loss_fn(pred, y)
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        total += float(loss.detach()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def train(
    train_npz: str | Path,
    val_npz: str | Path,
    stats_path: str | Path,
    out_dir: str | Path,
    cfg: TrainConfig,
) -> dict:
    """Train the predictor; return a summary dict and write best checkpoint."""
    set_seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TrajectoryDataset(train_npz, stats_path)
    val_ds = TrajectoryDataset(val_npz, stats_path)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = LSTMPredictor(LSTMConfig(
        hidden_size=cfg.hidden, num_layers=cfg.layers, horizon=cfg.H,
    ))
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def loss_fn(p, t):
        return weighted_trajectory_loss(p, t, cfg.lambda_vel)

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history = []
    ckpt_path = out_dir / "lstm_predictor_best.pt"

    for epoch in range(cfg.epochs):
        tr = _epoch(model, train_dl, loss_fn, optim)
        va = _epoch(model, val_dl, loss_fn) if len(val_ds) else tr
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        if va < best_val - 1e-6:
            best_val = va
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {"model_state": model.state_dict(),
                 "config": asdict(cfg)}, ckpt_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    return {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt_path),
        "history": history,
    }


def _load_cfg(path: str | Path) -> TrainConfig:
    with open(path) as f:
        raw = json.load(f) if str(path).endswith(".json") else _yaml_load(f)
    return TrainConfig(**{k: v for k, v in raw.items() if k in TrainConfig().__dict__})


def _yaml_load(f):
    import yaml
    return yaml.safe_load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the CCA-NMPC LSTM predictor")
    ap.add_argument("--config", help="training config (yaml/json)")
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args(argv)

    cfg = _load_cfg(args.config) if args.config else TrainConfig()
    summary = train(args.train_npz, args.val_npz, args.stats, args.out_dir, cfg)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
