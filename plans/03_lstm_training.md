# LSTM Training & Export Plan

## Overview

Train the offline human-trajectory LSTM and export a frozen artifact consumed at runtime by `prediction_node`, per `docs/03_pipeline.md` (offline pipeline) and `docs/01_mathematical_model.md` Section 6, 13. Pure Python; implementable and testable now on Windows (CPU is fine for the smoke path).

## Environment Note

- Windows, no ROS2. Standalone PyTorch project.
- GPU/CUDA optional. Training tests use a tiny synthetic dataset on CPU so they run anywhere.
- ONNX export runs on CPU. TensorRT engine build for the perception detector is a separate concern (see `01_human_perception.md`); this LSTM is exported to ONNX (and optionally TorchScript) for `prediction_node`.

## Project Type

**BACKEND / ML (offline, pure Python).** Primary agent: `backend-specialist`; skills: `python-patterns`, `clean-code`; supporting: `test-engineer`, `performance-optimizer`.

## Success Criteria

- Loads `.npz` splits + `normalization_stats.json` from the dataset plan.
- Defines an LSTM mapping `(L, 4) -> (H, 4)` per Eq. 13.1.
- Trains with loss `L = MSE_position + lambda * MSE_velocity` (Eq. 13.2).
- Reports val/test metrics; supports early stopping and checkpointing.
- Exports a frozen model (`.onnx`, optional `.pt` TorchScript) named with dataset version, e.g. `lstm_predictor_v1_on_lstm_dataset_v1.onnx`.
- Deterministic smoke training run is unit tested on CPU.

## Tech Stack

- Python 3, PyTorch, NumPy.
- onnx / onnxruntime for export + numerical parity check.
- `pytest` for tests.

## File Structure

```text
tools/lstm_training/
├── __init__.py
├── dataset.py           # torch Dataset over .npz + stats
├── model.py             # LSTM predictor (L,4)->(H,4)
├── loss.py              # weighted position/velocity MSE, Eq. 13.2
├── train.py             # training loop, early stop, checkpoint
├── evaluate.py          # val/test metrics (ADE/FDE, MSE)
├── export.py            # ONNX (+ TorchScript) export + parity check
└── tests/
    ├── test_model.py
    ├── test_loss.py
    ├── test_train_smoke.py
    └── test_export_parity.py
```

## Task Breakdown

### TR-01 — Torch dataset over npz

Agent: `backend-specialist`; skills: `python-patterns`; priority: P0; dependencies: dataset plan DS-06.

INPUT → `.npz` splits and `normalization_stats.json`.
OUTPUT → `dataset.py` yielding normalized `(L,4)`/`(H,4)` tensors; exposes denormalization for evaluation.
VERIFY → `pytest ... test_model.py -k dataset` loads a synthetic npz and returns correct shapes/dtypes.

### TR-02 — LSTM model

Agent: `backend-specialist`; skills: `python-patterns`; priority: P0; dependencies: TR-01.

INPUT → config: input size 4, hidden size, layers, horizon `H`.
OUTPUT → `model.py` LSTM encoder + decoder/linear head producing `(H, 4)`; no required-arg surprises, default config provided.
VERIFY → `test_model.py` forward pass on random `(batch, L, 4)` returns `(batch, H, 4)`.

### TR-03 — Weighted loss

Agent: `backend-specialist`; skills: `clean-code`; priority: P1; dependencies: TR-02.

INPUT → predicted vs target `(H,4)`, `lambda`.
OUTPUT → `loss.py` implementing `MSE_position + lambda * MSE_velocity` (channels 0,1 position; 2,3 velocity).
VERIFY → `test_loss.py` checks known-value cases and that lambda scales the velocity term.

### TR-04 — Training loop

Agent: `backend-specialist`; skills: `python-patterns`; priority: P1; dependencies: TR-03.

INPUT → train/val loaders, optimizer, epochs, patience, seed.
OUTPUT → `train.py` with checkpointing, early stopping, and a deterministic seed path.
VERIFY → `test_train_smoke.py` trains 1–2 epochs on tiny synthetic data on CPU and confirms loss decreases and a checkpoint is written.

### TR-05 — Evaluation metrics

Agent: `test-engineer`; skills: `testing-patterns`; priority: P2; dependencies: TR-04.

INPUT → trained model, test loader, stats.
OUTPUT → `evaluate.py` reporting ADE/FDE and per-channel MSE in denormalized units.
VERIFY → metrics computed on synthetic data match a hand-computed reference within tolerance.

### TR-06 — Export + parity

Agent: `backend-specialist`; skills: `python-patterns`; priority: P1; dependencies: TR-04.

INPUT → trained checkpoint, sample input.
OUTPUT → `export.py` writes ONNX (+ optional TorchScript) named with dataset version; runs onnxruntime and compares outputs to PyTorch.
VERIFY → `test_export_parity.py` asserts ONNX vs PyTorch outputs match within tolerance on random input.

### TR-07 — Config + CLI

Agent: `backend-specialist`; skills: `clean-code`; priority: P2; dependencies: TR-04, TR-06.

INPUT → a training config file (yaml/json) with `L, H, lambda, hidden, layers, lr, epochs`.
OUTPUT → CLI `train.py --config ...` and `export.py --checkpoint ...`; artifacts land in a `models/` output dir.
VERIFY → `python -m tools.lstm_training.train --help` and `--config` on tiny data produce a checkpoint and exported model.

## Phase X: Verification (Windows now)

- `pytest tools/lstm_training/tests/` passes on CPU.
- Smoke train + export on synthetic data produces an ONNX file and passes parity.
- Deferred (needs real data / GPU): full training run, ADE/FDE on real test split, model-version pinning for `prediction_node`.

## Notes and Risks

- Freeze normalization stats with the model; runtime must not recompute them.
- Keep the exported ONNX input/output signature documented so `prediction_node` (plan 04) binds to the exact tensor names/shapes.
- Position vs velocity channel indices must match the dataset channel order `[x, y, vx, vy]`.
