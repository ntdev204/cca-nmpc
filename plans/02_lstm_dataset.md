# LSTM Dataset Preprocessing Plan

## Overview

Build the offline dataset pipeline that turns recorded human trajectories into windowed tensors for LSTM training, per `docs/04_dataset_specification.md`. This is pure Python (no ROS2 runtime needed), so it can be fully implemented and tested now on Windows.

## Environment Note

- Windows, no ROS2. This package is standalone Python; do not import `rclpy`.
- Tests run now with `pytest`. Reading rosbag2 (`.db3`) is optional and gated behind an availability check so tests do not require ROS2.

## Project Type

**BACKEND / DATA PIPELINE (offline, pure Python).** Primary agent: `backend-specialist`; skills: `python-patterns`, `clean-code`; supporting: `test-engineer`.

## Success Criteria

- Reads raw trajectory records (CSV first; rosbag2 optional) matching the Section 2.2 schema.
- Resamples to fixed `dt`, windows into `(L, 4)` input / `(H, 4)` target tensors per Eq. 6.1–6.2.
- Splits by trajectory (not timestep) 70/15/15 with no leakage.
- Computes z-score normalization stats on the train split only and saves `normalization_stats.json`.
- Writes `train/val/test` as `.npz` (and/or `.parquet`).
- Deterministic given a fixed seed; unit tested.

## Tech Stack

- Python 3, NumPy, pandas.
- Optional: pyarrow (parquet), rosbags library for `.db3` (optional path only).
- `pytest` for tests.

## File Structure

```text
tools/lstm_dataset/
├── __init__.py
├── schema.py            # record dataclasses + column contract
├── loaders.py           # CSV loader (+ optional rosbag2 loader behind guard)
├── resample.py          # fixed-dt resampling, velocity re-derivation
├── windowing.py         # sliding L+H windows, gap discarding
├── normalize.py         # train-only z-score stats, apply/save/load
├── split.py             # trajectory-level train/val/test split
├── build_dataset.py     # CLI entry: raw -> train/val/test + stats
└── tests/
    ├── test_resample.py
    ├── test_windowing.py
    ├── test_normalize.py
    └── test_split.py
```

## Task Breakdown

### DS-01 — Record schema and CSV loader

Agent: `backend-specialist`; skills: `python-patterns`; priority: P0; dependencies: none.

INPUT → Section 2.2 schema (timestamp, track_id, x, y, vx, vy, c).
OUTPUT → `schema.py` column contract and `loaders.py` CSV loader returning per-track ordered records.
VERIFY → `pytest tools/lstm_dataset/tests/test_windowing.py -k loader` (or dedicated test) loads a synthetic CSV and groups by `track_id` in time order.

### DS-02 — Resampling and velocity re-derivation

Agent: `backend-specialist`; skills: `python-patterns`; priority: P1; dependencies: DS-01.

INPUT → per-track records and target `dt`.
OUTPUT → `resample.py` linear-interpolates position to fixed `dt`, finite-difference re-derives velocity where gaps exceed a threshold.
VERIFY → test with known linear motion recovers expected resampled positions/velocities; large-gap segments are flagged.

### DS-03 — Windowing

Agent: `backend-specialist`; skills: `python-patterns`; priority: P1; dependencies: DS-02.

INPUT → resampled per-track series, `L`, `H`, max-gap threshold.
OUTPUT → `windowing.py` produces `(N, L, 4)` inputs and `(N, H, 4)` targets, discarding windows spanning track-loss gaps.
VERIFY → test asserts output shapes, channel order `[x, y, vx, vy]`, and that gap-crossing windows are dropped.

### DS-04 — Trajectory-level split

Agent: `backend-specialist`; skills: `clean-code`; priority: P1; dependencies: DS-03.

INPUT → windows tagged by source trajectory ID, seed.
OUTPUT → `split.py` deterministic 70/15/15 split by trajectory (never by timestep), optional scenario stratification.
VERIFY → test confirms no trajectory appears in more than one split and ratios are within tolerance.

### DS-05 — Normalization (train-only stats)

Agent: `backend-specialist`; skills: `python-patterns`; priority: P1; dependencies: DS-04.

INPUT → train split tensors.
OUTPUT → `normalize.py` computes per-channel mean/std on train only, applies to all splits, saves/loads `normalization_stats.json`.
VERIFY → test confirms val/test use train stats; round-trip apply/inverse recovers original within tolerance.

### DS-06 — Build CLI and file output

Agent: `backend-specialist`; skills: `clean-code`; priority: P2; dependencies: DS-05.

INPUT → raw CSV path, output dir, `L`, `H`, `dt`, seed.
OUTPUT → `build_dataset.py` writes `train.npz`, `val.npz`, `test.npz`, and `normalization_stats.json`; prints a manifest with counts/checksums.
VERIFY → running the CLI on a synthetic input produces all expected files with consistent shapes and a version tag (`lstm_dataset_v1`).

### DS-07 — Optional rosbag2 loader (guarded)

Agent: `backend-specialist`; skills: `python-patterns`; priority: P3; dependencies: DS-01.

INPUT → rosbag2 `.db3` with `/human_states`.
OUTPUT → optional loader behind an import/availability guard so the module imports cleanly on Windows without ROS2.
VERIFY → test skips gracefully when the rosbag reader dependency is absent; documents how to run it where available.

## Phase X: Verification (Windows now)

- `pytest tools/lstm_dataset/tests/` passes.
- `python -m tools.lstm_dataset.build_dataset --help` runs.
- Build on a synthetic CSV; confirm shapes, split disjointness, and stats file.
- Deferred (needs data/ROS2): run on real recorded trajectories and a `.db3` bag.

## Notes and Risks

- Keep the pipeline seedable and deterministic for reproducibility claims (Section 4 of dataset doc).
- Do not couple to ROS message types; use plain arrays/dataclasses so this runs anywhere.
- Store `normalization_stats.json` alongside the model so `prediction_node` reuses identical stats at inference.
