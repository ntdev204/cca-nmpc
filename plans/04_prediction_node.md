# Prediction Node Plan

## Overview

Build `cca_nmpc_prediction`: subscribes `/human_states`, maintains a rolling buffer of length `L` per track, runs the offline-trained LSTM (ONNX) to produce `H`-step predictions, computes rolling prediction-error variance `sigma_h`, and publishes `/human_predictions` and `/human_pred_uncertainty`. Per `docs/02_system_architecture.md` Section 3.2, `docs/01_mathematical_model.md` Sections 6 and 13.

## Environment Note

- Windows, no ROS2: implement code now; ROS-dependent tests run later. Keep the numerical core (buffering, sigma, ONNX inference wrapper) in pure-Python modules that import without `rclpy`, so they are testable now with `pytest` + onnxruntime.
- ROS glue (`prediction_node.py`) is thin and tested under ROS2 later.

## Project Type

**BACKEND / ROBOTICS ROS2 package.** Primary agent: `backend-specialist`; skills: `python-patterns`, `api-patterns`, `clean-code`; supporting: `test-engineer`.

## Success Criteria

- Per-track rolling buffer of length `L`; only predict when a track has `L` samples.
- ONNX inference produces `(H, 4)` predictions using the frozen `normalization_stats.json`.
- `sigma_h` computed as rolling prediction-error variance over last `W` cycles (Eq. 6.3); grows conservatively while a prediction is held stale between refreshes (Eq. 13.3).
- Publishes `HumanPredictionArray` with `prediction_stamp` and `HumanUncertaintyArray`.
- Runs at `f_lstm_hz` decoupled from the control loop.
- Numerical core unit tested now; ROS integration verified later.

## Tech Stack

- `ament_python`, `rclpy`, `message_filters` not required (single subscription).
- onnxruntime, NumPy.
- Messages: existing `cca_nmpc_msgs` (`HumanStateArray`, `HumanPredictionArray`, `HumanUncertaintyArray`).

## File Structure

```text
src/cca_nmpc_prediction/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/cca_nmpc_prediction
├── cca_nmpc_prediction/
│   ├── __init__.py
│   ├── prediction_node.py     # ROS glue (thin)
│   ├── track_buffer.py        # per-track rolling buffers (pure python)
│   ├── lstm_infer.py          # ONNX inference + normalization (pure python)
│   └── uncertainty.py         # sigma_h rolling variance + stale growth (pure python)
├── config/prediction.yaml
├── launch/prediction.launch.py
└── test/
    ├── test_track_buffer.py
    ├── test_uncertainty.py
    └── test_lstm_infer.py
```

## Task Breakdown

### PR-01 — Package skeleton

Agent: `backend-specialist`; skills: `python-patterns`, `clean-code`; priority: P0; dependencies: msgs package.

INPUT → layout above.
OUTPUT → `ament_python` package with entry point `prediction_node`, config, launch, test.
VERIFY (later) → `colcon build --packages-select cca_nmpc_prediction`. VERIFY (now) → pure-Python modules import without `rclpy`.

### PR-02 — YAML parameters

Agent: `backend-specialist`; skills: `clean-code`; priority: P0; dependencies: PR-01.

INPUT → `docs/07_yaml_parameters.md` prediction block.
OUTPUT → `prediction.yaml` with `model_path` (ONNX), `normalization_stats_path`, `L`, `H`, `f_lstm_hz`, `uncertainty_window_W`, `sigma_growth_rate_beta`, `sigma_max`.
VERIFY → node reads all keys; fails clearly if model/stats missing. Note: `sigma_max` also exists under `context_node` in docs/07 and MUST hold the same value (shared saturation ceiling, Eq. 8.3/13.3).

### PR-03 — Per-track rolling buffer

Agent: `backend-specialist`; skills: `python-patterns`; priority: P1; dependencies: PR-01.

INPUT → sequence of per-track states matched by `track_id`.
OUTPUT → `track_buffer.py` maintains length-`L` buffers, evicts stale tracks, exposes "ready" tracks.
VERIFY → `pytest test_track_buffer.py`: buffer fills to L, evicts on age, matches by track_id not index.

### PR-04 — ONNX inference wrapper

Agent: `backend-specialist`; skills: `python-patterns`; priority: P1; dependencies: PR-02, PR-03.

INPUT → `(L,4)` window + normalization stats + ONNX model.
OUTPUT → `lstm_infer.py` normalizes input, runs onnxruntime, denormalizes to `(H,4)`.
VERIFY → `test_lstm_infer.py` with a tiny exported model (from training plan) returns `(H,4)`; normalization round-trips.

### PR-05 — Uncertainty (sigma_h)

Agent: `backend-specialist`; skills: `clean-code`; priority: P1; dependencies: PR-04.

INPUT → realized vs previously-predicted states, elapsed time since last refresh.
OUTPUT → `uncertainty.py`: rolling variance over last `W` (Eq. 6.3); monotone growth `min(sigma_max, sigma_base + beta*dt)` while stale (Eq. 13.3); clipped `sigma_h_tilde` (Eq. 8.3).
VERIFY → `test_uncertainty.py`: variance matches reference; growth is monotone and capped at `sigma_max`; resets on refresh.

### PR-06 — Node glue + publishers

Agent: `backend-specialist`; skills: `api-patterns`; priority: P1; dependencies: PR-03, PR-04, PR-05.

INPUT → `/human_states` subscription; timer at `f_lstm_hz`.
OUTPUT → `prediction_node.py` publishes `HumanPredictionArray` (each `HumanPrediction` sets `horizon_length = H` and `prediction_stamp`; all `*_hat` arrays length `H`) and `HumanUncertaintyArray`, matched by `track_id`. Reads `HumanState.header.stamp` from the incoming array (not a bare `stamp`).
VERIFY (later) → `ros2 topic echo /human_predictions` shows valid arrays under a replayed `/human_states`.

### PR-07 — Launch + tests

Agent: `test-engineer`; skills: `testing-patterns`; priority: P2; dependencies: PR-06.

INPUT → all modules.
OUTPUT → `prediction.launch.py`; pure-logic tests green now.
VERIFY (now) → `pytest src/cca_nmpc_prediction/test/`. VERIFY (later) → `colcon test`.

## Phase X: Verification

- Now (Windows): `pytest` on buffer/uncertainty/inference passes; modules import without ROS2.
- Later (ROS2): `colcon build` + `colcon test`; replay a `/human_states` bag and confirm prediction/uncertainty topics and rate `f_lstm_hz`.

## Notes and Risks

- Match tensors by `track_id`, never array index (message design note).
- Reuse the exact frozen normalization stats from training; do not recompute.
- Keep ONNX inference and sigma logic ROS-free so they are testable on Windows now.
