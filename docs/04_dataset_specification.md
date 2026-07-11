# CCA-NMPC — Dataset Specification

## 1. Purpose

Two distinct datasets are required:

1. **LSTM trajectory-prediction dataset** — trains the offline human motion predictor (Section 6, 13 of the Mathematical Model document).
2. **Context-calibration dataset** — used only for the two-stage weight calibration procedure in Section 8.1 of the Mathematical Model document (much smaller, collected on the target platform).

These are kept separate because they serve different purposes: the LSTM dataset needs volume and trajectory diversity; the calibration dataset needs to be representative of the *specific* robot/sensor/environment the context weights will be deployed in.

---

## 2. LSTM Trajectory-Prediction Dataset

### 2.1 Source options
- Public human-trajectory datasets (e.g., ETH/UCY-style pedestrian datasets) for pretraining, **and**
- Platform-collected trajectories from the robot's own Kalman-filtered `/human_states` output, for fine-tuning to match the actual sensor noise characteristics.

### 2.2 Raw record schema (per timestep, per tracked human)

| Field | Type | Description |
|---|---|---|
| `timestamp` | float64 (s) | ROS time |
| `track_id` | int | Persistent per-human ID from the tracker |
| `x`, `y` | float64 (m) | Position in map frame |
| `vx`, `vy` | float64 (m/s) | Velocity in map frame |
| `c` | float64 [0,1] | Detection confidence at that timestep |

### 2.3 Preprocessing

- Express every input/target window position relative to the final observed
  position; retain velocity components. Runtime inference translates predicted
  positions back into the map frame. This prevents memorization of map origin.
- **Resampling** to a fixed control period $\Delta t$ (matching the eventual `f_LSTM`/context loop rate) via linear interpolation on position, finite-difference re-derivation of velocity where gaps exceed a threshold.
- **Windowing:** sliding windows of length $L$ (input) + $H$ (prediction horizon) per Eq. (6.1)–(6.2), discarding windows with a track-loss gap longer than a configurable threshold.
- **Normalization:** z-score normalization of $x, y, v_x, v_y$ per-channel, statistics computed on the **training split only** and reused for val/test and at inference time (stored alongside the frozen model).
- **Train / validation / test split:** by *trajectory* (not by timestep) to avoid leakage — recommended 70/15/15, stratified by scenario type if scenario labels are available (e.g., passing, crossing, following, static).

### 2.4 Output tensor shapes

| Tensor | Shape | Notes |
|---|---|---|
| Input sequence | $(N, L, 4)$ | $N$ windows, $L$ timesteps, channels $[x,y,v_x,v_y]$ |
| Target sequence | $(N, H, 4)$ | $H$ future timesteps, same channels |

### 2.5 File format
- Stored as `.npz` (numpy) or `.parquet` per split (`train.parquet`, `val.parquet`, `test.parquet`), plus a `normalization_stats.json` with per-channel mean/std.
- Raw un-windowed trajectories additionally kept as rosbag2 (`.db3`) for traceability back to the original sensor run.

---

## 3. Context-Calibration Dataset (Section 8.1 of Mathematical Model)

### 3.1 Collection protocol
- 30–60 minutes of logged robot-human interaction on the **target platform**, covering at minimum:
  - near-passing (human passes close to robot),
  - crossing (perpendicular paths),
  - following (human/robot moving in same general direction),
  - static/idle human presence.
- Logged fields per timestep: robot state $X_r$, all tracked humans' $S_h$, and any observable evasive-reaction proxy (e.g., sudden human deceleration or lateral step, detected via velocity-change threshold) used as the fitting label described in Section 8.1.

### 3.2 Schema

| Field | Type | Description |
|---|---|---|
| `timestamp` | float64 | ROS time |
| `robot_x`, `robot_y`, `robot_theta` | float64 | Robot pose |
| `human_states` | array of `HumanState` | See Message Definitions doc |
| `min_future_distance` | float64 | Minimum $d_h$ observed over the next $T_{label}$ seconds (post-hoc label, computed offline) |
| `evasive_reaction_flag` | bool | Whether an evasive reaction proxy was detected in the same window |

### 3.3 Use
Feeds directly into the two-stage calibration in Section 8.1 of the Mathematical Model document: stage 1 fits $w_d, w_v, w_\theta, w_u, b$ against `min_future_distance` / `evasive_reaction_flag` proxies; stage 2 performs the $\pm 20\%$ weight-perturbation sensitivity sweep against logged NMPC solver success rate.

---

## 4. Versioning & Reproducibility

Raw collection records use the following minimum schema:
`dataset_version, session_id, run_id, sequence_id, track_id, subject_id,
scenario_id, environment_id, timestamp, frame_id, x, y, vx, vy, confidence,
is_observed`. Dataset 01 contains tracker observations (pseudo-labels), not
independent ground truth. ADE/FDE claims must therefore be reported separately
for tracker pseudo-label evaluation and for an independently annotated/mocap
reference subset.
- Each dataset version tagged with a semantic version (`lstm_dataset_v1`, `context_calib_v1`) and a checksum manifest.
- Frozen LSTM model artifacts named to include the dataset version they were trained on (e.g., `lstm_predictor_v1_on_lstm_dataset_v1.onnx`) so `prediction_node` config can pin an exact pairing.
