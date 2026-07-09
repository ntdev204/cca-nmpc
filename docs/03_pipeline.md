# CCA-NMPC — Pipeline Specification

## 1. Offline Pipeline (training-time)

```text
Recorded human trajectories (rosbag / CSV)
        │
        ▼
Preprocessing (resampling, normalization, windowing)
        │
        ▼
Train/Val/Test split
        │
        ▼
LSTM training (see Dataset Specification doc)
        │
        ▼
Frozen model export (.pt / .onnx)
        │
        ▼
Loaded by prediction_node at runtime
```

Training loss (Eq. 13.2 of math spec): $L = MSE_{position} + \lambda\, MSE_{velocity}$.

Context-weight calibration (Section 8.1 of math spec) is a **separate** offline procedure run once per platform/environment, producing $w_d, w_v, w_\theta, w_u, b$ values stored in the YAML parameter file.

---

## 2. Online Pipeline (run-time, per math-spec Section 14)

```text
RGB-D Camera
      │
      ▼
YOLO26m  ───────────────────────────────┐
      │                                  │  f_perception (e.g. 15–30 Hz)
      ▼                                  │
Depth Projection                          │
      │                                  │
      ▼                                  │
TF (camera → map)                         │
      │                                  │
      ▼                                  │
Kalman Filter  →  Human State (x,y,vx,vy,c)
      │
      ├──────────────────────────────────────────────┐
      │                                               │
      ▼                                               │  (current state, always fresh)
LSTM Prediction (+ rolling uncertainty σ_h)            │
   f_LSTM (e.g. 5–10 Hz, slower than control loop)      │
      │                                               │
      ▼                                               ▼
Continuous Context Estimation (corrected z, per-human φ_j)
   uses: latest current state (fresh) + latest held prediction (may be stale)
      │
      ▼
EMA filtering → φ_j^filtered  (Eq. 12.6)
      │
      ▼
Dwell-time gate → φ_j^used  (Eq. 12.7)
      │
      ▼
Adaptive Parameter Generation (saturated v_max, per-human d_safe, Q(φ))
      │
      ▼
CCA-NMPC solve
   f_NMPC (e.g. 20–50 Hz, fastest loop)
   per-human slack constraints, EMA + dwell-time gated φ
      │
      ▼
Velocity Command (/cmd_vel)
```

---

## 3. Timing Model

| Stage                                   | Approx. rate          | Notes                                                                                                        |
| --------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------ |
| Perception (YOLO + depth + TF + Kalman) | 15–30 Hz              | Bound by camera FPS and detector inference time                                                              |
| LSTM prediction                         | 5–10 Hz               | Slower by design; buffers input sequence of length $L$                                                       |
| Context estimation                      | Runs every NMPC cycle | Geometric terms recomputed fresh; predicted trajectory held between LSTM updates (Section 13.1 of math spec) |
| Adaptive parameter generation           | Runs every NMPC cycle | Cheap, closed-form (Eqs. 10.1–10.3)                                                                          |
| NMPC solve                              | 20–50 Hz              | Horizon $H$, solved via acados/CasADi RTI scheme recommended for real-time                                   |

**Rule:** $f_{NMPC} \ge f_{context} \ge f_{LSTM}$, consistent with Section 13.1 of the Mathematical Model document. Exact target rates depend on onboard compute and should be measured empirically (Section 12.1 feasibility logging covers solve time as well as feasibility).

---

## 4. Data Flow Contracts

Each arrow in the diagrams above corresponds to a ROS2 topic + message pair, specified in the **ROS Topics** and **Message Definitions** documents. Every numerical constant appearing in this pipeline ($f_{LSTM}, f_{NMPC}, L, H, W$, etc.) is externalized to the **YAML Parameters** document rather than hard-coded, so timing and horizon can be re-tuned without code changes.
