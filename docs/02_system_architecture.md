# CCA-NMPC — System Architecture

## 1. Design Goals

- Real-time operation on a Mecanum omnidirectional robot in ROS2.
- Clear separation between **perception**, **prediction**, **context estimation**, and **control**, each as an independent ROS2 node — matching the three-layer model in the Mathematical Model document.
- Perception/prediction run slower and asynchronously from the control loop (see Section 13.1 of the math spec); the architecture must tolerate this without stalling the controller.
- All nodes written in Python 3 (`rclpy`), using `ament_python` packaging.

---

## 2. High-Level Component Diagram

```text
┌─────────────────────┐
│  perception_node     │  RGB-D → YOLO26m → depth projection → TF → Kalman filter
│  (Python, rclpy)     │  publishes: /human_states
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│  prediction_node     │  Offline-trained LSTM inference
│  (Python, rclpy)     │  subscribes: /human_states
│                       │  publishes: /human_predictions, /human_pred_uncertainty
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│  context_node        │  Continuous context estimator (Eq. 8.1–8.3)
│  (Python, rclpy)     │  subscribes: /human_states, /human_predictions,
│                       │               /human_pred_uncertainty, /robot_odom
│                       │  publishes: /context_index (per-human φ_j + aggregate φ)
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│  adaptive_param_node │  Adaptive Parameter Generator (Eq. 10.1–10.3)
│  (Python, rclpy)     │  subscribes: /context_index
│                       │  publishes: /adaptive_params (d_safe, v_max, ω_max, Q(φ))
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│  nmpc_controller_node│  CCA-NMPC solver (CasADi/IPOPT reference; acados deferred)
│  (Python, rclpy)     │  subscribes: /adaptive_params, /human_predictions,
│                       │               /robot_odom, /local_costmap
│                       │  publishes: /cmd_vel, /nmpc_diagnostics
└──────────────────────┘
```

---

## 3. Node Responsibilities

### 3.1 `perception_node`

- Subscribes to RGB + depth camera topics.
- Runs YOLO26m human detection on RGB frames.
- Projects 2D detections to 3D using depth image, transforms camera → map frame via `tf2`.
- Feeds detections into a per-track Kalman filter (constant-velocity model) to produce $(x_h, y_h, v_x, v_y, c)$ per Eq. (5.1).
- Maintains a simple track manager (nearest-neighbor + track ID persistence) so identities remain stable across frames for the LSTM's input sequence.

### 3.2 `prediction_node`

- Maintains a rolling buffer of length $L$ per tracked human (Eq. 6.1).
- Runs the offline-trained LSTM (loaded from a frozen model file, see Dataset Specification doc) to produce $\hat{S}_h(k+1..k+H)$ (Eq. 6.2).
- Computes rolling prediction-error variance $\sigma_h$ (Eq. 6.3) using the last $W$ realized vs. predicted states.
- Publishes at $f_{LSTM}$ (slower than control loop, Section 13.1 of math spec).

### 3.3 `context_node`

- Computes relative-motion quantities (Eqs. 7.1–7.5) per tracked human.
- Computes corrected context score $z$ and $\phi_j = \sigma(z)$ (Eqs. 8.1–8.3), including the uncertainty term $u_h$.
- Applies EMA smoothing (Eq. 12.6) to produce $\phi_j^{filtered}$, then a dwell-time gate (Eq. 12.7) to produce $\phi_j^{used}$.
- Publishes $\phi_j$, $\phi_j^{filtered}$, $\phi_j^{used}$, and the aggregate $\phi = \max_j(\phi_j)$ (Eq. 9.1).

### 3.4 `adaptive_param_node`

- Maps $\phi$ (aggregate) and $\phi_j$ (per-human) to $d_{safe}(\phi_j)$, saturated $v_{x,max}, v_{y,max}, \omega_{max}$, and $Q(\phi)$ (Eqs. 10.1–10.3).
- Enforces the minimum-motion floors $v_{x,\min}, v_{y,\min}, \omega_{\min}$ from YAML parameters.

### 3.5 `nmpc_controller_node`

- Builds and solves the NMPC problem (Eqs. 11.1–11.2, 12.1–12.7) at $f_{NMPC}$.
- Uses per-human slack variables $s_j$ (Eq. 12.3) to guarantee feasibility.
- Publishes `/cmd_vel` and diagnostic info (solver status, slack activation, solve time) for the empirical feasibility validation described in Section 12.1 of the math spec.

---

## 4. Cross-Cutting Concerns

- **Synchronization:** `context_node` always reads the _latest_ Kalman-filtered human state for geometric terms, and only holds the LSTM-predicted trajectory constant between prediction updates (Section 13.1 of math spec) — implemented via message timestamps + `message_filters` approximate-time sync where needed.
- **Failure handling:** if `prediction_node` or `perception_node` drop out (no message within a timeout), `context_node` falls back to a conservative default ($\phi = 1$, i.e., maximally cautious) rather than holding a stale low-risk value.
- **Logging:** all nodes log to a common rosbag profile for reproducibility of experiments (see Dataset Specification doc for the same convention applied to LSTM training data).

---

## 5. Package Layout (proposed)

```text
cca_nmpc/
├── cca_nmpc_perception/
│   ├── perception_node.py
│   └── kalman_filter.py
├── cca_nmpc_prediction/
│   ├── prediction_node.py
│   └── lstm_model.py
├── cca_nmpc_context/
│   └── context_node.py
├── cca_nmpc_adaptive_params/
│   └── adaptive_param_node.py
├── cca_nmpc_control/
│   ├── nmpc_controller_node.py
│   └── nmpc_solver.py
├── cca_nmpc_msgs/          # custom message definitions (see Message Definitions doc)
├── cca_nmpc_bringup/
│   └── launch/cca_nmpc.launch.py
└── config/
    └── cca_nmpc_params.yaml   # (see YAML Parameters doc)
```
