# CCA-NMPC — YAML Parameters

All tunable constants referenced across the Mathematical Model, System Architecture, and Pipeline documents are externalized here (`config/cca_nmpc_params.yaml`), loaded per-node at launch via ROS2 parameter files. No constant listed below should be hard-coded in node source.

```yaml
# config/cca_nmpc_params.yaml

perception_node:
  ros__parameters:
    yolo_engine_path: "models/yolo26m_human.engine" # TensorRT serialized engine (built per target GPU)
    detection_confidence_threshold: 0.4 # minimum c to accept a raw detection
    max_track_age_sec: 1.0 # drop a track if unseen this long
    kalman:
      process_noise_std: 0.1 # Q process noise (Kalman, constant-velocity model)
      measurement_noise_std: 0.15 # R measurement noise
    # Topics — defaults match the Astra camera driver (astra.launch.xml), namespaced under /camera
    rgb_image_topic: "/camera/color/image_raw"
    depth_image_topic: "/camera/aligned_depth_to_color/image_raw" # aligned depth-to-color (matches require_depth_alignment: true)
    camera_info_topic: "/camera/color/camera_info"
    # Sensor topics use best-effort SensorDataQoS to match the Astra publisher (color_qos/depth_qos: "default")
    sensor_qos: "sensor_data" # one of: sensor_data | reliable
    # Frames — project into the OPTICAL frame reported by CameraInfo.header.frame_id, NOT camera_link.
    # camera_link is not axis-aligned with the optical frame; pinhole projection must use the optical frame.
    camera_optical_frame: "camera_color_optical_frame" # overridden by incoming CameraInfo/image header.frame_id at runtime
    map_frame: "map"
    # Depth-to-color alignment is REQUIRED for correct projection. The Astra driver ships with
    # depth_registration:=false by default, so either launch it with depth_registration:=true
    # (hardware D2C) or subscribe to an explicitly aligned depth-to-color topic.
    require_depth_alignment: true

prediction_node:
  ros__parameters:
    model_path: "models/lstm_predictor_v1_on_lstm_dataset_v1.onnx"
    normalization_stats_path: "models/normalization_stats.json"
    L: 8 # input sequence length, Eq. 6.1
    H: 12 # prediction horizon length, Eq. 6.2
    f_lstm_hz: 8.0 # f_LSTM, Section 13.1
    uncertainty_window_W: 20 # W in Eq. 6.3
    sigma_growth_rate_beta: 0.05 # beta in Eq. 13.3, sigma_h growth rate while a prediction is held stale between LSTM refreshes

context_node:
  ros__parameters:
    d0: 3.0 # d_0 in Eq. 8.2 [m]
    v_max_ref: 1.5 # v_max normalizer in Eq. 8.2 [m/s]
    epsilon: 1.0e-3 # epsilon in Eqs. 7.4, 7.5
    sigma_max: 0.5 # sigma_max in Eq. 8.3
    weights: # calibrated per Section 8.1 — do not hand-edit without re-running calibration
      w_d: 1.2
      w_v: 0.8
      w_theta: 0.6
      w_u: 1.0
      b: -0.5
    ema_filter:
      alpha: 0.7 # Eq. 12.6 EMA smoothing factor, produces phi_j_filtered
    dwell_time:
      T_dwell_cycles: 4 # Eq. 12.7 minimum cycles between phi_j_used updates, design constant, Section 12.1
    fallback_phi_on_dropout: 1.0 # conservative default per System Architecture doc Section 4

adaptive_param_node:
  ros__parameters:
    d_safe0: 0.6 # d_safe,0 in Eq. 10.1 [m]
    k_d: 0.8 # Eq. 10.1
    v_x0: 1.0 # nominal max forward speed [m/s]
    v_y0: 0.8 # nominal max lateral speed [m/s]
    omega_0: 1.2 # nominal max angular speed [rad/s]
    k_x: 0.6
    k_y: 0.5
    k_omega: 0.7
    v_x_min: 0.08 # floor, Section 10 ("frozen robot" fix
    v_y_min: 0.06
    omega_min: 0.1
    Q0_diag: [5.0, 5.0, 2.0] # Q_0 in Eq. 10.3 (diag over tracking-error states)
    Qh_diag: [8.0, 8.0, 3.0] # Q_h in Eq. 10.3

nmpc_controller_node:
  ros__parameters:
    solver_backend: "acados" # or "casadi"
    horizon_N: 20
    dt: 0.05 # control period [s] -> f_NMPC = 20 Hz
    R_diag: [0.1, 0.1, 0.05] # control effort weight, Eq. 11.2
    Rd_diag: [0.05, 0.05, 0.02] # control smoothness weight, Eq. 11.2
    w_h: 3.0 # human-avoidance weight, Eq. 11.2
    w_obstacle: 2.0 # obstacle-cost weight, Eq. 11.2
    P_diag: [10.0, 10.0, 4.0] # terminal cost weight, Eq. 11.2
    w_slack: 50.0 # slack penalty, Eq. 12.5
    C_collision: 0.9 # obstacle-cost collision threshold, Eq. 12.4
    max_humans_in_solver: 6 # cap on per-human constraints (Eq. 12.3) for real-time solve bound
    # Real-time fallback / warm-start invalidation (Solver Design doc, Sections 5.2, 9)
    timeout_hold_cycles: 3 # max consecutive failed/late solves to hold previous control before safe-stop (Section 9)
    odom_jump_pos_thresh_m: 0.30 # position discontinuity between consecutive /odom beyond this -> reset() (Section 5.2)
    odom_jump_yaw_thresh_rad: 0.35 # heading discontinuity beyond this -> reset() (Section 5.2)
    safe_stop_decel_limit: 1.0 # max deceleration [m/s^2] for the safe-stop ramp (Section 9)

calibration:
  ros__parameters:
    # Only used by the offline Section 8.1 calibration script, not at runtime.
    dataset_path: "datasets/context_calib_v1/"
    perturbation_pct: 0.20 # +-20% sensitivity sweep, Section 8.1
    max_allowed_solve_degradation_pct: 10.0 # threshold that triggers weight re-scaling
```

---

## Notes

- **Naming mirrors the math spec:** every symbol from the Mathematical Model document (e.g., $k_d$, $w_d$, $\alpha$, $T_{dwell}$, $\beta$) has a 1:1 YAML key, so the paper's methodology section and the codebase never drift apart.
- **Calibrated vs. hand-set:** parameters under `context_node.weights` are explicitly commented as calibration outputs (Section 8.1) rather than free hand-tuned constants — this distinction matters for the paper's reproducibility claims.
- **Rate consistency:** `prediction_node.f_lstm_hz` (8 Hz) < `nmpc_controller_node.dt` implied rate (20 Hz) enforces the ordering $f_{NMPC} \ge f_{context} \ge f_{LSTM}$ required by the Pipeline document.
- Values shown (e.g., $k_d = 0.8$, $d_{safe,0}=0.6$) are **placeholders** to illustrate structure — all must be replaced by the Section 8.1 calibration procedure and platform-specific tuning before real experiments.
