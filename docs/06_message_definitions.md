# CCA-NMPC — ROS2 Message Definitions

Package: `cca_nmpc_msgs`. All custom `.msg` files below correspond directly to the mathematical objects defined in the Mathematical Model document (equation references noted per field).

---

## 1. `HumanState.msg`

Corresponds to $S_h$ (Eq. 5.1).

```
# HumanState.msg
int32 track_id
float64 x              # x_h, map frame [m]
float64 y              # y_h, map frame [m]
float64 vx             # v_x, human velocity x [m/s]
float64 vy             # v_y, human velocity y [m/s]
float64 confidence     # c, detection confidence [0,1]
builtin_interfaces/Time stamp
```

## 2. `HumanStateArray.msg`

```
# HumanStateArray.msg
std_msgs/Header header
HumanState[] humans
```

---

## 3. `HumanPrediction.msg`

Corresponds to $\hat{S}_h$ (Eq. 6.2), one predicted trajectory per tracked human.

```
# HumanPrediction.msg
int32 track_id
float64[] x_hat        # \hat{x}_h over horizon H
float64[] y_hat        # \hat{y}_h over horizon H
float64[] vx_hat       # \hat{v}_x over horizon H
float64[] vy_hat       # \hat{v}_y over horizon H
builtin_interfaces/Time prediction_stamp   # time the prediction was generated (for staleness tracking, Section 13.1)
```

## 4. `HumanPredictionArray.msg`

```
# HumanPredictionArray.msg
std_msgs/Header header
HumanPrediction[] predictions
```

---

## 5. `HumanUncertainty.msg`

Corresponds to $\sigma_h$ / $\tilde\sigma_h$ (Eq. 6.3, 8.3).

```
# HumanUncertainty.msg
int32 track_id
float64 sigma_h         # rolling prediction-error variance, Eq. 6.3
float64 sigma_h_clipped # \tilde{sigma}_h, clipped to [0,1], Eq. 8.3
```

## 6. `HumanUncertaintyArray.msg`

```
# HumanUncertaintyArray.msg
std_msgs/Header header
HumanUncertainty[] uncertainties
```

---

## 7. `ContextIndex.msg`

Corresponds to per-human $\phi_j$, $\phi_j^{filtered}$, and $\phi_j^{used}$ (Eqs. 8.1, 9.1, 12.6, 12.7).

```
# ContextIndex.msg
int32 track_id
float64 z               # raw context score, Eq. 8.2
float64 phi_j            # sigma(z), Eq. 8.1
float64 phi_j_filtered   # EMA-smoothed value, Eq. 12.6
float64 phi_j_used       # dwell-time-gated value (post-EMA), Eq. 12.7 -- this is what feeds adaptive_param_node / the solver
float64 d_j              # current distance to this human, Eq. 7.2
```

## 8. `ContextIndexArray.msg`

```
# ContextIndexArray.msg
std_msgs/Header header
ContextIndex[] contexts
float64 phi_aggregate    # max_j(phi_j), Eq. 9.1
float64 d_h_aggregate    # min_j(d_j), Eq. 9.2
```

---

## 9. `AdaptiveParams.msg`

Corresponds to Section 10 of the Mathematical Model document.

```
# AdaptiveParams.msg
float64 d_safe_aggregate     # for monitoring/diagnostics only; per-human d_safe used internally by solver
float64 vx_max               # Eq. 10.2
float64 vy_max               # Eq. 10.2
float64 omega_max            # Eq. 10.2
float64[3] Q_diag            # diagonal of Q(phi), Eq. 10.3 -- nx=3 state [x_r, y_r, theta_r] (Section 3.1 of Solver Design doc), NOT a flattened 3x3 matrix
float64 phi_aggregate_used   # phi value this parameter set was generated from
```

---

## 10. `NmpcDiagnostics.msg`

Corresponds to Section 12.1 (feasibility/empirical validation requirements) and Section 9 of `08_solver_design.md` (real-time timeout / fallback strategy).

```
# NmpcDiagnostics.msg
std_msgs/Header header
bool solver_success
float64 solve_time_ms
float64[] slack_values        # s_j per tracked human, Eq. 12.3
int32 num_humans_active
float64 cost_total            # J, Eq. 11.1
float64 cost_track
float64 cost_control
float64 cost_smooth
float64 cost_human
float64 cost_obstacle
float64 cost_terminal
bool fallback_triggered        # true if this cycle published a held/decelerating control or a safe stop
                                # instead of a fresh solve, per the fallback hierarchy (Solver Design doc, Section 9)
string fallback_reason         # one of: "" (no fallback), "solver_failed", "timeout",
                                # "held_previous", "safe_stop" -- reason the fallback hierarchy was entered
```

---

## 11. Notes on Message Design

- All arrays (`HumanState[]`, `HumanPrediction[]`, etc.) are indexed consistently by `track_id`, **not** by array position, so consumers must match on `track_id` rather than assuming aligned indices across topics published at different rates.
- `builtin_interfaces/Time prediction_stamp` in `HumanPrediction.msg` is the field that lets `context_node` compute prediction staleness for the conservative $\sigma_h$ growth described in Section 13.1 of the Mathematical Model document.
- `NmpcDiagnostics.msg` is intentionally verbose (per-cost-term breakdown) to directly support the empirical feasibility logging required by Section 12.1, without needing to re-instrument the solver later for evaluation/paper figures.
- `fallback_triggered` / `fallback_reason` make the fallback hierarchy in `08_solver_design.md` Section 9 directly reportable from logged rosbags, the same way `slack_values` already makes Section 9 of the Mathematical Model doc (feasibility) reportable — fallback frequency and slack activation frequency are logged the same way, and both are meant to appear together in the paper's real-time/feasibility results.
