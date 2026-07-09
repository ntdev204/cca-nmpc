# CCA-NMPC — ROS2 Message Definitions

Package: `cca_nmpc_msgs`. All custom `.msg` files below correspond directly to the mathematical objects defined in the Mathematical Model document (equation references noted per field). This document is kept in sync with the implemented package under `src/cca_nmpc_msgs/msg/`.

**Conventions:**
- Every array/complex message carries a `std_msgs/Header header`; `header.stamp` semantics are noted per message.
- All arrays (`HumanState[]`, `SlackValue[]`, `HumanSafetyDistance[]`, etc.) are matched by `track_id`, **not** by array position — consumers must join on `track_id` across topics published at different rates.

---

## 1. `HumanState.msg`

Corresponds to $S_h$ (Eq. 5.1). `header.stamp` = observation time from the detector.

```
# HumanState.msg
std_msgs/Header header
int32 track_id
float64 x              # x_h, map frame [m]
float64 y              # y_h, map frame [m]
float64 vx             # v_x, human velocity x [m/s]
float64 vy             # v_y, human velocity y [m/s]
float64 confidence     # c, detection confidence [0,1] — MUST validate at runtime
```

## 2. `HumanStateArray.msg`

`header.stamp` = time of the most recent human observation in the array.

```
# HumanStateArray.msg
std_msgs/Header header
HumanState[] humans
```

---

## 3. `HumanPrediction.msg`

Corresponds to $\hat{S}_h$ (Eq. 6.2), one predicted trajectory per tracked human. All `*_hat` arrays MUST have length `horizon_length` = $H$.

```
# HumanPrediction.msg
int32 track_id
int32 horizon_length   # H, number of prediction steps (for validation)
float64[] x_hat        # \hat{x}_h over horizon H
float64[] y_hat        # \hat{y}_h over horizon H
float64[] vx_hat       # \hat{v}_x over horizon H
float64[] vy_hat       # \hat{v}_y over horizon H
builtin_interfaces/Time prediction_stamp   # time the prediction was generated (staleness tracking, Section 13.1)
```

## 4. `HumanPredictionArray.msg`

`header.stamp` = prediction generation time.

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

`header.stamp` = uncertainty computation time.

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
float64 phi_j_used       # dwell-time-gated value (post-EMA), Eq. 12.7 -- feeds adaptive_param_node / solver
float64 d_j              # current distance to this human, Eq. 7.2
```

## 8. `ContextIndexArray.msg`

`header.stamp` = context computation time. Publishes **both** the raw aggregate and the gated aggregate: `phi_aggregate` (raw, for diagnostics) and `phi_aggregate_used` (gated, the value `adaptive_param_node` actually consumes).

```
# ContextIndexArray.msg
std_msgs/Header header
ContextIndex[] contexts
float64 phi_aggregate         # max_j(phi_j), raw value, Eq. 9.1
float64 phi_aggregate_used    # max_j(phi_j_used), gated value used by adaptive params
float64 d_h_aggregate         # min_j(d_j), Eq. 9.2
```

---

## 9. `HumanSafetyDistance.msg`

Per-human adaptive safety distance $d_{safe}(\phi_j)$ (Eq. 10.1), computed by `adaptive_param_node` and consumed by the solver's per-human constraint (Eq. 12.3).

```
# HumanSafetyDistance.msg
int32 track_id
float64 d_safe         # d_safe(phi_j) for this human
```

## 10. `AdaptiveParams.msg`

Corresponds to Section 10 of the Mathematical Model document. `header.stamp` = parameter computation time. The solver receives already-computed per-human `d_safe` values (see `08_solver_design.md` Section 15.1, resolved) via `d_safe_per_human`; the scalar `d_safe_aggregate` is monitoring only.

```
# AdaptiveParams.msg
std_msgs/Header header
float64 d_safe_aggregate     # for monitoring/diagnostics only
float64 vx_max               # Eq. 10.2
float64 vy_max               # Eq. 10.2
float64 omega_max            # Eq. 10.2
float64[3] Q_diag            # diagonal of Q(phi), Eq. 10.3 -- state [x_r, y_r, theta_r], NOT a flattened 3x3
float64 phi_aggregate_used   # phi value this parameter set was generated from
HumanSafetyDistance[] d_safe_per_human  # per-human d_safe(phi_j), matched by track_id
```

---

## 11. `SlackValue.msg`

Per-human slack variable for constraint relaxation (Eq. 12.3).

```
# SlackValue.msg
int32 track_id
float64 slack          # s_j, slack value for this human's safety constraint
```

## 12. `NmpcDiagnostics.msg`

Corresponds to Section 12.1 (feasibility/empirical validation) and Section 9 of `08_solver_design.md` (fallback). `header.stamp` = NMPC solve completion time. Slack is a `SlackValue[]` (per-human, matched by `track_id`), not a positional `float64[]`. Fallback is reported as both a machine-readable `fallback_code` and an optional human-readable `fallback_reason`.

```
# NmpcDiagnostics.msg

# Diagnostic level constants (mirrors diagnostic_msgs/DiagnosticStatus)
uint8 LEVEL_OK=0
uint8 LEVEL_WARN=1
uint8 LEVEL_ERROR=2
uint8 LEVEL_STALE=3

# Fallback reason constants
uint8 FALLBACK_NONE=0
uint8 FALLBACK_SOLVER_FAILED=1
uint8 FALLBACK_TIMEOUT=2
uint8 FALLBACK_HELD_PREVIOUS=3
uint8 FALLBACK_SAFE_STOP=4

std_msgs/Header header
uint8 diagnostic_level         # see LEVEL_* constants
bool solver_success
float64 solve_time_ms
SlackValue[] slacks            # s_j per tracked human, Eq. 12.3
int32 num_humans_active
float64 cost_total             # J, Eq. 11.1
float64 cost_track
float64 cost_control
float64 cost_smooth
float64 cost_human
float64 cost_obstacle
float64 cost_terminal
bool fallback_triggered        # true if this cycle published a held/decelerating control or safe stop
uint8 fallback_code            # see FALLBACK_* constants
string fallback_reason         # human-readable debug string, optional
```

---

## 13. Notes on Message Design

- Arrays are matched by `track_id`, never by array position, across all topics.
- `HumanPrediction.horizon_length` lets consumers validate that all `*_hat` arrays have length $H$ before indexing.
- `HumanPrediction.prediction_stamp` is what `context_node` uses to compute prediction staleness for the conservative $\sigma_h$ growth (Section 13.1 of the Mathematical Model document).
- `ContextIndexArray` publishes both `phi_aggregate` (raw) and `phi_aggregate_used` (gated). `adaptive_param_node` MUST use `phi_aggregate_used` for global caps / $Q(\phi)$; `phi_aggregate` is diagnostics only.
- `AdaptiveParams.d_safe_per_human` carries the per-human $d_{safe}(\phi_j)$ so the solver stays "dumb" (receives values, does not recompute) — this resolves `08_solver_design.md` open question 15.1.
- `NmpcDiagnostics` is intentionally verbose (per-cost-term breakdown, per-human slack, fallback code) to directly support the empirical feasibility logging in Section 12.1 without re-instrumenting the solver for evaluation/paper figures.
```
```
