# Context Node Plan

## Overview

Build `cca_nmpc_context`: computes the continuous context index per tracked human, per `docs/01_mathematical_model.md` Sections 7–9, 12.1 and `docs/02_system_architecture.md` Section 3.3. Subscribes `/human_states`, `/human_predictions`, `/human_pred_uncertainty`, `/odom`; publishes `/context_index` (per-human `phi_j`, filtered, gated, plus aggregate).

This node is almost entirely pure math, so the core is fully implementable and testable now on Windows without ROS2.

## Environment Note

- Windows, no ROS2: implement the math core as ROS-free modules tested now with `pytest`; ROS glue is a thin wrapper verified later.

## Project Type

**BACKEND / ROBOTICS ROS2 package (math-heavy).** Primary agent: `backend-specialist`; skills: `python-patterns`, `clean-code`; supporting: `test-engineer`.

## Success Criteria

- Relative-motion terms `d_h`, `v_rel`, `cos(delta_theta)` computed correctly (Eqs. 7.1–7.5) using the latest Kalman-filtered current state, not stale predicted position.
- Corrected context score `z` and `phi_j = sigma(z)` with uncertainty term `u_h` (Eqs. 8.1–8.3).
- EMA smoothing (Eq. 12.6) then dwell-time gate (Eq. 12.7) producing `phi_j_filtered` and `phi_j_used`.
- Aggregate `phi = max_j phi_j`, `d_h = min_j d_j` (Eqs. 9.1–9.2).
- Conservative fallback `phi = 1` on perception/prediction dropout (Architecture Section 4).
- Math core unit tested now.

## Tech Stack

- `ament_python`, `rclpy`, NumPy.
- Messages: existing `cca_nmpc_msgs` (`HumanStateArray`, `HumanPredictionArray`, `HumanUncertaintyArray`, `ContextIndexArray`) and `nav_msgs/Odometry`.

## File Structure

```text
src/cca_nmpc_context/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/cca_nmpc_context
├── cca_nmpc_context/
│   ├── __init__.py
│   ├── context_node.py     # ROS glue (thin)
│   ├── relative_motion.py  # Eqs 7.1-7.5 (pure)
│   ├── context_score.py    # z, phi_j, u_h (Eqs 8.1-8.3) (pure)
│   ├── smoothing.py        # EMA + dwell-time gate (Eqs 12.6-12.7) (pure)
│   └── aggregate.py        # max/min aggregation (Eqs 9.1-9.2) (pure)
├── config/context.yaml
├── launch/context.launch.py
└── test/
    ├── test_relative_motion.py
    ├── test_context_score.py
    ├── test_smoothing.py
    └── test_aggregate.py
```

## Task Breakdown

### CX-01 — Package skeleton + YAML

Agent: `backend-specialist`; skills: `python-patterns`, `clean-code`; priority: P0; dependencies: msgs package.

INPUT → layout above; `docs/07_yaml_parameters.md` context block.
OUTPUT → package with `context.yaml` (`d0`, `v_max_ref`, `epsilon`, `sigma_max`, weights `w_d/w_v/w_theta/w_u/b`, `ema alpha`, `T_dwell_cycles`, `fallback_phi_on_dropout`).
VERIFY (now) → pure modules import without `rclpy`. VERIFY (later) → `colcon build`.

### CX-02 — Relative motion

Agent: `backend-specialist`; skills: `python-patterns`; priority: P1; dependencies: CX-01.

INPUT → robot pose/velocity + human state.
OUTPUT → `relative_motion.py`: `d_h` (Eq. 7.2), robot velocity in world (Eq. 7.3), unit direction `e` (Eq. 7.4), `cos(delta_theta)` (Eq. 7.5) with `epsilon` guards.
VERIFY → `test_relative_motion.py`: known geometry cases (head-on, crossing, receding) yield expected signs/values; epsilon prevents divide-by-zero.

### CX-03 — Context score

Agent: `backend-specialist`; skills: `python-patterns`; priority: P1; dependencies: CX-02.

INPUT → `d_h`, `|v_h|`, `cos(delta_theta)`, confidence `c`, `sigma_h_tilde`, weights.
OUTPUT → `context_score.py`: `u_h = 1 - c*(1 - sigma_tilde)` (Eq. 8.3), `z` (Eq. 8.2), `phi_j = sigma(z)` (Eq. 8.1).
VERIFY → `test_context_score.py`: low confidence or high uncertainty raises `phi_j`; sanity that phi in [0,1].

### CX-04 — EMA + dwell-time gate

Agent: `backend-specialist`; skills: `clean-code`; priority: P1; dependencies: CX-03.

INPUT → per-cycle `phi_j`, `alpha`, `T_dwell_cycles`, per-track gate state.
OUTPUT → `smoothing.py`: EMA `phi_j_filtered` (Eq. 12.6), then dwell gate `phi_j_used` holding for `T_dwell` cycles (Eq. 12.7).
VERIFY → `test_smoothing.py`: EMA converges; gated value only updates every `T_dwell` cycles; state is per-track.

### CX-05 — Aggregation + fallback

Agent: `backend-specialist`; skills: `clean-code`; priority: P1; dependencies: CX-04.

INPUT → per-human `phi_j`, `phi_j_used`, `d_j`; dropout/timeout signal.
OUTPUT → `aggregate.py`: `phi_aggregate = max_j phi_j` (raw, diagnostics), `phi_aggregate_used = max_j phi_j_used` (gated, consumed by adaptive), `d_h_aggregate = min_j d_j`; conservative `phi = 1` fallback on dropout applied to the used aggregate.
VERIFY → `test_aggregate.py`: raw vs used aggregates computed independently; correct max/min; empty/stale input yields conservative fallback (`phi_aggregate_used = 1`), not a stale low value.

### CX-06 — Node glue

Agent: `backend-specialist`; skills: `api-patterns`; priority: P1; dependencies: CX-02..CX-05.

INPUT → four subscriptions; runs every control cycle; holds predicted trajectory between LSTM updates but recomputes geometric terms from the latest current state (Section 13.1).
OUTPUT → `context_node.py` publishes `ContextIndexArray` with `header`, per-human `ContextIndex[]` (`track_id, z, phi_j, phi_j_filtered, phi_j_used, d_j`), and both aggregates `phi_aggregate`, `phi_aggregate_used`, `d_h_aggregate`.
VERIFY (later) → `ros2 topic echo /context_index` with replayed inputs shows both aggregates; dropout of a source drives `phi_aggregate_used -> 1`.

## Phase X: Verification

- Now (Windows): `pytest src/cca_nmpc_context/test/` passes; math core imports without ROS2.
- Later (ROS2): `colcon build` + `colcon test`; integration with replayed `/human_states` + `/human_predictions` + `/odom`.

## Notes and Risks

- Recompute geometric terms from the freshest current state each cycle; only the predicted future trajectory is held between LSTM refreshes.
- Fallback must be conservative (`phi = 1`) on missing/stale inputs — never hold a low-risk value.
- Keep EMA and dwell-gate distinct: EMA bounds noise, the gate bounds update frequency; do not merge them.
