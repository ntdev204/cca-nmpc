# Adaptive Parameter Node Plan

## Overview

Build `cca_nmpc_adaptive_params`: maps the context index to NMPC runtime parameters, per `docs/01_mathematical_model.md` Section 10 and `docs/02_system_architecture.md` Section 3.4. Subscribes `/context_index`; publishes `/adaptive_params` (`d_safe`, saturated `v_max`/`omega_max`, `Q(phi)` diag).

Pure closed-form math — fully implementable and testable now on Windows.

## Environment Note

- Windows, no ROS2: the parameter-mapping core is ROS-free and tested now with `pytest`; ROS glue verified later.

## Project Type

**BACKEND / ROBOTICS ROS2 package (math-light, closed-form).** Primary agent: `backend-specialist`; skills: `python-patterns`, `clean-code`; supporting: `test-engineer`.

## Success Criteria

- `d_safe(phi) = d_safe0 + k_d*phi` (Eq. 10.1).
- Saturated limits with floors (Eq. 10.2): `v = max(v_min, v0 - k*phi)` for `vx, vy, omega`.
- `Q(phi) = Q0 + phi*Qh` diagonal (Eq. 10.3).
- Consumes `ContextIndexArray.phi_aggregate_used` (gated) for global caps/`Q` — NOT the raw `phi_aggregate`.
- Computes per-human `d_safe(phi_j)` and publishes it in `AdaptiveParams.d_safe_per_human` (`HumanSafetyDistance[]`, matched by `track_id`) for the solver (resolved decision, docs/08 §15.1).
- Publishes `AdaptiveParams` with `header`, `Q_diag` as a length-3 diagonal, and `d_safe_aggregate` (monitoring only).
- Unit tested now.

## Tech Stack

- `ament_python`, `rclpy`, NumPy.
- Messages: existing `cca_nmpc_msgs` (`ContextIndexArray`, `AdaptiveParams`).

## File Structure

```text
src/cca_nmpc_adaptive_params/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/cca_nmpc_adaptive_params
├── cca_nmpc_adaptive_params/
│   ├── __init__.py
│   ├── adaptive_param_node.py   # ROS glue (thin)
│   └── param_map.py             # closed-form maps (pure)
├── config/adaptive_params.yaml
├── launch/adaptive_params.launch.py
└── test/
    └── test_param_map.py
```

## Task Breakdown

### AP-01 — Package skeleton + YAML

Agent: `backend-specialist`; skills: `python-patterns`, `clean-code`; priority: P0; dependencies: msgs package.

INPUT → `docs/07_yaml_parameters.md` adaptive block.
OUTPUT → package + `adaptive_params.yaml` (`d_safe0`, `k_d`, `v_x0/v_y0/omega_0`, `k_x/k_y/k_omega`, `v_x_min/v_y_min/omega_min`, `Q0_diag`, `Qh_diag`).
VERIFY (now) → `param_map.py` imports without `rclpy`. VERIFY (later) → `colcon build`.

### AP-02 — Closed-form parameter maps

Agent: `backend-specialist`; skills: `python-patterns`; priority: P1; dependencies: AP-01.

INPUT → aggregate `phi_aggregate_used` and per-human `phi_j_used`.
OUTPUT → `param_map.py`: `d_safe(phi_j)`, saturated `vx_max/vy_max/omega_max` with floors, `Q_diag(phi) = Q0_diag + phi*Qh_diag`.
VERIFY → `test_param_map.py`: at `phi=0` returns nominal; at `phi=1` respects floors and never goes below `v_min`; `Q` scales linearly; per-human `d_safe(phi_j)` monotone in `phi_j`.

### AP-03 — Node glue

Agent: `backend-specialist`; skills: `api-patterns`; priority: P1; dependencies: AP-02.

INPUT → `/context_index` (`phi_aggregate_used` + per-human `ContextIndex[]` with `phi_j_used`).
OUTPUT → `adaptive_param_node.py` publishes `AdaptiveParams` with `header`, `d_safe_aggregate` (monitoring), `vx_max/vy_max/omega_max`, `Q_diag`, `phi_aggregate_used`, and `d_safe_per_human` (`HumanSafetyDistance[]`, one entry per active track by `track_id`).
VERIFY (later) → `ros2 topic echo /adaptive_params` reacts to changing `phi`; `d_safe_per_human` has one entry per active human keyed by `track_id`.

### AP-04 — Launch + tests

Agent: `test-engineer`; skills: `testing-patterns`; priority: P2; dependencies: AP-03.

OUTPUT → `adaptive_params.launch.py`; tests green now.
VERIFY (now) → `pytest src/cca_nmpc_adaptive_params/test/`. VERIFY (later) → `colcon test`.

## Phase X: Verification

- Now (Windows): `pytest` on `param_map` passes; module imports without ROS2.
- Later (ROS2): `colcon build`/`colcon test`; live reaction to `/context_index`.

## Notes and Risks

- `AdaptiveParams.Q_diag` is a length-3 diagonal over `[x_r, y_r, theta_r]`, NOT a flattened 3x3 — keep consistent with the message.
- RESOLVED (docs/08 §15.1): this node computes per-human `d_safe(phi_j)` and publishes it in `AdaptiveParams.d_safe_per_human`; the solver consumes those values directly and does not recompute from `phi_j`.
- Use `phi_aggregate_used` (gated), not raw `phi_aggregate`, for global caps and `Q`.
- Floors (`v_min`) prevent the "frozen robot" failure; never let saturation drive a bound to zero.
