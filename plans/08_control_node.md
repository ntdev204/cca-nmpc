# NMPC Controller Node Plan

## Overview

Build `cca_nmpc_control`'s ROS2 node `nmpc_controller_node` that wraps the solver module (plan 07) into the control loop, per `docs/02_system_architecture.md` Section 3.5, `docs/08_solver_design.md` Sections 6.2, 9, 10. Subscribes `/adaptive_params`, `/human_predictions`, `/odom`, `/local_costmap/costmap`; publishes `/cmd_vel` and `/nmpc_diagnostics`.

The node is thin ROS glue over the (ROS-free) solver. Its non-ROS logic (fallback state machine, timing accounting, message assembly helpers) is testable now on Windows.

## Environment Note

- Windows, no ROS2: implement now; extract fallback/timeout state machine and diagnostics assembly into ROS-free helpers tested with `pytest`. Full node runs under ROS2 later.

## Project Type

**BACKEND / ROBOTICS ROS2 package.** Primary agent: `backend-specialist`; skills: `api-patterns`, `clean-code`; supporting: `test-engineer`, `security-auditor` (velocity safety), `performance-optimizer`.

## Success Criteria

- Constructs a `SolverInterface` backend from `nmpc_controller_node.solver_backend` YAML param.
- Every cycle: assemble reference + human predictions + adaptive params, call `solve(x0)` under a wall-clock deadline.
- Fallback hierarchy (Section 9): success -> publish `u0`; recent solution -> held/decayed control (bounded by `timeout_hold_cycles`); else safe-stop ramp + `reset()`.
- Warm-start invalidation on goal change / relocalization / lift / odom jump (Section 5.2).
- Publishes `/cmd_vel` and full `NmpcDiagnostics` (solver status, solve time, slack, cost breakdown, fallback flag/reason).
- Timing breakdown logged (Section 10).
- Fallback + timing helpers unit tested now.

## Tech Stack

- `ament_python`, `rclpy`, NumPy.
- Depends on `cca_nmpc_control.nmpc_solver` (plan 07) and `cca_nmpc_msgs`.
- Inputs: `AdaptiveParams`, `HumanPredictionArray`, `nav_msgs/Odometry`, `nav2_msgs/Costmap`; outputs `geometry_msgs/Twist`, `NmpcDiagnostics`.

## File Structure

```text
src/cca_nmpc_control/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/cca_nmpc_control
├── cca_nmpc_control/
│   ├── __init__.py
│   ├── nmpc_controller_node.py   # ROS glue
│   ├── fallback.py               # timeout/fallback state machine (pure)
│   ├── timing.py                 # cycle timing accounting (pure)
│   ├── invalidation.py           # odom-jump/goal-change detection (pure)
│   └── diagnostics.py            # NmpcDiagnostics assembly helpers (pure)
├── nmpc_solver/                  # from plan 07 (same package)
├── config/nmpc.yaml
├── launch/nmpc.launch.py
└── test/
    ├── test_fallback.py
    ├── test_timing.py
    └── test_invalidation.py
```

## Task Breakdown

### CT-01 — Package skeleton + YAML

Agent: `backend-specialist`; skills: `python-patterns`, `clean-code`; priority: P0; dependencies: solver plan SV-01.

INPUT → `docs/07_yaml_parameters.md` `nmpc_controller_node` block (now includes `timeout_hold_cycles`, `odom_jump_pos_thresh_m`, `odom_jump_yaw_thresh_rad`, `safe_stop_decel_limit`).
OUTPUT → package + `nmpc.yaml` (`solver_backend`, `horizon_N`, `dt`, `R/Rd/P_diag`, `w_h`, `w_obstacle`, `w_slack`, `C_collision`, `max_humans_in_solver`, `timeout_hold_cycles`, `odom_jump_pos_thresh_m`, `odom_jump_yaw_thresh_rad`, `safe_stop_decel_limit`).
VERIFY (now) → pure helpers import without `rclpy`. VERIFY (later) → `colcon build`.

### CT-02 — Fallback/timeout state machine

Agent: `backend-specialist`; skills: `clean-code`; priority: P1; dependencies: CT-01.

INPUT → solve success/timeout, recent-solution age, current speed.
OUTPUT → `fallback.py`: SUCCESS/FALLBACK logic (Section 6.2/9); held-vs-safe-stop with decelerating ramp (`safe_stop_decel_limit`); escalates after `timeout_hold_cycles`; emits a `fallback_code` matching `NmpcDiagnostics` FALLBACK_* constants.
VERIFY → `test_fallback.py`: success publishes u0 (FALLBACK_NONE); solver-fail/timeout map to the right codes; N consecutive failures escalate to safe stop (FALLBACK_SAFE_STOP); ramp respects `safe_stop_decel_limit`.

### CT-03 — Invalidation detection

Agent: `backend-specialist`; skills: `clean-code`; priority: P1; dependencies: CT-01.

INPUT → consecutive `/odom`, goal messages, relocalization/lift events; thresholds `odom_jump_pos_thresh_m`, `odom_jump_yaw_thresh_rad`.
OUTPUT → `invalidation.py`: detect goal change, odom position/yaw jump beyond thresholds, lift, relocalization; emit a reset signal.
VERIFY → `test_invalidation.py`: position/yaw jumps beyond thresholds trigger reset; normal motion does not.

### CT-04 — Timing accounting

Agent: `performance-optimizer`; skills: `performance-profiling`; priority: P2; dependencies: CT-01.

INPUT → per-segment timestamps.
OUTPUT → `timing.py`: `parameter_update / solver / publish / total_cycle` accounting (Section 10) vs `dt`.
VERIFY → `test_timing.py`: totals sum correctly; over-budget cycles flagged.

### CT-05 — Diagnostics assembly

Agent: `backend-specialist`; skills: `clean-code`; priority: P2; dependencies: CT-02.

INPUT → SolveResult + fallback state + timing.
OUTPUT → `diagnostics.py`: build `NmpcDiagnostics` per the real message — `header`, `diagnostic_level` (LEVEL_* const), `solver_success`, `solve_time_ms`, `slacks` (`SlackValue[]`, per-human by `track_id`), `num_humans_active`, per-cost terms, `fallback_triggered`, `fallback_code` (FALLBACK_* const), optional `fallback_reason` string.
VERIFY → helper produces a fully populated message struct from sample inputs; `slacks` is a list of `SlackValue`, NOT a positional `float64[]`; `fallback_code` maps to the FALLBACK_* constants.

### CT-06 — Node glue

Agent: `backend-specialist`; skills: `api-patterns`; priority: P1; dependencies: CT-02..CT-05, solver SV-04.

INPUT → four subscriptions; timer at `f_NMPC` (`1/dt`).
OUTPUT → `nmpc_controller_node.py`: builds backend from YAML, runs the cycle with a deadline, applies fallback, publishes `/cmd_vel` + `/nmpc_diagnostics`, calls `reset()` on invalidation.
VERIFY (later) → `ros2 run` with replayed inputs publishes bounded `/cmd_vel` and diagnostics; injected solver failure exercises fallback.

### CT-07 — Launch + tests

Agent: `test-engineer`; skills: `testing-patterns`; priority: P2; dependencies: CT-06.

OUTPUT → `nmpc.launch.py`; pure-logic tests green now.
VERIFY (now) → `pytest src/cca_nmpc_control/test/`. VERIFY (later) → `colcon test`.

## Phase X: Verification

- Now (Windows): `pytest` on fallback/timing/invalidation passes; helpers import without ROS2; solver (CasADi) drives an offline batch-replay of a scenario without ROS.
- Later (ROS2): `colcon build`/`colcon test`; live loop with `/adaptive_params` + `/human_predictions` + `/odom`; verify `/cmd_vel` bounds and fallback logging.

## Notes and Risks

- Security/safety: `/cmd_vel` must always be bounded and default to safe-stop on any uncertainty; never publish an unbounded or stale-indefinite command.
- Safe stop should decelerate within actuator limits, not command instant zero at speed (Section 9).
- The node must depend only on `SolverInterface`, no backend branching in node code.
- Offline batch-replay harness (solver + recorded inputs, no ROS) is the way to validate control logic now on Windows.
