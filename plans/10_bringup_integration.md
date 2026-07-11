# Bringup & Integration Plan

## Overview

Tie the CCA-NMPC packages together for launch and end-to-end runtime, per `docs/02_system_architecture.md` and `docs/03_pipeline.md`. Provides the combined parameter file, a bringup launch chaining perception → prediction → context → adaptive params → control, and an offline (no-ROS) integration harness so the full data-flow can be validated on Windows before hardware.

## Environment Note

- Windows, no ROS2: build the ROS-free end-to-end harness now (recorded/synthetic human states → solver → cmd_vel), tested with `pytest`. Launch files are authored now and executed under ROS2 later.

## Project Type

**BACKEND / ROBOTICS integration.** Primary agent: `devops-engineer` (launch/config) + `backend-specialist` (harness); supporting: `test-engineer`.

## Success Criteria

- Single combined `cca_nmpc_params.yaml` (or per-node files) consistent with `docs/07_yaml_parameters.md`.
- `cca_nmpc_bringup` launch starts all nodes with correct remaps, QoS, and Astra depth alignment enabled.
- Rate ordering respected: `f_NMPC >= f_context >= f_LSTM`.
- Offline integration harness runs the full pipeline without ROS2 and produces a `cmd_vel` stream from synthetic inputs.
- Repo hygiene items required for a clean baseline are captured as tasks.

## Tech Stack

- `ament_python` bringup package + ROS2 launch.
- Offline harness: pure Python wiring of plan-01..08 core modules.
- `pytest`.

## File Structure

```text
src/cca_nmpc_bringup/
├── package.xml
├── setup.py
├── config/cca_nmpc_params.yaml
└── launch/
    ├── cca_nmpc.launch.py          # full stack
    └── perception_runtime.launch.py

tools/integration_harness/
├── __init__.py
├── pipeline.py     # wires scenario states->prediction->context->adaptive->solver
├── run_harness.py  # CLI: synthetic/recorded input -> cmd_vel log
└── tests/
    └── test_pipeline_e2e.py
```

## Task Breakdown

### BR-01 — Combined parameter file

Agent: `devops-engineer`; skills: `deployment-procedures`; priority: P0; dependencies: node plans' YAML.

INPUT → all per-node YAML blocks.
OUTPUT → `cca_nmpc_params.yaml` mirroring `docs/07_yaml_parameters.md` exactly (including the TensorRT/alignment/QoS updates).
VERIFY → keys match the docs 1:1; rate ordering holds.

### BR-02 — Offline integration harness

Agent: `backend-specialist`; skills: `python-patterns`, `clean-code`; priority: P1; dependencies: plans 01,04,05,06,07 core modules.

INPUT → synthetic or recorded human states + robot odom + goal.
OUTPUT → `pipeline.py` wires scenario states → prediction → context → adaptive → solver and emits `cmd_vel`.
VERIFY → `test_pipeline_e2e.py`: a scripted crossing scenario produces bounded `cmd_vel`, non-trivial `phi`, and no solver infeasibility.

### BR-03 — Full-stack launch

Agent: `devops-engineer`; skills: `deployment-procedures`, `bash-linux`; priority: P2; dependencies: node packages.

INPUT → all node packages + Astra camera launch.
OUTPUT → `cca_nmpc.launch.py` starts camera (depth_registration:=true), perception, prediction, context, adaptive, control with the combined params and correct QoS.
VERIFY (later, ROS2) → `ros2 launch cca_nmpc_bringup cca_nmpc.launch.py`; `ros2 node list` shows all five nodes; `/cmd_vel` published.

### BR-04 — End-to-end rosbag validation

Agent: `test-engineer`; skills: `webapp-testing`/`testing-patterns`; priority: P2; dependencies: BR-03.

INPUT → recorded sensor bag.
OUTPUT → replay procedure + expected topic checklist.
VERIFY (later, ROS2) → replay produces `/human_states`→`/human_predictions`→`/context_index`→`/adaptive_params`→`/cmd_vel`; diagnostics show low fallback/slack rates.

### BR-05 — Repo hygiene baseline

Agent: `devops-engineer`; skills: `clean-code`, `bash-linux`; priority: P2; dependencies: none.

INPUT → current repo state (empty README/LICENSE, stale `.pyc`, mismatched SLAM package names, empty `rai_sensors` bug already fixed by user).
OUTPUT → tasks: fill README/LICENSE, add `.gitignore` (ignore `build/ install/ log/ __pycache__/ *.engine models/`), remove stale bytecode, commit baseline on a `feature/*` branch per project convention.
VERIFY → `git status` clean of stray artifacts; baseline committed on a dedicated branch.

## Phase X: Verification

- Now (Windows): `pytest tools/integration_harness/tests/` passes; harness emits a `cmd_vel` log from synthetic input; combined YAML matches docs.
- Later (ROS2): `colcon build` whole workspace; full launch; rosbag replay end-to-end with diagnostics.

## Notes and Risks

- The offline harness is the key Windows deliverable: it proves the math pipeline end-to-end without ROS2 or hardware.
- Enforce rate ordering `f_NMPC >= f_context >= f_LSTM` in config and document it.
- Keep large binaries (`.engine`, model files, datasets) out of git; reference by path/version.
- Follow project convention: major changes on a dedicated `feature/*` branch.

## Implementation Status (2026-07-11)

- [x] BR-01: combined parameter file authored and aligned with the runtime TensorRT engine contract.
- [x] BR-02: ROS-free integration harness and crossing-scenario tests pass on Windows.
- [x] BR-03: bringup package and full-stack launch authored; ROS2 execution remains target-only verification.
- [ ] BR-04: rosbag replay requires a recorded bag and ROS2 target runtime.
- [x] BR-05: ignore rules cover generated Python, ROS build, model, engine, and dataset artifacts.

The implementation deliberately does not invent a Semantic A* reference topic: the current docs
describe `P_ref` conceptually but do not define its ROS topic/message contract. Likewise, the
Nav2 costmap subscription and differentiable soft-cost adapter remain integration work because a
raw `nav2_msgs/Costmap` grid cannot be inserted directly into the NLP described by `docs/08`.
