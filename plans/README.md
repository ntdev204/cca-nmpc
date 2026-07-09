# CCA-NMPC Implementation Plans

Task plans to complete the full CCA-NMPC system described in `docs/`. Each plan is self-contained (Overview, Success Criteria, Tech Stack, File Structure, Task Breakdown, Phase X verification, Notes/Risks).

## Environment assumptions

- **Windows, no ROS2 installed.** Code is written now; ROS-dependent tests run later once ROS2 is available.
- Every plan splits logic into **ROS-free core modules** (unit-tested now with `pytest`) and **thin ROS glue** (verified later with `colcon build` / `colcon test`).
- Two verification tiers per plan: **VERIFY (now)** on Windows, and **VERIFY (later)** under ROS2.

## Baseline already done

- `src/cca_nmpc_msgs` — all 10 custom messages implemented.
- `docs/07_yaml_parameters.md` — updated to TensorRT `.engine`, Astra topic defaults, optical-frame projection, depth-alignment requirement, sensor QoS.

## Plans and suggested order

| # | Plan | What it delivers | ROS2 needed to finish? |
|---|------|------------------|------------------------|
| 01 | [Human perception](01_human_perception.md) | `cca_nmpc_perception`: RGB-D → YOLO26m TensorRT → depth/TF → Kalman → `/human_states` | Runtime yes; core testable now |
| 02 | [LSTM dataset](02_lstm_dataset.md) | Offline dataset preprocessing → windowed tensors + norm stats | No (pure Python) |
| 03 | [LSTM training](03_lstm_training.md) | Train + export ONNX predictor | No (CPU smoke now) |
| 04 | [Prediction node](04_prediction_node.md) | `cca_nmpc_prediction`: LSTM inference + `sigma_h` → predictions/uncertainty | Runtime yes; core now |
| 05 | [Context node](05_context_node.md) | `cca_nmpc_context`: `phi_j`, EMA, dwell gate, aggregation | Mostly now (math core) |
| 06 | [Adaptive params](06_adaptive_param_node.md) | `cca_nmpc_adaptive_params`: `d_safe`, saturated caps, `Q(phi)` | Mostly now (closed-form) |
| 07 | [NMPC solver](07_nmpc_solver.md) | Solver lib: `SolverInterface` + CasADi (now) + acados (later) | CasADi now; acados later |
| 08 | [Control node](08_control_node.md) | `cca_nmpc_control` node: solve loop + fallback + diagnostics | Runtime yes; core now |
| 09 | [Context calibration](09_context_calibration.md) | Offline 2-stage weight calibration + sensitivity sweep | No (uses CasADi) |
| 10 | [Bringup & integration](10_bringup_integration.md) | Combined params, launch, offline E2E harness, repo hygiene | Launch later; harness now |

## Dependency graph (high level)

```text
02 dataset ──> 03 training ──> 04 prediction ──┐
                                               ├─> 05 context ──> 06 adaptive ──┐
01 perception ─────────────────> /human_states ┘                               │
                                                                                ▼
                                        07 solver ──────────────> 08 control node
                                            │                            │
                        05 context math ──> 09 calibration               │
                                                                         ▼
                              01,04,05,06,07,08 ──> 10 bringup + offline E2E harness
```

## What can be built fully on Windows now

- 02, 03, 09 (offline pure Python / PyTorch / CasADi).
- 07 CasADi backend and all its tests.
- The ROS-free math cores of 04, 05, 06, 08, plus the offline integration harness in 10.

## What waits for a ROS2 machine

- ROS glue nodes, `colcon build`/`colcon test`, launch files, live topic/rosbag validation.
- 07 acados backend and CasADi/acados parity.
- 01 hardware runtime with the Astra camera and TensorRT engine.
