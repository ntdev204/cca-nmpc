# CCA-NMPC — Deferred Work Roadmap (09)

## 1. Purpose

This document tracks work that is **intentionally deferred** rather than
missing by oversight. Each item states *why* it is deferred, *what unblocks it*,
and the *acceptance criteria* that mark it done. Reviewers should read a
`NotImplementedError` or a "smoke-test-only" launch in this repo as "see this
roadmap", not "forgotten".

Status legend: 🔴 not started · 🟡 in progress · 🟢 done.

---

## 2. acados deployment backend — 🔴 Deferred

**Where:** `src/cca_nmpc_control/cca_nmpc_control/nmpc_solver/acados_solver.py`
(every method raises `NotImplementedError`).

**Design:** fully specified in `docs/08_solver_design.md` Sections 2, 4, 6, 16.

**Why deferred:** acados uses a native C code-generation toolchain
(`acados_template` + HPIPM) that is built on the **target Linux platform**. The
current development environment is Windows, where the codegen toolchain is not
available. CasADi + IPOPT is the shipped reference backend and is sufficient for
formulation development, unit testing, and offline replay.

**What unblocks it:** access to the deployment Linux/Jetson platform with the
acados toolchain installed.

**Acceptance criteria:**

1. `AcadosSolver` implements the full `SolverInterface` (no `NotImplementedError`).
2. `nmpc_controller_node.solver_backend: "acados"` runs end-to-end on target HW.
3. **Parity test** (`docs/08_solver_design.md` Section 15, item 3) passes: on
   identical logged scenarios, `AcadosSolver` and `CasadiSolver` agree on `u0`,
   per-term `cost_breakdown`, and `SolverDiagnostics` within numeric tolerance.
4. Section 10 timing breakdown measured on target HW to substantiate the
   real-time claim.

**Interim contract (shipped now):** the stub raises `NotImplementedError` with a
message pointing here, so selecting the acados backend fails loudly and early
rather than silently degrading. Covered by
`nmpc_solver/tests/test_acados_stub.py`.

---

## 3. Gazebo Mecanum (holonomic) simulation — 🔴 Deferred for control results

**Where:** `src/turn_on_rai_robot/urdf/mini_mec_gazebo.urdf.xacro`,
`src/turn_on_rai_robot/launch/gazebo_sim.launch.py`.

**Why deferred:** the shipped launch uses `libgazebo_ros_diff_drive.so`, which
**ignores `cmd_vel.linear.y`**. That is fine for perception/logger smoke tests
but cannot reproduce holonomic Mecanum lateral motion, so no RMSE / smoothness /
safety claim involving lateral motion may be based on it.

**What unblocks it:** a holonomic plugin
(`libgazebo_ros_planar_move.so` / mecanum drive plugin) or a custom four-wheel
velocity mapper wired to the Mecanum URDF.

**Acceptance criteria:**

1. A commanded `cmd_vel.linear.y` produces lateral robot motion in Gazebo.
2. The launch used for control experiments is documented as holonomic.
3. The diff-drive launch is retained and clearly labelled smoke-test-only.

**Interim contract (shipped now):** `gazebo_sim.launch.py` accepts a
`drive: "diff" | "holonomic"` argument. `holonomic` loads the planar-move
plugin; `diff` stays the labelled smoke-test default.

---

## 4. Independent ground-truth validation subset — 🔴 Deferred

**Why deferred:** Dataset 01 records tracker observations (YOLO/depth/Kalman) —
pseudo-ground-truth, not an independent reference. Collecting a mocap /
overhead-camera / AprilTag reference subset needs lab hardware time.

**What unblocks it:** access to an independent reference (mocap, calibrated
overhead camera, AprilTag markers, or a measured floor path).

**Acceptance criteria:**

1. A small independently-annotated subset exists alongside Dataset 01.
2. Paper reports ADE/FDE **separately** for tracker pseudo-labels (training) and
   the independent reference subset (accuracy). See
   `docs/04_dataset_specification.md` Section 4 and Section 2.6.

---

## 5. Subject-held-out generalization split — 🟡 Partial

**Why partial:** trajectory-level split (shipped) prevents window leakage but the
same person can still appear in train/val/test, so it supports "seen-subject
trajectory prediction" but not "generalizes to new walking styles".

**Acceptance criteria:**

1. `subject_id` retained through `TrajectoryRecord` (shipped).
2. Optional group split by `subject_id` available in the builder.
3. Paper reports both trajectory split and subject-held-out split where subject
   labels exist.
