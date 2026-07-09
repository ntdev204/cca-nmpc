---
name: implementation-plan-state
description: CCA-NMPC full-implementation orchestration state — what each plan delivers, deps, and progress
metadata:
  type: project
---

Goal: implement all 10 plans, mathematically correct per docs/01_mathematical_model.md, runnable, codex-accepted. Learn from each failure (project skills+memory).

**Dependency tiers (build order):**
- Tier 0 (offline pure Python, no cross-deps): plan 02 `tools/lstm_dataset/`, plan 03 `tools/lstm_training/`, plan 07 `src/cca_nmpc_control/nmpc_solver/` (CasADi).
- Tier 1: plan 04 `cca_nmpc_prediction` (needs ONNX+stats from 02/03), plan 05 `cca_nmpc_context` (math core), plan 06 `cca_nmpc_adaptive_params` (needs 05 output shape), plan 08 `cca_nmpc_control` node (needs 07 solver).
- Tier 2: plan 09 `tools/context_calibration/` (needs 05 features + 07 solver), plan 10 `cca_nmpc_bringup` + `tools/integration_harness/` (needs 01,04,05,06,07,08 cores).

**Cross-cutting contracts (DO NOT DRIFT):**
- Dataset↔training interface: `.npz` with keys `inputs` (N,L,4), `targets` (N,H,4); `normalization_stats.json` = per-channel mean/std for 4 channels ordered [x,y,vx,vy]. L=8, H=12 (docs/07).
- msg `AdaptiveParams.q_diag` is LOWERCASE, length-3 diag over [x_r,y_r,theta_r]. `d_safe_per_human` = `HumanSafetyDistance[]{track_id,d_safe}`.
- `ContextIndexArray` has `phi_aggregate` (raw) AND `phi_aggregate_used` (gated). adaptive_param_node consumes `phi_aggregate_used` for global caps/Q, per-human `phi_j_used` for per-human d_safe.
- Solver receives PRECOMPUTED per-human d_safe via AdaptiveParams.d_safe_per_human; does NOT recompute from phi_j (docs/08 §15.1).
- Rate ordering f_NMPC(20Hz) >= f_context >= f_LSTM(8Hz).

**Status (updated):**
- plan 01 perception: field-name bug fixed, codex-accepted, pushed (1d52b89). DONE (needs WSL colcon).
- plan 02 lstm_dataset: DONE, 25 pytest pass. Windows npz-lock gotcha fixed. NOT yet codex-reviewed.
- plan 07 nmpc_solver (CasADi): DONE, 25 pytest pass incl. real solves w/ human avoidance+slack. acados stub. NOT yet codex-reviewed.
- plan 05 context: DONE (4 pure modules + node glue + yaml + launch + pkg), 26 pytest pass. NOT yet codex-reviewed.
- plan 03 lstm_training: NOT STARTED (agent was interrupted, left nothing). TODO.
- Remaining: 04 prediction, 06 adaptive, 08 control node, 09 calibration, 10 bringup+harness.

**CODEX QUOTA hit** (see [[codex-quota-limit]]) — reviews deferred until quota resets ~00:00 UTC. Must codex-review all packages before final done.

ocp_spec gotcha: use `ca.dot(diag, e*e)` NOT `ca.bilinear(ca.diag(ca.DM(diag)),...)` — DM can't wrap an MX runtime param (Q(phi)).

See [[env-toolchain]], [[wsl-sync-workflow]]. Skill: codex-review-loop.
