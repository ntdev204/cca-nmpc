# NMPC Solver Module Plan

## Overview

Implement the NMPC solver module designed in `docs/08_solver_design.md`: an abstract `SolverInterface` with a CasADi + IPOPT reference backend and an acados + HPIPM deployment backend, mapping the OCP in `docs/01_mathematical_model.md` (Eqs. 11.1–12.7). The solver is a plain Python library independent of ROS2, so the CasADi backend is implementable and testable now on Windows; acados is deferred to the target platform.

## Environment Note

- Windows, no ROS2: the solver is ROS-free and unit-testable now. **CasADi + IPOPT** installs via pip on Windows and is the primary "code now" target. **acados** requires a native build/codegen toolchain and is typically Linux/target-hardware; its plan tasks are written now but built/verified later.
- Keep the controller's dependency strictly on `SolverInterface` so backends are swappable.

## Project Type

**BACKEND / NUMERICAL OPTIMIZATION (pure Python library).** Primary agent: `backend-specialist`; skills: `python-patterns`, `clean-code`; supporting: `test-engineer`, `performance-optimizer`.

## Success Criteria

- `SolverInterface` (ABC) with `initialize/set_reference/set_human_predictions/set_adaptive_params/solve/get_diagnostics/reset/shutdown` (Section 6).
- OCP: state `[x_r,y_r,theta_r]`, control `[v_x,v_y,omega]`, Mecanum dynamics via backend integrator (Section 3).
- Cost terms: tracking/control/smooth/terminal as least-squares; human hinge and obstacle as external cost (Section 7).
- Per-human soft constraint with slack + penalty (Section 8; Eqs. 12.3, 12.5); fixed `max_humans_in_solver` slots with dummy-human fill (Section 3.3).
- Warm-start shift-and-append + invalidation triggers + `reset()` (Section 5).
- `SolveResult` (u0, trajectory, success, cost breakdown) and `SolverDiagnostics` (Sections 10–12).
- CasADi backend fully tested now; acados backend behind the same interface for later parity testing (Section 15.3).

## Tech Stack

- Python 3, CasADi (+ IPOPT), NumPy — Windows-friendly.
- acados + acados_template (deferred, target platform).
- `pytest`.

## File Structure

```text
src/cca_nmpc_control/nmpc_solver/
├── __init__.py
├── interface.py        # SolverInterface ABC, SolveResult, SolverDiagnostics
├── ocp_spec.py         # symbolic OCP: state/control/dynamics/cost/constraints
├── casadi_solver.py    # CasADi + IPOPT implementation (code now)
├── acados_solver.py    # acados + HPIPM implementation (deferred build)
├── warm_start.py       # shift-and-append + invalidation triggers
├── dummy_humans.py     # fixed-slot fill convention (d_j large, phi_j=0)
└── tests/
    ├── test_interface_contract.py
    ├── test_ocp_spec.py
    ├── test_casadi_solve.py
    ├── test_warm_start.py
    └── test_dummy_humans.py
```

## Task Breakdown

### SV-01 — Interface + data types

Agent: `backend-specialist`; skills: `python-patterns`, `clean-code`; priority: P0; dependencies: none.

INPUT → Section 6 interface.
OUTPUT → `interface.py`: `SolverInterface` ABC, `SolveResult`, `SolverDiagnostics` dataclasses.
VERIFY → `test_interface_contract.py`: a stub implementation satisfies the ABC; result/diagnostics fields match Sections 10–12.

### SV-02 — OCP spec (dynamics + cost + constraints)

Agent: `backend-specialist`; skills: `python-patterns`; priority: P0; dependencies: SV-01.

INPUT → math spec Eqs. 4.1, 11.1–12.7; default config from Section 4.
OUTPUT → `ocp_spec.py`: symbolic Mecanum dynamics, least-squares + external-cost terms, per-human soft constraints with slack, fixed `max_humans_in_solver` structure.
VERIFY → `test_ocp_spec.py`: dynamics match Eq. 4.1 at sample points; cost/constraint expressions build symbolically without error.

### SV-03 — Dummy-human slots

Agent: `backend-specialist`; skills: `clean-code`; priority: P1; dependencies: SV-02.

INPUT → variable active-human count, fixed slot count.
OUTPUT → `dummy_humans.py`: documented constants (e.g. `d_j = 999.0`, `phi_j = 0.0`) to fill unused slots (Section 15.2).
VERIFY → `test_dummy_humans.py`: unused slots contribute ~0 cost and are trivially feasible; not logged as near-misses.

### SV-04 — CasADi backend

Agent: `backend-specialist`; skills: `python-patterns`; priority: P1; dependencies: SV-02, SV-03.

INPUT → OCP spec, runtime params: references, predictions, and `AdaptiveParams` (caps, `Q_diag`, and `d_safe_per_human` keyed by `track_id`).
OUTPUT → `casadi_solver.py` implementing `SolverInterface` with explicit slack decision variables; `set_adaptive_params` binds each active human's constraint bound from `d_safe_per_human` (by `track_id`) and fills unused slots with the dummy-human convention; returns `SolveResult` + cost breakdown + per-human slack keyed by `track_id`.
VERIFY → `test_casadi_solve.py`: solves a simple scenario (goal ahead, one human) to a sane control; slack activates only when constraints conflict; cost breakdown populated; per-human `d_safe` consumed from input, not recomputed.

### SV-05 — Warm start + invalidation

Agent: `backend-specialist`; skills: `clean-code`; priority: P1; dependencies: SV-04.

INPUT → previous solution; triggers (goal change, relocalization, lift, odom jump).
OUTPUT → `warm_start.py`: shift-and-append with state re-propagation; `reset()` on any Section 5.2 trigger; re-init slack for newly active slots.
VERIFY → `test_warm_start.py`: shift produces expected next guess; each trigger forces reset; new-slot slack starts at zero.

### SV-06 — Diagnostics + cost breakdown

Agent: `backend-specialist`; skills: `clean-code`; priority: P2; dependencies: SV-04.

INPUT → solve internals.
OUTPUT → `get_diagnostics()` returns SQP iters, KKT residual, QP status, constraint violation, objective; `SolveResult.cost_breakdown` per-term (Sections 11–12).
VERIFY → diagnostics populated on a solved problem; near-zero constraint violation given slack.

### SV-07 — acados backend (deferred build)

Agent: `backend-specialist`; skills: `python-patterns`; priority: P3; dependencies: SV-02..SV-06.

INPUT → same OCP spec; acados_template.
OUTPUT → `acados_solver.py` implementing the interface with SQP_RTI / HPIPM / full condensing (Section 4); slack via `Zl/Zu/zl/zu`.
VERIFY (later, target platform) → parity test vs CasADi on identical logged scenarios (Section 15.3): trajectories/costs match within tolerance.

## Phase X: Verification

- Now (Windows): `pytest src/cca_nmpc_control/nmpc_solver/tests/` passes with the CasADi backend; interface, warm start, dummy slots, and diagnostics covered.
- Later (target): build acados backend; run CasADi/acados parity harness; measure `solver_time` on hardware.

## Notes and Risks

- Keep `ocp_spec.py` backend-agnostic so CasADi and acados share one formulation — this is the guard against formulation drift.
- Slack guarantees feasibility; log slack activation frequency for the paper.
- acados install is the main platform risk; the CasADi backend intentionally covers the full formulation so progress is not blocked on Windows.
- RESOLVED (docs/08 §15.1): the solver receives **precomputed** per-human `d_safe` via `AdaptiveParams.d_safe_per_human` (`HumanSafetyDistance[]`, matched by `track_id`); `set_adaptive_params` binds each active human's constraint bound from it and fills unused slots with the dummy-human convention. The solver does NOT recompute `d_safe` from `phi_j`.
- `SolveResult`/diagnostics slack is per-human keyed by `track_id` (mirrors `SlackValue[]` in `NmpcDiagnostics`), not a positional array.
