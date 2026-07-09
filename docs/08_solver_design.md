# CCA-NMPC — Solver Design (08)

## 1. Purpose and Scope

This document specifies the design of the NMPC solver module (referenced as a placeholder in `02_system_architecture.md`, Section 5) — the concrete numerical component that turns the optimization problem defined in `01_mathematical_model.md` (Eqs. 11.1–12.7) into a real-time-solvable OCP (Optimal Control Problem). This is the missing piece flagged previously: **no solver exists yet**; this document is the design that should be implemented next.

Scope: supported backends, default solver configuration, OCP formulation mapping, warm-starting (with invalidation triggers), an abstract solver interface with lifecycle and runtime state machine, cost structure, constraint hierarchy, real-time failure handling, timing and diagnostics, computational complexity, and the validation plan needed to turn "designed for real-time" into a measured, defensible claim for the IJAT manuscript.

---

## 2. Supported Backends

The solver module is designed against an abstract interface (Section 6) so the choice of numerical backend is an implementation detail, not an architectural commitment.

**Supported backends:**

- **CasADi + IPOPT** — general-purpose NLP solver, pure Python, fastest to iterate on during formulation development and debugging.
- **acados + HPIPM** — code-generated, RTI-capable solver, used for the real-time deployment.

**Implementation target: acados.** CasADi + IPOPT is retained as the reference/validation backend (Section 15, item 3) rather than being discarded after prototyping — having two independent numerical implementations of the same OCP is directly useful for catching formulation-drift bugs, not just a development convenience.

---

## 3. OCP Formulation Mapping

This section maps every symbol in the math spec to the corresponding OCP object, backend-agnostically, so the solver code and the paper's methodology section never drift apart (same principle already used for the YAML parameters in `07_yaml_parameters.md`).

### 3.1 State and control

```
state:   X = [x_r, y_r, theta_r]              (Eq. 3.1)         nx = 3
control: U = [v_x, v_y, omega]                (Eq. 3.1)         nu = 3
```

### 3.2 Dynamics (discrete-time model)

Eq. (4.1) integrated over one control step `dt` (Eq. 4.2) using each backend's built-in integrator (acados' `AcadosSimSolver`, CasADi's `integrator('rk')`) — **not** a hand-rolled RK4, to avoid subtle sign/step-size bugs and to get consistent Jacobians for the SQP step for free.

```
x_dot = v_x*cos(theta) - v_y*sin(theta)
y_dot = v_x*sin(theta) + v_y*cos(theta)
theta_dot = omega
```

### 3.3 `phi_j`, `d_safe(phi_j)`, `v_max(phi)` as runtime parameters, not decision variables

`phi_j`, `phi_j^used` (Eq. 12.7, post-EMA + dwell-time gate), and everything derived from them (`d_safe(phi_j)`, `v_x,max(phi)`, `Q(phi)`) are **not optimized by the NMPC** — they are computed upstream by `context_node` / `adaptive_param_node` (per `02_system_architecture.md`) and enter the solver purely as **runtime parameters**, updated before every solve. This keeps the OCP's _structure_ (sparsity pattern, cost type, constraint count) fixed across solves — required for acados' code-generated solver — while the _numerical values_ of the safety/velocity bounds still change every cycle as `/adaptive_params` and `/context_index` update.

**Fixed maximum human count:** because a code-generated solver needs a fixed maximum number of constraints, `max_humans_in_solver` (already present in `07_yaml_parameters.md`, e.g. 6) fixes the number of per-human constraint slots. When fewer humans are tracked, unused slots are filled with a dummy human placed far away (`d_j` set to a large constant, `phi_j = 0`) so the constraint is trivially satisfied and contributes ~0 to the cost.

---

## 4. Default Solver Configuration

Exhaustive parameter tuning is out of scope here (that belongs to `07_yaml_parameters.md`, and must be re-derived empirically on target hardware regardless of what is written here). This section only states the **default configuration this design targets**, so the mapping between this document and the YAML file is unambiguous:

| Setting                  | Default                      | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------------ | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nlp_solver_type`        | `SQP_RTI`                    | One SQP step per control cycle — see real-time justification already established for this project                                                                                                                                                                                                                                                                                                                                        |
| Integrator               | `ERK` (explicit Runge-Kutta) | Matches the RK4 discretization already specified in Eq. 4.2; acados re-linearizes every RTI step regardless                                                                                                                                                                                                                                                                                                                              |
| `qp_solver`              | `HPIPM`                      | Paired with acados as specified in Section 2                                                                                                                                                                                                                                                                                                                                                                                             |
| Condensing               | `FULL_CONDENSING_HPIPM`      | Justified here (unlike a generic default) because the state dimension is very small ($n_x = 3$, Section 3.1) — full condensing eliminates the state variables from the QP at negligible cost for a system this small, and is expected to solve faster than partial condensing at this scale; should be re-checked empirically once `max_humans_in_solver` constraints are added, since condensing cost also scales with constraint count |
| `qp_solver_iter_max`     | 50                           | Standard starting point                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `tol` (NLP/QP tolerance) | acados default (~1e-6)       | Loosening this is a legitimate lever if `solve_time_ms` (Section 11) is too high in practice, at the cost of solution accuracy — flagged here so it is a deliberate, documented trade-off if used, not a silent change                                                                                                                                                                                                                   |

This table exists so a reader of the methodology section can map "SQP_RTI, HPIPM, full condensing" directly back to `nmpc_controller_node` in `07_yaml_parameters.md` without needing to read solver source code.

---

## 5. Warm Start Strategy

Almost every practical NMPC implementation relies on warm-starting: initializing cycle $k+1$'s solve from cycle $k$'s solution, shifted by one step, rather than from scratch. This is not optional for meeting the real-time target — it is the single largest lever on `solve_time_ms` (Section 11), since it gives the SQP/RTI step a starting point already close to the new optimum (the horizon overlaps by $N-1$ steps between consecutive solves).

### 5.1 Shift-and-append scheme

```
Cycle k solution:      u0  u1  u2  ...  u_{N-1}
                         |   |   |          |
                         v   v   v          v
Cycle k+1 warm start:   u1  u2  u3  ...  u_{N-1}  u_{N-1}
```

The control sequence is shifted left by one step (the applied $u_0$ is dropped, since it has already been consumed), and the last entry is duplicated to fill the new horizon tail — a standard, cheap choice in the absence of a better guess for the newly-exposed final step. The corresponding state trajectory is re-propagated from the shifted controls using the integrator from Section 3.2, rather than shifted directly, to keep the warm-start state trajectory dynamically consistent.

**Interaction with dummy humans (Section 3.3):** when the number of active humans changes between cycles (a person enters/leaves the tracked set), the warm-started control/state trajectory itself is still valid, but the corresponding slack values $s_j$ for a slot that just became active should **not** be warm-started from the previous dummy value — they are re-initialized to zero, since a dummy slot's slack has no relationship to the newly-active human's actual distance.

### 5.2 Warm-Start Invalidation Triggers

The shift-and-append warm start (5.1) assumes the previous solution is still a _reasonable_ guess for the current problem. That assumption breaks in specific, enumerable situations, and the solver must not silently keep warm-starting through them — each of the following must call `reset()` (Section 6.1) rather than shift-and-append:

```
goal changed          -> reset()   (reference trajectory discontinuity invalidates J_tracking's warm start)
map relocalization     -> reset()   (e.g. AMCL jump; robot's belief of X_r changed discontinuously)
robot lifted            -> reset()   (wheel odometry no longer reflects true motion; state estimate is meaningless)
odom jump               -> reset()   (discontinuity in /odom beyond a configurable threshold, distinct from
                                       normal relocalization — e.g. an odometry fault)
```

Each of these is detected outside the solver itself (by `nmpc_controller_node`, comparing consecutive `/odom` or goal messages against a threshold, or subscribing to a relocalization event from the localization stack) and triggers an explicit `reset()` call before the next `solve()`. Making this list explicit here — rather than leaving it to be inferred at implementation time — is deliberate: an implementer who is not told this list will likely only discover "goal changed" and miss the other three until a real failure exposes them.

---

## 6. Abstract Solver Interface

The controller node must not know which numerical backend is running underneath it. This is standard practice for any component expected to be swapped or benchmarked (CasADi during development, acados in deployment) without touching the calling code.

```
SolverInterface   (abstract base class)
        |
        +-- AcadosSolver     (implements SolverInterface using acados_template)
        +-- CasadiSolver     (implements SolverInterface using CasADi + IPOPT)
```

```python
class SolverInterface(ABC):
    @abstractmethod
    def initialize(self, params: dict):
        """Load/build the model (code-generated solver for acados, or
        symbolic NLP for CasADi), allocate solver memory, set up the
        fixed OCP structure (Section 3), apply default configuration
        (Section 4). Called once at node startup."""

    @abstractmethod
    def set_reference(self, ref_trajectory): ...

    @abstractmethod
    def set_human_predictions(self, predictions, uncertainties): ...

    @abstractmethod
    def set_adaptive_params(self, adaptive_params_msg): ...

    @abstractmethod
    def solve(self, x0) -> SolveResult: ...

    @abstractmethod
    def get_diagnostics(self) -> SolverDiagnostics: ...

    @abstractmethod
    def reset(self):
        """Clear the warm-start trajectory (Section 5) and any solver-
        internal history, without reallocating memory or re-generating
        code. Called on any trigger in Section 5.2, or after a prolonged
        solver failure (Section 9) where the stored trajectory can no
        longer be trusted as a warm start."""

    @abstractmethod
    def shutdown(self):
        """Release solver memory / external process handles cleanly."""
```

`nmpc_controller_node.py` (the ROS2 node) depends only on `SolverInterface`. It is constructed with a backend instance chosen by a YAML parameter (`nmpc_controller_node.solver_backend: "acados" | "casadi"`, already present in `07_yaml_parameters.md`), and calls only the methods above — it has no branching logic on backend type anywhere. This also makes `SolverInterface` implementations independently unit-testable in plain Python/pytest without spinning up ROS2, and reusable for an offline batch-replay evaluation harness.

`SolveResult` and `SolverDiagnostics` are the two data-only return types produced by `solve()` / `get_diagnostics()`; their fields are specified in Sections 10 and 11 respectively.

### 6.1 Solver Lifecycle

```
initialize()
     |
     v
load model  ->  allocate memory  ->  build fixed OCP structure (Section 3)
     |
     v
solve()   <---- called every control cycle at f_NMPC -----+
     |                                                      |
     +---(normal)------------------------------------------+
     |
     +---(Section 5.2 trigger / prolonged failure, Section 9)--> reset() --+
     |
     v
shutdown()   (node destruction / clean exit)
```

`initialize()` is where the one-time cost of loading a code-generated acados solver (or building the CasADi NLP) is paid, kept strictly out of the per-cycle `solve()` path. `reset()` is intentionally cheap and separate from `initialize()` — it clears the warm-start state without repeating the expensive model-loading step, since a relocalization event should not pay the full startup cost again.

### 6.2 Solver State Machine

Distinct from the lifecycle above (which describes _method calls_), this is the runtime _state_ the solver module occupies at any instant — useful for implementation because `nmpc_controller_node` can check the current state before deciding whether to publish, hold, or fall back:

```
                    initialize()
    UNINITIALIZED ───────────────> READY
                                      │
                                      │ solve() called
                                      v
                                  SOLVING
                                   │     │
                        succeeds   │     │  fails / times out
                                   v     v
                              SUCCESS  FAILED
                                   │     │
                     publish u0,   │     │  Section 9 fallback logic
                     return to     │     │  (hold / safe stop)
                     READY         │     v
                                   │  FALLBACK
                                   │     │
                                   │     │ Section 5.2 trigger,
                                   │     │ or recovered
                                   v     v
                                  READY <┘
                                   │
                                   │ shutdown()
                                   v
                              (terminated)
```

`FAILED` and `FALLBACK` are deliberately separate states: `FAILED` is the solver's own return status for a single cycle (Section 9); `FALLBACK` is `nmpc_controller_node`'s response state while it is holding/decelerating instead of publishing a fresh solve — the solver module itself does not need to know it is in `FALLBACK`, but exposing this as a named state in the design (rather than leaving it as untracked control flow) is what lets `fallback_triggered` (Section 9) be logged consistently.

---

## 7. Cost Function

$$
J = J_{tracking} + J_{control} + J_{smooth} + J_{obstacle} + J_{human} + J_{terminal}
$$

This mirrors Eq. (11.1)–(11.2) of the math spec exactly, restated here so the solver's cost implementation is checkable term-by-term against the methodology section without cross-referencing another document.

| Term           | Math spec source                              | Implementation form                                                    |
| -------------- | --------------------------------------------- | ---------------------------------------------------------------------- |
| $J_{tracking}$ | Eq. 11.2, $\sum e^TQ(\phi)e$                  | Nonlinear least-squares (backend-native)                               |
| $J_{control}$  | Eq. 11.2, $\sum u^TRu$                        | Nonlinear least-squares                                                |
| $J_{smooth}$   | Eq. 11.2, $\sum \Delta u^TR_d\Delta u$        | Nonlinear least-squares                                                |
| $J_{obstacle}$ | Eq. 11.2, $\sum w_{obstacle}\,C(x,y)$         | **External Cost** — see Section 13 (resolved)                          |
| $J_{human}$    | Eq. 11.2, $\sum w_h\,\phi\,\max(0,d_0-d_h)^2$ | **External Cost** — hinge term is not a natural least-squares residual |
| $J_{terminal}$ | Eq. 11.2, $e_N^TPe_N$                         | Nonlinear least-squares (terminal stage)                               |

$J_{obstacle}$ and $J_{human}$ are implemented as **External Cost** terms (custom CasADi expressions attached directly to the stage cost) rather than forced into a least-squares residual form, since both involve a hinge ($\max(0,\cdot)$) or a non-quadratic map ($C(x,y)$) that a least-squares residual cannot represent naturally without distortion.

---

## 8. Human Constraint: Hard -> Soft -> Slack

The per-human safety constraint is best understood as a three-stage design decision, stated explicitly here because it is exactly the kind of derivation a control-oriented reviewer will look for:

```
Hard Constraint:   d_j(x) >= d_safe(phi_j)                       -- always desirable, not always feasible
        |
        v
Soft Constraint:   d_j(x) + s_j >= d_safe(phi_j),  s_j >= 0       -- relax by an explicit, penalized slack
        |
        v
Slack Penalty:     w_slack * s_j^2   added to the cost             -- makes violation costly but never infeasible
```

**Why this order matters:** starting from the hard constraint states the actual safety requirement without compromise. Converting it to a soft constraint is a deliberate, documented relaxation — not an accidental weakening — motivated directly by the multi-human deadlock risk described in `01_mathematical_model.md` Section 9 (two humans on opposite sides can make the hard constraint simultaneously infeasible). The slack penalty then restores an economic incentive to satisfy the original hard constraint whenever physically possible, while guaranteeing the QP/NLP always returns _some_ solution. Reporting slack activation frequency (Section 11, `NmpcDiagnostics.msg`) is what lets the paper show the soft relaxation was rarely needed in practice, rather than silently masking frequent safety-margin violations.

Maps to Eq. (12.3) and (12.5) of the math spec; implemented via each backend's native slack mechanism (acados: `Zl`/`Zu`/`zl`/`zu` slack-penalty parameters on nonlinear constraints; CasADi: explicit slack decision variables added to the NLP).

---

## 9. Real-Time Timeout and Fallback Strategy

`solve_time_ms` (Section 11) tells us _how long_ a solve took, but says nothing about _what the robot does_ when a solve fails to return a usable control within the control period — this is a separate, mandatory design decision, not an afterthought, since it directly determines robot behavior in the worst case (e.g. a solver hiccup exactly while a person is crossing close by).

**Failure modes to cover:**

- Solver returns `solver_success = False` (QP/NLP infeasible even with slack, or hit `qp_solver_iter_max`).
- Solver exceeds the control period `dt` before returning (timeout, detected by the calling `nmpc_controller_node` via a wall-clock deadline around `solve()`).

**Fallback hierarchy (evaluated in order):**

```
if solve() succeeds within dt:
        publish the computed u0                                  [state: SUCCESS]
else if a recent successful solution exists (age < timeout_hold_cycles):
        publish the previously computed control, held or decayed  [state: FALLBACK]
        (do NOT re-publish a stale u0 indefinitely — see below)
else:
        command a safe stop (v_x = v_y = omega = 0, or a bounded  [state: FALLBACK]
        decelerating ramp if an instantaneous stop is itself
        undesirable at the current speed), then call reset() so the
        next solve is not warm-started from a possibly-invalid trajectory
```

**Design notes:**

- Simply "publish previous control" is only safe for a bounded number of consecutive failures (`timeout_hold_cycles`, a new YAML parameter to add to `nmpc_controller_node`); repeated failures should escalate to a safe stop rather than letting the robot coast indefinitely on stale commands, since a human's position may have changed materially in the meantime.
- Safe stop should decelerate within actuator limits rather than commanding an instantaneous zero velocity if the robot is moving at speed, to avoid a different hazard (an abrupt stop being unpredictable/startling to a nearby human, echoing the smoothness motivation for $J_{smooth}$ in Section 7).
- Every fallback activation is logged (extends `NmpcDiagnostics.msg` with a `fallback_triggered` flag and `fallback_reason` field) so fallback frequency becomes another number the paper can report directly, the same way slack activation frequency is reported (Section 8) — a low fallback rate is itself evidence supporting the real-time claim.

---

## 10. Cycle Timing Breakdown

A single `solve_time_ms` number is not enough to find where time is actually going in the control cycle. The timing budget is broken into named segments, each attributable to a specific piece of the pipeline so bottlenecks are diagnosable directly from a log rather than requiring re-instrumentation later:

| Segment                 | Owned by                                   | Description                                                                                                                                                                                |
| ----------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `prediction_time`       | `prediction_node` (not the solver)         | LSTM inference time; logged separately in that node's own diagnostics, included here only for full end-to-end latency context since it feeds `/human_predictions` that the solver consumes |
| `parameter_update_time` | `nmpc_controller_node` / `SolverInterface` | Time spent in `set_reference`, `set_human_predictions`, `set_adaptive_params` before calling `solve()`                                                                                     |
| `solver_time`           | `SolverInterface.solve()`                  | The RTI/SQP solve itself — this is what `solve_time_ms` in `NmpcDiagnostics.msg` already measures                                                                                          |
| `publish_time`          | `nmpc_controller_node`                     | Constructing and publishing `/cmd_vel` and `/nmpc_diagnostics`                                                                                                                             |
| `total_cycle`           | `nmpc_controller_node`                     | Sum of the above, measured wall-clock, compared directly against `dt` (Section 4) to check the real-time budget is met                                                                     |

Example benchmark table format (values illustrative, to be replaced with measured data on target hardware — this is exactly the kind of table the IJAT manuscript's experiments section should contain once measured):

| Segment          | Example time                         |
| ---------------- | ------------------------------------ |
| Parameter update | 0.5 ms                               |
| Solver           | 8.2 ms                               |
| Publish          | 0.2 ms                               |
| **Total cycle**  | **8.9 ms** (vs. `dt` = 50 ms budget) |

Splitting the budget this way turns "is it real-time?" from a single pass/fail number into a breakdown that shows _where_ to optimize if the budget is ever missed (e.g. a slow `parameter_update_time` points at message deserialization, not the solver itself).

---

## 11. Solver Diagnostics (debug-only, not for the paper)

Separate from the paper-facing `NmpcDiagnostics.msg` fields (solver_success, solve_time_ms, slack_values, cost breakdown, fallback fields — Section 9 — already defined/extended above), the solver module additionally exposes a lower-level diagnostics bundle intended purely for development/debugging, not for inclusion in results:

```
SolverDiagnostics:
    sqp_iterations       # number of SQP/RTI iterations performed
    kkt_residual          # KKT stationarity residual at solution
    qp_status             # underlying QP solver return code (per backend)
    constraint_violation  # max constraint violation (should be ~0 given slack)
    objective_value       # raw scalar J at the solution
```

This is exposed via `get_diagnostics()` (Section 6) as a separate call from `solve()`'s `SolveResult`, so the ROS2 node can choose to log it only when a debug flag is set, without adding overhead to `/nmpc_diagnostics` in normal operation. In practice this is most useful during the CasADi/acados parity testing described in Section 15 — a solve that "succeeds" per `solver_success` but has a large `kkt_residual` or nonzero `constraint_violation` is a red flag worth catching before it shows up as a subtle discrepancy in experiment results.

---

## 12. Cost Breakdown Logging

Every `solve()` call populates `SolveResult.cost_breakdown` with the per-term values corresponding 1:1 to the `NmpcDiagnostics.msg` cost fields already defined in `06_message_definitions.md`:

```
tracking_cost
control_cost
smooth_cost
obstacle_cost
human_cost
terminal_cost
```

Because this is logged every cycle to `/nmpc_diagnostics`, a post-hoc analysis script (`plot_cost.py`) can reconstruct the full cost evolution over any recorded trial directly from the rosbag, with no need to re-run the solver or re-instrument anything after the fact — one script, one rosbag in, one figure out. This is the same "instrument once, analyze later" principle already applied to the feasibility logging in `01_mathematical_model.md` Section 12.1, and to the timing breakdown in Section 10 above.

---

## 13. Computational Complexity

No formal complexity proof is needed for this venue; a stated order-of-magnitude estimate is sufficient and is what IJAT-style applied venues expect in a solver design section:

$$
\text{Total per-cycle cost} \;\approx\; O(N \cdot n_h) \;+\; O(\text{QP solve})
$$

where $N$ is the NMPC horizon length (`horizon_N` in `07_yaml_parameters.md`) and $n_h$ is `max_humans_in_solver` (the per-human constraint evaluation is linear in both the horizon length and the number of active human slots). The QP solve term dominates in practice and is treated as a black box here — its empirical cost is exactly what `solver_time` (Section 10) measures directly, which is preferred over a theoretical QP complexity bound for this application (real hardware timing is the number that actually matters for the real-time claim).

---

## 14. Obstacle Handling — Resolved (previously an open question)

**Decision: soft cost only, no hard constraint for the Nav2 costmap.**

```
Obstacle (C(x,y))
        |
        v
   Soft Cost only   (J_obstacle, Section 7)
```

Raw Nav2 costmaps are not differentiable (piecewise-constant occupancy grids, or at best a non-smooth inflation gradient), and both acados and CasADi's SQP-based solvers require differentiable constraint functions to produce usable Jacobians. Forcing $C(x,y) < C_{collision}$ (Eq. 12.4) into a hard nonlinear constraint would require first building a smoothed/differentiable representation (e.g. a signed-distance field), which is extra infrastructure with no clear benefit here, given the soft-cost path already provides a usable avoidance gradient via $J_{obstacle}$. Eq. (12.4) is therefore treated as **advisory/monitoring only** — logged for diagnostics (e.g. flag if the realized trajectory ever crosses $C_{collision}$) but not enforced as a hard solver constraint.

---

## 15. Remaining Open Design Questions

1. **Where is `d_safe(phi_j)` computed — inside the solver's parameter-setting code, or upstream in `adaptive_param_node`?** Current design in `02_system_architecture.md` has `adaptive_param_node` own Eqs. 10.1–10.3, meaning the solver receives already-computed `d_safe` values as parameters rather than `phi_j` directly. This is preferred (keeps the solver backend-agnostic and "dumb," easier to unit-test) but should be confirmed before implementing `set_adaptive_params`.
2. **Dummy-human placeholder convention** (Section 3.3) needs a documented constant (e.g. `d_j = 999.0`, `phi_j = 0.0`) so unused constraint slots don't accidentally get logged as a real near-miss in diagnostics.
3. **CasADi/acados parity testing**: before trusting the acados deployment, both backends should be run on identical logged scenarios and their trajectories, costs, and `SolverDiagnostics` compared numerically, to catch formulation drift introduced during implementation (e.g. a slack penalty accidentally doubled, a sign flipped in an External Cost term). This is the primary justification for keeping the CasADi backend alive post-prototyping (Section 2) rather than deleting it once acados is working.
4. **`timeout_hold_cycles` value** (Section 9) needs to be chosen and justified — likely a small number of cycles (e.g. 2–3 at 20 Hz, i.e. ~100–150 ms) but should be tied to how fast a nearby human can materially change position, not chosen arbitrarily.
5. **Odom-jump threshold** (Section 5.2) needs a concrete numerical value (e.g. position discontinuity beyond N cm or heading beyond M degrees between consecutive `/odom` messages) — currently only the trigger category is specified, not the detection threshold.

---

## 16. Summary

| Item                                                              | Status                                                                                                                                                                                                                                                                                            |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Supported backends identified, default configuration specified    | Designed (Sections 2, 4)                                                                                                                                                                                                                                                                          |
| Abstract interface, lifecycle, and runtime state machine designed | Designed (Section 6)                                                                                                                                                                                                                                                                              |
| Warm-start strategy and invalidation triggers                     | Designed (Section 5)                                                                                                                                                                                                                                                                              |
| Real-time timeout / fallback behavior                             | Designed (Section 9)                                                                                                                                                                                                                                                                              |
| Timing breakdown (parameter update / solver / publish / total)    | Designed (Section 10)                                                                                                                                                                                                                                                                             |
| Solver code (`AcadosSolver`, `CasadiSolver`)                      | **Not yet implemented**                                                                                                                                                                                                                                                                           |
| Real-time claim                                                   | Design target only; requires the Section 10 timing breakdown measured on target hardware to become a validated claim                                                                                                                                                                              |
| Control-theoretic guarantees                                      | None beyond RTI's standard justification + empirical feasibility logging (Doc. 01, Sec. 12.1); no closed-form stability proof — intentionally out of scope, consistent with the applied-engineering (IJAT) positioning, which weighs RMSE / solve time / real-robot validation over formal proofs |
| Obstacle handling                                                 | Resolved: soft cost only (Section 14)                                                                                                                                                                                                                                                             |
