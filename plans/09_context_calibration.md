# Context Calibration Plan

## Overview

Implement the offline two-stage calibration for the context weights `w_d, w_v, w_theta, w_u, b`, per `docs/01_mathematical_model.md` Section 8.1 and `docs/04_dataset_specification.md` Section 3. Stage 1 fits weights against proxy labels (min future distance / evasive-reaction flags); stage 2 runs a ±20% sensitivity sweep against NMPC solve success. Pure Python; implementable and testable now on Windows.

## Environment Note

- Windows, no ROS2. Uses the calibration dataset (CSV/parquet) and the solver library (plan 07, CasADi backend) for the stage-2 sweep — both run on Windows.

## Project Type

**BACKEND / OFFLINE CALIBRATION (pure Python).** Primary agent: `backend-specialist`; skills: `python-patterns`, `clean-code`; supporting: `test-engineer`.

## Success Criteria

- Loads the context-calibration dataset (Section 3.2 schema: robot pose, human states, `min_future_distance`, `evasive_reaction_flag`).
- Reconstructs the context-score features per record (reuses the `context_score`/`relative_motion` logic from plan 05 to avoid drift).
- Stage 1: fits `w_*, b` to the proxy label (logistic-style fit of `phi` to danger proxy).
- Stage 2: perturbs each weight ±20% independently, evaluates change in `d_safe`, `v_x_max`, and NMPC solve success rate; re-scales weights whose perturbation degrades solve success >10%.
- Emits a calibrated `context_node.weights` YAML block + a sensitivity report.
- Deterministic; unit tested on synthetic data.

## Tech Stack

- Python 3, NumPy, pandas, scikit-learn or scipy (logistic fit).
- Depends on plan 05 context math and plan 07 CasADi solver (for solve-success evaluation).
- `pytest`.

## File Structure

```text
tools/context_calibration/
├── __init__.py
├── load.py            # calibration dataset loader + label extraction
├── features.py        # reuse plan-05 relative_motion + context_score
├── fit_weights.py     # stage 1 fit
├── sensitivity.py     # stage 2 +-20% sweep vs solve success
├── report.py          # weights YAML block + sensitivity report
├── calibrate.py       # CLI orchestrating stage1 -> stage2 -> report
└── tests/
    ├── test_fit_weights.py
    └── test_sensitivity.py
```

## Task Breakdown

### CB-01 — Dataset loader + labels

Agent: `backend-specialist`; skills: `python-patterns`; priority: P0; dependencies: dataset plan.

INPUT → calibration dataset schema (Section 3.2).
OUTPUT → `load.py` returns per-record features + proxy labels (`min_future_distance`, `evasive_reaction_flag`).
VERIFY → `pytest ... -k load` on synthetic CSV returns aligned features/labels.

### CB-02 — Feature reconstruction (reuse plan 05)

Agent: `backend-specialist`; skills: `clean-code`; priority: P1; dependencies: CB-01, context plan CX-02/CX-03.

INPUT → records.
OUTPUT → `features.py` computes `(d0-d_h)/d0`, `|v_h|/v_max`, `cos(delta_theta)`, `u_h` using the SAME functions as `cca_nmpc_context` so calibration and runtime never drift.
VERIFY → feature values match the context node's math on shared test vectors.

### CB-03 — Stage 1 weight fit

Agent: `backend-specialist`; skills: `python-patterns`; priority: P1; dependencies: CB-02.

INPUT → features + proxy danger label.
OUTPUT → `fit_weights.py` fits `w_d, w_v, w_theta, w_u, b` (logistic fit of `phi` to the proxy).
VERIFY → `test_fit_weights.py`: on synthetic data with a known separating boundary, recovered weights classify the proxy above a threshold accuracy.

### CB-04 — Stage 2 sensitivity sweep

Agent: `backend-specialist`; skills: `clean-code`; priority: P1; dependencies: CB-03, solver plan SV-04.

INPUT → fitted weights, ±20% perturbation, CasADi solver.
OUTPUT → `sensitivity.py` perturbs each weight, measures change in `d_safe`, `v_x_max`, and solve success rate; re-scales weights degrading solve success >10%.
VERIFY → `test_sensitivity.py`: a deliberately dominating weight is flagged and down-scaled; report records per-weight sensitivity.

### CB-05 — Report + CLI

Agent: `backend-specialist`; skills: `clean-code`; priority: P2; dependencies: CB-04.

OUTPUT → `report.py` emits a ready-to-paste `context_node.weights` YAML block + human-readable sensitivity table; `calibrate.py` CLI runs the full pipeline.
VERIFY → `python -m tools.context_calibration.calibrate --help`; run on synthetic data produces a weights block and report.

## Phase X: Verification

- Now (Windows): `pytest tools/context_calibration/tests/` passes; CLI runs on synthetic data.
- Deferred (needs real logged interaction data): calibrate on the real target-platform dataset; feed weights into `context_node` config.

## Notes and Risks

- Reuse plan-05 context math directly — do not reimplement feature formulas, or calibration and runtime will drift.
- Weights are calibration outputs, not hand-tuned; report must document the procedure for the paper's reproducibility claim.
- Stage 2 depends on the solver; the CasADi backend is sufficient for solve-success evaluation on Windows.
