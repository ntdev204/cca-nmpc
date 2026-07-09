# Continuous Context-Aware Nonlinear Model Predictive Control (CCA-NMPC)

## Mathematical Model

---

# 1. Overview

CCA-NMPC consists of three layers:

1. Human Perception
2. Human Motion Prediction
3. Adaptive Context-Aware NMPC

Only the perception module processes RGB-D information. After transforming detections into the global coordinate frame, the entire mathematical model operates in **2D**.

---

# 2. Overall Pipeline

```text
RGB Camera
      │
Depth Camera
      │
      ▼
YOLO26m Human Detection
      │
      ▼
Depth Projection
      │
      ▼
TF Transformation
(Camera → Map)
      │
      ▼
Kalman Filter
      │
      ▼
Human State
(x,y,vx,vy,c)
      │
      ▼
Offline Trained LSTM
      │
      ▼
Predicted Human States
      │
      ▼
Continuous Context Estimator
      │
      ▼
Adaptive Parameter Generator
      │
      ▼
CCA-NMPC
      │
      ▼
Robot Velocity Command
```

---

# 3. Robot Model

Robot state and control input:

$$
X_r =
\begin{bmatrix} x_r \\ y_r \\ \theta_r \end{bmatrix},
\qquad
u =
\begin{bmatrix} v_x \\ v_y \\ \omega \end{bmatrix}
\tag{3.1}
$$

| Symbol     | Description      |
| ---------- | ---------------- |
| $x_r$      | robot x position |
| $y_r$      | robot y position |
| $\theta_r$ | robot heading    |
| $v_x$      | forward velocity |
| $v_y$      | lateral velocity |
| $\omega$   | angular velocity |

---

# 4. Mecanum Robot Dynamics

$$
\begin{aligned}
\dot{x} &= v_x\cos\theta - v_y\sin\theta \\
\dot{y} &= v_x\sin\theta + v_y\cos\theta \\
\dot{\theta} &= \omega
\end{aligned}
\tag{4.1}
$$

The NMPC prediction integrates the dynamics using RK4. Discrete prediction:

$$
X_{k+1} = f(X_k, u_k)
\tag{4.2}
$$

---

# 5. Human State

$$
S_h =
\begin{bmatrix} x_h \\ y_h \\ v_x \\ v_y \\ c \end{bmatrix}
\tag{5.1}
$$

| Symbol | Description                         |
| ------ | ----------------------------------- |
| $x_h$  | human x position                    |
| $y_h$  | human y position                    |
| $v_x$  | human velocity x                    |
| $v_y$  | human velocity y                    |
| $c$    | detection confidence, $c \in [0,1]$ |

Estimated via: YOLO → Depth → TF → Kalman Filter.

---

# 6. Human Motion Prediction

Input / output sequences:

$$
S_h(k-L+1), \dots, S_h(k)
\;\;\longrightarrow\;\;
\hat{S}_h(k+1), \dots, \hat{S}_h(k+H)
\tag{6.1}
$$

$$
\hat{S}_h =
\begin{bmatrix} \hat{x}_h \\ \hat{y}_h \\ \hat{v}_x \\ \hat{v}_y \end{bmatrix}
\tag{6.2}
$$

Each prediction also carries a **predictive uncertainty** $\sigma_h$, estimated as the LSTM's rolling prediction-error variance over the last $W$ cycles (used in Section 8):

$$
\sigma_h(k) = \mathrm{Var}\Big(\hat{S}_h(k \mid k-1) - S_h(k)\Big)_{\text{last } W}
\tag{6.3}
$$

This uncertainty term is required to correctly fold detection/prediction confidence into the context score (fix for the sign error in v1 — see Section 8).

---

# 7. Relative Motion

$$
p_r = [x_r, y_r]^T, \qquad p_h = [x_h, y_h]^T
\tag{7.1}
$$

$$
d_h = \lVert p_r - p_h \rVert
\tag{7.2}
$$

$$
v_r =
\begin{bmatrix}
v_x\cos\theta - v_y\sin\theta \\
v_x\sin\theta + v_y\cos\theta
\end{bmatrix},
\qquad
v_{rel} = v_h - v_r
\tag{7.3}
$$

$$
e = \frac{p_r - p_h}{\max(d_h, \epsilon)}
\tag{7.4}
$$

$$
\cos(\Delta\theta) = \frac{v_{rel} \cdot e}{\max(\lVert v_{rel} \rVert, \epsilon)}
\tag{7.5}
$$

---

# 8. Continuous Context Estimation (fixed)

Continuous context index:

$$
\phi_h = \sigma(z), \qquad \sigma(z) = \frac{1}{1+e^{-z}}
\tag{8.1}
$$

**Corrected context score:**

$$
z = w_d\,\frac{d_0 - d_h}{d_0} + w_v\,\frac{\lVert v_h \rVert}{v_{max}} + w_\theta\,\cos(\Delta\theta) + w_u\,u_h + b
\tag{8.2}
$$

where the confidence/uncertainty term is now:

$$
u_h = 1 - c\cdot(1 - \tilde\sigma_h),
\qquad
\tilde\sigma_h = \mathrm{clip}\!\left(\frac{\sigma_h}{\sigma_{max}}, 0, 1\right)
\tag{8.3}
$$

**Rationale for the fix:** in v1, $z$ contained $+w_c\,c$ directly, meaning a drop in confidence (e.g., during occlusion) _lowered_ $z$ and therefore _lowered_ $\phi$ — making the robot behave as if the situation were _safer_ precisely when the human's state is least certain. This is backwards: uncertainty should make the robot **more** cautious. The corrected term $u_h$ is high (→ increases $z$, hence $\phi$) when confidence $c$ is low or predictive uncertainty $\tilde\sigma_h$ is high, and low when both detection and prediction are reliable.

| Symbol               | Meaning                                                         |
| -------------------- | --------------------------------------------------------------- |
| $d_h$                | human distance                                                  |
| $v_h$                | human speed                                                     |
| $\cos(\Delta\theta)$ | relative motion direction                                       |
| $u_h$                | detection/prediction uncertainty term (replaces raw confidence) |
| $b$                  | bias                                                            |

## 8.1 Parameter Calibration (new)

Weights $w_d, w_v, w_\theta, w_u, b$ are **not** hand-tuned constants. They are calibrated in two stages:

1. **Initialization** from a short human-robot interaction dataset collected on the target platform (recommended: 30–60 min of logged trajectories covering near-passing, crossing, and following scenarios), by fitting $\phi$ to a proxy label constructed from measured minimum future distance and observed human evasive reactions (standard practice in social-navigation literature; avoids requiring subjective hand-labeling of "danger").
2. **Sensitivity validation**: after initialization, each weight is perturbed by $\pm 20\%$ independently and the resulting change in $d_{safe}$, $v_{x,max}$, and NMPC solve success rate is logged. Weights producing $>10\%$ degradation in solve success rate under perturbation are re-scaled down, since large sensitivity indicates the term is dominating the aggregate score disproportionately.

This replaces the "hand-picked constants" criticism from v1 with a reproducible, data-grounded procedure suitable for reporting in a methodology section.

---

# 9. Multi-Human Aggregation (fixed — directional handling added)

For multiple tracked humans $j = 1, \dots, N$:

**Risk index** (unchanged — still the max, since one high-risk human should not be diluted by averaging with distant ones):

$$
\phi = \max_j (\phi_j)
\tag{9.1}
$$

**Distance** (unchanged, for the aggregate minimum-distance monitoring):

$$
d_h = \min_j (d_j)
\tag{9.2}
$$

**New — per-human safety constraint (fix for potential conflicting-constraint deadlock):**

Instead of applying a single scalar $d_{safe}(\phi)$ uniformly, the robot enforces a **per-human** constraint using each human's own context index:

$$
d_j \ge d_{safe}(\phi_j) - s_j, \qquad s_j \ge 0 \quad \forall j
\tag{9.3}
$$

with per-human slack $s_j$ and penalty $w_{slack}\sum_j s_j^2$ in the cost. This means two humans on opposite sides each contribute their **own** constraint (scaled by their own $\phi_j$) rather than both being forced through one global $\phi = \max_j \phi_j$ value, which in v1 could make an already-distant person impose an unnecessarily tight bound copied from a nearby person's risk level, or — in the opposite-side case — produce two simultaneously infeasible tight constraints with no valid velocity satisfying both. The scalar $\phi = \max_j(\phi_j)$ is retained only for the **global** cost-weighting term $Q(\phi)$ and speed caps (Section 10), where a single conservative bound is appropriate; it is no longer used to gate the individual distance constraints.

---

# 10. Adaptive Parameters (fixed — saturation added)

Safety distance:

$$
d_{safe}(\phi) = d_{safe,0} + k_d \phi
\tag{10.1}
$$

**Saturated velocity limits** (fix for negative/degenerate bounds):

$$
\begin{aligned}
v_{x,max}(\phi) &= \max\big(v_{x,\min},\; v_{x0} - k_x \phi\big) \\
v_{y,max}(\phi) &= \max\big(v_{y,\min},\; v_{y0} - k_y \phi\big) \\
\omega_{max}(\phi) &= \max\big(\omega_{\min},\; \omega_0 - k_\omega \phi\big)
\end{aligned}
\tag{10.2}
$$

where $v_{x,\min}, v_{y,\min}, \omega_{\min} > 0$ are minimum-motion floors (e.g., 5–10% of nominal max speed) chosen so the robot always retains a controllable, non-zero escape velocity even at $\phi = 1$, preventing the "frozen robot" failure mode present in v1.

Tracking weight:

$$
Q(\phi) = Q_0 + \phi\, Q_h
\tag{10.3}
$$

---

# 11. NMPC Optimization

$$
J = J_{track} + J_{control} + J_{smooth} + J_{human} + J_{obstacle} + J_{terminal}
\tag{11.1}
$$

$$
\begin{aligned}
J_{track} &= \sum e^T Q(\phi)\, e \\
J_{control} &= \sum u^T R\, u \\
J_{smooth} &= \sum \Delta u^T R_d\, \Delta u \\
J_{human} &= \sum_j w_h\, \phi_j\, \max(0, d_0 - d_j)^2 \\
J_{obstacle} &= \sum w_{obstacle}\, C(x,y) \\
J_{terminal} &= e_N^T P\, e_N
\end{aligned}
\tag{11.2}
$$

**Note (per-human consistency fix):** $J_{human}$ is summed **per tracked human** $j$, using each human's own $\phi_j$ and $d_j$ — not the aggregate $\phi = \max_j(\phi_j)$ / $d_h = \min_j(d_j)$ from Eqs. 9.1–9.2. This mirrors the per-human safety constraint (Eq. 9.3, 12.3): if only the aggregate were used, a second at-risk human on the opposite side would have a dedicated slack constraint but contribute nothing to the avoidance gradient in the cost, undermining the reason Section 9 moved to per-human constraints in the first place. The aggregate $\phi$ is still used elsewhere (Eq. 10.3, speed caps) where a single conservative global bound is appropriate — see Section 9.

---

# 12. Constraints

$$
X_{k+1} = f(X_k, u_k)
\tag{12.1}
$$

$$
|v_x| \le v_{x,max}(\phi), \qquad |v_y| \le v_{y,max}(\phi), \qquad |\omega| \le \omega_{max}(\phi)
\tag{12.2}
$$

Per-human safety constraint (Section 9):

$$
d_j + s_j \ge d_{safe}(\phi_j), \qquad s_j \ge 0 \quad \forall j
\tag{12.3}
$$

Obstacle constraint:

$$
C(x,y) < C_{collision}
\tag{12.4}
$$

Slack penalty:

$$
w_{slack} \sum_j s_j^2
\tag{12.5}
$$

## 12.1 Feasibility and Recursive Feasibility (new)

Because $\phi_j(k)$ depends on LSTM predictions that vary at every step, the constraint set $d_{safe}(\phi_j)$ is time-varying and coupled to an external, non-cooperative agent (the human). Two measures are introduced to bound this:

1. **Two-stage smoothing of $\phi$: continuous EMA, then a dwell-time gate.** The raw per-cycle context index $\phi_j(k)$ (Eq. 8.1) is not used directly by the adaptive-parameter / NMPC stages. It passes through two distinct, sequential mechanisms:

   **(a) EMA smoothing (continuous, every cycle)** — removes per-cycle noise in $\phi_j(k)$ before anything else touches it:

$$
\phi_j^{filtered}(k) = \alpha\, \phi_j^{filtered}(k-1) + (1-\alpha)\, \phi_j(k), \qquad \alpha \in (0,1)
\tag{12.6}
$$

**(b) Dwell-time gate (event-based, at most once per $T_{dwell}$ cycles)** — takes the already-smoothed $\phi_j^{filtered}$ and holds the _active_ value used by `adaptive_param_node` / the NMPC constant for at least $T_{dwell}$ cycles before allowing it to update, so the safety/velocity bounds cannot change faster than the controller can react to within a horizon:

$$
\phi_j^{used}(k) =
\begin{cases}
\phi_j^{filtered}(k), & k - k_j^{last} \ge T_{dwell} \\[4pt]
\phi_j^{used}(k-1), & \text{otherwise}
\end{cases}
\tag{12.7}
$$

where $k_j^{last}$ is the cycle index of $j$'s last gate update, reset to $k$ whenever the top branch fires. $T_{dwell}$ (a design constant, e.g. 3–5 control cycles) is chosen following the same dwell-time argument used for switched/adaptive MPC stability (cf. Rawlings et al., 2017, _Model Predictive Control: Theory, Computation, and Design_): a constraint set that can only switch at a bounded rate is easier to reason about within a fixed-horizon OCP than one that is free to change every cycle. EMA alone (12.6) bounds _how noisy_ $\phi_j$ is; the gate (12.7) separately bounds _how often_ the value the solver actually sees is allowed to change — the two serve different purposes and are not redundant with each other.

$\phi_j^{used}$ is the value that feeds `adaptive_param_node` (Section 10) and is what `nmpc_controller_node` receives as a runtime parameter every cycle.

2. **Guaranteed feasibility via slack**: the per-human slack $s_j \ge 0$ (Eq. 12.3) guarantees the QP/NLP always has a feasible solution regardless of how tight $d_{safe}(\phi_j)$ becomes; hard infeasibility is converted into a bounded, penalized soft constraint violation instead of a solver failure.

3. **Empirical validation requirement**: because a full closed-form stability proof is not the deliverable for this venue, feasibility is validated experimentally by logging the NMPC solver return status and slack activation frequency across all test trajectories; a paper-ready claim is "zero solver infeasibility events and slack activation in under X% of control cycles across N trials," rather than a formal theorem.

---

# 13. Offline Learning

Training dataset $\rightarrow$ LSTM $\rightarrow$ prediction:

$$
[x, y, v_x, v_y] \;\xrightarrow{\text{LSTM}}\; [\hat{x}, \hat{y}, \hat{v}_x, \hat{v}_y]
\tag{13.1}
$$

Training loss:

$$
L = MSE_{position} + \lambda\, MSE_{velocity}
\tag{13.2}
$$

## 13.1 LSTM–NMPC Synchronization (new)

- LSTM inference runs at $f_{LSTM}$, NMPC solves at $f_{NMPC}$, with $f_{NMPC} \ge f_{LSTM}$ in general (control loop is faster than perception/prediction refresh).
- Between LSTM updates, the context estimator reuses the last predicted trajectory but recomputes $d_h$, $\cos(\Delta\theta)$ from the latest Kalman-filtered _current_ human state (Section 5), not from a stale predicted position — only the predicted future trajectory used inside the NMPC horizon is held constant between LSTM updates, not the instantaneous geometric terms.
- $\sigma_h$ (Eq. 6.3) is incremented conservatively (increased, never decreased) between LSTM refreshes to reflect growing uncertainty as the held prediction ages, feeding directly into $u_h$ (Eq. 8.3) so context correctly becomes more cautious as the prediction goes stale. Explicitly, between two consecutive LSTM refreshes:

$$
\sigma_h(k) = \min\big(\sigma_{max},\; \sigma_{base} + \beta\, \Delta t\big)
\tag{13.3}
$$

where $\sigma_{base}$ is the last value of $\sigma_h$ computed from Eq. 6.3 at the most recent LSTM refresh, $\Delta t$ is the time elapsed since that refresh, $\beta > 0$ is a fixed growth-rate constant, and $\sigma_{max}$ is the same saturation ceiling already used to normalize $\tilde\sigma_h$ in Eq. 8.3. At the instant of each new LSTM refresh, $\sigma_{base}$ is reset to the freshly computed Eq. 6.3 value and $\Delta t$ resets to 0, so $\sigma_h$ only ever grows monotonically within a hold interval and never exceeds $\sigma_{max}$.

---

# 14. Online Execution

```text
RGB-D Camera
      │
      ▼
YOLO26m
      │
      ▼
Depth Projection
      │
      ▼
TF
      │
      ▼
Kalman Filter
      │
      ▼
Human State
      │
      ▼
LSTM Prediction (+ rolling uncertainty σ_h)
      │
      ▼
Continuous Context Estimation (corrected z, per-human φ_j)
      │
      ▼
Adaptive Parameter Generation (saturated v_max, per-human d_safe)
      │
      ▼
CCA-NMPC (per-human slack constraints, EMA + dwell-time gated φ)
      │
      ▼
Velocity Command
```

---

# Contributions (consolidated from 5 → 3)

1. **A continuous, uncertainty-aware context index** $\phi = \sigma(z)$ that fuses distance, relative motion, and a corrected confidence/prediction-uncertainty term, replacing discrete context-switching schemes used in prior human-aware navigation work, with a data-driven, sensitivity-validated calibration procedure for its weights (Sections 8, 8.1).
2. **A per-human, EMA-smoothed and dwell-time-bounded adaptation mechanism** that couples $\phi$ to NMPC safety distance, velocity limits, and tracking cost weight simultaneously, with saturation guarantees and per-human slack to preserve feasibility under multiple, potentially conflicting human constraints (Sections 9, 10, 12.1).
3. **An integrated perception–prediction–control pipeline** (YOLO26m + Kalman + LSTM + CCA-NMPC) with an explicit synchronization scheme between the (slower) prediction module and the (faster) control loop, validated empirically rather than through discrete-time system claims alone (Section 13.1).

_(Note: LSTM-based human trajectory prediction itself is not claimed as novel — it is a supporting component; the novelty claim rests on items 1–3 above.)_
