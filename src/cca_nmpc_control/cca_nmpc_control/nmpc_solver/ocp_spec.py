"""
Backend-Agnostic OCP Specification

Defines the Mecanum dynamics (Eq 4.1), cost terms (Eq 11.1-11.2),
and per-human soft constraints (Eq 12.3-12.5) using CasADi symbolics.
This module is backend-agnostic: both CasadiSolver and AcadosSolver
use the same symbolic expressions to prevent formulation drift.

Reference: docs/08_solver_design.md Sections 3, 7, 8
           docs/01_mathematical_model.md Eqs 4.1, 11.1-12.7
"""

import numpy as np
import casadi as ca


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NX = 3  # state dimension: [x_r, y_r, theta_r]
NU = 3  # control dimension: [v_x, v_y, omega]


def mecanum_dynamics(x: ca.MX, u: ca.MX) -> ca.MX:
    """
    Mecanum wheel robot continuous dynamics (Eq 4.1).

    State x = [x_r, y_r, theta_r]
    Control u = [v_x, v_y, omega]

        x_dot = v_x * cos(theta) - v_y * sin(theta)
        y_dot = v_x * sin(theta) + v_y * cos(theta)
        theta_dot = omega

    Args:
        x: CasADi symbolic state vector of length 3
        u: CasADi symbolic control vector of length 3

    Returns:
        xdot: CasADi expression for state derivative (length 3)
    """
    theta = x[2]
    vx = u[0]
    vy = u[1]
    omega = u[2]

    xdot = ca.vertcat(
        vx * ca.cos(theta) - vy * ca.sin(theta),
        vx * ca.sin(theta) + vy * ca.cos(theta),
        omega,
    )
    return xdot


def build_rk4_integrator(dt: float) -> ca.Function:
    """
    Build a CasADi RK4 integrator for the Mecanum dynamics.

    Uses CasADi's integrator with RK method for consistent Jacobians
    (not a hand-rolled loop) as specified in docs/08 Section 3.2.

    Args:
        dt: control step duration in seconds

    Returns:
        CasADi Function: (x, u) -> x_next
    """
    x = ca.MX.sym("x", NX)
    u = ca.MX.sym("u", NU)

    ode = {"x": x, "p": u, "ode": mecanum_dynamics(x, u)}
    opts = {
        "tf": dt,
        "simplify": True,
        "number_of_finite_elements": 4,
    }
    integrator = ca.integrator("integrator", "rk", ode, opts)

    # Wrap into a cleaner (x, u) -> x_next Function
    x_next = integrator(x0=x, p=u)["xf"]
    f_discrete = ca.Function("f_discrete", [x, u], [x_next], ["x", "u"], ["x_next"])
    return f_discrete


def build_human_distance(x_robot: ca.MX, x_h: float, y_h: float) -> ca.MX:
    """
    Euclidean distance from robot position to a human position (symbolic).

    Args:
        x_robot: symbolic robot state [x_r, y_r, theta_r]
        x_h: human x coordinate (runtime parameter, scalar float or MX)
        y_h: human y coordinate (runtime parameter, scalar float or MX)

    Returns:
        CasADi expression for distance
    """
    dx = x_robot[0] - x_h
    dy = x_robot[1] - y_h
    return ca.sqrt(dx**2 + dy**2 + 1e-6)  # epsilon avoids zero-gradient at exact coincidence


def build_tracking_cost_stage(
    e: ca.MX,
    q_diag: np.ndarray,
) -> ca.MX:
    """
    Tracking cost term at a single stage: e^T * Q(phi) * e (Eq 11.2).

    Args:
        e: tracking error vector (length 3)
        q_diag: diagonal of Q matrix (length 3)

    Returns:
        Scalar CasADi expression for the stage tracking cost

    ``q_diag`` may be a numeric array (fixed weight) OR a CasADi symbol (runtime
    parameter Q(phi)); ``dot(q_diag, e*e)`` works for both, unlike ``ca.DM(...)``
    which cannot wrap an MX.
    """
    return ca.dot(q_diag, e * e)


def build_control_cost_stage(u: ca.MX, r_diag: np.ndarray) -> ca.MX:
    """
    Control effort cost term: u^T * R * u (Eq 11.2).

    Args:
        u: control vector (length 3)
        r_diag: diagonal of R matrix (length 3)

    Returns:
        Scalar CasADi expression
    """
    return ca.dot(r_diag, u * u)


def build_smooth_cost_stage(du: ca.MX, rd_diag: np.ndarray) -> ca.MX:
    """
    Control smoothness cost: delta_u^T * Rd * delta_u (Eq 11.2).

    Args:
        du: control difference u_k - u_{k-1}
        rd_diag: diagonal of Rd matrix (length 3)

    Returns:
        Scalar CasADi expression
    """
    return ca.dot(rd_diag, du * du)


def build_human_hinge_cost_stage(
    x_robot: ca.MX,
    x_h: float,
    y_h: float,
    phi_j: float,
    d0: float,
    w_h: float,
) -> ca.MX:
    """
    Per-human hinge cost term: w_h * phi_j * max(0, d0 - d_j)^2 (Eq 11.2).

    This is an external cost (hinge) that cannot be expressed as a natural
    least-squares residual.

    Args:
        x_robot: symbolic robot state
        x_h: human predicted x
        y_h: human predicted y
        phi_j: per-human context index (runtime parameter)
        d0: influence radius
        w_h: human cost weight

    Returns:
        Scalar CasADi expression
    """
    d_j = build_human_distance(x_robot, x_h, y_h)
    hinge = ca.fmax(0.0, d0 - d_j)
    return w_h * phi_j * hinge**2


def build_terminal_cost(e_N: ca.MX, p_diag: np.ndarray) -> ca.MX:
    """
    Terminal tracking cost: e_N^T * P * e_N (Eq 11.2).

    Args:
        e_N: terminal tracking error (length 3)
        p_diag: diagonal of P matrix (length 3)

    Returns:
        Scalar CasADi expression
    """
    return ca.dot(p_diag, e_N * e_N)


def build_soft_constraint_slack_cost(s: ca.MX, w_slack: float) -> ca.MX:
    """
    Slack penalty: w_slack * s^2 (Eq 12.5).

    Args:
        s: slack variable (scalar, >= 0)
        w_slack: slack penalty weight

    Returns:
        Scalar CasADi expression
    """
    return w_slack * s**2


def build_soft_human_constraint(
    x_robot: ca.MX,
    x_h: float,
    y_h: float,
    s_j: ca.MX,
    d_safe_j: float,
) -> ca.MX:
    """
    Per-human soft constraint residual: d_j + s_j - d_safe_j >= 0 (Eq 12.3).

    Returns the constraint expression g >= 0 (for CasADi NLP lbg=0).

    Args:
        x_robot: symbolic robot state
        x_h: human predicted x
        y_h: human predicted y
        s_j: slack variable (>= 0)
        d_safe_j: safety distance lower bound for this human

    Returns:
        CasADi expression that must be >= 0
    """
    d_j = build_human_distance(x_robot, x_h, y_h)
    return d_j + s_j - d_safe_j


class OCPSpec:
    """
    Backend-agnostic OCP specification container.

    Holds the symbolic integrator and a function reference to build NLP structure.
    Both CasadiSolver and AcadosSolver share this to prevent formulation drift.

    Reference: docs/08_solver_design.md Section 3
    """

    def __init__(self, dt: float):
        self.dt = dt
        self.nx = NX
        self.nu = NU
        self.f_discrete: ca.Function = build_rk4_integrator(dt)

    def integrate(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """Apply one discrete integration step."""
        return self.f_discrete(x=x, u=u)["x_next"]

    def integrate_numpy(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Numerically integrate one step given numpy arrays.

        Args:
            x: state array (3,)
            u: control array (3,)

        Returns:
            x_next: np.ndarray of shape (3,)
        """
        result = self.f_discrete(x=x, u=u)
        return np.array(result["x_next"]).flatten()
