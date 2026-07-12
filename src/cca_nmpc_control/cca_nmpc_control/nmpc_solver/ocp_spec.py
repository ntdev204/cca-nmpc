import numpy as np
import casadi as ca


NX = 3
NU = 3


def mecanum_dynamics(x: ca.MX, u: ca.MX) -> ca.MX:
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
    x = ca.MX.sym("x", NX)
    u = ca.MX.sym("u", NU)

    ode = {"x": x, "p": u, "ode": mecanum_dynamics(x, u)}
    opts = {
        "simplify": True,
        "number_of_finite_elements": 4,
    }
    integrator = ca.integrator("integrator", "rk", ode, 0.0, dt, opts)

    x_next = integrator(x0=x, p=u)["xf"]
    f_discrete = ca.Function("f_discrete", [x, u], [x_next], ["x", "u"], ["x_next"])
    return f_discrete


def build_human_distance(x_robot: ca.MX, x_h: float, y_h: float) -> ca.MX:
    dx = x_robot[0] - x_h
    dy = x_robot[1] - y_h
    return ca.sqrt(dx**2 + dy**2 + 1e-6)


def build_tracking_cost_stage(
    e: ca.MX,
    q_diag: np.ndarray,
) -> ca.MX:
    return ca.dot(q_diag, e * e)


def build_control_cost_stage(u: ca.MX, r_diag: np.ndarray) -> ca.MX:
    return ca.dot(r_diag, u * u)


def build_smooth_cost_stage(du: ca.MX, rd_diag: np.ndarray) -> ca.MX:
    return ca.dot(rd_diag, du * du)


def _softplus_hinge(z: ca.MX, beta: float) -> ca.MX:
    """Smooth approximation of max(0, z) with a continuous derivative.

    softplus_beta(z) = (1/beta) * log(1 + exp(beta*z)) -> max(0, z) as beta->inf.
    Uses ca.log1p(exp(-|beta z|)) + fmax(beta z, 0) for numerical stability so
    large beta*z does not overflow exp(). Removes the kink at z=0 that fmax has,
    giving SQP/IPOPT a smooth Jacobian near the safety-distance boundary.
    """
    bz = beta * z
    stable = ca.fmax(bz, 0.0) + ca.log1p(ca.exp(-ca.fabs(bz)))
    return stable / beta


def build_human_hinge_cost_stage(
    x_robot: ca.MX,
    x_h: float,
    y_h: float,
    phi_j: float,
    d0: float,
    w_h: float,
    hinge_beta: float = 0.0,
) -> ca.MX:
    """Per-human avoidance cost w_h * phi_j * hinge(d0 - d_j)^2 (Eq. 11.2 / 12).

    hinge_beta <= 0 (default) uses the exact hinge max(0, d0 - d_j), matching
    docs/01_mathematical_model.md Eq. (11.2) literally. hinge_beta > 0 swaps in a
    softplus hinge with the same limit but a smooth derivative at the boundary,
    which can improve SQP/IPOPT robustness near the hinge kink (review P2). The
    default preserves exact math-model parity; smoothing is strictly opt-in.
    """
    d_j = build_human_distance(x_robot, x_h, y_h)
    z = d0 - d_j
    hinge = _softplus_hinge(z, hinge_beta) if hinge_beta > 0.0 else ca.fmax(0.0, z)
    return w_h * phi_j * hinge**2


def build_terminal_cost(e_N: ca.MX, p_diag: np.ndarray) -> ca.MX:
    return ca.dot(p_diag, e_N * e_N)


def build_soft_constraint_slack_cost(s: ca.MX, w_slack: float) -> ca.MX:
    return w_slack * s**2


def build_soft_human_constraint(
    x_robot: ca.MX,
    x_h: float,
    y_h: float,
    s_j: ca.MX,
    d_safe_j: float,
) -> ca.MX:
    d_j = build_human_distance(x_robot, x_h, y_h)
    return d_j + s_j - d_safe_j


class OCPSpec:

    def __init__(self, dt: float):
        self.dt = dt
        self.nx = NX
        self.nu = NU
        self.f_discrete: ca.Function = build_rk4_integrator(dt)

    def integrate(self, x: ca.MX, u: ca.MX) -> ca.MX:
        return self.f_discrete(x=x, u=u)["x_next"]

    def integrate_numpy(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        result = self.f_discrete(x=x, u=u)
        return np.array(result["x_next"]).flatten()
