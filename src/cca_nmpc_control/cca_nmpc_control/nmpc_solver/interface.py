"""
Abstract NMPC Solver Interface

Defines the contract between the controller node and the numerical backend.
Supports both CasADi + IPOPT and acados + HPIPM implementations.

Reference: docs/08_solver_design.md Section 6
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class HumanSafetyDistance:
    """Per-human safety distance bound."""
    track_id: int
    d_safe: float


@dataclass
class AdaptiveParamsInput:
    """
    Adaptive parameters from adaptive_param_node.

    Mirrors cca_nmpc_msgs/AdaptiveParams but is ROS-free for standalone testing.
    Field names match the ROS message exactly for transparent conversion.
    """
    vx_max: float
    vy_max: float
    omega_max: float
    q_diag: np.ndarray  # length-3 lowercase [qx, qy, qtheta]
    d_safe_aggregate: float
    d_safe_per_human: list[HumanSafetyDistance] = field(default_factory=list)


@dataclass
class SolveResult:
    """
    Result of a single solve() call.

    Reference: docs/08_solver_design.md Section 10
    """
    u0: np.ndarray  # shape (3,) — [v_x, v_y, omega]
    trajectory: dict  # {"x": np.ndarray, "y": np.ndarray, "theta": np.ndarray, "u": np.ndarray}
    success: bool
    solve_time_ms: float
    cost_breakdown: dict[str, float]  # tracking_cost, control_cost, smooth_cost, obstacle_cost, human_cost, terminal_cost
    slacks: dict[int, float]  # keyed by track_id (only active humans)


@dataclass
class SolverDiagnostics:
    """
    Low-level solver diagnostics for debugging.

    Reference: docs/08_solver_design.md Section 11
    """
    sqp_iterations: int
    kkt_residual: float
    qp_status: str
    constraint_violation: float
    objective_value: float


class SolverInterface(ABC):
    """
    Abstract interface for NMPC solver backends.

    The controller node depends only on this interface, not on any specific
    numerical backend (CasADi, acados). This enables backend swapping for
    benchmarking and validation without changing the controller code.

    Reference: docs/08_solver_design.md Section 6
    """

    @abstractmethod
    def initialize(self, params: dict) -> None:
        """
        Load/build the solver model and allocate memory.

        Called once at node startup. For acados, this is where code generation
        and solver compilation happens. For CasADi, this builds the symbolic NLP.

        Args:
            params: Solver configuration dict containing:
                - horizon_N: prediction horizon steps
                - dt: control period (s)
                - max_humans_in_solver: fixed number of human constraint slots
                - cost_weights: dict of cost term weights
                - default bounds, etc.
        """
        pass

    @abstractmethod
    def set_reference(self, ref_trajectory: dict) -> None:
        """
        Update the reference trajectory for tracking cost.

        Args:
            ref_trajectory: dict with keys "x", "y", "theta" (each np.ndarray of length horizon_N+1)
        """
        pass

    @abstractmethod
    def set_human_predictions(self, predictions: list, uncertainties: list) -> None:
        """
        Update human motion predictions for the horizon.

        Args:
            predictions: list per tracked human of (track_id, x_hat[H], y_hat[H], phi_j)
                where x_hat, y_hat are np.ndarray of length horizon_N+1, phi_j is scalar context index
            uncertainties: list per human of sigma_h values (currently not used in OCP but reserved)
        """
        pass

    @abstractmethod
    def set_adaptive_params(self, adaptive_params: AdaptiveParamsInput) -> None:
        """
        Update adaptive parameters (velocity bounds, tracking weight, per-human d_safe).

        The solver binds each active human's constraint bound from d_safe_per_human
        (matched by track_id) and fills unused slots with dummy-human convention.

        Args:
            adaptive_params: AdaptiveParamsInput with velocity caps, Q diagonal, and d_safe_per_human
        """
        pass

    @abstractmethod
    def solve(self, x0: np.ndarray) -> SolveResult:
        """
        Solve the OCP for the current state.

        Args:
            x0: current robot state [x_r, y_r, theta_r]

        Returns:
            SolveResult with u0 (control to apply), trajectory, success flag,
            solve time, cost breakdown, and per-human slacks keyed by track_id.
        """
        pass

    @abstractmethod
    def get_diagnostics(self) -> SolverDiagnostics:
        """
        Retrieve low-level solver diagnostics from the last solve.

        Returns:
            SolverDiagnostics with SQP iterations, KKT residual, QP status, etc.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Clear warm-start trajectory and solver internal state.

        Called on warm-start invalidation triggers (goal change, relocalization,
        robot lifted, odom jump) or after prolonged solver failure.

        Does NOT reallocate memory or regenerate code — only clears history.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Release solver resources cleanly.

        Called at node destruction.
        """
        pass
