from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class HumanSafetyDistance:
    track_id: int
    d_safe: float


@dataclass
class AdaptiveParamsInput:
    vx_max: float
    vy_max: float
    omega_max: float
    q_diag: np.ndarray
    d_safe_aggregate: float
    d_safe_per_human: list[HumanSafetyDistance] = field(default_factory=list)


@dataclass
class SolveResult:
    u0: np.ndarray
    trajectory: dict
    success: bool
    solve_time_ms: float
    cost_breakdown: dict[str, float]
    slacks: dict[int, float]


@dataclass
class SolverDiagnostics:
    sqp_iterations: int
    kkt_residual: float
    qp_status: str
    constraint_violation: float
    objective_value: float


class SolverInterface(ABC):

    @abstractmethod
    def initialize(self, params: dict) -> None:
        pass

    @abstractmethod
    def set_reference(self, ref_trajectory: dict) -> None:
        pass

    @abstractmethod
    def set_human_predictions(self, predictions: list, uncertainties: list) -> None:
        pass

    @abstractmethod
    def set_adaptive_params(self, adaptive_params: AdaptiveParamsInput) -> None:
        pass

    def set_obstacles(self, obstacle_points: np.ndarray) -> None:
        raise NotImplementedError("This solver backend does not implement obstacle cost")

    @abstractmethod
    def solve(self, x0: np.ndarray) -> SolveResult:
        pass

    @abstractmethod
    def get_diagnostics(self) -> SolverDiagnostics:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass
