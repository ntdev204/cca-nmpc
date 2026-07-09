"""
NMPC Solver Module

Backend-agnostic NMPC solver for CCA-NMPC control.
Implements the OCP from docs/01_mathematical_model.md (Eqs 11.1-12.7).

Supported backends:
- CasADi + IPOPT (reference, fully implemented)
- acados + HPIPM (deferred to target platform)
"""

from .interface import (
    SolverInterface,
    SolveResult,
    SolverDiagnostics,
    AdaptiveParamsInput,
    HumanSafetyDistance,
)
from .casadi_solver import CasadiSolver

__all__ = [
    "SolverInterface",
    "SolveResult",
    "SolverDiagnostics",
    "AdaptiveParamsInput",
    "HumanSafetyDistance",
    "CasadiSolver",
]
