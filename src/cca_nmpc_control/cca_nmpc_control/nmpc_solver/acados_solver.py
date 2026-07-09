"""
acados + HPIPM NMPC backend (deferred — target platform build).

This backend implements the same OCP as CasadiSolver (shared ocp_spec.py, the
anti-drift guard) using acados' code-generated SQP_RTI / HPIPM / full-condensing
solver (docs/08_solver_design.md Sections 2, 4). acados requires a native
build/codegen toolchain that is not available on the Windows dev machine, so the
implementation is deferred to the target platform; it is stubbed here behind the
same SolverInterface so the controller node code needs no changes when it lands.

Reference: docs/08_solver_design.md Sections 2, 4, 15 (item 3: CasADi/acados parity).
"""
from __future__ import annotations

import numpy as np

from .interface import (
    SolverInterface,
    SolveResult,
    SolverDiagnostics,
    AdaptiveParamsInput,
)

_UNAVAILABLE = (
    "AcadosSolver requires the acados code-generation toolchain, which is built "
    "on the target Linux platform. Use CasadiSolver on Windows / for the "
    "reference backend. See docs/08_solver_design.md Section 2."
)


class AcadosSolver(SolverInterface):
    """Deferred acados backend. Every method raises until built on target."""

    def initialize(self, params: dict) -> None:
        raise NotImplementedError(_UNAVAILABLE)

    def set_reference(self, ref_trajectory: dict) -> None:
        raise NotImplementedError(_UNAVAILABLE)

    def set_human_predictions(self, predictions: list, uncertainties: list) -> None:
        raise NotImplementedError(_UNAVAILABLE)

    def set_adaptive_params(self, adaptive_params: AdaptiveParamsInput) -> None:
        raise NotImplementedError(_UNAVAILABLE)

    def solve(self, x0: np.ndarray) -> SolveResult:
        raise NotImplementedError(_UNAVAILABLE)

    def get_diagnostics(self) -> SolverDiagnostics:
        raise NotImplementedError(_UNAVAILABLE)

    def reset(self) -> None:
        raise NotImplementedError(_UNAVAILABLE)

    def shutdown(self) -> None:
        raise NotImplementedError(_UNAVAILABLE)
