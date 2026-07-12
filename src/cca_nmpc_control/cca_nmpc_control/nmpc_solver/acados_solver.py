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
    "reference backend. This backend is intentionally deferred, not missing: "
    "see docs/08_solver_design.md Section 2 and the roadmap in "
    "docs/09_roadmap.md Section 2 (acceptance criteria + parity test)."
)


class AcadosSolver(SolverInterface):

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
