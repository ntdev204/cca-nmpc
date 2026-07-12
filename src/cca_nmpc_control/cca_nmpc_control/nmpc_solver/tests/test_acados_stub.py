"""Contract for the deferred acados backend.

acados is intentionally deferred (docs/09_roadmap.md Section 2): the codegen
toolchain lives on the target Linux platform. Until then the stub must fail
loudly and early on every SolverInterface method, and point the caller to the
roadmap, rather than silently degrading. These tests lock that contract so a
future half-finished implementation can't quietly ship a method that returns
None instead of raising.
"""
import numpy as np
import pytest

from cca_nmpc_control.nmpc_solver import SolverInterface
from cca_nmpc_control.nmpc_solver.acados_solver import AcadosSolver


def test_acados_implements_interface():
    assert issubclass(AcadosSolver, SolverInterface)
    assert isinstance(AcadosSolver(), SolverInterface)


def test_every_method_raises_not_implemented_with_roadmap_pointer():
    solver = AcadosSolver()
    calls = (
        lambda: solver.initialize({}),
        lambda: solver.set_reference({}),
        lambda: solver.set_human_predictions([], []),
        lambda: solver.set_adaptive_params(None),
        lambda: solver.solve(np.zeros(3)),
        lambda: solver.get_diagnostics(),
        lambda: solver.reset(),
        lambda: solver.shutdown(),
    )
    for call in calls:
        with pytest.raises(NotImplementedError) as exc:
            call()
        # Must route the reader to the deferral roadmap, not just fail blankly.
        assert "docs/09_roadmap.md" in str(exc.value)
