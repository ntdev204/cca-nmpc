"""End-to-end offline pipeline test (BR-02).

A scripted crossing scenario must produce bounded cmd_vel, a non-trivial phi
when the human is close, and no solver infeasibility across the run.
"""
import numpy as np

from tools.integration_harness.pipeline import (
    Pipeline, PipelineConfig, HumanObs,
)
from tools.integration_harness.run_harness import run


def test_single_step_bounded_cmd_vel():
    pipe = Pipeline(PipelineConfig())
    res = pipe.step(
        robot=(0.0, 0.0, 0.0, 0.0, 0.0),
        humans=[HumanObs(track_id=1, x=1.5, y=0.0, vx=0.0, vy=0.0)],
        goal=(3.0, 0.0, 0.0),
    )
    assert res.solver_success
    # bounded by nominal caps (vx<=1.0, vy<=0.8, omega<=1.2)
    assert abs(res.cmd_vel[0]) <= 1.0 + 1e-6
    assert abs(res.cmd_vel[1]) <= 0.8 + 1e-6
    assert abs(res.cmd_vel[2]) <= 1.2 + 1e-6


def test_close_human_raises_phi():
    pipe = Pipeline(PipelineConfig())
    res = pipe.step(
        robot=(0.0, 0.0, 0.0, 0.0, 0.0),
        humans=[HumanObs(track_id=1, x=0.3, y=0.0, vx=-0.5, vy=0.0)],
        goal=(3.0, 0.0, 0.0),
    )
    # close, approaching human -> non-trivial context
    assert res.phi_aggregate_used > 0.3


def test_crossing_scenario_no_infeasibility():
    log = run(steps=30, dt=0.1)
    assert len(log) == 30
    # every solve must succeed (slack guarantees feasibility)
    assert all(r.solver_success for r in log)
    # cmd_vel bounded throughout
    for r in log:
        assert np.all(np.abs(r.cmd_vel) <= np.array([1.0, 0.8, 1.2]) + 1e-6)


def test_no_humans_still_solves():
    pipe = Pipeline(PipelineConfig())
    res = pipe.step((0.0, 0.0, 0.0, 0.0, 0.0), [], (2.0, 0.0, 0.0))
    assert res.solver_success
    assert res.cmd_vel[0] > 0.0  # drives toward goal


def test_multi_human_all_constrained():
    pipe = Pipeline(PipelineConfig())
    res = pipe.step(
        robot=(0.0, 0.0, 0.0, 0.0, 0.0),
        humans=[
            HumanObs(track_id=1, x=1.5, y=0.5, vx=0.0, vy=0.0),
            HumanObs(track_id=2, x=1.5, y=-0.5, vx=0.0, vy=0.0),
        ],
        goal=(3.0, 0.0, 0.0),
    )
    assert res.solver_success
    assert set(res.slacks.keys()) == {1, 2}
