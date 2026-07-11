"""CLI: run the offline pipeline over a scripted scenario -> cmd_vel log (BR-02).

    python -m tools.integration_harness.run_harness [--steps 40]

Emits a per-step cmd_vel / phi / solver-success log for a scripted crossing
scenario, proving the full math pipeline end-to-end without ROS or hardware.
"""
from __future__ import annotations

import argparse

from .pipeline import Pipeline, PipelineConfig, HumanObs


def crossing_scenario(steps: int, dt: float):
    """Robot drives +x toward a goal; one human crosses its path along +y."""
    goal = (4.0, 0.0, 0.0)
    frames = []
    for k in range(steps):
        t = k * dt
        robot = (0.3 * t, 0.0, 0.0, 0.3, 0.0)          # moving forward
        human = HumanObs(track_id=1, x=2.0, y=-1.5 + 0.5 * t, vx=0.0, vy=0.5)
        frames.append((robot, [human], goal))
    return frames


def run(steps: int = 40, dt: float = 0.1) -> list:
    pipe = Pipeline(PipelineConfig(dt=dt))
    log = []
    for (robot, humans, goal) in crossing_scenario(steps, dt):
        res = pipe.step(robot, humans, goal)
        log.append(res)
    return log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Offline CCA-NMPC pipeline harness")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--dt", type=float, default=0.1)
    args = ap.parse_args(argv)
    log = run(args.steps, args.dt)
    n_ok = sum(1 for r in log if r.solver_success)
    print(f"steps={len(log)} solver_success={n_ok}/{len(log)}")
    for k, r in enumerate(log):
        print(f"  k={k:02d} cmd_vel=[{r.cmd_vel[0]:.3f},{r.cmd_vel[1]:.3f},"
              f"{r.cmd_vel[2]:.3f}] phi_used={r.phi_aggregate_used:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
