"""
Dummy Human Slot Constants

Defines constants and helpers for filling unused human constraint slots
in the NMPC solver. Unused slots are assigned a dummy human placed far away
so the constraint is trivially feasible and contributes ~0 to the cost.

This is required because the solver uses a fixed max_humans_in_solver
number of constraint slots (for code-generated acados compatibility).

Reference: docs/08_solver_design.md Sections 3.3, 15.2
"""

import numpy as np


# Documented constant: distance for dummy human (trivially feasible)
DUMMY_D_J: float = 999.0

# Documented constant: context index for dummy human (zero avoidance cost)
DUMMY_PHI_J: float = 0.0

# Dummy d_safe — must be much smaller than DUMMY_D_J so constraint is trivially satisfied
DUMMY_D_SAFE: float = 0.5

# Near-miss logging threshold — slack activations for dummy slots are never logged
DUMMY_NEAR_MISS_THRESHOLD: float = 1.0  # d_j must be < this to be logged as near-miss


def fill_dummy_human_slots(
    active_humans: list,
    max_slots: int,
    horizon_N: int,
) -> list:
    """
    Fill unused human slots with dummy data so all slots are trivially feasible.

    Args:
        active_humans: list of (track_id, x_hat[N+1], y_hat[N+1], phi_j) tuples
        max_slots: total number of human constraint slots in solver
        horizon_N: NMPC prediction horizon length

    Returns:
        list of exactly max_slots tuples (track_id, x_hat, y_hat, phi_j).
        Slots beyond len(active_humans) have track_id=-1 (dummy marker).
    """
    if len(active_humans) > max_slots:
        raise ValueError(
            f"Active humans ({len(active_humans)}) exceed max_humans_in_solver ({max_slots})."
            " Caller must pre-select the highest-risk humans."
        )

    dummy_x = np.full(horizon_N + 1, DUMMY_D_J)
    dummy_y = np.full(horizon_N + 1, DUMMY_D_J)

    result = list(active_humans)
    while len(result) < max_slots:
        result.append((-1, dummy_x.copy(), dummy_y.copy(), DUMMY_PHI_J))
    return result


def build_d_safe_slots(
    active_d_safe: dict[int, float],
    human_slots: list,
) -> list[float]:
    """
    Build ordered d_safe values for all solver slots.

    Args:
        active_d_safe: {track_id: d_safe} dict from AdaptiveParamsInput.d_safe_per_human
        human_slots: output of fill_dummy_human_slots (ordered slot list)

    Returns:
        list of d_safe values, one per slot (DUMMY_D_SAFE for dummy slots)
    """
    result = []
    for track_id, _x, _y, _phi in human_slots:
        if track_id == -1:
            result.append(DUMMY_D_SAFE)
        else:
            result.append(active_d_safe.get(track_id, DUMMY_D_SAFE))
    return result


def is_dummy_slot(track_id: int) -> bool:
    """Return True if the slot is a dummy (not a real tracked human)."""
    return track_id == -1


def should_log_as_near_miss(track_id: int, d_j: float) -> bool:
    """
    Return True only if this is a real human and close enough to log as near-miss.

    Dummy slots (track_id == -1) are never logged as near-misses.

    Args:
        track_id: slot track ID (-1 for dummy)
        d_j: current distance to the human
    """
    if is_dummy_slot(track_id):
        return False
    return d_j < DUMMY_NEAR_MISS_THRESHOLD
