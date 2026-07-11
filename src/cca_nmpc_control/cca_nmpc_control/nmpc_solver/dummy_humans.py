import numpy as np


DUMMY_D_J: float = 999.0

DUMMY_PHI_J: float = 0.0

DUMMY_D_SAFE: float = 0.5

DUMMY_NEAR_MISS_THRESHOLD: float = 1.0


def fill_dummy_human_slots(
    active_humans: list,
    max_slots: int,
    horizon_N: int,
) -> list:
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
    result = []
    for track_id, _x, _y, _phi in human_slots:
        if track_id == -1:
            result.append(DUMMY_D_SAFE)
        else:
            result.append(active_d_safe.get(track_id, DUMMY_D_SAFE))
    return result


def is_dummy_slot(track_id: int) -> bool:
    return track_id == -1


def should_log_as_near_miss(track_id: int, d_j: float) -> bool:
    if is_dummy_slot(track_id):
        return False
    return d_j < DUMMY_NEAR_MISS_THRESHOLD
