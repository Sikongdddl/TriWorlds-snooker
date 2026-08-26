"""Check deterministic distance-cell scheduling and candidate geometry."""

from __future__ import annotations

from collections import Counter

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_difficulty import (  # noqa: E402
    TASK_DIFFICULTY_CELLS,
    difficulty_profile,
    task_difficulty_index,
)
from snooker_env.midlevel_tasks import (  # noqa: E402
    DEFAULT_TRAIN_TASKS,
    DEFAULT_VALIDATION_TASKS,
    _generation_schedule,
    _sample_candidate,
)
from snooker_env.midlevel_two_ball import (  # noqa: E402
    POCKET_NAMES,
    TwoBallShotSimulator,
)


def _check_schedule(count: int, seed: int) -> None:
    schedule = _generation_schedule(count, seed)
    pocket_counts = Counter(pocket for pocket, _, _ in schedule)
    cell_counts = Counter(cell for _, cell, _ in schedule)
    joint_counts = Counter((pocket, cell) for pocket, cell, _ in schedule)
    all_pocket_counts = [pocket_counts[pocket] for pocket in POCKET_NAMES]
    all_cell_counts = [cell_counts[cell.index] for cell in TASK_DIFFICULTY_CELLS]
    if max(all_pocket_counts) - min(all_pocket_counts) > 1:
        raise RuntimeError("Difficulty schedule is not pocket-balanced.")
    if max(all_cell_counts) - min(all_cell_counts) > 1:
        raise RuntimeError("Difficulty schedule is not cell-balanced.")
    all_joint_counts = [
        joint_counts[(pocket_name, cell.index)]
        for pocket_name in POCKET_NAMES
        for cell in TASK_DIFFICULTY_CELLS
    ]
    if max(all_joint_counts) - min(all_joint_counts) > 1:
        raise RuntimeError("Difficulty schedule is not pocket/cell-balanced.")


def main() -> None:
    for count in range(1, 2 * len(POCKET_NAMES) * len(TASK_DIFFICULTY_CELLS) + 1):
        _check_schedule(count, count + 17)
    _check_schedule(DEFAULT_TRAIN_TASKS, 0)
    _check_schedule(DEFAULT_VALIDATION_TASKS, 1)
    if _generation_schedule(128, 0) == _generation_schedule(128, 1):
        raise RuntimeError("Train and validation difficulty schedules are identical.")

    simulator = TwoBallShotSimulator()
    cue_positions: list[np.ndarray] = []
    object_positions: list[np.ndarray] = []
    pocket_positions: list[np.ndarray] = []
    for pocket_index, pocket_name in enumerate(POCKET_NAMES):
        for cell in TASK_DIFFICULTY_CELLS:
            candidate = None
            for attempt in range(1, 501):
                candidate = _sample_candidate(
                    pocket_name,
                    cell.index,
                    10_000 * pocket_index + 500 * cell.index + attempt,
                    simulator,
                )
                if candidate is not None:
                    break
            if candidate is None:
                raise RuntimeError(
                    f"No geometrically valid candidate for {pocket_name}/{cell.name}."
                )
            cue, object_ball, _, _ = candidate
            actual_cell = task_difficulty_index(
                cue,
                object_ball,
                simulator.pocket_positions[pocket_name],
            )
            if actual_cell != cell.index:
                raise RuntimeError(
                    f"Candidate escaped {cell.name}: actual={actual_cell}."
                )
            cue_positions.append(cue)
            object_positions.append(object_ball)
            pocket_positions.append(simulator.pocket_positions[pocket_name])

    profile = difficulty_profile(
        np.stack(cue_positions),
        np.stack(object_positions),
        np.stack(pocket_positions),
    )
    counts = [
        int(profile["cells"][cell.name]["count"])
        for cell in TASK_DIFFICULTY_CELLS
    ]
    if counts != [len(POCKET_NAMES)] * len(TASK_DIFFICULTY_CELLS):
        raise RuntimeError(f"Candidate difficulty counts are not balanced: {counts}")
    print(
        "midlevel_difficulty=PASS "
        f"cells={len(TASK_DIFFICULTY_CELLS)} "
        f"candidates={len(cue_positions)} counts={counts}"
    )


if __name__ == "__main__":
    main()
