"""Audit two-ball task libraries across both distance difficulty axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_difficulty import (  # noqa: E402
    TASK_DIFFICULTY_CELLS,
)
from snooker_env.midlevel_tasks import (  # noqa: E402
    DEFAULT_TRAIN_TASKS,
    DEFAULT_VALIDATION_TASKS,
    TwoBallTaskDataset,
)
from snooker_env.midlevel_two_ball import POCKET_NAMES  # noqa: E402
from snooker_env.midlevel_bc import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        type=Path,
        default=[
            ROOT / "outputs/tasks/midlevel_two_ball_train.npz",
            ROOT / "outputs/tasks/midlevel_two_ball_validation.npz",
        ],
    )
    parser.add_argument(
        "--require-balanced",
        action="store_true",
        help=(
            "Fail unless every pocket/cell combination is represented and "
            "all pocket, cell, and joint totals differ by at most one task."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def audit_dataset(path: Path) -> dict[str, object]:
    dataset = TwoBallTaskDataset.load(path, validate_model=False)
    joint_counts = dataset.pocket_difficulty_counts()
    cell_counts = np.sum(joint_counts, axis=0)
    pocket_counts = np.sum(joint_counts, axis=1)
    balanced = bool(
        int(np.min(joint_counts)) > 0
        and int(np.max(pocket_counts) - np.min(pocket_counts)) <= 1
        and int(np.max(cell_counts) - np.min(cell_counts)) <= 1
        and int(np.max(joint_counts) - np.min(joint_counts)) <= 1
    )
    return {
        "path": str(path),
        "task_count": len(dataset),
        "expected_default_count": len(dataset)
        in (DEFAULT_TRAIN_TASKS, DEFAULT_VALIDATION_TASKS),
        "balanced": balanced,
        "cell_count_range": [
            int(np.min(cell_counts)),
            int(np.max(cell_counts)),
        ],
        "pocket_count_range": [
            int(np.min(pocket_counts)),
            int(np.max(pocket_counts)),
        ],
        "pocket_cell_count_range": [
            int(np.min(joint_counts)),
            int(np.max(joint_counts)),
        ],
        "missing_pocket_cell_combinations": int(np.sum(joint_counts == 0)),
        "pocket_cell_counts": {
            pocket_name: {
                cell.name: int(joint_counts[pocket_index, cell.index])
                for cell in TASK_DIFFICULTY_CELLS
            }
            for pocket_index, pocket_name in enumerate(POCKET_NAMES)
        },
        "difficulty_profile": dataset.difficulty_profile(),
    }


def main() -> None:
    args = parse_args()
    if not args.datasets:
        raise ValueError("At least one task dataset is required.")
    reports = [audit_dataset(path) for path in args.datasets]
    result = {"datasets": reports}
    if args.output is not None:
        write_json(args.output, result)
        print(f"report={args.output}", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_balanced:
        failures = [report["path"] for report in reports if not report["balanced"]]
        if failures:
            raise RuntimeError(
                "Task difficulty distribution is not balanced: "
                + ", ".join(map(str, failures))
            )


if __name__ == "__main__":
    main()
