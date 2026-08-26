"""Replay one batch-aligned range from a merged MJWarp task library."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_bc import write_json  # noqa: E402
from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.midlevel_mujoco_warp_tasks import (  # noqa: E402
    validate_mujoco_warp_task_dataset,
)
from snooker_env.midlevel_tasks import (  # noqa: E402
    MUJOCO_WARP_PHYSICS_BACKEND,
    TwoBallTaskDataset,
    require_balanced_task_difficulty,
    require_complete_task_difficulty,
)
from snooker_env.midlevel_two_ball import TwoBallShotSimulator  # noqa: E402
from snooker_env.mujoco_warp_sdf import (  # noqa: E402
    MUJOCO_WARP_NCONMAX,
    MUJOCO_WARP_NJMAX,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--start-task", type=int, required=True)
    parser.add_argument("--max-tasks", type=int, required=True)
    parser.add_argument("--num-worlds", type=int, default=4096)
    parser.add_argument("--physics-device", default="cuda:0")
    parser.add_argument("--chunk-steps", type=int, default=64)
    parser.add_argument("--check-interval-steps", type=int, default=8192)
    parser.add_argument("--nconmax", type=int, default=MUJOCO_WARP_NCONMAX)
    parser.add_argument("--njmax", type=int, default=MUJOCO_WARP_NJMAX)
    parser.add_argument("--max-shot-time", type=float, default=8.0)
    parser.add_argument("--stop-tolerance", type=float, default=5e-3)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--targeted-training-distribution",
        action="store_true",
        help="Require full pocket/cell coverage instead of uniform counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if args.start_task < 0 or args.max_tasks <= 0:
        raise ValueError("Task range must have a non-negative start and positive size.")
    if args.report_output.exists() and not args.overwrite:
        raise FileExistsError(
            "Validation report exists; pass --overwrite to replace: "
            f"{args.report_output}"
        )

    simulator = TwoBallShotSimulator(args.model, max_time=args.max_shot_time)
    dataset = TwoBallTaskDataset.load(
        args.dataset,
        simulator=simulator,
        expected_backend=MUJOCO_WARP_PHYSICS_BACKEND,
    )
    if args.targeted_training_distribution:
        require_complete_task_difficulty(dataset, context="Replay")
    else:
        require_balanced_task_difficulty(dataset, context="Replay")
    report = validate_mujoco_warp_task_dataset(
        dataset,
        model_path=args.model,
        start_task=args.start_task,
        max_tasks=args.max_tasks,
        num_worlds=args.num_worlds,
        device=args.physics_device,
        chunk_steps=args.chunk_steps,
        check_interval_steps=args.check_interval_steps,
        nconmax=args.nconmax,
        njmax=args.njmax,
        max_time=args.max_shot_time,
        stop_tolerance=args.stop_tolerance,
    )
    values: dict[str, object] = {
        "dataset": str(args.dataset),
        "dataset_content_sha256": dataset.content_sha256(),
        "dataset_task_count": len(dataset),
        "start_task": args.start_task,
        "end_task_exclusive": args.start_task + report.checked_count,
        "checked_count": report.checked_count,
        "passed_count": report.passed_count,
        "max_stop_replay_error_m": report.max_stop_replay_error,
        "failures": list(report.failures),
        "num_worlds": args.num_worlds,
        "physics_backend": dataset.physics_backend,
        "backend_sha256": dataset.backend_hash,
    }
    write_json(args.report_output, values)
    print(
        f"replay_range=[{args.start_task}, "
        f"{args.start_task + report.checked_count}) "
        f"passed={report.passed_count}/{report.checked_count} "
        f"max_stop_error={report.max_stop_replay_error:.6g}m "
        f"report={args.report_output}",
        flush=True,
    )
    if report.failures:
        raise RuntimeError("; ".join(report.failures[:10]))


if __name__ == "__main__":
    main()
