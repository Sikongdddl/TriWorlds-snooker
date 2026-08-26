#!/usr/bin/env python3
"""Generate one targeted local-speed augmentation shard from the core library."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.midlevel_local_speed_tasks import (  # noqa: E402
    generate_local_speed_augmentation,
    require_local_speed_provenance,
    save_local_speed_provenance,
)
from snooker_env.midlevel_tasks import (  # noqa: E402
    DEFAULT_LOCAL_SPEED_AUGMENTATION_TASKS,
    MUJOCO_WARP_PHYSICS_BACKEND,
    TwoBallTaskDataset,
    require_complete_task_difficulty,
)
from snooker_env.midlevel_two_ball import TwoBallShotSimulator  # noqa: E402
from snooker_env.mujoco_warp_sdf import (  # noqa: E402
    MUJOCO_WARP_NCONMAX,
    MUJOCO_WARP_NJMAX,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "outputs/tasks/midlevel_two_ball_train.npz",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument(
        "--global-task-count",
        type=int,
        default=DEFAULT_LOCAL_SPEED_AUGMENTATION_TASKS,
    )
    parser.add_argument("--global-seed", type=int, default=20260826)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--physics-device", default="cuda:0")
    parser.add_argument("--num-worlds", type=int, default=4096)
    parser.add_argument("--chunk-steps", type=int, default=64)
    parser.add_argument("--check-interval-steps", type=int, default=8192)
    parser.add_argument("--nconmax", type=int, default=MUJOCO_WARP_NCONMAX)
    parser.add_argument("--njmax", type=int, default=MUJOCO_WARP_NJMAX)
    parser.add_argument("--max-shot-time", type=float, default=8.0)
    parser.add_argument("--stop-tolerance", type=float, default=5e-3)
    parser.add_argument("--max-fixed-layout-rounds", type=int, default=8)
    parser.add_argument(
        "--resume-unvalidated",
        action="store_true",
        help="Audit and publish existing staged outputs without rerunning physics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = (args.output, args.provenance_output)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Local-speed shard output exists: " + ", ".join(map(str, existing))
        )
    simulator = TwoBallShotSimulator(args.model, max_time=args.max_shot_time)
    source = TwoBallTaskDataset.load(
        args.source,
        simulator=simulator,
        expected_backend=MUJOCO_WARP_PHYSICS_BACKEND,
    )
    require_complete_task_difficulty(source, context="Local-speed source")

    staged_dataset = args.output.with_name(
        f"{args.output.stem}.unvalidated{args.output.suffix}"
    )
    staged_provenance = args.provenance_output.with_name(
        f"{args.provenance_output.stem}.unvalidated{args.provenance_output.suffix}"
    )
    if args.resume_unvalidated:
        for path in (staged_dataset, staged_provenance):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Cannot resume without staged local-speed output: {path}"
                )
        dataset = TwoBallTaskDataset.load(
            staged_dataset,
            simulator=simulator,
            expected_backend=MUJOCO_WARP_PHYSICS_BACKEND,
        )
        from snooker_env.midlevel_local_speed_tasks import (
            load_local_speed_provenance,
        )

        metadata, arrays = load_local_speed_provenance(staged_provenance)
        require_local_speed_provenance(dataset, source, metadata, arrays)
        staged_dataset.replace(args.output)
        staged_provenance.replace(args.provenance_output)
        print(
            f"local_speed_shard={args.shard_index}: resumed=PASS "
            f"tasks={len(dataset)} content_sha256={dataset.content_sha256()} "
            f"output={args.output} provenance={args.provenance_output}",
            flush=True,
        )
        return

    def status(message: str) -> None:
        print(f"local_speed_shard={args.shard_index}: {message}", flush=True)

    augmentation = generate_local_speed_augmentation(
        source,
        global_task_count=args.global_task_count,
        global_seed=args.global_seed,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        model_path=args.model,
        num_worlds=args.num_worlds,
        device=args.physics_device,
        chunk_steps=args.chunk_steps,
        check_interval_steps=args.check_interval_steps,
        nconmax=args.nconmax,
        njmax=args.njmax,
        max_time=args.max_shot_time,
        stop_tolerance=args.stop_tolerance,
        max_fixed_layout_rounds=args.max_fixed_layout_rounds,
        status=status,
    )
    for path in (staged_dataset, staged_provenance):
        if path.exists():
            raise FileExistsError(f"Staged local-speed output exists: {path}")
    augmentation.dataset.save(staged_dataset)
    save_local_speed_provenance(augmentation, staged_provenance)
    from snooker_env.midlevel_local_speed_tasks import load_local_speed_provenance

    metadata, arrays = load_local_speed_provenance(staged_provenance)
    require_local_speed_provenance(
        augmentation.dataset,
        source,
        metadata,
        arrays,
    )
    staged_dataset.replace(args.output)
    staged_provenance.replace(args.provenance_output)
    print(
        f"local_speed_shard={args.shard_index}: status=PASS "
        f"tasks={len(augmentation.dataset)} "
        f"groups={len(augmentation.dataset) // 4} "
        f"content_sha256={augmentation.dataset.content_sha256()} "
        f"output={args.output} provenance={args.provenance_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
