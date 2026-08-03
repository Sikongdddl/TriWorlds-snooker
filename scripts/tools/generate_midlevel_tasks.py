"""Generate exact feasible train/validation libraries for mid-level PPO."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.midlevel_mujoco_warp_tasks import (  # noqa: E402
    generate_mujoco_warp_task_dataset,
    validate_mujoco_warp_task_dataset,
)
from snooker_env.midlevel_tasks import (  # noqa: E402
    DEFAULT_TRAIN_TASKS,
    DEFAULT_VALIDATION_TASKS,
    generate_task_dataset,
    generate_task_dataset_parallel,
    validate_task_dataset,
)
from snooker_env.midlevel_two_ball import TwoBallShotSimulator  # noqa: E402


def _generate(
    label: str,
    count: int,
    seed: int,
    output: Path,
    simulator: TwoBallShotSimulator,
    replay_check: int,
    workers: int,
    backend: str,
    physics_device: str,
    num_worlds: int,
    chunk_steps: int,
    check_interval_steps: int,
    nconmax: int,
    njmax: int,
    max_shot_time: float,
    prefilter_time: float,
    max_attempts_per_task: int,
) -> None:
    def progress(done: int, attempts: int, task: object) -> None:
        if done == 1 or done == count or done % 50 == 0:
            print(f"{label}: accepted={done}/{count} last_attempts={attempts}", flush=True)

    if backend == "mujoco-warp":
        dataset = generate_mujoco_warp_task_dataset(
            count,
            seed=seed,
            model_path=simulator.model_path,
            num_worlds=num_worlds,
            device=physics_device,
            chunk_steps=chunk_steps,
            check_interval_steps=check_interval_steps,
            nconmax=nconmax,
            njmax=njmax,
            max_time=max_shot_time,
            prefilter_time=prefilter_time,
            max_attempts_per_task=max_attempts_per_task,
            progress=progress,
        )
    else:
        generator = (
            generate_task_dataset_parallel if workers > 1 else generate_task_dataset
        )
        generator_kwargs = {
            "seed": seed,
            "simulator": simulator,
            "model_path": simulator.model_path,
            "max_attempts_per_task": max_attempts_per_task,
            "progress": progress,
        }
        if workers > 1:
            generator_kwargs["num_workers"] = workers
        dataset = generator(count, **generator_kwargs)
    # Persist the expensive generation result before replaying it.  The
    # dataset writer itself uses a same-directory temporary file and atomic
    # replace, so an interrupted write cannot leave a partial archive.  Keep
    # the staged archive when replay fails; it is essential for diagnosing a
    # rare task or world-slot divergence and avoids throwing away hours of GPU
    # work.  A previously validated output is not replaced until the new
    # archive passes its requested replay check.
    unvalidated_output = output.with_name(
        f"{output.stem}.unvalidated{output.suffix}"
    )
    dataset.save(unvalidated_output)
    print(
        f"{label}: staged={unvalidated_output} tasks={len(dataset)} "
        f"status=unvalidated",
        flush=True,
    )
    if replay_check != 0:
        replay_count = None if replay_check < 0 else min(replay_check, len(dataset))
        if backend == "mujoco-warp":
            report = validate_mujoco_warp_task_dataset(
                dataset,
                model_path=simulator.model_path,
                max_tasks=replay_count,
                num_worlds=num_worlds,
                device=physics_device,
                chunk_steps=chunk_steps,
                check_interval_steps=check_interval_steps,
                nconmax=nconmax,
                njmax=njmax,
                max_time=max_shot_time,
            )
        else:
            report = validate_task_dataset(
                dataset,
                simulator=simulator,
                max_tasks=replay_count,
            )
        print(
            f"{label}: replay={report.passed_count}/{report.checked_count} "
            f"max_stop_error={report.max_stop_replay_error:.6g}m"
        )
        if report.failures:
            raise RuntimeError("; ".join(report.failures[:10]))
    unvalidated_output.replace(output)
    print(
        f"{label}: output={output} tasks={len(dataset)} "
        f"backend={dataset.physics_backend} xml_sha256={dataset.xml_hash} "
        f"model_sha256={dataset.model_hash} backend_sha256={dataset.backend_hash}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_TASKS)
    parser.add_argument("--validation-count", type=int, default=DEFAULT_VALIDATION_TASKS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=("mujoco-warp", "cpu"),
        default="mujoco-warp",
        help="Generate and replay tasks on the same backend used for training.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Spawn this many independent MuJoCo generation workers.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=ROOT / "outputs" / "tasks" / "midlevel_two_ball_train.npz",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=ROOT / "outputs" / "tasks" / "midlevel_two_ball_validation.npz",
    )
    parser.add_argument(
        "--split",
        choices=("both", "train", "validation"),
        default="both",
    )
    parser.add_argument(
        "--replay-check",
        type=int,
        default=-1,
        help="Replay this many tasks after saving; -1 checks all and 0 disables.",
    )
    parser.add_argument("--physics-device", default="cuda:0")
    parser.add_argument("--num-worlds", type=int, default=1024)
    parser.add_argument("--chunk-steps", type=int, default=16)
    parser.add_argument("--check-interval-steps", type=int, default=2048)
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=1024)
    parser.add_argument("--max-shot-time", type=float, default=8.0)
    parser.add_argument("--prefilter-time", type=float, default=1.5)
    parser.add_argument("--max-attempts-per-task", type=int, default=2_000)
    args = parser.parse_args()

    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.num_worlds <= 0:
        raise ValueError("--num-worlds must be positive.")
    if args.backend == "mujoco-warp" and args.workers != 1:
        raise ValueError("--workers applies only to the CPU backend.")
    simulator = TwoBallShotSimulator(
        args.model,
        max_time=args.max_shot_time,
    )
    if args.split in ("both", "train"):
        _generate(
            "train",
            args.train_count,
            args.seed,
            args.train_output,
            simulator,
            args.replay_check,
            args.workers,
            args.backend,
            args.physics_device,
            args.num_worlds,
            args.chunk_steps,
            args.check_interval_steps,
            args.nconmax,
            args.njmax,
            args.max_shot_time,
            args.prefilter_time,
            args.max_attempts_per_task,
        )
    if args.split in ("both", "validation"):
        _generate(
            "validation",
            args.validation_count,
            args.seed + 1,
            args.validation_output,
            simulator,
            args.replay_check,
            args.workers,
            args.backend,
            args.physics_device,
            args.num_worlds,
            args.chunk_steps,
            args.check_interval_steps,
            args.nconmax,
            args.njmax,
            args.max_shot_time,
            args.prefilter_time,
            args.max_attempts_per_task,
        )


if __name__ == "__main__":
    main()
