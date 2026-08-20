"""Generate exact feasible train/validation libraries for learned mid-level policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.midlevel_mujoco_warp_tasks import (  # noqa: E402
    generate_mujoco_warp_task_dataset,
    repair_mujoco_warp_task_dataset,
    validate_mujoco_warp_task_dataset,
)
from snooker_env.midlevel_tasks import (  # noqa: E402
    DEFAULT_TRAIN_TASKS,
    DEFAULT_VALIDATION_TASKS,
    MUJOCO_WARP_PHYSICS_BACKEND,
    TwoBallTaskDataset,
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
    stability_stop_tolerance: float,
    max_attempts_per_task: int,
    canonical_reserve_per_pocket: int = 32,
    canonical_max_rounds: int = 8,
    resume_unvalidated: bool = False,
    recanonicalize_backend: bool = False,
) -> None:
    def progress(done: int, attempts: int, task: object) -> None:
        if done == 1 or done == count or done % 50 == 0:
            print(f"{label}: accepted={done}/{count} last_attempts={attempts}", flush=True)

    def status(message: str) -> None:
        print(f"{label}: {message}", flush=True)

    unvalidated_output = output.with_name(
        f"{output.stem}.unvalidated{output.suffix}"
    )
    if resume_unvalidated:
        if backend != "mujoco-warp":
            raise ValueError("--resume-unvalidated requires --backend mujoco-warp.")
        if not unvalidated_output.is_file():
            raise FileNotFoundError(
                f"No staged task library exists at {unvalidated_output}."
            )
        staged = TwoBallTaskDataset.load(
            unvalidated_output,
            simulator=simulator,
            expected_backend=MUJOCO_WARP_PHYSICS_BACKEND,
        )
        if len(staged) != count:
            raise ValueError(
                "Staged task count does not match the requested split: "
                f"{len(staged)} != {count}."
            )
        if staged.generation_seed != seed:
            raise ValueError(
                "Staged generation seed does not match the requested split: "
                f"{staged.generation_seed} != {seed}."
            )
        print(
            f"{label}: resuming={unvalidated_output} tasks={len(staged)}",
            flush=True,
        )
        dataset = repair_mujoco_warp_task_dataset(
            staged,
            model_path=simulator.model_path,
            replacement_tasks_per_pocket=canonical_reserve_per_pocket,
            num_worlds=num_worlds,
            device=physics_device,
            chunk_steps=chunk_steps,
            check_interval_steps=check_interval_steps,
            nconmax=nconmax,
            njmax=njmax,
            max_time=max_shot_time,
            prefilter_time=prefilter_time,
            stop_tolerance=stability_stop_tolerance,
            max_rounds=canonical_max_rounds,
            max_attempts_per_task=max_attempts_per_task,
            allow_backend_recanonicalization=recanonicalize_backend,
            status=status,
        )
    elif backend == "mujoco-warp":
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
            stability_stop_tolerance=stability_stop_tolerance,
            canonical_reserve_per_pocket=canonical_reserve_per_pocket,
            canonical_max_rounds=canonical_max_rounds,
            max_attempts_per_task=max_attempts_per_task,
            progress=progress,
            status=status,
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
                stop_tolerance=stability_stop_tolerance,
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
    parser.add_argument(
        "--stability-stop-tolerance",
        type=float,
        default=5e-3,
        help=(
            "Accept a MJWarp candidate only when its first rollout and "
            "independent canonical replay stop within this distance (meters)."
        ),
    )
    parser.add_argument(
        "--canonical-reserve-per-pocket",
        type=int,
        default=32,
        help=(
            "Maximum stable reserve tasks generated per represented pocket; "
            "failed final-layout slots are replaced in place from this pool."
        ),
    )
    parser.add_argument(
        "--canonical-max-rounds",
        type=int,
        default=8,
        help="Maximum fixed-layout replay/replace rounds per world batch.",
    )
    parser.add_argument(
        "--resume-unvalidated",
        action="store_true",
        help=(
            "Repair the existing .unvalidated MJWarp library in immutable "
            "final slots instead of regenerating the split from scratch."
        ),
    )
    parser.add_argument(
        "--recanonicalize-backend",
        action="store_true",
        help=(
            "Allow an existing staged MJWarp library with an older backend "
            "hash to undergo full fixed-slot double replay on the active "
            "backend. Requires --resume-unvalidated."
        ),
    )
    parser.add_argument("--max-attempts-per-task", type=int, default=2_000)
    args = parser.parse_args()

    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.num_worlds <= 0:
        raise ValueError("--num-worlds must be positive.")
    if args.canonical_reserve_per_pocket <= 0:
        raise ValueError("--canonical-reserve-per-pocket must be positive.")
    if args.canonical_max_rounds <= 0:
        raise ValueError("--canonical-max-rounds must be positive.")
    if args.recanonicalize_backend and not args.resume_unvalidated:
        raise ValueError(
            "--recanonicalize-backend requires --resume-unvalidated."
        )
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
            args.stability_stop_tolerance,
            args.max_attempts_per_task,
            args.canonical_reserve_per_pocket,
            args.canonical_max_rounds,
            args.resume_unvalidated,
            args.recanonicalize_backend,
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
            args.stability_stop_tolerance,
            args.max_attempts_per_task,
            args.canonical_reserve_per_pocket,
            args.canonical_max_rounds,
            args.resume_unvalidated,
            args.recanonicalize_backend,
        )


if __name__ == "__main__":
    main()
