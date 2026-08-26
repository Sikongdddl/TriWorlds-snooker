"""Evaluate one direct-BC mid-level checkpoint in complete shot rollouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_bc import (  # noqa: E402
    DirectBCPolicy,
    validate_policy_for_dataset,
    write_json,
)
from snooker_env.midlevel_difficulty import (  # noqa: E402
    DIFFICULTY_LEVEL_NAMES,
    TASK_DIFFICULTY_CELLS,
    TASK_DIFFICULTY_VERSION,
)
from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.midlevel_mujoco_warp_vec_env import (  # noqa: E402
    MJWarpMidLevelVecEnv,
)
from snooker_env.midlevel_two_ball_env import MidLevelTwoBallEnv  # noqa: E402
from snooker_env.midlevel_tasks import (  # noqa: E402
    CPU_PHYSICS_BACKEND,
    MUJOCO_WARP_PHYSICS_BACKEND,
    TwoBallTaskDataset,
    require_balanced_task_difficulty,
)
from snooker_env.midlevel_two_ball import MAX_ANGLE_RESIDUAL  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "outputs/tasks/midlevel_two_ball_validation.npz",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--backend",
        choices=("mujoco-warp", "cpu"),
        default="mujoco-warp",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--physics-device", default="cuda:0")
    parser.add_argument("--chunk-steps", type=int, default=16)
    parser.add_argument("--check-interval-steps", type=int, default=2048)
    parser.add_argument("--max-shot-time", type=float, default=8.0)
    parser.add_argument(
        "--details-output",
        type=Path,
        help="Save deterministic per-task actions and outcomes as NPZ.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        help="Save the aggregate JSON report.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace requested report/detail outputs if they already exist.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.tasks.is_file():
        raise FileNotFoundError(args.tasks)
    if args.max_tasks is not None and args.max_tasks <= 0:
        raise ValueError("--max-tasks must be positive when provided.")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")
    if args.details_output is not None and args.details_output.suffix != ".npz":
        raise ValueError("--details-output must use the .npz extension.")
    if args.report_output is not None and args.report_output.suffix != ".json":
        raise ValueError("--report-output must use the .json extension.")
    for path in (args.details_output, args.report_output):
        if path is not None and path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Evaluation output exists; pass --overwrite to replace: {path}"
            )


def _evaluate_mujoco_warp(
    policy: DirectBCPolicy,
    dataset: TwoBallTaskDataset,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[dict[str, object]], int]:
    task_count = len(dataset)
    count = task_count if args.max_tasks is None else min(task_count, args.max_tasks)
    batch_size = min(args.num_envs, count)
    environment = MJWarpMidLevelVecEnv(
        dataset,
        args.model,
        num_envs=batch_size,
        device=args.physics_device,
        chunk_steps=args.chunk_steps,
        check_interval_steps=args.check_interval_steps,
        max_time=args.max_shot_time,
    )
    actions: list[np.ndarray] = []
    infos: list[dict[str, object]] = []
    try:
        for start in range(0, count, batch_size):
            valid_count = min(batch_size, count - start)
            indices = [
                start + min(offset, valid_count - 1)
                for offset in range(batch_size)
            ]
            environment.set_options(
                [{"task_index": task_index} for task_index in indices]
            )
            observation = environment.reset()
            action = policy.predict(observation)
            _, rewards, dones, batch_infos = environment.step(action)
            if not bool(np.all(dones)):
                raise RuntimeError("Every mid-level shot must terminate in one step.")
            for offset in range(valid_count):
                info = dict(batch_infos[offset])
                info["reward"] = float(rewards[offset])
                infos.append(info)
                actions.append(np.asarray(action[offset], dtype=np.float32))
            print(f"evaluated={start + valid_count}/{count}", flush=True)
    finally:
        environment.close()
    return actions, infos, batch_size


def _evaluate_cpu(
    policy: DirectBCPolicy,
    dataset: TwoBallTaskDataset,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[dict[str, object]], int]:
    environment = MidLevelTwoBallEnv(
        dataset,
        args.model,
        max_time=args.max_shot_time,
    )
    count = len(dataset) if args.max_tasks is None else min(
        len(dataset),
        args.max_tasks,
    )
    actions: list[np.ndarray] = []
    infos: list[dict[str, object]] = []
    try:
        for task_index in range(count):
            observation, _ = environment.reset(
                options={"task_index": task_index}
            )
            action = policy.predict(observation)
            _, reward, terminated, truncated, info = environment.step(action)
            if not terminated or truncated:
                raise RuntimeError("A complete mid-level shot did not terminate.")
            result = dict(info)
            result["reward"] = float(reward)
            infos.append(result)
            actions.append(np.asarray(action, dtype=np.float32))
            if (task_index + 1) % 50 == 0 or task_index + 1 == count:
                print(f"evaluated={task_index + 1}/{count}", flush=True)
    finally:
        environment.close()
    return actions, infos, 1


def _aggregate_report(
    checkpoint: Path,
    backend: str,
    device: str,
    actions: np.ndarray,
    infos: list[dict[str, object]],
    difficulty_indices: np.ndarray,
    parallel_num_envs: int,
) -> dict[str, object]:
    difficulty_indices = np.asarray(difficulty_indices, dtype=np.int64)
    if difficulty_indices.shape != (len(infos),):
        raise ValueError("Difficulty indices do not match evaluation outcomes.")
    stop_errors = np.asarray(
        [float(info["stop_error"]) for info in infos],
        dtype=np.float64,
    )
    cue_speeds = np.asarray(
        [float(info["cue_speed"]) for info in infos],
        dtype=np.float64,
    )
    stopped_correct = np.asarray(
        [
            bool(info["correct_pot"])
            and not bool(info["cue_scratch"])
            and bool(info["stopped"])
            for info in infos
        ],
        dtype=np.bool_,
    )
    stopped_correct_errors = stop_errors[stopped_correct]

    def outcome_group(selected: np.ndarray) -> dict[str, float | int | None]:
        indices = np.flatnonzero(selected)
        selected_stop_errors = stop_errors[indices]
        selected_stopped_correct = stopped_correct[indices]
        successful_stop_errors = selected_stop_errors[selected_stopped_correct]
        return {
            "task_count": int(len(indices)),
            "correct_pot_rate": float(
                np.mean([bool(infos[index]["correct_pot"]) for index in indices])
            ),
            "joint_success_rate": float(
                np.mean([bool(infos[index]["joint_success"]) for index in indices])
            ),
            "scratch_rate": float(
                np.mean([bool(infos[index]["cue_scratch"]) for index in indices])
            ),
            "mean_stop_error_m": float(np.mean(selected_stop_errors)),
            "p95_stop_error_m": float(np.percentile(selected_stop_errors, 95)),
            "mean_stop_error_given_stopped_correct_pot_m": (
                float(np.mean(successful_stop_errors))
                if successful_stop_errors.size
                else None
            ),
        }

    per_pocket: dict[str, dict[str, float | int]] = {}
    for pocket_name in sorted({str(info["pocket_name"]) for info in infos}):
        selected = np.asarray(
            [str(info["pocket_name"]) == pocket_name for info in infos],
            dtype=np.bool_,
        )
        indices = np.flatnonzero(selected)
        per_pocket[pocket_name] = {
            "task_count": int(len(indices)),
            "correct_pot_rate": float(
                np.mean([bool(infos[index]["correct_pot"]) for index in indices])
            ),
            "joint_success_rate": float(
                np.mean([bool(infos[index]["joint_success"]) for index in indices])
            ),
            "scratch_rate": float(
                np.mean([bool(infos[index]["cue_scratch"]) for index in indices])
            ),
        }
    per_difficulty_cell = {
        cell.name: outcome_group(difficulty_indices == cell.index)
        for cell in TASK_DIFFICULTY_CELLS
        if bool(np.any(difficulty_indices == cell.index))
    }
    per_difficulty_level = {}
    for level, level_name in enumerate(DIFFICULTY_LEVEL_NAMES):
        level_cells = [
            cell.index for cell in TASK_DIFFICULTY_CELLS if cell.level == level
        ]
        selected = np.isin(difficulty_indices, level_cells)
        if bool(np.any(selected)):
            per_difficulty_level[level_name] = outcome_group(selected)
    angle_residual_deg = (
        actions[:, 0].astype(np.float64)
        * math.degrees(MAX_ANGLE_RESIDUAL)
    )
    return {
        "algorithm": "direct_behavior_cloning",
        "checkpoint": str(checkpoint),
        "task_count": len(infos),
        "physics_backend": backend,
        "policy_device": device,
        "parallel_num_envs": parallel_num_envs,
        "correct_pot_rate": float(
            np.mean([bool(info["correct_pot"]) for info in infos])
        ),
        "joint_success_rate": float(
            np.mean([bool(info["joint_success"]) for info in infos])
        ),
        "scratch_rate": float(
            np.mean([bool(info["cue_scratch"]) for info in infos])
        ),
        "wrong_pocket_rate": float(
            np.mean([bool(info["wrong_pocket"]) for info in infos])
        ),
        "timeout_rate": float(
            np.mean([bool(info["timed_out"]) for info in infos])
        ),
        "numerical_failure_rate": float(
            np.mean([bool(info["numerical_failure"]) for info in infos])
        ),
        "mean_stop_error_m": float(np.mean(stop_errors)),
        "p95_stop_error_m": float(np.percentile(stop_errors, 95)),
        "stopped_correct_pot_rate": float(np.mean(stopped_correct)),
        "mean_stop_error_given_stopped_correct_pot_m": (
            float(np.mean(stopped_correct_errors))
            if stopped_correct_errors.size
            else None
        ),
        "p95_stop_error_given_stopped_correct_pot_m": (
            float(np.percentile(stopped_correct_errors, 95))
            if stopped_correct_errors.size
            else None
        ),
        "angle_residual_deg_mean": float(np.mean(angle_residual_deg)),
        "angle_residual_deg_p95_abs": float(
            np.percentile(np.abs(angle_residual_deg), 95)
        ),
        "cue_speed_mean_mps": float(np.mean(cue_speeds)),
        "cue_speed_std_mps": float(np.std(cue_speeds)),
        "per_pocket": per_pocket,
        "difficulty_version": TASK_DIFFICULTY_VERSION,
        "per_difficulty_cell": per_difficulty_cell,
        "per_difficulty_level": per_difficulty_level,
    }


def _write_details(
    path: Path,
    checkpoint: Path,
    tasks: Path,
    dataset: TwoBallTaskDataset,
    actions: np.ndarray,
    infos: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": str(checkpoint),
        "tasks": str(tasks),
        "task_library_content_sha256": dataset.content_sha256(),
        "task_count": len(infos),
        "difficulty_version": TASK_DIFFICULTY_VERSION,
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary_path,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
            task_index=np.asarray(
                [int(info["task_index"]) for info in infos],
                dtype=np.int64,
            ),
            candidate_seed=dataset.candidate_seeds[: len(infos)],
            pocket_index=dataset.pocket_indices[: len(infos)],
            difficulty_cell_index=dataset.difficulty_indices()[: len(infos)],
            action=actions.astype(np.float32),
            reward=np.asarray(
                [float(info["reward"]) for info in infos],
                dtype=np.float32,
            ),
            stop_error_m=np.asarray(
                [float(info["stop_error"]) for info in infos],
                dtype=np.float32,
            ),
            cue_speed_mps=np.asarray(
                [float(info["cue_speed"]) for info in infos],
                dtype=np.float32,
            ),
            correct_pot=np.asarray(
                [bool(info["correct_pot"]) for info in infos],
                dtype=np.bool_,
            ),
            cue_scratch=np.asarray(
                [bool(info["cue_scratch"]) for info in infos],
                dtype=np.bool_,
            ),
            wrong_pocket=np.asarray(
                [bool(info["wrong_pocket"]) for info in infos],
                dtype=np.bool_,
            ),
            stopped=np.asarray(
                [bool(info["stopped"]) for info in infos],
                dtype=np.bool_,
            ),
            timed_out=np.asarray(
                [bool(info["timed_out"]) for info in infos],
                dtype=np.bool_,
            ),
            numerical_failure=np.asarray(
                [bool(info["numerical_failure"]) for info in infos],
                dtype=np.bool_,
            ),
            joint_success=np.asarray(
                [bool(info["joint_success"]) for info in infos],
                dtype=np.bool_,
            ),
        )
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    dataset = TwoBallTaskDataset.load(args.tasks, validate_model=False)
    require_balanced_task_difficulty(dataset, context="Evaluation")
    expected_backend = (
        MUJOCO_WARP_PHYSICS_BACKEND
        if args.backend == "mujoco-warp"
        else CPU_PHYSICS_BACKEND
    )
    if dataset.physics_backend != expected_backend:
        raise ValueError(
            f"--backend {args.backend!r} cannot evaluate "
            f"{dataset.physics_backend!r} tasks."
        )
    if dataset.execution_max_time != args.max_shot_time:
        raise ValueError(
            "--max-shot-time does not match the task dataset."
        )
    policy = DirectBCPolicy.load(args.checkpoint, device=args.device)
    validate_policy_for_dataset(policy, dataset)

    if args.backend == "mujoco-warp":
        action_rows, infos, parallel_num_envs = _evaluate_mujoco_warp(
            policy,
            dataset,
            args,
        )
    else:
        action_rows, infos, parallel_num_envs = _evaluate_cpu(
            policy,
            dataset,
            args,
        )
    actions = np.stack(action_rows)
    report = _aggregate_report(
        args.checkpoint,
        args.backend,
        args.device,
        actions,
        infos,
        dataset.difficulty_indices()[: len(infos)],
        parallel_num_envs,
    )
    if args.details_output is not None:
        _write_details(
            args.details_output,
            args.checkpoint,
            args.tasks,
            dataset,
            actions,
            infos,
        )
        print(f"details={args.details_output}", flush=True)
    if args.report_output is not None:
        write_json(args.report_output, report)
        print(f"report={args.report_output}", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
