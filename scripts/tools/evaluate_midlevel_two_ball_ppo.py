"""Evaluate a PPO checkpoint on a fixed two-ball validation library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.midlevel_mujoco_warp_vec_env import (  # noqa: E402
    MJWarpMidLevelVecEnv,
)
from snooker_env.midlevel_ppo import (  # noqa: E402
    MIDLEVEL_TRAINING_MANIFEST_VERSION,
    require_checkpoint_manifest_subset,
)
from snooker_env.midlevel_ppo_env import MidLevelTwoBallPPOEnv  # noqa: E402
from snooker_env.midlevel_tasks import (  # noqa: E402
    CPU_PHYSICS_BACKEND,
    MUJOCO_WARP_PHYSICS_BACKEND,
    TwoBallTaskDataset,
)
from snooker_env.midlevel_two_ball import SHOT_EXECUTION_VERSION  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "outputs" / "tasks" / "midlevel_two_ball_validation.npz",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
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
    args = parser.parse_args()

    if args.max_tasks is not None and args.max_tasks <= 0:
        raise ValueError("--max-tasks must be positive when provided.")
    dataset = TwoBallTaskDataset.load(args.tasks, validate_model=False)
    requested_physics_backend = (
        MUJOCO_WARP_PHYSICS_BACKEND
        if args.backend == "mujoco-warp"
        else CPU_PHYSICS_BACKEND
    )
    if dataset.physics_backend != requested_physics_backend:
        raise ValueError(
            f"--backend {args.backend!r} cannot evaluate a "
            f"{dataset.physics_backend!r} task dataset."
        )
    if dataset.execution_max_time != args.max_shot_time:
        raise ValueError(
            "--max-shot-time does not match the task dataset execution setting."
        )
    policy = PPO.load(str(args.checkpoint), device=args.device)
    require_checkpoint_manifest_subset(
        policy,
        {
            "manifest_version": MIDLEVEL_TRAINING_MANIFEST_VERSION,
            "shot_execution_version": SHOT_EXECUTION_VERSION,
            "physics": {
                "xml_sha256": dataset.xml_hash,
                "model_sha256": dataset.model_hash,
                "backend": dataset.physics_backend,
                "backend_sha256": dataset.backend_hash,
            },
            "environment": {
                "backend": args.backend,
                "max_shot_time": args.max_shot_time,
            },
        },
        context="Evaluation",
    )
    infos: list[dict[str, object]] = []
    actions: list[np.ndarray] = []
    if args.backend == "mujoco-warp":
        if args.num_envs <= 0:
            raise ValueError("--num-envs must be positive.")
        task_count = len(dataset)
        count = task_count if args.max_tasks is None else min(task_count, args.max_tasks)
        batch_size = min(args.num_envs, count)
        env = MJWarpMidLevelVecEnv(
            dataset,
            args.model,
            num_envs=batch_size,
            device=args.physics_device,
            chunk_steps=args.chunk_steps,
            check_interval_steps=args.check_interval_steps,
            max_time=args.max_shot_time,
        )
        try:
            for start in range(0, count, batch_size):
                valid_count = min(batch_size, count - start)
                indices = [
                    start + min(offset, valid_count - 1)
                    for offset in range(batch_size)
                ]
                env.set_options(
                    [{"task_index": task_index} for task_index in indices]
                )
                observation = env.reset()
                action, _ = policy.predict(
                    observation,
                    deterministic=not args.stochastic,
                )
                _, rewards, _, batch_infos = env.step(action)
                for offset in range(valid_count):
                    info = batch_infos[offset]
                    info["reward"] = float(rewards[offset])
                    infos.append(info)
                    actions.append(np.asarray(action[offset], dtype=np.float64))
                print(f"evaluated={start + valid_count}/{count}", flush=True)
        finally:
            env.close()
    else:
        env = MidLevelTwoBallPPOEnv(
            args.tasks,
            args.model,
            max_time=args.max_shot_time,
        )
        count = len(env.tasks) if args.max_tasks is None else min(
            len(env.tasks),
            args.max_tasks,
        )
        try:
            for task_index in range(count):
                observation, _ = env.reset(options={"task_index": task_index})
                action, _ = policy.predict(
                    observation,
                    deterministic=not args.stochastic,
                )
                _, reward, _, _, info = env.step(action)
                info["reward"] = reward
                infos.append(info)
                actions.append(np.asarray(action, dtype=np.float64))
                if (task_index + 1) % 50 == 0 or task_index + 1 == count:
                    print(f"evaluated={task_index + 1}/{count}", flush=True)
        finally:
            env.close()

    action_array = np.stack(actions)
    stop_errors = np.asarray([float(info["stop_error"]) for info in infos])
    cue_speeds = np.asarray([float(info["cue_speed"]) for info in infos])
    angle_residuals_deg = action_array[:, 0] * 15.0
    report = {
        "task_count": count,
        "physics_backend": args.backend,
        "correct_pot_rate": float(np.mean([bool(info["correct_pot"]) for info in infos])),
        "joint_success_rate": float(np.mean([bool(info["joint_success"]) for info in infos])),
        "mean_stop_error_m": float(np.mean(stop_errors)),
        "p95_stop_error_m": float(np.percentile(stop_errors, 95)),
        "scratch_rate": float(np.mean([bool(info["cue_scratch"]) for info in infos])),
        "wrong_pocket_rate": float(np.mean([bool(info["wrong_pocket"]) for info in infos])),
        "timeout_rate": float(np.mean([bool(info["timed_out"]) for info in infos])),
        "numerical_failure_rate": float(
            np.mean([bool(info["numerical_failure"]) for info in infos])
        ),
        "angle_residual_deg_mean": float(np.mean(angle_residuals_deg)),
        "angle_residual_deg_std": float(np.std(angle_residuals_deg)),
        "angle_residual_deg_min": float(np.min(angle_residuals_deg)),
        "angle_residual_deg_max": float(np.max(angle_residuals_deg)),
        "angle_residual_deg_p05": float(np.percentile(angle_residuals_deg, 5)),
        "angle_residual_deg_p95": float(np.percentile(angle_residuals_deg, 95)),
        "cue_speed_mean_mps": float(np.mean(cue_speeds)),
        "cue_speed_std_mps": float(np.std(cue_speeds)),
        "cue_speed_min_mps": float(np.min(cue_speeds)),
        "cue_speed_max_mps": float(np.max(cue_speeds)),
        "acceptance_targets": {
            "correct_pot_rate": 0.90,
            "joint_success_rate": 0.70,
            "max_scratch_rate": 0.02,
            "max_wrong_pocket_rate": 0.01,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
