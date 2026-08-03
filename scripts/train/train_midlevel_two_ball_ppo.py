"""Train or resume single-step contextual PPO with CPU or batched MJWarp physics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecMonitor

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.midlevel_mujoco_warp_vec_env import (  # noqa: E402
    MUJOCO_WARP_SHOT_EXECUTION_VERSION,
    MJWarpMidLevelVecEnv,
)
from snooker_env.midlevel_ppo import (  # noqa: E402
    MIDLEVEL_TRAINING_MANIFEST_VERSION,
    BoundedActorCriticPolicy,
    behavior_clone_policy,
    require_checkpoint_manifest,
    set_independent_action_std,
)
from snooker_env.midlevel_ppo_env import MidLevelTwoBallPPOEnv  # noqa: E402
from snooker_env.midlevel_tasks import (  # noqa: E402
    CPU_PHYSICS_BACKEND,
    MUJOCO_WARP_PHYSICS_BACKEND,
    TwoBallTaskDataset,
)
from snooker_env.midlevel_two_ball import SHOT_EXECUTION_VERSION  # noqa: E402


def make_cpu_env(
    dataset: Path,
    model: Path,
    rank: int,
    seed: int,
    max_time: float,
) -> Callable[[], gym.Env]:
    """Return a spawn-safe factory for one monitored MuJoCo environment."""

    def initialize() -> gym.Env:
        env = MidLevelTwoBallPPOEnv(dataset, model, max_time=max_time)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return initialize


def build_environments(args: argparse.Namespace) -> VecEnv:
    """Build the requested vector backend with identical PPO contracts."""

    if args.backend == "mujoco-warp":
        environment = MJWarpMidLevelVecEnv(
            args.tasks,
            args.model,
            num_envs=args.num_envs,
            seed=args.seed,
            device=args.physics_device,
            chunk_steps=args.chunk_steps,
            check_interval_steps=args.check_interval_steps,
            nconmax=args.nconmax,
            njmax=args.njmax,
            max_time=args.max_shot_time,
        )
        environment.seed(args.seed)
        return VecMonitor(environment)
    return SubprocVecEnv(
        [
            make_cpu_env(
                args.tasks,
                args.model,
                rank,
                args.seed,
                args.max_shot_time,
            )
            for rank in range(args.num_envs)
        ],
        start_method="spawn",
    )


def build_training_manifest(
    args: argparse.Namespace,
    dataset: TwoBallTaskDataset,
) -> dict[str, object]:
    """Describe every input that changes rollout or PPO update semantics."""

    environment_settings: dict[str, object] = {
        "backend": args.backend,
        "num_envs": args.num_envs,
        "max_shot_time": args.max_shot_time,
    }
    if args.backend == "mujoco-warp":
        environment_settings.update(
            {
                "chunk_steps": args.chunk_steps,
                "check_interval_steps": args.check_interval_steps,
                "nconmax": args.nconmax,
                "njmax": args.njmax,
                "mujoco_warp_shot_execution_version": (
                    MUJOCO_WARP_SHOT_EXECUTION_VERSION
                ),
            }
        )
    return {
        "manifest_version": MIDLEVEL_TRAINING_MANIFEST_VERSION,
        "shot_execution_version": SHOT_EXECUTION_VERSION,
        "physics": {
            "xml_sha256": dataset.xml_hash,
            "model_sha256": dataset.model_hash,
            "backend": dataset.physics_backend,
            "backend_sha256": dataset.backend_hash,
        },
        "task_dataset": {
            "content_sha256": dataset.content_sha256(),
            "task_count": len(dataset),
            "generation_seed": dataset.generation_seed,
            "execution_max_time": dataset.execution_max_time,
            "stop_speed": dataset.stop_speed,
            "stop_hold_time": dataset.stop_hold_time,
        },
        "environment": environment_settings,
        "algorithm": {
            "name": "PPO",
            "policy": "BoundedActorCriticPolicy",
            "net_arch": [256, 256],
            "initial_action_std": [
                args.angle_action_std,
                args.speed_action_std,
            ],
            "n_steps": 1,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "gamma": 1.0,
            "gae_lambda": 1.0,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "seed": args.seed,
        },
        "behavior_cloning": {
            "epochs": args.bc_epochs,
            "batch_size": args.bc_batch_size,
            "learning_rate": args.bc_learning_rate,
            "angle_weight": args.bc_angle_weight,
            "max_grad_norm": args.bc_max_grad_norm,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "outputs" / "tasks" / "midlevel_two_ball_train.npz",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument(
        "--backend",
        choices=("mujoco-warp", "cpu"),
        default="mujoco-warp",
        help="Batched GPU physics is the default; cpu retains the legacy subprocess path.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--angle-action-std",
        type=float,
        default=0.05,
        help="Initial latent Gaussian std for angle residual (0.05 is about 0.75 degrees).",
    )
    parser.add_argument(
        "--speed-action-std",
        type=float,
        default=0.25,
        help="Initial latent Gaussian std for normalized cue speed.",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=100,
        help="Generated-action behavior-cloning epochs before a fresh PPO run.",
    )
    parser.add_argument("--bc-batch-size", type=int, default=1024)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bc-angle-weight", type=float, default=4.0)
    parser.add_argument("--bc-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Stable-Baselines3 policy device.",
    )
    parser.add_argument(
        "--physics-device",
        default="cuda:0",
        help="CUDA device used by the MJWarp physics batch.",
    )
    parser.add_argument("--chunk-steps", type=int, default=16)
    parser.add_argument("--check-interval-steps", type=int, default=2048)
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=1024)
    parser.add_argument("--max-shot-time", type=float, default=8.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "checkpoints" / "midlevel_two_ball_ppo",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    args = parser.parse_args()

    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive.")
    if args.max_shot_time <= 0.0:
        raise ValueError("--max-shot-time must be positive.")
    if args.n_epochs <= 0:
        raise ValueError("--n-epochs must be positive.")
    for name in (
        "learning_rate",
        "angle_action_std",
        "speed_action_std",
        "bc_learning_rate",
        "bc_angle_weight",
        "bc_max_grad_norm",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    if args.bc_epochs < 0:
        raise ValueError("--bc-epochs must be non-negative.")
    if args.bc_batch_size <= 0:
        raise ValueError("--bc-batch-size must be positive.")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive.")
    rollout_size = args.num_envs
    if not 2 <= args.batch_size <= rollout_size:
        raise ValueError(
            f"--batch-size must be in [2, num_envs={args.num_envs}] because n_steps=1."
        )
    if rollout_size % args.batch_size != 0:
        raise ValueError("--batch-size must divide num_envs for complete PPO minibatches.")
    if not args.tasks.exists():
        raise FileNotFoundError(args.tasks)

    dataset = TwoBallTaskDataset.load(args.tasks, validate_model=False)
    required_task_backend = (
        MUJOCO_WARP_PHYSICS_BACKEND
        if args.backend == "mujoco-warp"
        else CPU_PHYSICS_BACKEND
    )
    if dataset.physics_backend != required_task_backend:
        raise ValueError(
            f"--backend {args.backend!r} requires a {required_task_backend!r} "
            f"task dataset, received {dataset.physics_backend!r}."
        )
    if dataset.execution_max_time != args.max_shot_time:
        raise ValueError(
            "--max-shot-time does not match the task dataset execution setting."
        )
    manifest = build_training_manifest(args, dataset)
    set_random_seed(args.seed)
    print(
        f"building backend={args.backend} num_envs={args.num_envs} "
        f"physics_device={args.physics_device if args.backend == 'mujoco-warp' else 'cpu'}",
        flush=True,
    )
    environments = build_environments(args)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        bc_report_data: dict[str, object] | None = None
        if args.resume is not None:
            policy = PPO.load(str(args.resume), env=environments, device=args.device)
            require_checkpoint_manifest(policy, manifest, context="Resume")
            stored_bc_report = getattr(
                policy,
                "midlevel_behavior_cloning_report",
                None,
            )
            if isinstance(stored_bc_report, dict):
                bc_report_data = dict(stored_bc_report)
            reset_num_timesteps = False
        else:
            policy = PPO(
                BoundedActorCriticPolicy,
                environments,
                seed=args.seed,
                device=args.device,
                verbose=1,
                n_steps=1,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                learning_rate=args.learning_rate,
                gamma=1.0,
                gae_lambda=1.0,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs={
                    "net_arch": [256, 256],
                    "log_std_init": -1.0,
                },
            )
            set_independent_action_std(
                policy.policy,
                (args.angle_action_std, args.speed_action_std),
            )
            if args.bc_epochs > 0:
                bc_report = behavior_clone_policy(
                    policy.policy,
                    dataset,
                    epochs=args.bc_epochs,
                    batch_size=args.bc_batch_size,
                    learning_rate=args.bc_learning_rate,
                    angle_weight=args.bc_angle_weight,
                    seed=args.seed,
                    max_grad_norm=args.bc_max_grad_norm,
                )
                bc_report_data = bc_report.as_dict()
                policy.midlevel_behavior_cloning_report = bc_report_data
                print(
                    "behavior_cloning="
                    + json.dumps(
                        bc_report_data,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            reset_num_timesteps = True
        policy.midlevel_training_manifest = manifest

        callback = CheckpointCallback(
            save_freq=max(1, args.checkpoint_every // args.num_envs),
            save_path=str(args.output.parent),
            name_prefix=f"{args.output.name}_step",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        policy.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
        policy.save(str(args.output))
        manifest_path = Path(f"{args.output}.manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if bc_report_data is not None:
            Path(f"{args.output}.bc.json").write_text(
                json.dumps(bc_report_data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if args.backend == "mujoco-warp":
            throughput = float(
                environments.get_attr(
                    "last_world_steps_per_second",
                    indices=0,
                )[0]
            )
            rollout_wall_seconds = float(
                environments.get_attr(
                    "last_rollout_wall_seconds",
                    indices=0,
                )[0]
            )
            print(
                f"last_mujoco_warp_world_steps_per_second={throughput:.0f} "
                f"last_rollout_wall_seconds={rollout_wall_seconds:.3f}",
                flush=True,
            )
    finally:
        environments.close()
    checkpoint_path = (
        args.output
        if args.output.suffix == ".zip"
        else Path(f"{args.output}.zip")
    )
    print(f"checkpoint={checkpoint_path} manifest={args.output}.manifest.json")


if __name__ == "__main__":
    main()
