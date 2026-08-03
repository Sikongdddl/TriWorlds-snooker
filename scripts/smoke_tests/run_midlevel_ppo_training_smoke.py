"""Run a four-environment PPO rollout, checkpoint, reload, and resume."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from _bootstrap import add_src_to_path

add_src_to_path()

from run_midlevel_two_ball_ppo_env_smoke import _fixture_task  # noqa: E402
from snooker_env.midlevel_ppo import (  # noqa: E402
    MIDLEVEL_TRAINING_MANIFEST_VERSION,
    BoundedActorCriticPolicy,
    PPOCheckpointMidLevelPolicy,
    behavior_clone_policy,
    require_checkpoint_manifest,
    set_independent_action_std,
)
from snooker_env.midlevel_ppo_env import MidLevelTwoBallPPOEnv  # noqa: E402
from snooker_env.midlevel_tasks import TwoBallTaskDataset  # noqa: E402
from snooker_env.midlevel_two_ball import TwoBallShotSimulator  # noqa: E402
from snooker_env.pipeline_types import (  # noqa: E402
    BallState,
    SceneState,
    ShotIntent,
    SkillCommand,
    SkillId,
    TableState,
)


def _factory(task_path: Path, rank: int) -> Callable[[], gym.Env]:
    def initialize() -> gym.Env:
        env = MidLevelTwoBallPPOEnv(task_path)
        env.reset(seed=100 + rank)
        return Monitor(env)

    return initialize


def main() -> None:
    simulator = TwoBallShotSimulator()
    task = _fixture_task(simulator)
    dataset = TwoBallTaskDataset.from_tasks([task], simulator, generation_seed=7)
    with tempfile.TemporaryDirectory(prefix="midlevel-ppo-smoke-") as directory:
        directory_path = Path(directory)
        task_path = directory_path / "tasks.npz"
        checkpoint = directory_path / "ppo_smoke"
        dataset.save(task_path)
        env = SubprocVecEnv(
            [_factory(task_path, rank) for rank in range(4)], start_method="spawn"
        )
        policy = PPO(
            BoundedActorCriticPolicy,
            env,
            seed=5,
            device="cpu",
            n_steps=1,
            batch_size=4,
            n_epochs=1,
            gamma=1.0,
            gae_lambda=1.0,
            learning_rate=3e-4,
            policy_kwargs={"net_arch": [32, 32], "log_std_init": -1.0},
        )
        set_independent_action_std(policy.policy, (0.05, 0.25))
        bc_report = behavior_clone_policy(
            policy.policy,
            dataset,
            epochs=100,
            batch_size=1,
            learning_rate=1e-3,
            angle_weight=4.0,
            seed=5,
        )
        action_std = np.exp(policy.policy.log_std.detach().cpu().numpy())
        if not np.allclose(action_std, (0.05, 0.25), atol=1e-7, rtol=0.0):
            raise RuntimeError("Per-axis PPO exploration std was not preserved.")
        if (
            bc_report.final_loss >= bc_report.initial_loss
            or bc_report.final_angle_mae_deg > 0.05
            or bc_report.final_speed_mae_mps > 0.01
        ):
            raise RuntimeError(
                f"Behavior cloning did not reconstruct the generated action: {bc_report}"
            )
        manifest = {
            "manifest_version": MIDLEVEL_TRAINING_MANIFEST_VERSION,
            "fixture": dataset.content_sha256(),
        }
        policy.midlevel_behavior_cloning_report = bc_report.as_dict()
        policy.midlevel_training_manifest = manifest
        policy.learn(total_timesteps=4)
        policy.save(checkpoint)
        reloaded = PPO.load(checkpoint, env=env, device="cpu")
        require_checkpoint_manifest(reloaded, manifest, context="Smoke resume")
        if getattr(reloaded, "midlevel_behavior_cloning_report", None) != (
            bc_report.as_dict()
        ):
            raise RuntimeError("Behavior-cloning report did not survive checkpoint load.")
        reloaded.learn(total_timesteps=4, reset_num_timesteps=False)
        reloaded.train()
        observation = env.reset()
        action, _ = reloaded.predict(observation, deterministic=True)
        parameters_finite = all(
            bool(np.all(np.isfinite(parameter.detach().cpu().numpy())))
            for parameter in reloaded.policy.parameters()
        )
        rollout_finite = all(
            bool(np.all(np.isfinite(values)))
            for values in (
                reloaded.rollout_buffer.rewards,
                reloaded.rollout_buffer.advantages,
                reloaded.rollout_buffer.returns,
            )
        )
        loss_values = {
            name: float(value)
            for name, value in reloaded.logger.name_to_value.items()
            if name in ("train/loss", "train/policy_gradient_loss", "train/value_loss")
        }
        losses_finite = bool(loss_values) and all(
            np.isfinite(value) for value in loss_values.values()
        )
        adapter = PPOCheckpointMidLevelPolicy(checkpoint)
        state = SceneState(
            time=0.0,
            table=TableState(),
            balls={
                "cue_ball": BallState(
                    "cue_ball", np.array([*task.cue_position, 1.0785], dtype=np.float64)
                ),
                "object_ball_0": BallState(
                    "object_ball_0",
                    np.array([*task.object_position, 1.0785], dtype=np.float64),
                ),
            },
        )
        commands = adapter.rollout(
            SkillCommand(
                skill_id=SkillId.POSITION_SHOT,
                intent=ShotIntent(
                    cue_ball_name="cue_ball",
                    object_ball_name="object_ball_0",
                    target_pocket=task.pocket_name,
                    desired_cue_ball_position=task.target_stop_position,
                ),
            ),
            state,
        )
        env.close()

    print(
        f"timesteps={reloaded.num_timesteps} action={np.round(action[0], 4).tolist()} "
        f"rollout_finite={rollout_finite} losses_finite={losses_finite} "
        f"parameters_finite={parameters_finite} cue_commands={len(commands)}"
        f" bc_final_loss={bc_report.final_loss:.6g}"
    )
    if reloaded.num_timesteps != 8:
        raise RuntimeError("PPO checkpoint did not resume its timestep count.")
    if not rollout_finite or not losses_finite:
        raise RuntimeError("PPO produced non-finite rewards, advantages, returns, or losses.")
    if not parameters_finite or not np.all(np.isfinite(action)):
        raise RuntimeError("PPO produced non-finite parameters or actions.")
    if len(commands) != 3 or not all(
        np.all(np.isfinite(command.linear_velocity)) for command in commands
    ):
        raise RuntimeError("Checkpoint did not adapt to finite CueCommand outputs.")


if __name__ == "__main__":
    main()
