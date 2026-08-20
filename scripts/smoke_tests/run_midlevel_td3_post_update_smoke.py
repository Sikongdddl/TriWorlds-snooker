"""Check that TD3 logs and checkpoints are emitted after each full update."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile

import gymnasium as gym
import numpy as np
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_sac_her import (  # noqa: E402
    ConservativeResidualTD3Policy,
    PostUpdateTrainingHook,
    SingleStepCuePositionHerReplayBuffer,
    SingleStepTD3BC,
    resolve_replay_buffer_path,
)


class _OneStepEnv(gym.Env[np.ndarray, np.ndarray]):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        return np.zeros(8, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        return (
            np.zeros(8, dtype=np.float32),
            float(1.0 + 0.1 * action[1]),
            True,
            False,
            {
                "correct_pot": False,
                "cue_scratch": False,
                "stopped": True,
                "cue_ball_final_position": np.zeros(3, dtype=np.float64),
            },
        )


def main() -> None:
    environment = DummyVecEnv([_OneStepEnv for _ in range(4)])
    model = SingleStepTD3BC(
        ConservativeResidualTD3Policy,
        environment,
        seed=31,
        device="cpu",
        learning_rate=3.0e-4,
        actor_learning_rate=1.0e-5,
        critic_learning_rate=3.0e-4,
        actor_update_interval=4,
        actor_learning_starts=8,
        residual_l2_weight=0.02,
        buffer_size=64,
        batch_size=4,
        learning_starts=0,
        gamma=0.0,
        train_freq=(1, "step"),
        gradient_steps=4,
        replay_buffer_class=SingleStepCuePositionHerReplayBuffer,
        replay_buffer_kwargs={
            "her_ratio": 0.0,
            "success_ratio": 0.0,
            "failure_ratio": 0.0,
        },
        policy_kwargs={"net_arch": [32, 32]},
    )
    model.configure_conservative_speed_residual(
        max_speed_residual_mps=0.12,
        exploration_initial_std=0.35,
        exploration_final_std=0.05,
        exploration_decay_timesteps=8,
    )
    model.configure_discrete_candidate_ranking((-0.12, 0.0, 0.12))

    with tempfile.TemporaryDirectory(prefix="midlevel-td3-post-update-") as directory:
        output_directory = Path(directory)
        checkpoint_base = output_directory / "post_update"
        model.set_logger(configure(str(output_directory), ["csv"]))
        hook = PostUpdateTrainingHook(
            checkpoint_base,
            checkpoint_every=8,
            initial_timesteps=0,
        )
        model.set_post_update_hook(hook)
        try:
            model.learn(total_timesteps=8, log_interval=None)
        finally:
            model.set_post_update_hook(None)

        checkpoint = output_directory / "post_update_step_8_steps.zip"
        if not checkpoint.is_file():
            raise RuntimeError("Post-update checkpoint was not saved at timestep 8.")
        replay_path = resolve_replay_buffer_path(checkpoint)
        if not replay_path.is_file():
            raise RuntimeError("Post-update replay companion is missing.")
        restored = SingleStepTD3BC.load(
            checkpoint,
            env=environment,
            device="cpu",
        )
        restored.load_replay_buffer(replay_path)
        restored_count = restored.replay_buffer.size() * restored.replay_buffer.n_envs
        if (
            restored.num_timesteps != 8
            or restored_count != 8
            or restored._n_updates != 8
            or restored._actor_updates != 1
        ):
            raise RuntimeError(
                "Checkpoint timestep, replay, critic, and actor state are not aligned."
            )
        if restored._post_update_hook is not None:
            raise RuntimeError("Checkpoint serialized its process-local update hook.")

        with (output_directory / "progress.csv").open(
            newline="",
            encoding="utf-8",
        ) as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 2:
            raise RuntimeError("Expected exactly one log row per completed rollout.")
        expected = ((4, 4), (8, 8))
        observed = tuple(
            (
                int(float(row["replay/stored_transitions"])),
                int(float(row["train/n_updates"])),
            )
            for row in rows
        )
        if observed != expected:
            raise RuntimeError(
                f"Post-update log rows are stale: expected={expected}, observed={observed}."
            )

    environment.close()
    print(
        "checkpoint_timesteps=8 replay_transitions=8 critic_updates=8 "
        "actor_updates=1 log_rows=2 aligned=True"
    )


if __name__ == "__main__":
    main()
