"""Check TD3 critics, delayed actor updates, BC anchoring, and resume state."""

from __future__ import annotations

from pathlib import Path
import tempfile

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_ppo_env import MAX_TERMINAL_REWARD  # noqa: E402
from snooker_env.midlevel_sac_her import (  # noqa: E402
    ConservativeResidualTD3Policy,
    SingleStepCuePositionHerReplayBuffer,
    SingleStepTD3BC,
    replay_buffer_path,
)


class _OneStepEnv(gym.Env[np.ndarray, np.ndarray]):
    observation_space = gym.spaces.Box(
        -1.0,
        1.0,
        shape=(8,),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(
        -1.0,
        1.0,
        shape=(2,),
        dtype=np.float32,
    )

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
        del action
        return np.zeros(8, dtype=np.float32), 0.0, True, False, {}


def main() -> None:
    environment = DummyVecEnv([_OneStepEnv for _ in range(4)])
    policy = SingleStepTD3BC(
        ConservativeResidualTD3Policy,
        environment,
        seed=11,
        device="cpu",
        learning_rate=3.0e-4,
        actor_learning_rate=1.0e-4,
        critic_learning_rate=3.0e-4,
        actor_update_interval=2,
        buffer_size=64,
        batch_size=4,
        learning_starts=0,
        gamma=0.0,
        train_freq=(1, "step"),
        gradient_steps=4,
        replay_buffer_class=SingleStepCuePositionHerReplayBuffer,
        replay_buffer_kwargs={
            "her_ratio": 0.25,
            "success_ratio": 0.25,
            "failure_ratio": 0.25,
        },
        policy_kwargs={"net_arch": [32, 32]},
    )
    policy.set_logger(configure(format_strings=[]))
    observations = np.zeros((4, 8), dtype=np.float32)
    observations[:, 0] = np.linspace(-0.3, 0.3, 4)
    actions = np.zeros((4, 2), dtype=np.float32)
    actions[:, 1] = np.linspace(-0.2, 0.2, 4)
    policy.configure_conservative_speed_residual(
        max_speed_residual_mps=0.12,
        exploration_initial_std=0.35,
        exploration_final_std=0.05,
        exploration_decay_timesteps=16,
    )
    policy.configure_discrete_candidate_ranking((-0.12, 0.0, 0.12))
    policy.configure_behavior_cloning_reference(
        observations,
        actions,
        initial_weight=1.0,
        final_weight=0.1,
        decay_actor_updates=4,
        batch_size=4,
        angle_weight=4.0,
        speed_weight=4.0,
    )
    infos = [
        {
            "correct_pot": True,
            "cue_scratch": False,
            "stopped": True,
            "cue_ball_final_position": np.array([0.0, 0.0, 0.0]),
        },
        {
            "correct_pot": False,
            "cue_scratch": True,
            "stopped": False,
            "cue_ball_final_position": np.array([0.0, 0.0, 0.0]),
        },
        {
            "correct_pot": False,
            "wrong_pocket": True,
            "stopped": True,
            "cue_ball_final_position": np.array([0.0, 0.0, 0.0]),
        },
        {
            "correct_pot": False,
            "cue_scratch": False,
            "stopped": True,
            "cue_ball_final_position": np.array([0.0, 0.0, 0.0]),
        },
    ]
    policy.replay_buffer.add(
        observations,
        observations.copy(),
        actions,
        np.array([MAX_TERMINAL_REWARD, 0.0, 0.0, 0.2], dtype=np.float32),
        np.ones(4, dtype=np.bool_),
        infos,
    )
    replay_data = policy.replay_buffer.sample(4)
    with torch.no_grad():
        critic_values = policy._critic_values(
            replay_data.observations,
            replay_data.actions,
        )
        if len(critic_values) != 2:
            raise RuntimeError("TD3-style optimizer did not build twin critics.")
        expected_direct_reward_loss = float(
            (
                0.5
                * sum(
                    torch.nn.functional.smooth_l1_loss(
                        value,
                        replay_data.rewards,
                    )
                    for value in critic_values
                )
            ).item()
        )
        minimum, _ = policy._minimum_critic_value(
            replay_data.observations,
            replay_data.actions,
        )
        expected_minimum = torch.minimum(critic_values[0], critic_values[1])
        if not torch.equal(minimum, expected_minimum):
            raise RuntimeError("Pessimistic critic value did not select min(Q1, Q2).")
    observed_direct_reward_loss = policy._critic_update(replay_data)
    if not np.isclose(
        observed_direct_reward_loss,
        expected_direct_reward_loss,
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        raise RuntimeError("Critics did not regress the immediate reward directly.")
    actor_before = [parameter.detach().clone() for parameter in policy.actor.parameters()]
    warmup = policy.warmup_critic(4, batch_size=4)
    if any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, policy.actor.parameters())
    ):
        raise RuntimeError("Critic warmup changed the BC actor.")
    policy.train(gradient_steps=4, batch_size=4)
    if policy._actor_updates != 2:
        raise RuntimeError("Delayed actor updates did not use the configured interval.")
    if policy.replay_buffer.last_sample_composition != {
        "uniform": 1,
        "success": 1,
        "failure": 1,
        "local_probe": 0,
        "hindsight": 1,
    }:
        raise RuntimeError("Optimizer did not consume stratified replay.")

    with tempfile.TemporaryDirectory(prefix="midlevel-td3-her-optimizer-") as directory:
        checkpoint = Path(directory) / "optimizer_smoke"
        policy.save(checkpoint)
        policy.save_replay_buffer(replay_buffer_path(checkpoint))
        restored = SingleStepTD3BC.load(
            checkpoint,
            env=environment,
            device="cpu",
        )
        restored.set_logger(configure(format_strings=[]))
        restored.load_replay_buffer(replay_buffer_path(checkpoint))
        if restored._actor_updates != 2:
            raise RuntimeError("Checkpoint lost delayed actor update state.")
        if restored.bc_reference_actions is None:
            raise RuntimeError("Checkpoint lost the online BC reference set.")
        restored.train(gradient_steps=2, batch_size=4)
        if restored._actor_updates != 3:
            raise RuntimeError("Resumed delayed actor schedule is inconsistent.")

    environment.close()
    print(
        f"warmup_initial={warmup['initial_loss']:.6g} "
        f"warmup_final={warmup['final_loss']:.6g} "
        f"actor_updates={restored._actor_updates} "
        f"bc_weight={restored._behavior_cloning_weight():.3f} "
        "direct_reward=True twin_min=True stratified=True resume=True"
    )


if __name__ == "__main__":
    main()
