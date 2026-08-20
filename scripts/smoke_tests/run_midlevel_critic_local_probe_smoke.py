"""Check BC-centered safety ranking on grouped local candidates."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_ppo_env import MAX_TERMINAL_REWARD  # noqa: E402
from snooker_env.midlevel_sac_her import (  # noqa: E402
    ConservativeResidualTD3Policy,
    SingleStepCuePositionHerReplayBuffer,
    SingleStepTD3BC,
    critic_local_speed_diagnostics,
)


class _OneStepEnv(gym.Env[np.ndarray, np.ndarray]):
    observation_space = gym.spaces.Box(-1.0, 1.0, (8,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

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


class _SinglePocketDataset:
    def __init__(self, count: int) -> None:
        self.pocket_indices = np.zeros(count, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.pocket_indices)


def main() -> None:
    task_count = 32
    probe_offsets = np.asarray((-0.03, -0.01, 0.0, 0.01, 0.03))
    env_count = task_count
    environment = DummyVecEnv([_OneStepEnv for _ in range(env_count)])
    model = SingleStepTD3BC(
        ConservativeResidualTD3Policy,
        environment,
        seed=7,
        device="cuda:0",
        learning_rate=3.0e-4,
        actor_learning_rate=1.0e-5,
        critic_learning_rate=3.0e-4,
        critic_probe_ranking_weight=5.0,
        critic_probe_ranking_margin=0.1,
        critic_probe_minimum_reward_difference=0.05,
        critic_probe_holdout_fraction=0.2,
        critic_probe_holdout_seed=3,
        buffer_size=env_count * 8,
        batch_size=env_count,
        learning_starts=0,
        gamma=0.0,
        train_freq=(1, "step"),
        gradient_steps=1,
        replay_buffer_class=SingleStepCuePositionHerReplayBuffer,
        replay_buffer_kwargs={
            "her_ratio": 0.10,
            "success_ratio": 0.20,
            "failure_ratio": 0.20,
            "local_probe_ratio": 0.25,
            "probe_holdout_fraction": 0.20,
            "probe_holdout_seed": 3,
        },
        policy_kwargs={"net_arch": [64, 64], "n_critics": 2},
    )
    model.set_logger(configure(format_strings=[]))
    rng = np.random.default_rng(3)
    observations = rng.uniform(
        -0.7,
        0.7,
        size=(task_count, 8),
    ).astype(np.float32)
    baseline, _ = model.predict(observations, deterministic=True)
    baseline = np.asarray(baseline, dtype=np.float32)
    baseline[:, 0] = 0.0
    certified = baseline.copy()
    correction_mps = np.where(observations[:, 0] >= 0.0, 0.03, -0.03)
    certified[:, 1] += correction_mps.astype(np.float32) / 1.1

    model.configure_conservative_speed_residual(
        max_speed_residual_mps=0.12,
        exploration_initial_std=0.35,
        exploration_final_std=0.05,
        exploration_decay_timesteps=100,
    )
    model.configure_discrete_candidate_ranking(tuple(probe_offsets))
    model.configure_behavior_cloning_reference(
        observations,
        certified,
        initial_weight=1.0,
        final_weight=1.0,
        decay_actor_updates=1,
        batch_size=task_count,
        angle_weight=1.0,
        speed_weight=8.0,
    )

    def infos(offsets_mps: np.ndarray | None = None) -> list[dict[str, object]]:
        values: list[dict[str, object]] = []
        for env_id in range(env_count):
            info: dict[str, object] = {
                "correct_pot": True,
                "cue_scratch": False,
                "wrong_pocket": False,
                "stopped": True,
                "timed_out": False,
                "numerical_failure": False,
                "cue_ball_final_position": np.zeros(3, dtype=np.float64),
            }
            if offsets_mps is not None:
                info.update(
                    local_speed_probe=True,
                    local_speed_probe_offset_mps=float(offsets_mps[env_id]),
                    local_speed_probe_task_index=env_id,
                )
            values.append(info)
        return values

    model.replay_buffer.add(
        observations,
        observations.copy(),
        certified,
        np.full(env_count, MAX_TERMINAL_REWARD, dtype=np.float32),
        np.ones(env_count, dtype=np.bool_),
        infos(),
    )
    rows_by_offset: dict[float, int] = {}
    for offset in probe_offsets:
        batch_offsets = np.full(task_count, offset, dtype=np.float64)
        probe_actions = baseline.copy()
        probe_actions[:, 1] += np.float32(offset / 1.1)
        reward = np.clip(
            MAX_TERMINAL_REWARD
            - 800.0 * np.square(offset - correction_mps),
            0.0,
            MAX_TERMINAL_REWARD,
        ).astype(np.float32)
        storage_row = model.replay_buffer.pos
        model.replay_buffer.add(
            observations,
            observations.copy(),
            probe_actions,
            reward,
            np.ones(env_count, dtype=np.bool_),
            infos(batch_offsets),
        )
        rows_by_offset[float(offset)] = storage_row
    model.replay_buffer.finalize_local_probe_group(rows_by_offset)
    expected_probe_count = task_count * len(probe_offsets)
    if model.replay_buffer.local_probe_group_count != expected_probe_count:
        raise RuntimeError("Synthetic speed probes were not grouped in replay.")
    if model.replay_buffer.local_probe_group_anchor_count != task_count:
        raise RuntimeError("Synthetic speed probes lack one center anchor per task.")

    warmup = model.warmup_critic(1000, batch_size=env_count)
    diagnostics = critic_local_speed_diagnostics(
        model,
        _SinglePocketDataset(task_count),
        observations,
        certified,
        task_count=task_count,
        seed=3,
        batch_size=task_count,
        minimum_bc_error_mps=0.005,
    )
    environment.close()
    agreement = float(diagnostics["pairwise_both_critics_agreement"])
    if agreement < 0.75:
        raise RuntimeError(
            f"BC-centered critic ranking agreement is too low: {agreement:.3%}."
        )
    print(
        f"warmup_final={warmup['final_loss']:.6g} "
        f"probe_q_mae={diagnostics['probe_q_mae']:.6g} "
        f"pairwise_ranking_agreement={agreement:.3%} "
        "bc_centered_action=True grouped_probe_ranking=True"
    )


if __name__ == "__main__":
    main()
