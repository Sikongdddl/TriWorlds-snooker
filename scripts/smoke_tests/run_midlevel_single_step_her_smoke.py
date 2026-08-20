"""Validate successful-pot-only cue-position hindsight replay."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_sac_her import (  # noqa: E402
    HINDSIGHT_SUCCESS_REWARD,
    SingleStepCuePositionHerReplayBuffer,
)


def main() -> None:
    observation_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    replay = SingleStepCuePositionHerReplayBuffer(
        16,
        observation_space,
        action_space,
        device="cpu",
        n_envs=2,
        her_ratio=1.0,
        success_ratio=0.0,
        failure_ratio=0.0,
        probe_holdout_fraction=0.0,
    )
    observations = np.array(
        [
            [-0.2, 0.0, 0.4, 0.0, 0.9, 0.0, -0.5, 0.1],
            [0.1, 0.2, -0.3, 0.1, -0.9, 0.0, 0.4, -0.2],
        ],
        dtype=np.float32,
    )
    actions = np.array([[0.0, -0.2], [0.1, 0.3]], dtype=np.float32)
    replay.add(
        observations,
        observations.copy(),
        actions,
        np.array([1.2, 0.4], dtype=np.float32),
        np.ones(2, dtype=np.bool_),
        [
            {
                "correct_pot": True,
                "cue_scratch": False,
                "stopped": True,
                "cue_ball_final_position": np.array([0.30, -0.70, 1.0785]),
                "local_speed_probe": True,
                "local_speed_probe_offset_mps": -0.02,
                "local_speed_probe_task_index": 0,
            },
            {
                "correct_pot": False,
                "cue_scratch": False,
                "stopped": True,
                "cue_ball_final_position": np.array([0.0, 0.0, 1.0785]),
            },
        ],
    )
    replay.add(
        observations,
        observations.copy(),
        actions,
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.bool_),
        [
            {
                "correct_pot": True,
                "cue_scratch": True,
                "stopped": False,
                "cue_ball_final_position": np.array([0.0, 0.0, 1.0785]),
            },
            {
                "correct_pot": True,
                "cue_scratch": False,
                "stopped": False,
                "cue_ball_final_position": np.array([0.0, 0.0, 1.0785]),
            },
        ],
    )
    replay.add(
        observations,
        observations.copy(),
        actions,
        np.ones(2, dtype=np.float32),
        np.zeros(2, dtype=np.bool_),
        [
            {
                "correct_pot": True,
                "cue_scratch": False,
                "stopped": True,
                "cue_ball_final_position": np.array([0.1, 0.1, 1.0785]),
            },
            {
                "correct_pot": True,
                "cue_scratch": False,
                "stopped": True,
                "cue_ball_final_position": np.array([-0.1, -0.1, 1.0785]),
            },
        ],
    )

    if replay.eligible_transition_count != 1:
        raise RuntimeError(
            "HER admitted a miss, scratch, non-stopped, or non-terminal transition."
        )
    if replay.successful_transition_count != 1:
        raise RuntimeError("Replay did not classify the legal success.")
    if replay.failure_transition_count != 1:
        raise RuntimeError("Replay did not classify the scratch as a failure.")
    if replay.local_probe_transition_count != 1:
        raise RuntimeError("Replay did not classify the real local speed probe.")
    samples = replay.sample(32)
    sampled_observations = samples.observations.cpu().numpy()
    sampled_rewards = samples.rewards.cpu().numpy().reshape(-1)
    expected_stop_goal = np.array([0.4, -0.5], dtype=np.float32)
    if not np.allclose(sampled_observations[:, 6:8], expected_stop_goal):
        raise RuntimeError("HER did not relabel the cue-stop coordinates.")
    if not np.allclose(sampled_observations[:, 4:6], observations[0, 4:6]):
        raise RuntimeError("HER changed the requested target pocket.")
    if not np.allclose(sampled_rewards, HINDSIGHT_SUCCESS_REWARD):
        raise RuntimeError("HER virtual rewards are not exact terminal successes.")
    if not np.allclose(samples.dones.cpu().numpy(), 1.0):
        raise RuntimeError("HER changed one-step terminal semantics.")

    replay.her_ratio = 0.25
    replay.success_ratio = 0.25
    replay.failure_ratio = 0.25
    replay.local_probe_ratio = 0.25
    replay.sample(32)
    if replay.last_sample_composition != {
        "uniform": 0,
        "success": 8,
        "failure": 8,
        "local_probe": 8,
        "hindsight": 8,
    }:
        raise RuntimeError("Replay did not enforce the configured strata.")

    split_replay = SingleStepCuePositionHerReplayBuffer(
        8,
        observation_space,
        action_space,
        device="cpu",
        n_envs=2,
        her_ratio=0.0,
        success_ratio=0.0,
        failure_ratio=0.0,
        probe_holdout_fraction=0.20,
        probe_holdout_seed=20_000,
    )
    candidate_indices = np.arange(10_000, dtype=np.int64)
    holdout_flags = split_replay.probe_holdout_mask(candidate_indices)
    held_out_task = int(candidate_indices[np.flatnonzero(holdout_flags)[0]])
    training_task = int(candidate_indices[np.flatnonzero(~holdout_flags)[0]])
    split_observations = np.zeros((2, 8), dtype=np.float32)
    split_observations[:, 0] = (0.75, -0.75)
    split_replay.add(
        split_observations,
        split_observations.copy(),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.bool_),
        [
            {"task_index": held_out_task},
            {"task_index": training_task},
        ],
    )
    if split_replay.critic_holdout_transition[0].tolist() != [True, False]:
        raise RuntimeError("Task-level Critic holdout was not applied at insertion.")
    split_samples = split_replay.sample(32).observations.cpu().numpy()
    if not np.allclose(split_samples[:, 0], -0.75):
        raise RuntimeError("Critic holdout tasks leaked into uniform replay samples.")

    print(
        f"eligible={replay.eligible_transition_count} "
        f"local_probes={replay.local_probe_transition_count} "
        f"sampled={len(sampled_rewards)} reward={sampled_rewards[0]:.1f} "
        f"cue_goal={expected_stop_goal.tolist()} pocket_preserved=True "
        "task_holdout_isolated=True"
    )


if __name__ == "__main__":
    main()
