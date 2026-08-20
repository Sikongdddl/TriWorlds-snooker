"""Fit and audit held-out Critic rankings on persisted real-physics curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.logger import configure

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_sac_her import (  # noqa: E402
    SingleStepCuePositionHerReplayBuffer,
    SingleStepTD3BC,
    critic_local_speed_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--critic-warmup-updates", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--ranking-weight", type=float, default=5.0)
    parser.add_argument("--ranking-margin", type=float, default=0.10)
    parser.add_argument("--delta-weight", type=float, default=5.0)
    parser.add_argument("--supervision-batch-size", type=int, default=256)
    parser.add_argument("--action-center-scale-mps", type=float, default=0.06)
    parser.add_argument("--candidate-min-q-improvement", type=float, default=0.10)
    parser.add_argument("--candidate-min-safe-q", type=float, default=1.5)
    parser.add_argument(
        "--candidate-max-critic-disagreement",
        type=float,
        default=0.25,
    )
    parser.add_argument("--minimum-ranking-agreement", type=float, default=0.75)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20_000)
    args = parser.parse_args()

    with np.load(args.curve, allow_pickle=False) as archive:
        curve = {name: np.asarray(archive[name]) for name in archive.files}
    offsets = curve["offsets_mps"].astype(np.float64)
    observations = curve["observation"].astype(np.float32)
    actions = curve["action"].astype(np.float32)
    rewards = curve["reward"].astype(np.float32)
    task_indices = curve["task_indices"].astype(np.int64)
    if observations.shape[:2] != (len(offsets), len(task_indices)):
        raise ValueError("Real curve observation dimensions are inconsistent.")
    if not np.allclose(observations, observations[:1], atol=0.0, rtol=0.0):
        raise ValueError("Real curve points changed their task observations.")

    model = SingleStepTD3BC.load(args.checkpoint, device=args.device)
    replay = model.replay_buffer
    if not isinstance(replay, SingleStepCuePositionHerReplayBuffer):
        raise TypeError("Checkpoint did not construct the ranking replay buffer.")
    if len(task_indices) % replay.n_envs != 0 or replay.size() != 0:
        raise ValueError("Real curve smoke requires an empty, slot-aligned replay.")
    model.set_logger(configure(format_strings=[]))
    model.critic_action_center_scale_mps = args.action_center_scale_mps
    model.critic_probe_delta_weight = args.delta_weight
    model.critic_probe_ranking_weight = args.ranking_weight
    model.critic_probe_ranking_margin = args.ranking_margin
    model.critic_probe_minimum_reward_difference = 0.05
    model.critic_supervision_batch_size = args.supervision_batch_size
    model.actor_candidate_min_q_improvement = args.candidate_min_q_improvement
    model.actor_candidate_min_safe_q = args.candidate_min_safe_q
    model.actor_candidate_max_critic_disagreement = (
        args.candidate_max_critic_disagreement
    )
    model.critic_probe_holdout_fraction = replay.probe_holdout_fraction
    model.critic_probe_holdout_seed = replay.probe_holdout_seed
    model.configure_conservative_speed_residual(
        max_speed_residual_mps=float(np.max(np.abs(offsets))),
        exploration_initial_std=0.35,
        exploration_final_std=0.05,
        exploration_decay_timesteps=65_536,
    )
    model.configure_discrete_candidate_ranking(tuple(offsets))

    dones = np.ones(replay.n_envs, dtype=np.bool_)
    for task_start in range(0, len(task_indices), replay.n_envs):
        task_slice = slice(task_start, task_start + replay.n_envs)
        rows_by_offset: dict[float, int] = {}
        for offset_index, offset in enumerate(offsets):
            infos: list[dict[str, object]] = []
            for relative_env, task_index in enumerate(task_indices[task_slice]):
                task_row = task_start + relative_env
                infos.append(
                    {
                        "correct_pot": bool(
                            curve["correct_pot"][offset_index, task_row]
                        ),
                        "cue_scratch": bool(
                            curve["cue_scratch"][offset_index, task_row]
                        ),
                        "wrong_pocket": bool(
                            curve["wrong_pocket"][offset_index, task_row]
                        ),
                        "stopped": bool(
                            curve["stopped"][offset_index, task_row]
                        ),
                        "timed_out": bool(
                            curve["timed_out"][offset_index, task_row]
                        ),
                        "numerical_failure": bool(
                            curve["numerical_failure"][offset_index, task_row]
                        ),
                        "joint_success": bool(
                            curve["joint_success"][offset_index, task_row]
                        ),
                        "cue_ball_final_position": curve["cue_final"][
                            offset_index,
                            task_row,
                        ],
                        "local_speed_probe": True,
                        "local_speed_probe_offset_mps": float(offset),
                        "local_speed_probe_task_index": int(task_index),
                        "TimeLimit.truncated": False,
                    }
                )
            storage_row = replay.pos
            replay.add(
                observations[offset_index, task_slice],
                observations[offset_index, task_slice].copy(),
                actions[offset_index, task_slice],
                rewards[offset_index, task_slice],
                dones,
                infos,
            )
            rows_by_offset[float(offset)] = storage_row
        replay.finalize_local_probe_group(rows_by_offset)

    warmup = model.warmup_critic(
        args.critic_warmup_updates,
        batch_size=args.batch_size,
    )
    diagnostics = critic_local_speed_diagnostics(
        model,
        None,
        observations[0],
        curve["generated_actions"],
        task_count=len(task_indices),
        seed=args.seed,
        minimum_physical_reward_difference=0.05,
        batch_size=args.batch_size,
    )
    print("critic_warmup=" + json.dumps(warmup, sort_keys=True), flush=True)
    print(
        "real_curve_ranking=" + json.dumps(diagnostics, sort_keys=True),
        flush=True,
    )
    agreement = float(diagnostics["pairwise_both_critics_agreement"])
    if agreement < args.minimum_ranking_agreement:
        raise RuntimeError(
            "Held-out real-curve ranking agreement is too low: "
            f"{agreement:.3%}."
        )


if __name__ == "__main__":
    main()
