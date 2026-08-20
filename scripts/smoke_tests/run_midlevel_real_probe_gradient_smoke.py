"""Validate held-out critic rankings on real fixed-library speed probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3.common.vec_env import VecMonitor

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_mujoco_warp_vec_env import MJWarpMidLevelVecEnv  # noqa: E402
from snooker_env.midlevel_ppo import generated_behavior_cloning_data  # noqa: E402
from snooker_env.midlevel_sac_her import (  # noqa: E402
    SingleStepTD3BC,
    collect_local_speed_probes,
    critic_local_speed_diagnostics,
    prefill_certified_replay_buffer,
    replay_buffer_path,
)
from snooker_env.midlevel_tasks import TwoBallTaskDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--probe-task-count", type=int, default=4096)
    parser.add_argument("--critic-warmup-updates", type=int, default=4096)
    parser.add_argument("--replay-buffer", type=Path)
    parser.add_argument("--critic-probe-delta-weight", type=float, default=1.0)
    parser.add_argument("--critic-probe-ranking-weight", type=float, default=0.0)
    parser.add_argument(
        "--critic-probe-holdout-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument("--chunk-steps", type=int, default=64)
    parser.add_argument("--check-interval-steps", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = TwoBallTaskDataset.load(args.tasks, validate_model=False)
    reference_observations, reference_actions = generated_behavior_cloning_data(
        dataset
    )
    environment = None
    try:
        if args.replay_buffer is None:
            base_env = MJWarpMidLevelVecEnv(
                dataset,
                num_envs=args.num_envs,
                seed=args.seed,
                device=args.device,
                chunk_steps=args.chunk_steps,
                check_interval_steps=args.check_interval_steps,
            )
            environment = VecMonitor(base_env)
            model = SingleStepTD3BC.load(
                args.checkpoint,
                env=environment,
                device=args.device,
            )
            # The input checkpoint may predate this smoke's action-coordinate
            # addition while containing the exact same BC actor.
            model.critic_action_center_scale_mps = 0.03
            model.critic_probe_holdout_fraction = (
                args.critic_probe_holdout_fraction
            )
            model.critic_probe_holdout_seed = args.seed + 20_000
            model.configure_conservative_speed_residual(
                max_speed_residual_mps=0.03,
                exploration_initial_std=0.35,
                exploration_final_std=0.05,
                exploration_decay_timesteps=65_536,
            )
            model.configure_discrete_candidate_ranking(
                (-0.03, -0.01, 0.0, 0.01, 0.03)
            )
            model.configure_behavior_cloning_reference(
                reference_observations,
                reference_actions,
                initial_weight=1.0,
                final_weight=1.0,
                decay_actor_updates=1,
                batch_size=1024,
                angle_weight=1.0,
                speed_weight=8.0,
            )
            prefill_report = prefill_certified_replay_buffer(
                model,
                dataset,
                reference_observations,
                reference_actions,
            )
            probe_report = collect_local_speed_probes(
                model,
                environment,
                dataset,
                reference_observations,
                reference_actions,
                task_count=args.probe_task_count,
                offsets_mps=(-0.03, -0.01, 0.0, 0.01, 0.03),
                seed=args.seed + 10_000,
            )
        else:
            model = SingleStepTD3BC.load(
                args.checkpoint,
                device=args.device,
            )
            model.load_replay_buffer(args.replay_buffer)
            if not model.residual_policy_enabled:
                model.critic_action_center_scale_mps = 0.03
                model.configure_conservative_speed_residual(
                    max_speed_residual_mps=0.03,
                    exploration_initial_std=0.35,
                    exploration_final_std=0.05,
                    exploration_decay_timesteps=65_536,
                )
                model.configure_discrete_candidate_ranking(
                    (-0.03, -0.01, 0.0, 0.01, 0.03)
                )
                model.configure_behavior_cloning_reference(
                    reference_observations,
                    reference_actions,
                    initial_weight=1.0,
                    final_weight=1.0,
                    decay_actor_updates=1,
                    batch_size=1024,
                    angle_weight=1.0,
                    speed_weight=8.0,
                )
            if not model.candidate_ranking_enabled:
                model.configure_discrete_candidate_ranking(
                    (-0.03, -0.01, 0.0, 0.01, 0.03)
                )
            prefill_report = getattr(
                model,
                "midlevel_certified_prefill_report",
                {"reused_replay": True},
            )
            probe_report = getattr(
                model,
                "midlevel_local_probe_report",
                {
                    "probe_center": "frozen_bc_action",
                    "reused_replay": True,
                },
            )
        model.critic_probe_delta_weight = args.critic_probe_delta_weight
        model.critic_probe_ranking_weight = args.critic_probe_ranking_weight
        model.actor_candidate_min_q_improvement = 0.10
        model.actor_candidate_min_safe_q = 1.5
        model.actor_candidate_max_critic_disagreement = 0.25
        model.critic_probe_holdout_fraction = (
            args.critic_probe_holdout_fraction
        )
        model.critic_probe_holdout_seed = args.seed + 20_000
        warmup_report = model.warmup_critic(
            args.critic_warmup_updates,
            batch_size=1024,
        )
        diagnostics = critic_local_speed_diagnostics(
            model,
            dataset,
            reference_observations,
            reference_actions,
            task_count=args.probe_task_count,
            seed=args.seed + 10_000,
        )
        model.midlevel_certified_prefill_report = prefill_report
        model.midlevel_local_probe_report = probe_report
        model.midlevel_critic_warmup_report = warmup_report
        model.midlevel_critic_speed_diagnostics = diagnostics
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.output)
        model.save_replay_buffer(replay_buffer_path(args.output))
    finally:
        if environment is not None:
            environment.close()

    agreement = float(diagnostics["pairwise_both_critics_agreement"])
    print(
        "local_speed_probes="
        + json.dumps(probe_report, sort_keys=True),
        flush=True,
    )
    print(
        "critic_warmup="
        + json.dumps(warmup_report, sort_keys=True),
        flush=True,
    )
    print(
        "critic_local_speed_diagnostics="
        + json.dumps(diagnostics, sort_keys=True),
        flush=True,
    )
    if agreement < 0.75:
        raise RuntimeError(
            f"Real-probe critic ranking agreement is too low: {agreement:.3%}."
        )


if __name__ == "__main__":
    main()
