"""Train deterministic single-step TD3+BC with cue-position HER."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
)

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.midlevel_offline_curves import (  # noqa: E402
    OFFLINE_CURVE_TRAINING_VERSION,
    OfflineSpeedCurveDataset,
)
from snooker_env.midlevel_mujoco_warp_vec_env import (  # noqa: E402
    MUJOCO_WARP_SHOT_EXECUTION_VERSION,
    MJWarpMidLevelVecEnv,
)
from snooker_env.midlevel_ppo import (  # noqa: E402
    MIDLEVEL_TRAINING_MANIFEST_VERSION,
    generated_behavior_cloning_data,
    require_checkpoint_manifest,
)
from snooker_env.midlevel_ppo_env import (  # noqa: E402
    CUE_POSITION_REWARD_WEIGHT,
    CUE_POSITION_REWARD_DISTANCE_SCALE,
    JOINT_SUCCESS_REWARD_BONUS,
    MAX_TERMINAL_REWARD,
    MIDLEVEL_REWARD_VERSION,
    OBJECT_BALL_REWARD_WEIGHT,
    OBJECT_POCKET_REWARD_DISTANCE_SCALE,
    MidLevelTwoBallEnv,
)
from snooker_env.midlevel_sac_her import (  # noqa: E402
    ConservativeResidualTD3Policy,
    MIDLEVEL_GEOMETRIC_FEATURE_DIM,
    MIDLEVEL_GEOMETRIC_FEATURE_VERSION,
    MidLevelGeometricFeatures,
    SINGLE_STEP_HER_VERSION,
    SINGLE_STEP_TD3_VERSION,
    PostUpdateTrainingHook,
    SingleStepTD3BC,
    SingleStepCuePositionHerReplayBuffer,
    behavior_clone_td3_policy,
    behavior_clone_td3_policy_with_offline_curves,
    collect_local_speed_probes,
    critic_local_speed_diagnostics,
    prefill_certified_replay_buffer,
    replay_buffer_path,
    resolve_replay_buffer_path,
    td3_behavior_cloning_metrics,
)
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
        env = MidLevelTwoBallEnv(dataset, model, max_time=max_time)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return initialize


class OfflineMidLevelSpaceEnv(gym.Env):
    """Space-only environment for Actor/Critic training without rollouts."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=(8,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
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
        del options
        super().reset(seed=seed)
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        del action
        raise RuntimeError("Offline-only training must not execute environment steps.")


def build_environments(args: argparse.Namespace) -> VecEnv:
    """Build a vector backend with one terminal transition per world step."""

    if args.offline_only or args.bc_only or args.residual_only:
        print(
            f"offline_vector_environment=space_only num_envs={args.num_envs}",
            flush=True,
        )
        return DummyVecEnv(
            [OfflineMidLevelSpaceEnv for _ in range(args.num_envs)]
        )
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


class MidLevelTD3DiagnosticsCallback(BaseCallback):
    """Record terminal outcome rates for the post-update log row."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        if infos:
            for key in (
                "correct_pot",
                "joint_success",
                "cue_scratch",
                "wrong_pocket",
                "timed_out",
                "numerical_failure",
            ):
                values = [float(bool(info.get(key, False))) for info in infos]
                self.logger.record(
                    f"rollout/{key}_rate",
                    float(sum(values) / len(values)),
                )
        return True


def validate_loaded_replay_buffer(
    policy: SingleStepTD3BC,
    args: argparse.Namespace,
) -> None:
    """Reject a mismatched replay companion before a TD3+BC resume."""

    replay = policy.replay_buffer
    if not isinstance(replay, SingleStepCuePositionHerReplayBuffer):
        raise TypeError(
            "Resume replay buffer is not the single-step cue-position HER buffer."
        )
    if getattr(replay, "her_version", None) != SINGLE_STEP_HER_VERSION:
        raise ValueError("Resume replay buffer has an incompatible HER version.")
    if replay.n_envs != args.num_envs:
        raise ValueError(
            f"Resume replay buffer has n_envs={replay.n_envs}, "
            f"expected {args.num_envs}."
        )
    expected_rows = max(args.buffer_size // args.num_envs, 1)
    if replay.buffer_size != expected_rows:
        raise ValueError(
            "Resume replay buffer capacity does not match "
            "--buffer-size and --num-envs."
        )
    for name, actual, expected in (
        ("HER", replay.her_ratio, args.her_ratio),
        ("success", replay.success_ratio, args.success_replay_ratio),
        ("failure", replay.failure_ratio, args.failure_replay_ratio),
        ("local probe", replay.local_probe_ratio, args.local_probe_replay_ratio),
        (
            "probe holdout",
            replay.probe_holdout_fraction,
            args.critic_probe_holdout_fraction,
        ),
    ):
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Resume replay buffer {name} ratio does not match its argument."
            )
    if replay.probe_holdout_seed != args.critic_probe_holdout_seed:
        raise ValueError("Resume replay buffer probe holdout seed does not match.")


def critic_actor_gate_report(
    diagnostics: dict[str, object],
    *,
    minimum_pairwise_agreement: float,
    minimum_candidate_count: int,
    minimum_improvement_precision: float,
    minimum_improvement_precision_lower_95: float,
    minimum_reward_improvement: float,
    minimum_safe_improvement: float,
    minimum_joint_success_improvement: float,
    maximum_failure_increase: float,
) -> dict[str, object]:
    """Require real held-out physical benefit before enabling actor updates."""

    requirements = {
        "pairwise_both_critics_agreement": (
            minimum_pairwise_agreement,
            "minimum",
        ),
        "candidate_nonzero_selection_count": (
            minimum_candidate_count,
            "minimum",
        ),
        "candidate_nonzero_true_improvement_precision": (
            minimum_improvement_precision,
            "minimum",
        ),
        "candidate_nonzero_true_improvement_precision_lower_95": (
            minimum_improvement_precision_lower_95,
            "minimum",
        ),
        "candidate_selected_physical_reward_improvement_mean": (
            minimum_reward_improvement,
            "minimum",
        ),
        "candidate_selected_physical_safe_improvement_mean": (
            minimum_safe_improvement,
            "minimum",
        ),
        "candidate_selected_physical_joint_success_improvement_mean": (
            minimum_joint_success_improvement,
            "minimum",
        ),
        "candidate_selected_physical_failure_increase_mean": (
            maximum_failure_increase,
            "maximum",
        ),
    }
    checks: dict[str, dict[str, object]] = {}
    for metric, (threshold, direction) in requirements.items():
        value = float(diagnostics[metric])
        finite = math.isfinite(value)
        passed = finite and (
            value >= threshold if direction == "minimum" else value <= threshold
        )
        checks[metric] = {
            "value": value,
            "threshold": threshold,
            "direction": direction,
            "passed": passed,
        }
    return {
        "version": "held-out-real-physics-candidate-gate-v1",
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
    }


def structured_critic_actor_gate_report(
    diagnostics: dict[str, object],
    *,
    minimum_pairwise_agreement: float,
    minimum_canonical_selection_rate: float,
    maximum_reward_mae: float,
    maximum_cue_endpoint_mae_m: float,
    maximum_safe_brier: float,
) -> dict[str, object]:
    """Gate online use on task-disjoint grouped physical-curve accuracy."""

    requirements = {
        "pairwise_both_critics_agreement": (
            minimum_pairwise_agreement,
            "minimum",
        ),
        "canonical_selection_rate": (
            minimum_canonical_selection_rate,
            "minimum",
        ),
        "reward_pessimistic_mae": (maximum_reward_mae, "maximum"),
        "cue_endpoint_mae_m_on_safe": (
            maximum_cue_endpoint_mae_m,
            "maximum",
        ),
        "safe_brier": (maximum_safe_brier, "maximum"),
    }
    checks: dict[str, dict[str, object]] = {}
    for metric, (threshold, direction) in requirements.items():
        value = float(diagnostics[metric])
        finite = math.isfinite(value)
        passed = finite and (
            value >= threshold if direction == "minimum" else value <= threshold
        )
        checks[metric] = {
            "value": value,
            "threshold": threshold,
            "direction": direction,
            "passed": passed,
        }
    return {
        "version": "task-disjoint-structured-curve-gate-v1",
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
    }


def build_training_manifest(
    args: argparse.Namespace,
    dataset: TwoBallTaskDataset,
    offline_curves: OfflineSpeedCurveDataset | None,
) -> dict[str, object]:
    """Describe every input that changes TD3+BC rollout or HER semantics."""

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
        "reward": {
            "version": MIDLEVEL_REWARD_VERSION,
            "object_pocket_distance_scale": OBJECT_POCKET_REWARD_DISTANCE_SCALE,
            "cue_position_distance_scale": CUE_POSITION_REWARD_DISTANCE_SCALE,
            "object_ball_weight": OBJECT_BALL_REWARD_WEIGHT,
            "cue_position_weight": CUE_POSITION_REWARD_WEIGHT,
            "joint_success_bonus": JOINT_SUCCESS_REWARD_BONUS,
            "maximum_terminal_reward": MAX_TERMINAL_REWARD,
            "scratch_reward": 0.0,
        },
        "algorithm": {
            "name": "SingleStepTD3BC",
            "version": SINGLE_STEP_TD3_VERSION,
            "policy": "ConservativeResidualTD3Policy",
            "net_arch": [512, 512, 256],
            "features": {
                "version": MIDLEVEL_GEOMETRIC_FEATURE_VERSION,
                "dimension": MIDLEVEL_GEOMETRIC_FEATURE_DIM,
            },
            "actor_learning_rate": args.actor_learning_rate,
            "critic_learning_rate": args.critic_learning_rate,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "learning_starts": args.learning_starts,
            "train_freq_vector_steps": 1,
            "gradient_steps_per_rollout": args.gradient_steps,
            "critic_warmup_updates": args.critic_warmup_updates,
            "critic_probe_delta_weight": args.critic_probe_delta_weight,
            "critic_probe_ranking_weight": args.critic_probe_ranking_weight,
            "critic_probe_ranking_margin": args.critic_probe_ranking_margin,
            "critic_probe_minimum_reward_difference": (
                args.critic_probe_minimum_reward_difference
            ),
            "critic_supervision_batch_size": (
                args.critic_supervision_batch_size
            ),
            "critic_probe_holdout_fraction": (
                args.critic_probe_holdout_fraction
            ),
            "critic_probe_holdout_seed": args.critic_probe_holdout_seed,
            "critic_action_parameterization": (
                "absolute_normalized_speed_and_bc_centered_physical_offset"
            ),
            "critic_action_center_scale_mps": (
                args.critic_action_center_scale_mps
            ),
            "critic_minimum_pairwise_ranking_agreement": (
                args.critic_min_pairwise_ranking_agreement
            ),
            "critic_actor_gate": {
                "minimum_candidate_count": (
                    args.critic_min_candidate_selection_count
                ),
                "minimum_improvement_precision": (
                    args.critic_min_candidate_improvement_precision
                ),
                "minimum_improvement_precision_lower_95": (
                    args.critic_min_candidate_improvement_precision_lower_95
                ),
                "minimum_reward_improvement": (
                    args.critic_min_candidate_reward_improvement
                ),
                "minimum_safe_improvement": (
                    args.critic_min_candidate_safe_improvement
                ),
                "minimum_joint_success_improvement": (
                    args.critic_min_candidate_joint_success_improvement
                ),
                "maximum_failure_increase": (
                    args.critic_max_candidate_failure_increase
                ),
            },
            "actor_update_interval": args.actor_update_interval,
            "actor_learning_starts": args.actor_learning_starts,
            "actor_update_objective": (
                "offline_exact_action_and_her_residual_supervision"
            ),
            "actor_candidate_offsets_mps": list(
                args.actor_candidate_offsets_mps
            ),
            "actor_candidate_supervision_weight": (
                args.actor_candidate_supervision_weight
            ),
            "actor_physical_probe_supervision_weight": (
                args.actor_physical_probe_supervision_weight
            ),
            "actor_candidate_min_q_improvement": (
                args.actor_candidate_min_q_improvement
            ),
            "actor_candidate_min_safe_q": args.actor_candidate_min_safe_q,
            "actor_candidate_max_critic_disagreement": (
                args.actor_candidate_max_critic_disagreement
            ),
            "residual_l2_weight": args.residual_l2_weight,
            "fixed_ghost_ball_angle": True,
            "max_speed_residual_mps": args.max_speed_residual_mps,
            "residual_exploration_initial_std": (
                args.residual_exploration_initial_std
            ),
            "residual_exploration_final_std": (
                args.residual_exploration_final_std
            ),
            "residual_exploration_decay_timesteps": (
                args.residual_exploration_decay_timesteps
            ),
            "gamma": 0.0,
            "terminal_critic_target": "immediate_reward",
            "critic_count": 2,
            "actor_q_objective": None,
            "critic_usage": "structured_double_critic_safety_gate",
            "deterministic_actor": True,
            "certified_training_rollout_baseline": (
                args.certified_rollout_baseline
            ),
            "stop_after_residual_actor": args.residual_only,
            "target_network_updates": False,
            "seed": args.seed,
        },
        "hindsight_replay": {
            "version": SINGLE_STEP_HER_VERSION,
            "ratio": args.her_ratio,
            "success_ratio": args.success_replay_ratio,
            "failure_ratio": args.failure_replay_ratio,
            "local_probe_ratio": args.local_probe_replay_ratio,
            "uniform_ratio": 1.0
            - args.her_ratio
            - args.success_replay_ratio
            - args.failure_replay_ratio
            - args.local_probe_replay_ratio,
            "eligible_outcome": "correct_target_pot_no_scratch_stopped",
            "relabelled_goal": "cue_stop_position_only",
            "target_pocket_relabelled": False,
            "certified_task_prefill": not args.residual_only,
            "certified_prefill_reward": MAX_TERMINAL_REWARD,
        },
        "local_speed_probes": {
            "task_count": (
                0 if offline_curves is not None else args.local_probe_task_count
            ),
            "offsets_mps": list(args.local_probe_offsets_mps),
            "center": "frozen_bc_action",
            "selection": "balanced_complete_execution_batches",
            "world_slot_aligned": True,
            "offset_execution": "serial_same_world_slot",
            "seed": args.seed + args.local_probe_seed_offset,
            "real_physics_rewards": True,
            "collected_before_critic_warmup": True,
        },
        "offline_speed_curves": (
            {
                **offline_curves.report(),
                "training_version": OFFLINE_CURVE_TRAINING_VERSION,
                "task_holdout_fraction": args.offline_task_holdout_fraction,
                "task_holdout_seed": args.offline_task_holdout_seed,
                "actor_hindsight_fraction": args.bc_hindsight_fraction,
                "actor_physical_loss_weight": args.bc_physical_loss_weight,
                "actor_physical_distance_scale_m": (
                    args.bc_physical_distance_scale_m
                ),
                "actor_sensitivity_minimum": args.bc_sensitivity_minimum,
                "actor_sensitivity_maximum": args.bc_sensitivity_maximum,
                "residual_warmup_updates": (
                    args.offline_residual_warmup_updates
                ),
                "residual_warmup_learning_rate": (
                    args.offline_residual_learning_rate
                ),
                "residual_supervision_weight": (
                    args.offline_actor_supervision_weight
                ),
                "residual_margin_loss_weight": (
                    args.offline_actor_margin_loss_weight
                ),
                "residual_success_margin_m": (
                    args.offline_actor_success_margin_m
                ),
                "residual_success_interval_loss_weight": (
                    args.offline_actor_success_interval_loss_weight
                ),
                "residual_success_interval_scale_mps": (
                    args.offline_actor_success_interval_scale_mps
                ),
                "residual_middle_pocket_weight": (
                    args.offline_actor_middle_pocket_weight
                ),
                "residual_hard_task_weight": (
                    args.offline_actor_hard_task_weight
                ),
                "residual_hard_task_quantile": (
                    args.offline_actor_hard_task_quantile
                ),
                "residual_hard_task_metric": (
                    args.offline_actor_hard_task_metric
                ),
                "safe_candidate_classifier": {
                    "enabled": args.offline_safe_candidate_classifier,
                    "offsets_mps": list(args.safe_candidate_offsets_mps),
                    "updates": args.safe_candidate_updates,
                    "learning_rate": args.safe_candidate_learning_rate,
                    "batch_size": args.safe_candidate_batch_size,
                    "ensemble_size": args.safe_candidate_ensemble_size,
                    "pocket_head_count": args.safe_candidate_pocket_heads,
                    "hidden_sizes": list(args.safe_candidate_hidden_sizes),
                    "positive_weight": args.safe_candidate_positive_weight,
                    "selection_loss_weight": (
                        args.safe_candidate_selection_loss_weight
                    ),
                    "selection_target": args.safe_candidate_selection_target,
                    "unknown_weight": args.safe_candidate_unknown_weight,
                    "label_tolerance_mps": (
                        args.safe_candidate_label_tolerance_mps
                    ),
                    "minimum_probability": (
                        args.safe_candidate_min_probability
                    ),
                    "minimum_improvement": (
                        args.safe_candidate_min_improvement
                    ),
                    "maximum_disagreement": (
                        args.safe_candidate_max_disagreement
                    ),
                    "uncertainty_scale": (
                        args.safe_candidate_uncertainty_scale
                    ),
                    "residual_penalty": args.safe_candidate_residual_penalty,
                    "weight_decay": args.safe_candidate_weight_decay,
                },
                "effective_residual_gate": {
                    "maximum_holdout_speed_p95_mps": float(
                        np.max(np.abs(offline_curves.offsets_mps))
                    ),
                    "maximum_holdout_stop_p90_m": (
                        args.bc_max_offline_stop_p90_m
                    ),
                    "maximum_validation_speed_mae_mps": (
                        args.bc_max_validation_speed_mae_mps
                    ),
                    "maximum_validation_speed_p95_mps": min(
                        args.bc_max_validation_speed_p95_mps,
                        float(np.max(np.abs(offline_curves.offsets_mps))),
                    ),
                },
                "structured_critic": {
                    "learning_rate": args.structured_critic_learning_rate,
                    "warmup_updates": args.structured_critic_warmup_updates,
                    "task_batch_size": args.structured_critic_task_batch_size,
                    "reward_weight": args.structured_reward_weight,
                    "reward_delta_weight": (
                        args.structured_reward_delta_weight
                    ),
                    "ranking_weight": args.structured_ranking_weight,
                    "ranking_temperature": (
                        args.structured_ranking_temperature
                    ),
                    "cue_delta_weight": args.structured_cue_delta_weight,
                    "event_weight": args.structured_event_weight,
                },
            }
            if offline_curves is not None
            else None
        ),
        "behavior_cloning": {
            "training_mode": args.bc_training_mode,
            "stop_after_behavior_cloning": args.bc_only,
            "initialize_from_bc": (
                str(args.initialize_from_bc)
                if args.initialize_from_bc is not None
                else None
            ),
            "epochs": args.bc_epochs,
            "batch_size": args.bc_batch_size,
            "learning_rate": args.bc_learning_rate,
            "final_learning_rate": args.bc_final_learning_rate,
            "angle_weight": args.bc_angle_weight,
            "speed_weight": args.bc_speed_weight,
            "max_grad_norm": args.bc_max_grad_norm,
            "validation_tasks": (
                str(args.bc_validation_tasks)
                if args.bc_validation_tasks is not None
                else None
            ),
            "maximum_validation_speed_mae_mps": (
                args.bc_max_validation_speed_mae_mps
            ),
            "maximum_validation_speed_p95_mps": (
                args.bc_max_validation_speed_p95_mps
            ),
            "maximum_offline_stop_p90_m": (
                args.bc_max_offline_stop_p90_m
            ),
            "online_regularization_initial_weight": (
                args.bc_regularization_initial_weight
            ),
            "online_regularization_final_weight": (
                args.bc_regularization_final_weight
            ),
            "online_regularization_decay_actor_updates": (
                args.bc_regularization_decay_actor_updates
            ),
            "online_regularization_batch_size": (
                args.bc_regularization_batch_size
            ),
            "online_regularization_residual_weight": (
                args.bc_regularization_residual_weight
            ),
            "bc_only_checkpoint": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "outputs" / "tasks" / "midlevel_two_ball_train.npz",
    )
    parser.add_argument(
        "--offline-speed-curves",
        type=Path,
        default=ROOT
        / "outputs"
        / "diagnostics"
        / "midlevel_speed_perturbations_196608.npz",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Stop after offline Actor/Critic training and held-out audit.",
    )
    parser.add_argument(
        "--bc-only",
        action="store_true",
        help=(
            "Stop immediately after saving the behavior-cloned Actor; do not "
            "configure residual learning, train Critics, or run physics."
        ),
    )
    parser.add_argument(
        "--residual-only",
        action="store_true",
        help=(
            "Stop after fitting and auditing the frozen-BC speed residual; "
            "do not train Critics or execute online physics rollouts."
        ),
    )
    parser.add_argument(
        "--bc-training-mode",
        choices=("canonical", "offline-curves"),
        default="offline-curves",
        help=(
            "Fit only exact generated actions ('canonical') or mix canonical, "
            "legal hindsight, and measured curve losses ('offline-curves')."
        ),
    )
    parser.add_argument(
        "--certified-rollout-baseline",
        action="store_true",
        help=(
            "Center stochastic online exploration on each task's certified "
            "action; deterministic inference remains Actor-only."
        ),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--total-timesteps", type=int, default=65_536)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument(
        "--backend",
        choices=("mujoco-warp", "cpu"),
        default="mujoco-warp",
    )
    parser.add_argument("--buffer-size", type=int, default=327_680)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gradient-steps", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=0)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-warmup-updates", type=int, default=4096)
    parser.add_argument(
        "--critic-probe-delta-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--critic-probe-ranking-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--critic-probe-ranking-margin",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--critic-probe-minimum-reward-difference",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--critic-supervision-batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--critic-probe-holdout-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--critic-probe-holdout-seed",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "--offline-task-holdout-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--offline-task-holdout-seed",
        type=int,
        default=30_000,
    )
    parser.add_argument(
        "--structured-critic-learning-rate",
        type=float,
        default=3.0e-4,
    )
    parser.add_argument(
        "--structured-critic-warmup-updates",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--structured-critic-task-batch-size",
        type=int,
        default=256,
    )
    parser.add_argument("--structured-reward-weight", type=float, default=1.0)
    parser.add_argument(
        "--structured-reward-delta-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument("--structured-ranking-weight", type=float, default=1.0)
    parser.add_argument(
        "--structured-ranking-temperature",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--structured-cue-delta-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--structured-cue-delta-scale-m",
        type=float,
        default=0.10,
    )
    parser.add_argument("--structured-event-weight", type=float, default=1.0)
    parser.add_argument(
        "--structured-event-balance-clip",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--structured-gate-min-pairwise-agreement",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--structured-gate-min-canonical-selection-rate",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--structured-gate-max-reward-mae",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--structured-gate-max-cue-endpoint-mae-m",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--structured-gate-max-safe-brier",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--structured-actor-min-safe-probability",
        type=float,
        default=0.98,
    )
    parser.add_argument(
        "--structured-actor-max-scratch-probability",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--structured-actor-min-reward-improvement",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--structured-actor-max-reward-disagreement",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--critic-action-center-scale-mps",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--critic-min-pairwise-ranking-agreement",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--critic-min-candidate-selection-count",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--critic-min-candidate-improvement-precision",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--critic-min-candidate-improvement-precision-lower-95",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--critic-min-candidate-reward-improvement",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--critic-min-candidate-safe-improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--critic-min-candidate-joint-success-improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--critic-max-candidate-failure-increase",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--actor-candidate-supervision-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--actor-physical-probe-supervision-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--actor-candidate-min-q-improvement",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--actor-candidate-min-safe-q",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--actor-candidate-max-critic-disagreement",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--actor-candidate-offsets-mps",
        type=float,
        nargs="+",
        default=(-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03),
    )
    parser.add_argument("--actor-update-interval", type=int, default=16)
    parser.add_argument("--actor-learning-starts", type=int, default=0)
    parser.add_argument("--residual-l2-weight", type=float, default=0.02)
    parser.add_argument("--max-speed-residual-mps", type=float, default=0.12)
    parser.add_argument(
        "--residual-exploration-initial-std",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--residual-exploration-final-std",
        type=float,
        default=1.0 / 240.0,
    )
    parser.add_argument(
        "--residual-exploration-decay-timesteps",
        type=int,
        default=65_536,
    )
    parser.add_argument("--her-ratio", type=float, default=0.10)
    parser.add_argument("--success-replay-ratio", type=float, default=0.20)
    parser.add_argument("--failure-replay-ratio", type=float, default=0.20)
    parser.add_argument("--local-probe-replay-ratio", type=float, default=0.0)
    parser.add_argument("--local-probe-task-count", type=int, default=16_384)
    parser.add_argument(
        "--local-probe-offsets-mps",
        type=float,
        nargs="+",
        default=(-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03),
    )
    parser.add_argument("--local-probe-seed-offset", type=int, default=10_000)
    parser.add_argument("--bc-epochs", type=int, default=200)
    parser.add_argument("--bc-batch-size", type=int, default=2048)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bc-final-learning-rate", type=float, default=3e-5)
    parser.add_argument("--bc-angle-weight", type=float, default=1.0)
    parser.add_argument("--bc-speed-weight", type=float, default=8.0)
    parser.add_argument("--bc-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--bc-hindsight-fraction", type=float, default=0.50)
    parser.add_argument("--bc-physical-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--bc-physical-distance-scale-m",
        type=float,
        default=0.05,
    )
    parser.add_argument("--bc-sensitivity-minimum", type=float, default=0.25)
    parser.add_argument("--bc-sensitivity-maximum", type=float, default=4.0)
    parser.add_argument(
        "--bc-validation-tasks",
        type=Path,
        default=ROOT
        / "outputs"
        / "tasks"
        / "midlevel_two_ball_validation.npz",
    )
    parser.add_argument(
        "--bc-max-validation-speed-mae-mps",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--bc-max-validation-speed-p95-mps",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--bc-max-offline-stop-p90-m",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--bc-regularization-initial-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--bc-regularization-final-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--bc-regularization-decay-actor-updates",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--bc-regularization-batch-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--bc-regularization-residual-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--offline-residual-warmup-updates",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--offline-residual-learning-rate",
        type=float,
        default=3.0e-4,
    )
    parser.add_argument(
        "--offline-safe-candidate-classifier",
        action="store_true",
        help=(
            "Replace continuous residual regression with an ensemble that "
            "classifies a BC-relative set of measured-safe speed candidates."
        ),
    )
    parser.add_argument("--safe-candidate-step-mps", type=float, default=0.005)
    parser.add_argument("--safe-candidate-updates", type=int, default=4096)
    parser.add_argument(
        "--safe-candidate-learning-rate",
        type=float,
        default=3.0e-4,
    )
    parser.add_argument("--safe-candidate-batch-size", type=int, default=2048)
    parser.add_argument("--safe-candidate-ensemble-size", type=int, default=3)
    parser.add_argument(
        "--safe-candidate-pocket-heads",
        type=int,
        choices=(1, 6),
        default=6,
    )
    parser.add_argument(
        "--safe-candidate-hidden-sizes",
        type=int,
        nargs="+",
        default=(512, 512, 256),
    )
    parser.add_argument(
        "--safe-candidate-positive-weight",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--safe-candidate-selection-loss-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--safe-candidate-selection-target",
        choices=("set", "nearest"),
        default="set",
    )
    parser.add_argument(
        "--safe-candidate-unknown-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--safe-candidate-label-tolerance-mps",
        type=float,
        default=0.0025,
    )
    parser.add_argument(
        "--safe-candidate-min-probability",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--safe-candidate-min-improvement",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--safe-candidate-max-disagreement",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--safe-candidate-uncertainty-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--safe-candidate-residual-penalty",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--safe-candidate-weight-decay",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--offline-actor-supervision-weight",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--offline-actor-margin-loss-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--offline-actor-success-margin-m",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--offline-actor-success-interval-loss-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--offline-actor-success-interval-scale-mps",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--offline-actor-middle-pocket-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--offline-actor-hard-task-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--offline-actor-hard-task-quantile",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--offline-actor-hard-task-metric",
        choices=("canonical_speed_error", "success_interval_distance"),
        default="canonical_speed_error",
    )
    parser.add_argument(
        "--offline-actor-batch-size",
        type=int,
        default=2048,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--physics-device", default="cuda:0")
    parser.add_argument("--chunk-steps", type=int, default=16)
    parser.add_argument("--check-interval-steps", type=int, default=2048)
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=1024)
    parser.add_argument("--max-shot-time", type=float, default=8.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs"
        / "checkpoints"
        / "midlevel_two_ball_td3_her_v8",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-replay-buffer", type=Path)
    parser.add_argument(
        "--initialize-from-bc",
        type=Path,
        help=(
            "Initialize a fresh residual/critic run from a compatible BC-only "
            "checkpoint without repeating BC optimization."
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=32_768)
    args = parser.parse_args()

    if args.bc_only and args.residual_only:
        raise ValueError("--bc-only and --residual-only are mutually exclusive.")
    if args.offline_only and args.residual_only:
        raise ValueError("--offline-only and --residual-only are mutually exclusive.")
    if args.residual_only and args.resume is not None:
        raise ValueError(
            "--residual-only requires a fresh run, optionally initialized with "
            "--initialize-from-bc."
        )
    if args.bc_only and (
        args.resume is not None or args.initialize_from_bc is not None
    ):
        raise ValueError(
            "--bc-only trains a fresh Actor and cannot be combined with "
            "--resume or --initialize-from-bc."
        )
    if args.num_envs <= 0 or args.total_timesteps <= 0:
        raise ValueError("--num-envs and --total-timesteps must be positive.")
    if args.buffer_size < args.num_envs:
        raise ValueError("--buffer-size must hold at least one full vector rollout.")
    if args.batch_size <= 0 or args.gradient_steps <= 0:
        raise ValueError("--batch-size and --gradient-steps must be positive.")
    if args.learning_starts != 0:
        raise ValueError(
            "--learning-starts must be 0 so all rollout exploration remains "
            "inside the bounded BC residual."
        )
    if args.actor_learning_starts < 0:
        raise ValueError("--actor-learning-starts must be non-negative.")
    for name in (
        "her_ratio",
        "success_replay_ratio",
        "failure_replay_ratio",
        "local_probe_replay_ratio",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    if (
        args.her_ratio
        + args.success_replay_ratio
        + args.failure_replay_ratio
        + args.local_probe_replay_ratio
        > 1.0 + 1.0e-12
    ):
        raise ValueError("Replay sampling ratios must sum to at most one.")
    for name in (
        "actor_learning_rate",
        "critic_learning_rate",
        "bc_learning_rate",
        "bc_final_learning_rate",
        "bc_angle_weight",
        "bc_speed_weight",
        "bc_max_grad_norm",
        "bc_max_validation_speed_mae_mps",
        "bc_max_validation_speed_p95_mps",
        "bc_max_offline_stop_p90_m",
        "critic_action_center_scale_mps",
        "max_speed_residual_mps",
        "max_shot_time",
        "offline_actor_success_margin_m",
        "offline_actor_success_interval_scale_mps",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    for name in (
        "bc_physical_loss_weight",
        "offline_actor_margin_loss_weight",
        "offline_actor_success_interval_loss_weight",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative."
            )
    for name in (
        "offline_actor_middle_pocket_weight",
        "offline_actor_hard_task_weight",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 1.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and at least one."
            )
    if not 0.0 < args.offline_actor_hard_task_quantile < 1.0:
        raise ValueError(
            "--offline-actor-hard-task-quantile must be in (0, 1)."
        )
    if (
        not math.isfinite(args.actor_candidate_supervision_weight)
        or args.actor_candidate_supervision_weight < 0.0
    ):
        raise ValueError(
            "--actor-candidate-supervision-weight must be finite and non-negative."
        )
    if (
        not math.isfinite(args.actor_physical_probe_supervision_weight)
        or args.actor_physical_probe_supervision_weight < 0.0
    ):
        raise ValueError(
            "--actor-physical-probe-supervision-weight must be finite and "
            "non-negative."
        )
    if (
        not math.isfinite(args.residual_l2_weight)
        or args.residual_l2_weight < 0.0
    ):
        raise ValueError("--residual-l2-weight must be finite and non-negative.")
    if (
        not math.isfinite(args.residual_exploration_initial_std)
        or not 0.0 <= args.residual_exploration_initial_std < 1.0
    ):
        raise ValueError(
            "--residual-exploration-initial-std must be in [0, 1)."
        )
    if (
        not math.isfinite(args.residual_exploration_final_std)
        or not 0.0 <= args.residual_exploration_final_std < 1.0
        or args.residual_exploration_final_std
        > args.residual_exploration_initial_std
    ):
        raise ValueError(
            "--residual-exploration-final-std must be in [0, initial-std]."
        )
    if args.residual_exploration_decay_timesteps <= 0:
        raise ValueError(
            "--residual-exploration-decay-timesteps must be positive."
        )
    needs_offline_curves = not (
        args.bc_only and args.bc_training_mode == "canonical"
    )
    if needs_offline_curves and not args.offline_speed_curves.exists():
        raise FileNotFoundError(args.offline_speed_curves)
    if (
        args.initialize_from_bc is not None
        and not args.initialize_from_bc.exists()
    ):
        raise FileNotFoundError(args.initialize_from_bc)
    if (
        not 0.0 < args.offline_task_holdout_fraction < 0.5
        or args.offline_task_holdout_seed < 0
    ):
        raise ValueError("Offline task holdout settings are invalid.")
    if not 0.0 <= args.bc_hindsight_fraction <= 1.0:
        raise ValueError("--bc-hindsight-fraction must be in [0, 1].")
    if not 0.0 < args.bc_sensitivity_minimum <= 1.0 <= args.bc_sensitivity_maximum:
        raise ValueError("BC sensitivity bounds must bracket one.")
    for name in (
        "bc_physical_distance_scale_m",
        "offline_actor_supervision_weight",
        "offline_residual_learning_rate",
        "structured_critic_learning_rate",
        "structured_reward_weight",
        "structured_reward_delta_weight",
        "structured_ranking_weight",
        "structured_ranking_temperature",
        "structured_cue_delta_weight",
        "structured_cue_delta_scale_m",
        "structured_event_weight",
        "structured_event_balance_clip",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    if (
        args.offline_residual_warmup_updates <= 0
        or args.offline_actor_batch_size <= 0
        or args.structured_critic_warmup_updates <= 0
        or args.structured_critic_task_batch_size <= 0
    ):
        raise ValueError("Offline Actor/Critic update and batch counts must be positive.")
    for name in (
        "structured_gate_min_pairwise_agreement",
        "structured_gate_min_canonical_selection_rate",
        "structured_actor_min_safe_probability",
        "structured_actor_max_scratch_probability",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    for name in (
        "structured_gate_max_reward_mae",
        "structured_gate_max_cue_endpoint_mae_m",
        "structured_gate_max_safe_brier",
        "structured_actor_min_reward_improvement",
        "structured_actor_max_reward_disagreement",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative."
            )
    probe_offsets = np.asarray(args.local_probe_offsets_mps, dtype=np.float64)
    if args.local_probe_task_count <= 0 or len(probe_offsets) < 3:
        raise ValueError("Local speed probes require tasks and at least three offsets.")
    if (
        not np.all(np.isfinite(probe_offsets))
        or len(np.unique(probe_offsets)) != len(probe_offsets)
        or not np.all(np.diff(probe_offsets) > 0.0)
        or np.count_nonzero(probe_offsets == 0.0) != 1
        or not np.allclose(probe_offsets, -probe_offsets[::-1], atol=1.0e-12)
    ):
        raise ValueError(
            "--local-probe-offsets-mps must be sorted, finite, unique, "
            "symmetric, and include zero."
        )
    candidate_offsets = np.asarray(
        args.actor_candidate_offsets_mps,
        dtype=np.float64,
    )
    if (
        len(candidate_offsets) < 3
        or not np.all(np.isfinite(candidate_offsets))
        or len(np.unique(candidate_offsets)) != len(candidate_offsets)
        or not np.all(np.diff(candidate_offsets) > 0.0)
        or np.count_nonzero(candidate_offsets == 0.0) != 1
        or not np.allclose(
            candidate_offsets,
            -candidate_offsets[::-1],
            atol=1.0e-12,
        )
    ):
        raise ValueError(
            "--actor-candidate-offsets-mps must be sorted, finite, unique, "
            "symmetric, and include zero."
        )
    if args.local_probe_task_count % args.num_envs != 0:
        raise ValueError(
            "--local-probe-task-count must be a multiple of --num-envs so "
            "probes preserve complete batches and stable world slots."
        )
    if np.max(np.abs(probe_offsets)) > args.max_speed_residual_mps:
        raise ValueError("Local speed probe offsets exceed the residual bound.")
    if np.max(np.abs(candidate_offsets)) > args.max_speed_residual_mps:
        raise ValueError("Actor candidate offsets exceed the residual bound.")
    if args.offline_safe_candidate_classifier and not args.residual_only:
        raise ValueError(
            "The safe-candidate classifier is currently audited in "
            "--residual-only mode before online Critic training."
        )
    if (
        args.safe_candidate_updates <= 0
        or args.safe_candidate_batch_size <= 0
        or args.safe_candidate_ensemble_size <= 0
        or any(size <= 0 for size in args.safe_candidate_hidden_sizes)
    ):
        raise ValueError("Safe-candidate training sizes must be positive.")
    for name in (
        "safe_candidate_step_mps",
        "safe_candidate_learning_rate",
        "safe_candidate_positive_weight",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be positive and finite."
            )
    for name in (
        "safe_candidate_label_tolerance_mps",
        "safe_candidate_selection_loss_weight",
        "safe_candidate_min_probability",
        "safe_candidate_min_improvement",
        "safe_candidate_max_disagreement",
        "safe_candidate_uncertainty_scale",
        "safe_candidate_residual_penalty",
        "safe_candidate_weight_decay",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative."
            )
    if not 0.0 <= args.safe_candidate_unknown_weight <= 1.0:
        raise ValueError("--safe-candidate-unknown-weight must be in [0, 1].")
    if (
        args.safe_candidate_min_probability > 1.0
        or args.safe_candidate_min_improvement > 1.0
    ):
        raise ValueError("Safe-candidate probability gates cannot exceed one.")
    candidate_radius_steps = round(
        args.max_speed_residual_mps / args.safe_candidate_step_mps
    )
    if (
        candidate_radius_steps <= 0
        or not math.isclose(
            candidate_radius_steps * args.safe_candidate_step_mps,
            args.max_speed_residual_mps,
            abs_tol=1.0e-9,
        )
    ):
        raise ValueError(
            "--safe-candidate-step-mps must evenly divide the residual bound."
        )
    args.safe_candidate_offsets_mps = tuple(
        float(index * args.safe_candidate_step_mps)
        for index in range(-candidate_radius_steps, candidate_radius_steps + 1)
    )
    if not np.array_equal(
        np.round(probe_offsets, 9),
        np.round(candidate_offsets, 9),
    ):
        raise ValueError(
            "Local probe offsets and actor candidates must match exactly so "
            "every candidate has a held-out physical outcome."
        )
    if args.bc_epochs <= 0 or args.bc_batch_size <= 0:
        raise ValueError(
            "Offline curve training requires positive behavior-cloning epochs "
            "and batch size."
        )
    if args.bc_final_learning_rate > args.bc_learning_rate:
        raise ValueError("BC final learning rate cannot exceed its initial rate.")
    if args.critic_warmup_updates <= 0 or args.actor_update_interval <= 0:
        raise ValueError("Critic warmup and actor update interval must be positive.")
    if args.critic_supervision_batch_size <= 0:
        raise ValueError("Critic supervision batch size must be positive.")
    for name in (
        "critic_probe_ranking_weight",
        "critic_probe_delta_weight",
        "critic_probe_ranking_margin",
        "critic_probe_minimum_reward_difference",
        "actor_candidate_min_q_improvement",
        "actor_candidate_min_safe_q",
        "actor_candidate_max_critic_disagreement",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative."
            )
    if not 0.0 < args.critic_probe_holdout_fraction < 0.5:
        raise ValueError(
            "--critic-probe-holdout-fraction must be in (0, 0.5)."
        )
    if args.critic_probe_holdout_seed < 0:
        raise ValueError("--critic-probe-holdout-seed must be non-negative.")
    if not 0.0 <= args.critic_min_pairwise_ranking_agreement <= 1.0:
        raise ValueError(
            "--critic-min-pairwise-ranking-agreement must be in [0, 1]."
        )
    if args.critic_min_candidate_selection_count <= 0:
        raise ValueError(
            "--critic-min-candidate-selection-count must be positive."
        )
    for name in (
        "critic_min_candidate_improvement_precision",
        "critic_min_candidate_improvement_precision_lower_95",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    for name in (
        "critic_min_candidate_reward_improvement",
        "critic_min_candidate_safe_improvement",
        "critic_min_candidate_joint_success_improvement",
        "critic_max_candidate_failure_increase",
    ):
        value = getattr(args, name)
        if not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite.")
    if (
        args.bc_regularization_initial_weight < 0.0
        or args.bc_regularization_final_weight < 0.0
        or args.bc_regularization_final_weight
        > args.bc_regularization_initial_weight
    ):
        raise ValueError("Invalid BC regularization weights.")
    if (
        args.bc_regularization_decay_actor_updates <= 0
        or args.bc_regularization_batch_size <= 0
    ):
        raise ValueError("BC regularization decay and batch size must be positive.")
    if (
        not math.isfinite(args.bc_regularization_residual_weight)
        or args.bc_regularization_residual_weight <= 0.0
    ):
        raise ValueError(
            "--bc-regularization-residual-weight must be positive and finite."
        )
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive.")
    if not args.tasks.exists():
        raise FileNotFoundError(args.tasks)

    dataset = TwoBallTaskDataset.load(args.tasks, validate_model=False)
    if args.local_probe_task_count > len(dataset):
        raise ValueError("--local-probe-task-count exceeds the task library.")
    if args.local_probe_task_count > (len(dataset) // args.num_envs) * args.num_envs:
        raise ValueError(
            "--local-probe-task-count exceeds the complete execution batches "
            "in the task library."
        )
    replay_capacity_rows = max(args.buffer_size // args.num_envs, 1)
    certified_rows = (
        0
        if args.offline_only or args.bc_only or args.residual_only
        else math.ceil(len(dataset) / args.num_envs)
    )
    probe_rows = 0
    online_rows = (
        0
        if args.offline_only or args.bc_only or args.residual_only
        else math.ceil(args.total_timesteps / args.num_envs)
    )
    if certified_rows + probe_rows + online_rows > replay_capacity_rows:
        raise ValueError(
            "Replay buffer cannot retain certified, local-probe, and online "
            "transitions for this run."
        )
    required_task_backend = (
        MUJOCO_WARP_PHYSICS_BACKEND
        if args.backend == "mujoco-warp"
        else CPU_PHYSICS_BACKEND
    )
    if dataset.physics_backend != required_task_backend:
        raise ValueError(
            f"--backend {args.backend!r} requires {required_task_backend!r} tasks."
        )
    if dataset.execution_max_time != args.max_shot_time:
        raise ValueError(
            "--max-shot-time does not match the task dataset execution setting."
        )
    bc_validation_dataset: TwoBallTaskDataset | None = None
    if args.bc_validation_tasks is not None:
        if not args.bc_validation_tasks.exists():
            raise FileNotFoundError(args.bc_validation_tasks)
        bc_validation_dataset = TwoBallTaskDataset.load(
            args.bc_validation_tasks,
            validate_model=False,
        )
        if (
            bc_validation_dataset.xml_hash != dataset.xml_hash
            or bc_validation_dataset.model_hash != dataset.model_hash
            or bc_validation_dataset.physics_backend != dataset.physics_backend
            or bc_validation_dataset.backend_hash != dataset.backend_hash
            or bc_validation_dataset.execution_max_time
            != dataset.execution_max_time
        ):
            raise ValueError(
                "BC validation tasks do not match the training physics fingerprint."
            )
    reference_observations, reference_actions = generated_behavior_cloning_data(
        dataset
    )
    offline_curves: OfflineSpeedCurveDataset | None = None
    if needs_offline_curves:
        offline_curves = OfflineSpeedCurveDataset.load(
            args.offline_speed_curves,
            task_dataset=dataset,
            reference_observations=reference_observations,
            reference_actions=reference_actions,
        )
        print(
            "offline_speed_curves="
            + json.dumps(offline_curves.report(), sort_keys=True),
            flush=True,
        )
    else:
        print(
            "offline_speed_curves=SKIP reason=canonical_bc_only",
            flush=True,
        )

    manifest = build_training_manifest(args, dataset, offline_curves)
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
            policy = SingleStepTD3BC.load(
                str(args.resume),
                env=environments,
                device=args.device,
            )
            require_checkpoint_manifest(policy, manifest, context="Resume")
            buffer_input = args.resume_replay_buffer or resolve_replay_buffer_path(
                args.resume
            )
            if not buffer_input.exists():
                raise FileNotFoundError(
                    f"TD3+BC resume requires its replay buffer: {buffer_input}"
                )
            policy.load_replay_buffer(buffer_input)
            validate_loaded_replay_buffer(policy, args)
            policy.attach_offline_speed_curves_for_resume(
                offline_curves,
                seed=args.seed,
            )
            stored_gate = getattr(
                policy,
                "midlevel_critic_actor_gate_report",
                None,
            )
            if (
                not isinstance(stored_gate, dict)
                or not bool(stored_gate.get("passed", False))
            ) and not policy.certified_rollout_baseline_enabled:
                raise ValueError(
                    "Resume checkpoint has neither a reliable structured "
                    "Critic gate nor measured-safe rollout actions."
                )
            stored_residual_gate = getattr(
                policy,
                "midlevel_offline_residual_actor_gate_report",
                None,
            )
            if (
                not isinstance(stored_residual_gate, dict)
                or not bool(stored_residual_gate.get("passed", False))
            ) and not policy.certified_rollout_baseline_enabled:
                raise ValueError(
                    "Resume checkpoint has neither a deployable residual Actor "
                    "nor a certified stochastic rollout baseline."
                )
            stored_bc_report = getattr(
                policy,
                "midlevel_behavior_cloning_report",
                None,
            )
            if isinstance(stored_bc_report, dict):
                bc_report_data = dict(stored_bc_report)
            reset_num_timesteps = False
        else:
            policy = SingleStepTD3BC(
                ConservativeResidualTD3Policy,
                environments,
                seed=args.seed,
                device=args.device,
                verbose=1,
                learning_rate=args.critic_learning_rate,
                actor_learning_rate=args.actor_learning_rate,
                critic_learning_rate=args.critic_learning_rate,
                actor_update_interval=args.actor_update_interval,
                actor_learning_starts=args.actor_learning_starts,
                actor_candidate_supervision_weight=(
                    args.actor_candidate_supervision_weight
                ),
                actor_physical_probe_supervision_weight=(
                    args.actor_physical_probe_supervision_weight
                ),
                actor_candidate_min_q_improvement=(
                    args.actor_candidate_min_q_improvement
                ),
                actor_candidate_min_safe_q=args.actor_candidate_min_safe_q,
                actor_candidate_max_critic_disagreement=(
                    args.actor_candidate_max_critic_disagreement
                ),
                residual_l2_weight=args.residual_l2_weight,
                critic_probe_ranking_weight=(
                    args.critic_probe_ranking_weight
                ),
                critic_probe_delta_weight=args.critic_probe_delta_weight,
                critic_probe_ranking_margin=(
                    args.critic_probe_ranking_margin
                ),
                critic_probe_minimum_reward_difference=(
                    args.critic_probe_minimum_reward_difference
                ),
                critic_supervision_batch_size=(
                    args.critic_supervision_batch_size
                ),
                critic_probe_holdout_fraction=(
                    args.critic_probe_holdout_fraction
                ),
                critic_probe_holdout_seed=args.critic_probe_holdout_seed,
                critic_action_center_scale_mps=(
                    args.critic_action_center_scale_mps
                ),
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                gamma=0.0,
                train_freq=(1, "step"),
                gradient_steps=args.gradient_steps,
                replay_buffer_class=SingleStepCuePositionHerReplayBuffer,
                replay_buffer_kwargs={
                    "her_ratio": args.her_ratio,
                    "success_ratio": args.success_replay_ratio,
                    "failure_ratio": args.failure_replay_ratio,
                    "local_probe_ratio": args.local_probe_replay_ratio,
                    "probe_holdout_fraction": (
                        args.critic_probe_holdout_fraction
                    ),
                    "probe_holdout_seed": args.critic_probe_holdout_seed,
                },
                policy_kwargs={
                    "net_arch": [512, 512, 256],
                    "n_critics": 2,
                    "features_extractor_class": MidLevelGeometricFeatures,
                },
            )
            policy.midlevel_training_manifest = manifest
            if args.initialize_from_bc is not None:
                bc_source = SingleStepTD3BC.load(
                    str(args.initialize_from_bc),
                    device=args.device,
                )
                source_manifest = getattr(
                    bc_source,
                    "midlevel_training_manifest",
                    None,
                )
                source_task = (
                    source_manifest.get("task_dataset", {})
                    if isinstance(source_manifest, dict)
                    else {}
                )
                source_algorithm = (
                    source_manifest.get("algorithm", {})
                    if isinstance(source_manifest, dict)
                    else {}
                )
                if (
                    source_task.get("content_sha256")
                    != dataset.content_sha256()
                    or source_algorithm.get("version")
                    not in {
                        SINGLE_STEP_TD3_VERSION,
                        "offline-curve-her-structured-td3-bc-v7",
                    }
                ):
                    raise ValueError(
                        "BC initializer does not match the task library or "
                        "algorithm version."
                    )
                policy.policy.actor.load_state_dict(
                    bc_source.policy.actor.state_dict()
                )
                source_report = getattr(
                    bc_source,
                    "midlevel_behavior_cloning_report",
                    None,
                )
                if not isinstance(source_report, dict):
                    raise ValueError(
                        "BC initializer has no behavior-cloning report."
                    )
                bc_report_data = dict(source_report)
                bc_report_data["initialized_from"] = str(
                    args.initialize_from_bc
                )
                policy.midlevel_behavior_cloning_report = bc_report_data
                print(
                    "behavior_cloning_initializer="
                    + json.dumps(bc_report_data, sort_keys=True),
                    flush=True,
                )
                del bc_source
            else:
                if args.bc_training_mode == "canonical":
                    bc_report = behavior_clone_td3_policy(
                        policy.policy,
                        dataset,
                        epochs=args.bc_epochs,
                        batch_size=args.bc_batch_size,
                        learning_rate=args.bc_learning_rate,
                        final_learning_rate=args.bc_final_learning_rate,
                        angle_weight=args.bc_angle_weight,
                        speed_weight=args.bc_speed_weight,
                        seed=args.seed,
                        max_grad_norm=args.bc_max_grad_norm,
                    )
                else:
                    if offline_curves is None:
                        raise RuntimeError(
                            "Offline-curve BC requires measured speed curves."
                        )
                    bc_report = behavior_clone_td3_policy_with_offline_curves(
                        policy.policy,
                        offline_curves,
                        epochs=args.bc_epochs,
                        batch_size=args.bc_batch_size,
                        learning_rate=args.bc_learning_rate,
                        final_learning_rate=args.bc_final_learning_rate,
                        angle_weight=args.bc_angle_weight,
                        speed_weight=args.bc_speed_weight,
                        hindsight_fraction=args.bc_hindsight_fraction,
                        physical_loss_weight=args.bc_physical_loss_weight,
                        physical_distance_scale_m=(
                            args.bc_physical_distance_scale_m
                        ),
                        sensitivity_minimum=args.bc_sensitivity_minimum,
                        sensitivity_maximum=args.bc_sensitivity_maximum,
                        holdout_fraction=args.offline_task_holdout_fraction,
                        holdout_seed=args.offline_task_holdout_seed,
                        seed=args.seed,
                        max_grad_norm=args.bc_max_grad_norm,
                    )
                bc_report_data = bc_report.as_dict()
                bc_report_data["training_mode"] = args.bc_training_mode
                bc_gate_checks: dict[str, dict[str, object]] = {
                    "reconstruction_loss_decreased": {
                        "value": bc_report.final_loss,
                        "threshold": bc_report.initial_loss,
                        "direction": "maximum_exclusive",
                        "passed": bc_report.final_loss < bc_report.initial_loss,
                    },
                    "training_speed_p95_within_residual_support": {
                        "value": bc_report.final_speed_p95_mps,
                        "threshold": args.max_speed_residual_mps,
                        "direction": "maximum",
                        "passed": (
                            bc_report.final_speed_p95_mps
                            <= args.max_speed_residual_mps
                        ),
                    },
                }
                if args.bc_training_mode == "offline-curves":
                    bc_gate_checks["holdout_estimated_stop_p90"] = {
                        "value": bc_report.final_estimated_stop_p90_m,
                        "threshold": args.bc_max_offline_stop_p90_m,
                        "direction": "maximum",
                        "passed": (
                            bc_report.final_estimated_stop_p90_m
                            <= args.bc_max_offline_stop_p90_m
                        ),
                    }
                if bc_validation_dataset is not None:
                    validation_report = td3_behavior_cloning_metrics(
                        policy.policy,
                        bc_validation_dataset,
                        batch_size=args.bc_batch_size,
                        angle_weight=args.bc_angle_weight,
                        speed_weight=args.bc_speed_weight,
                    )
                    bc_report_data["validation"] = validation_report
                    print(
                        "behavior_cloning_validation="
                        + json.dumps(validation_report, sort_keys=True),
                        flush=True,
                    )
                    validation_p95_limit = min(
                        args.bc_max_validation_speed_p95_mps,
                        args.max_speed_residual_mps,
                    )
                    bc_gate_checks["validation_speed_mae"] = {
                        "value": validation_report["speed_mae_mps"],
                        "threshold": args.bc_max_validation_speed_mae_mps,
                        "direction": "maximum",
                        "passed": (
                            validation_report["speed_mae_mps"]
                            <= args.bc_max_validation_speed_mae_mps
                        ),
                    }
                    bc_gate_checks[
                        "validation_speed_p95_within_residual_support"
                    ] = {
                        "value": validation_report["speed_p95_mps"],
                        "threshold": validation_p95_limit,
                        "direction": "maximum",
                        "passed": (
                            validation_report["speed_p95_mps"]
                            <= validation_p95_limit
                        ),
                    }
                bc_gate = {
                    "version": "bc-baseline-coverage-diagnostic-v2",
                    "passed": all(
                        bool(check["passed"])
                        for check in bc_gate_checks.values()
                    ),
                    "checks": bc_gate_checks,
                }
                bc_report_data["baseline_coverage"] = bc_gate
                policy.midlevel_behavior_cloning_report = bc_report_data
                print(
                    "behavior_cloning="
                    + json.dumps(bc_report_data, sort_keys=True),
                    flush=True,
                )
                # SB3 only appends ``.zip`` when the supplied path has no
                # suffix.  ``.bc_only`` is itself treated as a suffix, so
                # make the archive extension explicit.
                bc_checkpoint_base = Path(f"{args.output}.bc_only")
                bc_checkpoint = Path(f"{bc_checkpoint_base}.zip")
                policy.save(str(bc_checkpoint))
                Path(f"{bc_checkpoint_base}.manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                Path(f"{bc_checkpoint_base}.bc.json").write_text(
                    json.dumps(bc_report_data, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(f"bc_only_checkpoint={bc_checkpoint}", flush=True)
                if not bool(bc_gate["passed"]):
                    failed_metrics = [
                        name
                        for name, check in bc_gate_checks.items()
                        if not bool(check["passed"])
                    ]
                    print(
                        "behavior_cloning_baseline_requires_residual="
                        + json.dumps(
                            {"failed_metrics": failed_metrics},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if args.bc_only:
                    print(
                        "behavior_cloning_only=COMPLETE "
                        f"gate_passed={bool(bc_gate['passed'])}",
                        flush=True,
                    )
                    return
            if offline_curves is None:
                raise RuntimeError(
                    "Residual Actor and Critic training require offline speed curves."
                )
            policy.configure_conservative_speed_residual(
                max_speed_residual_mps=args.max_speed_residual_mps,
                exploration_initial_std=(
                    args.residual_exploration_initial_std
                ),
                exploration_final_std=args.residual_exploration_final_std,
                exploration_decay_timesteps=(
                    args.residual_exploration_decay_timesteps
                ),
            )
            policy.configure_discrete_candidate_ranking(
                tuple(args.actor_candidate_offsets_mps)
            )
            print(
                "conservative_residual="
                + json.dumps(
                    {
                        "fixed_angle_residual": 0.0,
                        "max_speed_residual_mps": args.max_speed_residual_mps,
                        "exploration_initial_std": (
                            args.residual_exploration_initial_std
                        ),
                        "exploration_final_std": (
                            args.residual_exploration_final_std
                        ),
                        "exploration_decay_timesteps": (
                            args.residual_exploration_decay_timesteps
                        ),
                        "actor_learning_starts": args.actor_learning_starts,
                        "candidate_offsets_mps": list(
                            args.actor_candidate_offsets_mps
                        ),
                        "actor_update": "discrete_candidate_supervision",
                        "residual_l2_weight": args.residual_l2_weight,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            policy.configure_behavior_cloning_reference(
                reference_observations,
                reference_actions,
                initial_weight=args.bc_regularization_initial_weight,
                final_weight=args.bc_regularization_final_weight,
                decay_actor_updates=(
                    args.bc_regularization_decay_actor_updates
                ),
                batch_size=args.bc_regularization_batch_size,
                angle_weight=args.bc_angle_weight,
                speed_weight=args.bc_speed_weight,
                residual_weight=args.bc_regularization_residual_weight,
            )
            policy.configure_offline_curve_actor_supervision(
                offline_curves,
                supervision_weight=args.offline_actor_supervision_weight,
                hindsight_fraction=args.bc_hindsight_fraction,
                batch_size=args.offline_actor_batch_size,
                angle_weight=args.bc_angle_weight,
                speed_weight=args.bc_speed_weight,
                physical_loss_weight=args.bc_physical_loss_weight,
                physical_distance_scale_m=args.bc_physical_distance_scale_m,
                sensitivity_minimum=args.bc_sensitivity_minimum,
                sensitivity_maximum=args.bc_sensitivity_maximum,
                holdout_fraction=args.offline_task_holdout_fraction,
                holdout_seed=args.offline_task_holdout_seed,
                seed=args.seed + 1_000,
                margin_loss_weight=args.offline_actor_margin_loss_weight,
                success_margin_m=args.offline_actor_success_margin_m,
                success_interval_loss_weight=(
                    args.offline_actor_success_interval_loss_weight
                ),
                success_interval_scale_mps=(
                    args.offline_actor_success_interval_scale_mps
                ),
                middle_pocket_weight=(
                    args.offline_actor_middle_pocket_weight
                ),
                hard_task_weight=args.offline_actor_hard_task_weight,
                hard_task_quantile=args.offline_actor_hard_task_quantile,
                hard_task_metric=args.offline_actor_hard_task_metric,
                continuous_residual_inference=True,
            )
            offline_sampling_report = policy.offline_actor_sampling_report()
            policy.midlevel_offline_residual_sampling_report = (
                offline_sampling_report
            )
            print(
                "offline_residual_sampling="
                + json.dumps(offline_sampling_report, sort_keys=True),
                flush=True,
            )
            safe_candidate_configuration: dict[str, object] | None = None
            if args.offline_safe_candidate_classifier:
                safe_candidate_configuration = (
                    policy.configure_safe_candidate_classifier(
                        offline_curves,
                        offsets_mps=tuple(args.safe_candidate_offsets_mps),
                        ensemble_size=args.safe_candidate_ensemble_size,
                        pocket_head_count=args.safe_candidate_pocket_heads,
                        hidden_sizes=tuple(args.safe_candidate_hidden_sizes),
                        batch_size=args.safe_candidate_batch_size,
                        learning_rate=args.safe_candidate_learning_rate,
                        weight_decay=args.safe_candidate_weight_decay,
                        positive_weight=args.safe_candidate_positive_weight,
                        selection_loss_weight=(
                            args.safe_candidate_selection_loss_weight
                        ),
                        selection_target=args.safe_candidate_selection_target,
                        unknown_weight=args.safe_candidate_unknown_weight,
                        label_tolerance_mps=(
                            args.safe_candidate_label_tolerance_mps
                        ),
                        min_probability=args.safe_candidate_min_probability,
                        min_improvement=args.safe_candidate_min_improvement,
                        max_disagreement=(
                            args.safe_candidate_max_disagreement
                        ),
                        uncertainty_scale=(
                            args.safe_candidate_uncertainty_scale
                        ),
                        residual_penalty=args.safe_candidate_residual_penalty,
                        seed=args.seed + 5_000,
                    )
                )
                policy.midlevel_safe_candidate_configuration = (
                    safe_candidate_configuration
                )
                print(
                    "safe_candidate_configuration="
                    + json.dumps(
                        safe_candidate_configuration,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            policy.configure_certified_rollout_baseline(
                offline_curves,
                enabled=args.certified_rollout_baseline,
                seed=args.seed + 4_000,
            )
            print(
                "certified_rollout_baseline="
                + json.dumps(
                    {
                        "enabled": args.certified_rollout_baseline,
                        "stochastic_training_only": True,
                        "deterministic_inference_uses_actor": True,
                        "selection": "uniform_measured_safe_curve_action",
                        "offsets_mps": [
                            float(value)
                            for value in offline_curves.offsets_mps
                        ],
                        "seed": args.seed + 4_000,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if args.residual_only:
                prefill_report = {
                    "skipped": True,
                    "reason": "residual_actor_does_not_use_replay",
                    "transition_count": 0,
                }
            else:
                prefill_report = prefill_certified_replay_buffer(
                    policy,
                    dataset,
                    reference_observations,
                    reference_actions,
                )
            policy.midlevel_certified_prefill_report = prefill_report
            print(
                "certified_replay_prefill="
                + json.dumps(prefill_report, sort_keys=True),
                flush=True,
            )
            if args.offline_safe_candidate_classifier:
                offline_residual_report = (
                    policy.warmup_safe_candidate_classifier(
                        args.safe_candidate_updates,
                        max_grad_norm=args.bc_max_grad_norm,
                    )
                )
                print(
                    "offline_safe_candidate_classifier="
                    + json.dumps(offline_residual_report, sort_keys=True),
                    flush=True,
                )
            else:
                offline_residual_report = policy.warmup_offline_residual_actor(
                    args.offline_residual_warmup_updates,
                    max_grad_norm=args.bc_max_grad_norm,
                    learning_rate=args.offline_residual_learning_rate,
                )
            policy.midlevel_offline_residual_actor_report = (
                offline_residual_report
            )
            print(
                "offline_residual_actor="
                + json.dumps(offline_residual_report, sort_keys=True),
                flush=True,
            )
            residual_diagnostics = policy.offline_residual_actor_diagnostics(
                batch_size=args.bc_batch_size,
            )
            safe_candidate_training_diagnostics: dict[str, object] | None = None
            safe_candidate_holdout_diagnostics: dict[str, object] | None = None
            if args.offline_safe_candidate_classifier:
                safe_candidate_training_diagnostics = (
                    policy.safe_candidate_classifier_diagnostics(
                        holdout=False,
                        batch_size=args.bc_batch_size,
                    )
                )
                safe_candidate_holdout_diagnostics = (
                    policy.safe_candidate_classifier_diagnostics(
                        holdout=True,
                        batch_size=args.bc_batch_size,
                    )
                )
                print(
                    "safe_candidate_training_audit="
                    + json.dumps(
                        safe_candidate_training_diagnostics,
                        sort_keys=True,
                    ),
                    flush=True,
                )
                print(
                    "safe_candidate_holdout_audit="
                    + json.dumps(
                        safe_candidate_holdout_diagnostics,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            residual_curve_radius_mps = float(
                np.max(np.abs(offline_curves.offsets_mps))
            )
            residual_gate_checks: dict[str, dict[str, object]] = {
                "holdout_speed_p95_within_curve": {
                    "value": residual_diagnostics["speed_p95_mps"],
                    "threshold": residual_curve_radius_mps,
                    "direction": "maximum",
                    "passed": (
                        residual_diagnostics["speed_p95_mps"]
                        <= residual_curve_radius_mps
                    ),
                },
                "holdout_estimated_stop_p90": {
                    "value": residual_diagnostics["estimated_stop_p90_m"],
                    "threshold": args.bc_max_offline_stop_p90_m,
                    "direction": "maximum",
                    "passed": (
                        residual_diagnostics["estimated_stop_p90_m"]
                        <= args.bc_max_offline_stop_p90_m
                    ),
                },
            }
            if safe_candidate_holdout_diagnostics is not None:
                residual_gate_checks["holdout_safe_set_net_improvement"] = {
                    "value": safe_candidate_holdout_diagnostics[
                        "safe_rate_improvement"
                    ],
                    "threshold": 0.005,
                    "direction": "minimum",
                    "passed": (
                        safe_candidate_holdout_diagnostics[
                            "safe_rate_improvement"
                        ]
                        >= 0.005
                    ),
                }
                residual_gate_checks[
                    "holdout_paired_improvement_precision"
                ] = {
                    "value": safe_candidate_holdout_diagnostics[
                        "paired_improvement_precision"
                    ],
                    "threshold": 0.60,
                    "direction": "minimum",
                    "passed": (
                        safe_candidate_holdout_diagnostics[
                            "paired_improvement_precision"
                        ]
                        >= 0.60
                    ),
                }
                residual_gate_checks[
                    "holdout_baseline_regression_rate"
                ] = {
                    "value": safe_candidate_holdout_diagnostics[
                        "baseline_regression_rate"
                    ],
                    "threshold": 0.02,
                    "direction": "maximum",
                    "passed": (
                        safe_candidate_holdout_diagnostics[
                            "baseline_regression_rate"
                        ]
                        <= 0.02
                    ),
                }
            residual_validation_report: dict[str, float | int] | None = None
            if bc_validation_dataset is not None:
                validation_observations, validation_actions = (
                    generated_behavior_cloning_data(bc_validation_dataset)
                )
                residual_validation_report = (
                    policy.effective_actor_action_metrics(
                        validation_observations,
                        validation_actions,
                        batch_size=args.bc_batch_size,
                    )
                )
                residual_validation_p95_limit = min(
                    args.bc_max_validation_speed_p95_mps,
                    residual_curve_radius_mps,
                )
                residual_gate_checks["validation_speed_mae"] = {
                    "value": residual_validation_report["speed_mae_mps"],
                    "threshold": args.bc_max_validation_speed_mae_mps,
                    "direction": "maximum",
                    "passed": (
                        residual_validation_report["speed_mae_mps"]
                        <= args.bc_max_validation_speed_mae_mps
                    ),
                }
                residual_gate_checks[
                    "validation_speed_p95_within_curve"
                ] = {
                    "value": residual_validation_report["speed_p95_mps"],
                    "threshold": residual_validation_p95_limit,
                    "direction": "maximum",
                    "passed": (
                        residual_validation_report["speed_p95_mps"]
                        <= residual_validation_p95_limit
                    ),
                }
            residual_gate = {
                "version": "effective-residual-coverage-gate-v1",
                "passed": all(
                    bool(check["passed"])
                    for check in residual_gate_checks.values()
                ),
                "checks": residual_gate_checks,
            }
            residual_audit = {
                "warmup": offline_residual_report,
                "holdout": residual_diagnostics,
                "validation": residual_validation_report,
                "safe_candidate_configuration": safe_candidate_configuration,
                "safe_candidate_training": (
                    safe_candidate_training_diagnostics
                ),
                "safe_candidate_holdout": safe_candidate_holdout_diagnostics,
                "gate": residual_gate,
            }
            policy.midlevel_offline_residual_actor_diagnostics = (
                residual_diagnostics
            )
            policy.midlevel_offline_residual_actor_gate_report = residual_gate
            print(
                "offline_residual_actor_audit="
                + json.dumps(residual_audit, sort_keys=True),
                flush=True,
            )
            residual_base = Path(f"{args.output}.residual_only")
            residual_checkpoint = Path(f"{residual_base}.zip")
            policy.save(str(residual_checkpoint))
            if not args.residual_only:
                policy.save_replay_buffer(replay_buffer_path(residual_base))
            Path(f"{residual_base}.manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            Path(f"{residual_base}.json").write_text(
                json.dumps(residual_audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"residual_only_checkpoint={residual_checkpoint}",
                flush=True,
            )
            if not bool(residual_gate["passed"]):
                failed_metrics = [
                    name
                    for name, check in residual_gate_checks.items()
                    if not bool(check["passed"])
                ]
                if (
                    not args.residual_only
                    and not args.certified_rollout_baseline
                ):
                    raise RuntimeError(
                        "Effective residual Actor failed its task-disjoint gate; "
                        "Critic training was not started. Failed metrics: "
                        + ", ".join(failed_metrics)
                        + f". Residual checkpoint: {residual_checkpoint}"
                    )
                print(
                    "deployable_actor_gate=FAIL "
                    "online_collection_mode="
                    + (
                        "SKIP_RESIDUAL_ONLY "
                        if args.residual_only
                        else "certified_baseline "
                    )
                    + "failed_metrics="
                    + ",".join(failed_metrics),
                    flush=True,
                )
            if args.residual_only:
                print(
                    "residual_actor_training=COMPLETE "
                    "critic_training=SKIP online_physics=SKIP",
                    flush=True,
                )
                return
            policy.configure_structured_curve_critics(
                offline_curves,
                learning_rate=args.structured_critic_learning_rate,
                task_batch_size=args.structured_critic_task_batch_size,
                reward_weight=args.structured_reward_weight,
                reward_delta_weight=args.structured_reward_delta_weight,
                ranking_weight=args.structured_ranking_weight,
                ranking_temperature=args.structured_ranking_temperature,
                cue_delta_weight=args.structured_cue_delta_weight,
                cue_delta_scale_m=args.structured_cue_delta_scale_m,
                event_weight=args.structured_event_weight,
                event_balance_clip=args.structured_event_balance_clip,
                holdout_fraction=args.offline_task_holdout_fraction,
                holdout_seed=args.offline_task_holdout_seed,
                seed=args.seed + 2_000,
            )
            critic_warmup_report = policy.warmup_structured_curve_critics(
                args.structured_critic_warmup_updates
            )
            policy.midlevel_critic_warmup_report = critic_warmup_report
            print(
                "structured_critic_warmup="
                + json.dumps(critic_warmup_report, sort_keys=True),
                flush=True,
            )
            critic_speed_report = policy.structured_curve_critic_diagnostics(
                minimum_reward_difference=(
                    args.critic_probe_minimum_reward_difference
                ),
                seed=args.seed + 3_000,
            )
            policy.midlevel_critic_speed_diagnostics = critic_speed_report
            print(
                "structured_critic_diagnostics="
                + json.dumps(critic_speed_report, sort_keys=True),
                flush=True,
            )
            gate_report = structured_critic_actor_gate_report(
                critic_speed_report,
                minimum_pairwise_agreement=(
                    args.structured_gate_min_pairwise_agreement
                ),
                minimum_canonical_selection_rate=(
                    args.structured_gate_min_canonical_selection_rate
                ),
                maximum_reward_mae=args.structured_gate_max_reward_mae,
                maximum_cue_endpoint_mae_m=(
                    args.structured_gate_max_cue_endpoint_mae_m
                ),
                maximum_safe_brier=args.structured_gate_max_safe_brier,
            )
            policy.midlevel_critic_actor_gate_report = gate_report
            policy.configure_structured_actor_gate(
                enabled=bool(gate_report["passed"]),
                minimum_safe_probability=(
                    args.structured_actor_min_safe_probability
                ),
                maximum_scratch_probability=(
                    args.structured_actor_max_scratch_probability
                ),
                minimum_reward_improvement=(
                    args.structured_actor_min_reward_improvement
                ),
                maximum_reward_disagreement=(
                    args.structured_actor_max_reward_disagreement
                ),
            )
            audit_base = Path(f"{args.output}.critic_audit")
            audit_checkpoint = Path(f"{audit_base}.zip")
            audit_report_path = Path(f"{audit_base}.json")
            audit_report_path.write_text(
                json.dumps(
                    {
                        "gate": gate_report,
                        "diagnostics": critic_speed_report,
                        "warmup": critic_warmup_report,
                        "offline_speed_curves": offline_curves.report(),
                        "offline_residual_actor": offline_residual_report,
                        "offline_residual_actor_diagnostics": (
                            residual_diagnostics
                        ),
                        "offline_residual_actor_gate": residual_gate,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            policy.save(str(audit_checkpoint))
            policy.save_replay_buffer(replay_buffer_path(audit_base))
            Path(f"{audit_base}.manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                "critic_actor_gate="
                + json.dumps(gate_report, sort_keys=True),
                flush=True,
            )
            if not bool(gate_report["passed"]):
                failed_metrics = [
                    name
                    for name, check in gate_report["checks"].items()
                    if not bool(check["passed"])
                ]
                if not args.certified_rollout_baseline:
                    raise RuntimeError(
                        "Critics failed the held-out real-physics Actor gate; "
                        "online learning was not started. Failed metrics: "
                        + ", ".join(failed_metrics)
                        + f". Audit: {audit_report_path}"
                    )
                print(
                    "structured_critic_gate=FAIL "
                    "online_action_selection=measured_safe_curve_only "
                    "failed_metrics="
                    + ",".join(failed_metrics),
                    flush=True,
                )
            reset_num_timesteps = True
        policy.midlevel_training_manifest = manifest

        if args.offline_only:
            print(
                "offline_curve_training=COMPLETE online_learning=SKIP",
                flush=True,
            )
        else:
            post_update_hook = PostUpdateTrainingHook(
                args.output,
                checkpoint_every=args.checkpoint_every,
                initial_timesteps=policy.num_timesteps,
            )
            print(
                f"single_step_td3_bc=START gamma=0 "
                f"gradient_steps={args.gradient_steps} "
                f"actor_update_interval={args.actor_update_interval} "
                f"actor_learning_starts={args.actor_learning_starts} "
                f"her_ratio={args.her_ratio} "
                f"local_probe_ratio={args.local_probe_replay_ratio} "
                f"buffer_size={args.buffer_size}",
                flush=True,
            )
            policy.set_post_update_hook(post_update_hook)
            try:
                policy.learn(
                    total_timesteps=args.total_timesteps,
                    callback=MidLevelTD3DiagnosticsCallback(),
                    # Disable SB3's pre-update dump inside collect_rollouts().
                    # The post-update hook emits one complete row after every
                    # rollout.
                    log_interval=None,  # type: ignore[arg-type]
                    reset_num_timesteps=reset_num_timesteps,
                )
            finally:
                policy.set_post_update_hook(None)
        policy.save(str(args.output))
        final_replay_path = replay_buffer_path(args.output)
        policy.save_replay_buffer(final_replay_path)
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
        if args.backend == "mujoco-warp" and not args.offline_only:
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
        args.output if args.output.suffix == ".zip" else Path(f"{args.output}.zip")
    )
    print(
        f"checkpoint={checkpoint_path} replay_buffer={replay_buffer_path(args.output)} "
        f"manifest={args.output}.manifest.json"
    )


if __name__ == "__main__":
    main()
