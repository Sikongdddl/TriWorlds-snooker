"""Check grouped offline curves, legal HER, structured Critics, and resume."""

from __future__ import annotations

from pathlib import Path
import tempfile

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_offline_curves import (  # noqa: E402
    OFFLINE_SPEED_CURVE_VERSION,
    OfflineSpeedCurveDataset,
)
from snooker_env.midlevel_ppo_env import (  # noqa: E402
    MAX_TERMINAL_REWARD,
    OBSERVATION_X_SCALE,
    OBSERVATION_Y_SCALE,
)
from snooker_env.midlevel_sac_her import (  # noqa: E402
    ConservativeResidualTD3Policy,
    SingleStepCuePositionHerReplayBuffer,
    SingleStepTD3BC,
    behavior_clone_td3_policy_with_offline_curves,
)
from snooker_env.midlevel_two_ball import (  # noqa: E402
    MAX_CUE_SPEED,
    MIN_CUE_SPEED,
    POCKET_POSITIONS,
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
        del options
        super().reset(seed=seed)
        return np.zeros(8, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        return np.zeros(8, dtype=np.float32), 0.0, True, False, {}


def _synthetic_curves() -> OfflineSpeedCurveDataset:
    task_count = 48
    offsets = np.asarray(
        (-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03),
        dtype=np.float64,
    )
    rng = np.random.default_rng(71)
    observations = rng.uniform(-0.6, 0.6, size=(task_count, 8)).astype(np.float32)
    pocket_positions = np.stack(
        [POCKET_POSITIONS[name] for name in sorted(POCKET_POSITIONS)]
    ).astype(np.float32)
    observations[:, 4:6] = (
        pocket_positions[np.arange(task_count) % len(pocket_positions)]
        / np.asarray(
            (OBSERVATION_X_SCALE, OBSERVATION_Y_SCALE),
            dtype=np.float32,
        )
    )
    target_stop = observations[:, 6:8] * np.asarray(
        (OBSERVATION_X_SCALE, OBSERVATION_Y_SCALE),
        dtype=np.float32,
    )
    center_action = np.zeros((task_count, 2), dtype=np.float32)
    center_action[:, 1] = np.clip(
        0.35 * observations[:, 0] - 0.2 * observations[:, 3],
        -0.7,
        0.7,
    )
    speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    actions = np.repeat(center_action[None, :, :], len(offsets), axis=0)
    actions[:, :, 1] += (offsets / speed_half_range)[:, None]
    cue_delta = np.zeros((len(offsets), task_count, 2), dtype=np.float32)
    sensitivity = 1.5 + 2.0 * np.abs(observations[:, 1])
    cue_delta[:, :, 0] = offsets[:, None] * sensitivity[None, :]
    cue_delta[:, :, 1] = (
        np.sign(offsets)[:, None]
        * np.square(offsets[:, None])
        * (8.0 + np.abs(observations[:, 2])[None, :])
    )
    cue_final = np.zeros((len(offsets), task_count, 3), dtype=np.float32)
    cue_final[:, :, :2] = target_stop[None, :, :] + cue_delta
    cue_final[:, :, 2] = 1.05
    object_final = np.zeros_like(cue_final)
    reward = (
        MAX_TERMINAL_REWARD
        - 20.0 * np.abs(offsets)[:, None]
        - 0.2 * np.abs(observations[:, 2])[None, :]
        * (offsets[:, None] != 0.0)
    ).astype(np.float32)
    center = int(np.flatnonzero(offsets == 0.0)[0])
    reward[center] = MAX_TERMINAL_REWARD
    correct_pot = np.ones((len(offsets), task_count), dtype=np.bool_)
    cue_scratch = np.zeros_like(correct_pot)
    cue_scratch[-1, ::5] = True
    stopped = np.ones_like(correct_pot)
    stopped[0, ::7] = False
    wrong_pocket = np.zeros_like(correct_pot)
    timed_out = ~stopped
    numerical_failure = np.zeros_like(correct_pot)
    safe = (
        correct_pot
        & ~cue_scratch
        & stopped
        & ~timed_out
    )
    joint_success = safe & (
        np.linalg.norm(cue_delta, axis=2) <= 0.05
    )
    joint_success[center] = True
    return OfflineSpeedCurveDataset(
        path=Path("synthetic-offline-curves.npz"),
        metadata={
            "format_version": OFFLINE_SPEED_CURVE_VERSION,
            "center_action_source": "canonical_generated_action",
            "world_slot_aligned": True,
            "task_count": task_count,
            "offset_count": len(offsets),
            "record_count": task_count * len(offsets),
        },
        offsets_mps=offsets,
        task_indices=np.arange(task_count, dtype=np.int64),
        observation=observations,
        action=actions.astype(np.float32),
        center_action=center_action,
        target_stop_position=target_stop,
        cue_final=cue_final,
        cue_final_delta_xy_m=cue_delta,
        object_final=object_final,
        reward=reward,
        reward_object_ball=np.ones_like(reward),
        reward_cue_position=np.maximum(reward - 1.0, 0.0),
        reward_joint_success_bonus=0.5 * joint_success.astype(np.float32),
        correct_pot=correct_pot,
        cue_scratch=cue_scratch,
        wrong_pocket=wrong_pocket,
        stopped=stopped,
        timed_out=timed_out,
        numerical_failure=numerical_failure,
        joint_success=joint_success,
        object_pocket_error=np.zeros_like(reward),
    )


def main() -> None:
    curves = _synthetic_curves()
    curves.validate(
        task_dataset=None,
        reference_observations=curves.observation,
        reference_actions=curves.center_action,
        center_stop_tolerance_m=1.0e-7,
    )
    holdout = curves.holdout_mask(fraction=0.20, seed=93)
    if not np.any(holdout) or not np.any(~holdout):
        raise RuntimeError("Task-level offline split is empty.")
    actor_batch = curves.sample_actor_batch(
        np.random.default_rng(94),
        batch_size=16,
        hindsight_fraction=0.5,
        task_mask=~holdout,
    )
    if int(np.count_nonzero(actor_batch.hindsight)) != 8:
        raise RuntimeError("Offline Actor batch did not preserve its HER ratio.")
    if actor_batch.curve_safe.shape != (16, curves.offset_count):
        raise RuntimeError("Offline Actor batch lost its curve safety mask.")
    if not np.allclose(
        actor_batch.observations[actor_batch.hindsight, 6:8]
        * np.asarray((OBSERVATION_X_SCALE, OBSERVATION_Y_SCALE)),
        actor_batch.desired_stop_xy[actor_batch.hindsight],
        atol=1.0e-6,
    ):
        raise RuntimeError("Offline HER did not replace only the cue-stop goal.")
    middle = curves.middle_pocket_mask
    weighted_batch = curves.sample_actor_batch(
        np.random.default_rng(941),
        batch_size=4096,
        hindsight_fraction=0.0,
        task_sampling_weights=np.where(middle, 3.0, 1.0),
    )
    weighted_middle_fraction = float(
        np.mean(middle[weighted_batch.task_indices])
    )
    if not 0.55 <= weighted_middle_fraction <= 0.65:
        raise RuntimeError("Offline Actor task weights were not applied.")
    zero_interval_distance = curves.success_interval_distances_mps(
        np.zeros(curves.task_count, dtype=np.float32)
    )
    if not np.allclose(zero_interval_distance, 0.0):
        raise RuntimeError("Canonical speeds left their safe success intervals.")

    environment = DummyVecEnv([_OneStepEnv for _ in range(4)])
    model = SingleStepTD3BC(
        ConservativeResidualTD3Policy,
        environment,
        seed=95,
        device="cpu",
        verbose=0,
        learning_rate=3.0e-4,
        actor_learning_rate=1.0e-3,
        critic_learning_rate=3.0e-4,
        actor_update_interval=4,
        actor_candidate_supervision_weight=0.0,
        buffer_size=256,
        batch_size=8,
        learning_starts=0,
        gamma=0.0,
        train_freq=(1, "step"),
        gradient_steps=1,
        replay_buffer_class=SingleStepCuePositionHerReplayBuffer,
        replay_buffer_kwargs={
            "her_ratio": 0.0,
            "success_ratio": 0.0,
            "failure_ratio": 0.0,
            "local_probe_ratio": 0.0,
        },
        policy_kwargs={"net_arch": [32, 32]},
    )
    actor_report = behavior_clone_td3_policy_with_offline_curves(
        model.policy,
        curves,
        epochs=2,
        batch_size=16,
        learning_rate=1.0e-3,
        final_learning_rate=5.0e-4,
        hindsight_fraction=0.5,
        holdout_fraction=0.20,
        holdout_seed=93,
        seed=96,
    )
    if actor_report.final_loss >= actor_report.initial_loss:
        raise RuntimeError("Offline Actor did not reduce held-out action loss.")
    model.configure_conservative_speed_residual(
        max_speed_residual_mps=0.03,
        exploration_initial_std=0.10,
        exploration_final_std=0.02,
        exploration_decay_timesteps=16,
    )
    model.configure_discrete_candidate_ranking(tuple(curves.offsets_mps))
    model.configure_offline_curve_actor_supervision(
        curves,
        supervision_weight=4.0,
        hindsight_fraction=0.5,
        batch_size=16,
        angle_weight=1.0,
        speed_weight=8.0,
        physical_loss_weight=1.0,
        physical_distance_scale_m=0.05,
        sensitivity_minimum=0.25,
        sensitivity_maximum=4.0,
        holdout_fraction=0.20,
        holdout_seed=93,
        seed=97,
        margin_loss_weight=2.0,
        success_margin_m=0.01,
        success_interval_loss_weight=1.0,
        success_interval_scale_mps=0.01,
        middle_pocket_weight=2.0,
        hard_task_weight=2.0,
        hard_task_quantile=0.75,
        hard_task_metric="success_interval_distance",
    )
    sampling_report = model.offline_actor_sampling_report()
    if (
        sampling_report["effective_task_count"]
        >= sampling_report["training_task_count"]
        or sampling_report["expected_middle_sample_fraction"] <= 1.0 / 3.0
    ):
        raise RuntimeError("Residual Actor weighted sampling report is invalid.")
    model.configure_certified_rollout_baseline(curves, enabled=True)
    exploratory, _ = model.predict(
        curves.observation[:4],
        deterministic=False,
    )
    speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    if np.any(
        np.abs(exploratory[:, 1] - curves.center_action[:4, 1])
        * speed_half_range
        > 0.030001
    ):
        raise RuntimeError("Certified rollout exploration left its hard bound.")
    residual_report = model.warmup_offline_residual_actor(
        4,
        max_grad_norm=1.0,
    )
    residual_diagnostics = model.offline_residual_actor_diagnostics(
        batch_size=8,
    )
    if not np.isfinite(float(residual_report["final_physical_margin_loss"])):
        raise RuntimeError("Residual Actor margin loss is non-finite.")
    if not np.isfinite(float(residual_report["final_success_interval_loss"])):
        raise RuntimeError("Residual Actor success-interval loss is non-finite.")
    if not np.isfinite(float(residual_diagnostics["estimated_stop_p90_m"])):
        raise RuntimeError("Residual Actor diagnostics are non-finite.")
    classifier_configuration = model.configure_safe_candidate_classifier(
        curves,
        offsets_mps=tuple(curves.offsets_mps),
        ensemble_size=2,
        pocket_head_count=6,
        hidden_sizes=(32, 32),
        batch_size=16,
        learning_rate=1.0e-3,
        weight_decay=1.0e-5,
        positive_weight=2.0,
        selection_loss_weight=0.5,
        selection_target="nearest",
        unknown_weight=0.25,
        label_tolerance_mps=0.005,
        min_probability=0.40,
        min_improvement=0.0,
        max_disagreement=0.50,
        uncertainty_scale=0.0,
        residual_penalty=0.0,
        seed=971,
    )
    classifier_report = model.warmup_safe_candidate_classifier(
        8,
        max_grad_norm=1.0,
    )
    classifier_diagnostics = model.safe_candidate_classifier_diagnostics(
        holdout=True,
        batch_size=8,
    )
    if classifier_configuration["task_with_positive_candidate_rate"] <= 0.0:
        raise RuntimeError("Safe-candidate labels contain no positive action.")
    if not np.isfinite(float(classifier_report["final_loss"])):
        raise RuntimeError("Safe-candidate loss is non-finite.")
    if not np.isfinite(
        float(classifier_diagnostics["safe_rate_improvement"])
    ):
        raise RuntimeError("Safe-candidate diagnostics are non-finite.")
    model.configure_structured_curve_critics(
        curves,
        learning_rate=3.0e-4,
        task_batch_size=8,
        reward_weight=1.0,
        reward_delta_weight=1.0,
        ranking_weight=1.0,
        ranking_temperature=0.25,
        cue_delta_weight=1.0,
        cue_delta_scale_m=0.10,
        event_weight=1.0,
        event_balance_clip=10.0,
        holdout_fraction=0.20,
        holdout_seed=93,
        seed=98,
    )
    critic_report = model.warmup_structured_curve_critics(4)
    diagnostics = model.structured_curve_critic_diagnostics(
        batch_tasks=8,
        max_tasks=8,
    )
    if not np.isfinite(float(diagnostics["reward_pessimistic_mae"])):
        raise RuntimeError("Structured Critic diagnostics are non-finite.")
    observations = curves.observation[:4]
    before, _ = model.predict(observations, deterministic=True)
    repeated, _ = model.predict(observations, deterministic=True)
    if not np.array_equal(before, repeated):
        raise RuntimeError(
            "Continuous-residual deterministic prediction added noise."
        )
    with tempfile.TemporaryDirectory(prefix="midlevel-offline-curve-") as directory:
        checkpoint = Path(directory) / "offline_curve"
        model.save(checkpoint)
        restored = SingleStepTD3BC.load(
            checkpoint,
            env=environment,
            device="cpu",
        )
        after, _ = restored.predict(observations, deterministic=True)
        if not np.allclose(before, after, atol=1.0e-7):
            raise RuntimeError("Structured Critic checkpoint changed Actor output.")
        if not restored.structured_curve_critic_enabled:
            raise RuntimeError("Checkpoint lost its structured Critics.")
        if not restored.safe_candidate_classifier_enabled:
            raise RuntimeError("Checkpoint lost its safe-candidate classifier.")
        restored.attach_offline_speed_curves_for_resume(curves, seed=101)
        restored.predict(observations, deterministic=False)
    environment.close()
    print(
        f"tasks={curves.task_count} records={curves.task_count * curves.offset_count} "
        f"her={len(curves.her_eligible_flat_indices())} "
        f"actor_loss={actor_report.final_loss:.6g} "
        f"residual_loss={residual_report['final_loss']:.6g} "
        f"classifier_loss={classifier_report['final_loss']:.6g} "
        f"critic_loss={critic_report['final_loss']:.6g} "
        "grouped=True task_holdout=True safe_set=True structured_twin=True "
        "resume=True"
    )


if __name__ == "__main__":
    main()
