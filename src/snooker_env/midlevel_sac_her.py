"""Single-step deterministic TD3+BC and cue-position HER support."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.td3.policies import TD3Policy
import torch
from torch import nn
from torch.nn import functional as F

from snooker_env.midlevel_ppo import (
    BehaviorCloningReport,
    CheckpointMidLevelPolicy,
    generated_behavior_cloning_data,
)
from snooker_env.midlevel_ppo_env import (
    MAX_TERMINAL_REWARD,
    OBSERVATION_X_SCALE,
    OBSERVATION_Y_SCALE,
)
from snooker_env.midlevel_two_ball import (
    BALL_RADIUS,
    MAX_ANGLE_RESIDUAL,
    MAX_CUE_SPEED,
    MIN_CUE_SPEED,
    POCKET_POSITIONS,
)


SINGLE_STEP_HER_VERSION = "stratified-successful-pot-cue-stop-ranking-v6"
SINGLE_STEP_TD3_VERSION = "deterministic-td3-bc-discrete-candidate-ranking-v6"
MIDLEVEL_GEOMETRIC_FEATURE_VERSION = "relative-shot-geometry-v1"
MIDLEVEL_GEOMETRIC_FEATURE_DIM = 47
HINDSIGHT_SUCCESS_REWARD = MAX_TERMINAL_REWARD
TARGET_STOP_OBSERVATION_SLICE = slice(6, 8)


class MidLevelGeometricFeatures(BaseFeaturesExtractor):
    """Expose relative shot geometry needed by the inverse-speed mapping.

    The raw observation contains four absolute XY positions.  A generic MLP
    can derive their relationships, but doing so consumed much of the BC
    model capacity and left a large speed error.  These deterministic features
    retain the raw coordinates and add pairwise deltas/distances, ghost-ball
    directions, target projections, pocket identity, and target-to-cushion
    distances.  The same representation is used by the actor and both critics.
    """

    def __init__(self, observation_space: Any) -> None:
        super().__init__(
            observation_space,
            features_dim=MIDLEVEL_GEOMETRIC_FEATURE_DIM,
        )
        self.register_buffer(
            "xy_scale",
            torch.tensor(
                [OBSERVATION_X_SCALE, OBSERVATION_Y_SCALE],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "delta_scale",
            torch.tensor(
                [2.0 * OBSERVATION_X_SCALE, 2.0 * OBSERVATION_Y_SCALE],
                dtype=torch.float32,
            ),
        )
        pocket_positions = np.stack(
            [POCKET_POSITIONS[name][:2] for name in sorted(POCKET_POSITIONS)]
        ).astype(np.float32)
        self.register_buffer(
            "pocket_positions",
            torch.as_tensor(pocket_positions, dtype=torch.float32),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        coordinates = observations.reshape(-1, 4, 2) * self.xy_scale
        cue, object_ball, pocket, target = (
            coordinates[:, index] for index in range(4)
        )
        pairwise = (
            object_ball - cue,
            pocket - cue,
            target - cue,
            pocket - object_ball,
            target - object_ball,
            target - pocket,
        )
        deltas = torch.cat(
            [delta / self.delta_scale for delta in pairwise],
            dim=1,
        )
        distances = torch.cat(
            [
                torch.linalg.vector_norm(delta, dim=1, keepdim=True) / 3.2
                for delta in pairwise
            ],
            dim=1,
        )
        object_to_pocket = pocket - object_ball
        pot_direction = object_to_pocket / torch.clamp(
            torch.linalg.vector_norm(
                object_to_pocket,
                dim=1,
                keepdim=True,
            ),
            min=1.0e-6,
        )
        ghost_position = object_ball - 2.0 * BALL_RADIUS * pot_direction
        cue_to_ghost = ghost_position - cue
        shot_direction = cue_to_ghost / torch.clamp(
            torch.linalg.vector_norm(cue_to_ghost, dim=1, keepdim=True),
            min=1.0e-6,
        )
        tangent = torch.stack(
            (-pot_direction[:, 1], pot_direction[:, 0]),
            dim=1,
        )
        cut_cosine = torch.sum(
            shot_direction * pot_direction,
            dim=1,
            keepdim=True,
        )
        cut_sine = (
            shot_direction[:, :1] * pot_direction[:, 1:]
            - shot_direction[:, 1:] * pot_direction[:, :1]
        )
        target_from_ghost = target - ghost_position
        target_projections = torch.cat(
            (
                torch.sum(
                    target_from_ghost * shot_direction,
                    dim=1,
                    keepdim=True,
                )
                / 3.2,
                torch.sum(
                    target_from_ghost * pot_direction,
                    dim=1,
                    keepdim=True,
                )
                / 3.2,
                torch.sum(
                    target_from_ghost * tangent,
                    dim=1,
                    keepdim=True,
                )
                / 3.2,
            ),
            dim=1,
        )
        pocket_index = torch.argmin(
            torch.sum(
                torch.square(
                    pocket[:, None, :] - self.pocket_positions[None, :, :]
                ),
                dim=2,
            ),
            dim=1,
        )
        pocket_one_hot = F.one_hot(
            pocket_index,
            num_classes=len(POCKET_POSITIONS),
        ).to(observations.dtype)
        target_to_cushions = torch.stack(
            (
                (target[:, 0] + OBSERVATION_X_SCALE)
                / (2.0 * OBSERVATION_X_SCALE),
                (OBSERVATION_X_SCALE - target[:, 0])
                / (2.0 * OBSERVATION_X_SCALE),
                (target[:, 1] + OBSERVATION_Y_SCALE)
                / (2.0 * OBSERVATION_Y_SCALE),
                (OBSERVATION_Y_SCALE - target[:, 1])
                / (2.0 * OBSERVATION_Y_SCALE),
            ),
            dim=1,
        )
        features = torch.cat(
            (
                observations,
                deltas,
                distances,
                pot_direction,
                shot_direction,
                tangent,
                cut_cosine,
                cut_sine,
                target_projections,
                pocket_one_hot,
                target_to_cushions,
            ),
            dim=1,
        )
        if features.shape[1] != MIDLEVEL_GEOMETRIC_FEATURE_DIM:
            raise RuntimeError(
                "Mid-level geometric feature dimension changed unexpectedly."
            )
        return features


class ConservativeResidualTD3Policy(TD3Policy):
    """Deterministic TD3 policy with a frozen behavior-cloned actor."""

    def _build(self, lr_schedule: Any) -> None:
        super()._build(lr_schedule)
        self.reference_actor = self.make_actor()
        self.snapshot_reference_actor()

    def snapshot_reference_actor(self) -> None:
        """Copy the current actor into the immutable residual baseline."""

        self.reference_actor.load_state_dict(self.actor.state_dict())
        for parameter in self.reference_actor.parameters():
            parameter.requires_grad_(False)
        self.reference_actor.set_training_mode(False)

    def set_training_mode(self, mode: bool) -> None:
        super().set_training_mode(mode)
        self.reference_actor.set_training_mode(False)


class SingleStepTD3BC(TD3):
    """Deterministic TD3-style contextual bandit with a strong BC anchor.

    Every transition is terminal, so both critics directly regress the
    immediate shot reward and target networks are deliberately unused.  The
    critics rank a small, BC-centered speed grid; the delayed deterministic
    actor is supervised toward a candidate only when both critics agree that
    it is safe and better than the BC center.  This avoids differentiating
    through the discontinuous pot/scratch/stop event boundary.  Rollout
    exploration remains an independently scheduled, bounded Gaussian speed
    residual around the frozen behavior-cloned actor.
    """

    def __init__(
        self,
        *args: Any,
        actor_learning_rate: float = 1.0e-4,
        critic_learning_rate: float = 3.0e-4,
        actor_update_interval: int = 8,
        actor_learning_starts: int = 0,
        actor_candidate_supervision_weight: float = 1.0,
        actor_physical_probe_supervision_weight: float = 4.0,
        actor_candidate_min_q_improvement: float = 0.05,
        actor_candidate_min_safe_q: float = 1.0,
        actor_candidate_max_critic_disagreement: float = 0.25,
        residual_l2_weight: float = 0.0,
        critic_probe_delta_weight: float = 1.0,
        critic_probe_ranking_weight: float = 0.0,
        critic_probe_ranking_margin: float = 0.10,
        critic_probe_minimum_reward_difference: float = 0.05,
        critic_supervision_batch_size: int = 256,
        critic_probe_holdout_fraction: float = 0.20,
        critic_probe_holdout_seed: int = 20_000,
        critic_action_center_scale_mps: float = 0.06,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("action_noise") is not None:
            raise ValueError(
                "SingleStepTD3BC owns its bounded exploration schedule; "
                "external action_noise is not supported."
            )
        policy_delay = int(kwargs.pop("policy_delay", actor_update_interval))
        if policy_delay != actor_update_interval:
            raise ValueError(
                "TD3 policy_delay must match actor_update_interval."
            )
        kwargs["policy_delay"] = policy_delay
        kwargs.setdefault("target_policy_noise", 0.0)
        kwargs.setdefault("target_noise_clip", 0.0)
        for name, value in (
            ("actor_learning_rate", actor_learning_rate),
            ("critic_learning_rate", critic_learning_rate),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if actor_update_interval <= 0:
            raise ValueError("actor_update_interval must be positive.")
        if actor_learning_starts < 0:
            raise ValueError("actor_learning_starts must be non-negative.")
        if not math.isfinite(residual_l2_weight) or residual_l2_weight < 0.0:
            raise ValueError("residual_l2_weight must be finite and non-negative.")
        for name, value in (
            (
                "actor_candidate_supervision_weight",
                actor_candidate_supervision_weight,
            ),
            (
                "actor_physical_probe_supervision_weight",
                actor_physical_probe_supervision_weight,
            ),
            (
                "actor_candidate_min_q_improvement",
                actor_candidate_min_q_improvement,
            ),
            ("actor_candidate_min_safe_q", actor_candidate_min_safe_q),
            (
                "actor_candidate_max_critic_disagreement",
                actor_candidate_max_critic_disagreement,
            ),
            ("critic_probe_ranking_weight", critic_probe_ranking_weight),
            ("critic_probe_delta_weight", critic_probe_delta_weight),
            ("critic_probe_ranking_margin", critic_probe_ranking_margin),
            (
                "critic_probe_minimum_reward_difference",
                critic_probe_minimum_reward_difference,
            ),
            ("critic_action_center_scale_mps", critic_action_center_scale_mps),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if critic_supervision_batch_size <= 0:
            raise ValueError("critic_supervision_batch_size must be positive.")
        if not 0.0 <= critic_probe_holdout_fraction < 0.5:
            raise ValueError("critic_probe_holdout_fraction must be in [0, 0.5).")
        if critic_probe_holdout_seed < 0:
            raise ValueError("critic_probe_holdout_seed must be non-negative.")
        if critic_action_center_scale_mps <= 0.0:
            raise ValueError("critic_action_center_scale_mps must be positive.")
        self.actor_learning_rate = float(actor_learning_rate)
        self.critic_learning_rate = float(critic_learning_rate)
        self.actor_update_interval = int(actor_update_interval)
        self.actor_learning_starts = int(actor_learning_starts)
        self.actor_candidate_supervision_weight = float(
            actor_candidate_supervision_weight
        )
        self.actor_physical_probe_supervision_weight = float(
            actor_physical_probe_supervision_weight
        )
        self.actor_candidate_min_q_improvement = float(
            actor_candidate_min_q_improvement
        )
        self.actor_candidate_min_safe_q = float(actor_candidate_min_safe_q)
        self.actor_candidate_max_critic_disagreement = float(
            actor_candidate_max_critic_disagreement
        )
        self.residual_l2_weight = float(residual_l2_weight)
        self.critic_probe_delta_weight = float(critic_probe_delta_weight)
        self.critic_probe_ranking_weight = float(critic_probe_ranking_weight)
        self.critic_probe_ranking_margin = float(critic_probe_ranking_margin)
        self.critic_probe_minimum_reward_difference = float(
            critic_probe_minimum_reward_difference
        )
        self.critic_supervision_batch_size = int(critic_supervision_batch_size)
        self.critic_probe_holdout_fraction = float(
            critic_probe_holdout_fraction
        )
        self.critic_probe_holdout_seed = int(critic_probe_holdout_seed)
        self.critic_action_center_scale_mps = float(
            critic_action_center_scale_mps
        )
        self._last_critic_base_loss = 0.0
        self._last_critic_probe_delta_loss = 0.0
        self._last_critic_probe_ranking_loss = 0.0
        self.residual_policy_enabled = False
        self.candidate_ranking_enabled = False
        self.actor_candidate_offsets_mps = np.zeros(1, dtype=np.float32)
        self.max_speed_residual_mps = 0.0
        self.speed_residual_action_scale = 0.0
        self.residual_exploration_initial_std = 0.0
        self.residual_exploration_final_std = 0.0
        self.residual_exploration_decay_timesteps = 1
        self._last_rollout_exploration_std = 0.0
        self.bc_reference_observations: np.ndarray | None = None
        self.bc_reference_actions: np.ndarray | None = None
        self.bc_regularization_initial_weight = 0.0
        self.bc_regularization_final_weight = 0.0
        self.bc_regularization_decay_actor_updates = 1
        self.bc_regularization_batch_size = 0
        self.bc_regularization_action_weights = np.ones(2, dtype=np.float32)
        self.bc_regularization_residual_weight = 0.25
        self._actor_updates = 0
        self._critic_warmup_updates_completed = 0
        self._post_update_hook: Callable[[SingleStepTD3BC], None] | None = None
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        """Keep process-local orchestration hooks out of model archives."""

        return [*super()._excluded_save_params(), "_post_update_hook"]

    def set_post_update_hook(
        self,
        hook: Callable[[SingleStepTD3BC], None] | None,
    ) -> None:
        """Run ``hook`` after one collected rollout has been fully trained."""

        self._post_update_hook = hook

    def _setup_model(self) -> None:
        super()._setup_model()
        if self.residual_policy_enabled:
            self._apply_residual_runtime_constraints()

    def _require_residual_policy(self) -> ConservativeResidualTD3Policy:
        if not isinstance(self.policy, ConservativeResidualTD3Policy):
            raise TypeError(
                "Conservative residual mode requires ConservativeResidualTD3Policy."
            )
        return self.policy

    def _apply_residual_runtime_constraints(self) -> None:
        policy = self._require_residual_policy()
        for parameter in policy.reference_actor.parameters():
            parameter.requires_grad_(False)
        policy.reference_actor.set_training_mode(False)
        for target_network in (self.actor_target, self.critic_target):
            target_network.set_training_mode(False)
        for parameter in self.actor_target.parameters():
            parameter.requires_grad_(False)
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)

    def configure_conservative_speed_residual(
        self,
        *,
        max_speed_residual_mps: float,
        exploration_initial_std: float,
        exploration_final_std: float,
        exploration_decay_timesteps: int,
    ) -> None:
        """Freeze the BC actor and train a deterministic bounded correction.

        The live actor emits a normalized residual. Its angle component is
        ignored; its speed component is scaled and added to the frozen BC
        action. Exploration noise is applied only during rollout and is clipped
        by the same hard speed-residual bound.
        """

        if (
            not math.isfinite(max_speed_residual_mps)
            or max_speed_residual_mps <= 0.0
        ):
            raise ValueError("Maximum speed residual must be positive and finite.")
        speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
        if max_speed_residual_mps > speed_half_range:
            raise ValueError("Maximum speed residual exceeds the action half-range.")
        for name, value in (
            ("initial", exploration_initial_std),
            ("final", exploration_final_std),
        ):
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(
                    f"Residual exploration {name} std must be finite and in [0, 1)."
                )
        if exploration_final_std > exploration_initial_std:
            raise ValueError("Residual exploration std must not increase over training.")
        if exploration_decay_timesteps <= 0:
            raise ValueError("Residual exploration decay timesteps must be positive.")
        if self.learning_starts != 0:
            raise ValueError(
                "Bounded residual exploration requires learning_starts=0; "
                "certified replay provides the initial training data."
            )

        policy = self._require_residual_policy()
        policy.snapshot_reference_actor()
        with torch.no_grad():
            output_layer = next(
                layer
                for layer in reversed(self.actor.mu)
                if isinstance(layer, nn.Linear)
            )
            output_layer.weight.zero_()
            output_layer.bias.zero_()
        self.max_speed_residual_mps = float(max_speed_residual_mps)
        self.speed_residual_action_scale = float(
            max_speed_residual_mps / speed_half_range
        )
        self.residual_exploration_initial_std = float(exploration_initial_std)
        self.residual_exploration_final_std = float(exploration_final_std)
        self.residual_exploration_decay_timesteps = int(
            exploration_decay_timesteps
        )
        self.residual_policy_enabled = True
        self._apply_residual_runtime_constraints()

    def configure_discrete_candidate_ranking(
        self,
        offsets_mps: tuple[float, ...],
    ) -> None:
        """Restrict deterministic improvement to a symmetric speed grid."""

        if not self.residual_policy_enabled:
            raise RuntimeError(
                "Configure the conservative BC residual before candidate ranking."
            )
        offsets = np.asarray(offsets_mps, dtype=np.float64)
        if (
            offsets.ndim != 1
            or len(offsets) < 3
            or not np.all(np.isfinite(offsets))
            or len(np.unique(offsets)) != len(offsets)
            or not np.all(np.diff(offsets) > 0.0)
            or np.count_nonzero(offsets == 0.0) != 1
            or not np.allclose(offsets, -offsets[::-1], atol=1.0e-12)
        ):
            raise ValueError(
                "Candidate offsets must be sorted, unique, symmetric, and include zero."
            )
        if np.max(np.abs(offsets)) > self.max_speed_residual_mps + 1.0e-12:
            raise ValueError(
                "Candidate offsets exceed the configured speed-residual bound."
            )
        self.actor_candidate_offsets_mps = offsets.astype(np.float32)
        self.candidate_ranking_enabled = True

    def _quantize_candidate_residual(
        self,
        residual_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Snap deterministic actor output to the audited physical grid."""

        if not self.candidate_ranking_enabled:
            return residual_actions
        candidates = torch.as_tensor(
            self.actor_candidate_offsets_mps / self.max_speed_residual_mps,
            dtype=residual_actions.dtype,
            device=residual_actions.device,
        )
        nearest = torch.argmin(
            torch.abs(residual_actions[:, 1:2] - candidates[None, :]),
            dim=1,
        )
        speed = candidates[nearest].reshape(-1, 1)
        return torch.cat((torch.zeros_like(speed), speed), dim=1)

    def _candidate_action_grid(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``(batch, candidate, action)`` around frozen BC speed."""

        if not self.candidate_ranking_enabled:
            raise RuntimeError("Discrete candidate ranking is not configured.")
        policy = self._require_residual_policy()
        with torch.no_grad():
            baseline = policy.reference_actor(observations)
        speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
        offsets = torch.as_tensor(
            self.actor_candidate_offsets_mps / speed_half_range,
            dtype=observations.dtype,
            device=observations.device,
        )
        candidate_count = len(self.actor_candidate_offsets_mps)
        speeds = torch.clamp(
            baseline[:, None, 1:2] + offsets[None, :, None],
            -1.0,
            1.0,
        )
        angles = torch.zeros(
            (len(observations), candidate_count, 1),
            dtype=observations.dtype,
            device=observations.device,
        )
        return torch.cat((angles, speeds), dim=2)

    def _effective_action_tensor(
        self,
        observations: torch.Tensor,
        residual_actions: torch.Tensor,
    ) -> torch.Tensor:
        if not self.residual_policy_enabled:
            return residual_actions
        policy = self._require_residual_policy()
        with torch.no_grad():
            baseline = policy.reference_actor(observations)
        speed = torch.clamp(
            baseline[:, 1:2]
            + self.speed_residual_action_scale * residual_actions[:, 1:2],
            -1.0,
            1.0,
        )
        angle = torch.zeros_like(speed)
        return torch.cat((angle, speed), dim=1)

    def _current_exploration_std(self) -> float:
        """Return linearly decayed rollout noise in residual-action units."""

        progress = min(
            max(self.num_timesteps, 0)
            / self.residual_exploration_decay_timesteps,
            1.0,
        )
        return float(
            self.residual_exploration_initial_std
            + progress
            * (
                self.residual_exploration_final_std
                - self.residual_exploration_initial_std
            )
        )

    def predict(
        self,
        observation: Any,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        if not self.residual_policy_enabled:
            return super().predict(
                observation,
                state=state,
                episode_start=episode_start,
                deterministic=deterministic,
            )
        del episode_start
        observation_tensor, vectorized = self.policy.obs_to_tensor(observation)
        with torch.no_grad():
            residual = self.actor(observation_tensor)
            if deterministic:
                residual = self._quantize_candidate_residual(residual)
            else:
                exploration_std = self._current_exploration_std()
                self._last_rollout_exploration_std = exploration_std
                speed = torch.clamp(
                    residual[:, 1:2]
                    + exploration_std * torch.randn_like(residual[:, 1:2]),
                    -1.0,
                    1.0,
                )
                residual = torch.cat(
                    (torch.zeros_like(speed), speed),
                    dim=1,
                )
            effective = self._effective_action_tensor(
                observation_tensor,
                residual,
            )
        actions = self.policy.unscale_action(effective.cpu().numpy())
        if not vectorized:
            actions = actions.squeeze(axis=0)
        return actions, state

    @staticmethod
    def _set_optimizer_learning_rate(
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
    ) -> None:
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    def _configure_optimizer_learning_rates(self) -> None:
        self._set_optimizer_learning_rate(
            self.actor.optimizer,
            self.actor_learning_rate,
        )
        self._set_optimizer_learning_rate(
            self.critic.optimizer,
            self.critic_learning_rate,
        )

    def configure_behavior_cloning_reference(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        initial_weight: float,
        final_weight: float,
        decay_actor_updates: int,
        batch_size: int,
        angle_weight: float,
        speed_weight: float,
        residual_weight: float = 0.25,
    ) -> None:
        """Attach certified task actions used to anchor every actor update."""

        reference_observations = np.asarray(observations, dtype=np.float32)
        reference_actions = np.asarray(actions, dtype=np.float32)
        if reference_observations.ndim != 2 or reference_observations.shape[1] != 8:
            raise ValueError("BC reference observations must have shape (N, 8).")
        if reference_actions.shape != (len(reference_observations), 2):
            raise ValueError("BC reference actions must have shape (N, 2).")
        if len(reference_observations) == 0:
            raise ValueError("BC reference data must not be empty.")
        if not np.all(np.isfinite(reference_observations)) or not np.all(
            np.isfinite(reference_actions)
        ):
            raise ValueError("BC reference data contains non-finite values.")
        for name, value in (
            ("initial_weight", initial_weight),
            ("final_weight", final_weight),
            ("angle_weight", angle_weight),
            ("speed_weight", speed_weight),
            ("residual_weight", residual_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"BC regularization {name} must be finite and non-negative.")
        if final_weight > initial_weight:
            raise ValueError("BC regularization final weight cannot exceed its initial weight.")
        if decay_actor_updates <= 0 or batch_size <= 0:
            raise ValueError("BC regularization decay and batch size must be positive.")
        self.bc_reference_observations = reference_observations.copy()
        self.bc_reference_actions = reference_actions.copy()
        self.bc_regularization_initial_weight = float(initial_weight)
        self.bc_regularization_final_weight = float(final_weight)
        self.bc_regularization_decay_actor_updates = int(decay_actor_updates)
        self.bc_regularization_batch_size = min(int(batch_size), len(reference_actions))
        self.bc_regularization_action_weights = np.asarray(
            [angle_weight, speed_weight],
            dtype=np.float32,
        )
        self.bc_regularization_residual_weight = float(residual_weight)

    def _behavior_cloning_weight(self) -> float:
        progress = min(
            self._actor_updates / self.bc_regularization_decay_actor_updates,
            1.0,
        )
        return float(
            self.bc_regularization_initial_weight
            + progress
            * (
                self.bc_regularization_final_weight
                - self.bc_regularization_initial_weight
            )
        )

    def _behavior_cloning_regularization_loss(self) -> torch.Tensor:
        if (
            self.bc_reference_observations is None
            or self.bc_reference_actions is None
            or self.bc_regularization_batch_size <= 0
        ):
            return torch.zeros((), dtype=torch.float32, device=self.device)
        indices = np.random.randint(
            0,
            len(self.bc_reference_actions),
            size=self.bc_regularization_batch_size,
        )
        observations = torch.as_tensor(
            self.bc_reference_observations[indices],
            dtype=torch.float32,
            device=self.device,
        )
        residual_predictions = self.actor(observations)
        if self.residual_policy_enabled:
            # Online regularization protects the empirically high-success BC
            # policy itself.  The generated action is a feasible certificate,
            # not a local optimum: real speed curves show that its correction
            # direction is close to random around the learned BC center.  Use
            # residual units here: measuring the same deviation in the broad
            # normalized physical action range made this supposedly strong
            # anchor roughly three orders of magnitude too small.
            return self.bc_regularization_residual_weight * torch.mean(
                torch.square(residual_predictions[:, 1])
            )
        targets = torch.as_tensor(
            self.bc_reference_actions[indices],
            dtype=torch.float32,
            device=self.device,
        )
        weights = torch.as_tensor(
            self.bc_regularization_action_weights,
            dtype=torch.float32,
            device=self.device,
        )
        predictions = self._effective_action_tensor(
            observations,
            residual_predictions,
        )
        return torch.mean(
            torch.sum(weights * torch.square(predictions - targets), dim=1)
        )

    def _critic_action_features(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Expose absolute speed and its physical offset from frozen BC."""

        if not self.residual_policy_enabled:
            return actions
        policy = self._require_residual_policy()
        with torch.no_grad():
            baseline = policy.reference_actor(observations)
        speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
        relative_speed = (
            (actions[:, 1:2] - baseline[:, 1:2])
            * speed_half_range
            / self.critic_action_center_scale_mps
        )
        # Angle is hard-locked at zero by the conservative policy.  Reusing
        # that otherwise constant input for absolute speed avoids forcing each
        # critic to relearn the BC inverse-speed map from state before it can
        # reason about a local offset.
        absolute_speed = actions[:, 1:2]
        return torch.cat((absolute_speed, relative_speed), dim=1)

    def _critic_values(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return self.critic(
            observations,
            self._critic_action_features(observations, actions),
        )

    def _local_probe_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit same-state reward deltas and rank the measured safe optimum.

        Absolute terminal reward varies much more across tasks than it does
        across the small action neighborhood used by the residual actor.  A
        plain reward-regression Critic can therefore explain its minibatch
        while ignoring speed.  Subtracting the measured BC-center reward and
        the predicted BC-center value removes that shortcut and directly
        identifies the local action effect.
        """

        replay = self.replay_buffer
        if (
            (
                self.critic_probe_delta_weight == 0.0
                and self.critic_probe_ranking_weight == 0.0
            )
            or not isinstance(replay, SingleStepCuePositionHerReplayBuffer)
        ):
            zero = torch.zeros((), dtype=torch.float32, device=self.device)
            return zero, zero
        row_count = replay.buffer_size if replay.full else replay.pos
        local_flat = np.flatnonzero(replay.local_probe_group_valid[:row_count])
        if len(local_flat):
            local_rows, local_envs = np.unravel_index(
                local_flat,
                (row_count, replay.n_envs),
            )
            training_mask = ~self._probe_holdout_mask(
                replay.local_probe_task_index[local_rows, local_envs]
            )
            local_flat = local_flat[training_mask]
        if len(local_flat) == 0:
            zero = torch.zeros((), dtype=torch.float32, device=self.device)
            return zero, zero
        sample_count = min(self.critic_supervision_batch_size, len(local_flat))
        sampled_flat = np.random.choice(
            local_flat,
            size=sample_count,
            replace=len(local_flat) < sample_count,
        )
        rows, envs = np.unravel_index(
            sampled_flat,
            (row_count, replay.n_envs),
        )
        observations = torch.as_tensor(
            replay.observations[rows, envs],
            dtype=torch.float32,
            device=self.device,
        )
        probe_actions = torch.as_tensor(
            replay.actions[rows, envs],
            dtype=torch.float32,
            device=self.device,
        )
        center_actions = torch.as_tensor(
            replay.local_probe_center_action[rows, envs],
            dtype=torch.float32,
            device=self.device,
        )
        best_actions = torch.as_tensor(
            replay.local_probe_best_action[rows, envs],
            dtype=torch.float32,
            device=self.device,
        )
        probe_q_values = self._critic_values(observations, probe_actions)
        center_q_values = self._critic_values(observations, center_actions)
        best_q_values = self._critic_values(observations, best_actions)
        reward_delta = torch.as_tensor(
            replay.rewards[rows, envs]
            - replay.local_probe_center_reward[rows, envs],
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 1)
        delta_loss = 0.5 * sum(
            F.smooth_l1_loss(
                probe_q - center_q,
                reward_delta,
            )
            for probe_q, center_q in zip(
                probe_q_values,
                center_q_values,
                strict=True,
            )
        )
        eligible = torch.as_tensor(
            replay.local_probe_best_reward[rows, envs]
            - replay.rewards[rows, envs]
            >= self.critic_probe_minimum_reward_difference,
            dtype=torch.bool,
            device=self.device,
        )
        if bool(torch.any(eligible)):
            ranking_loss = 0.5 * sum(
                torch.mean(
                    torch.square(
                        F.relu(
                            self.critic_probe_ranking_margin
                            - (best_q[eligible] - probe_q[eligible])
                        )
                    )
                )
                for probe_q, best_q in zip(
                    probe_q_values,
                    best_q_values,
                    strict=True,
                )
            )
        else:
            ranking_loss = torch.zeros(
                (),
                dtype=torch.float32,
                device=self.device,
            )
        return delta_loss, ranking_loss

    def _probe_holdout_mask(self, task_indices: np.ndarray) -> np.ndarray:
        """Return a reproducible task-level split shared by all probe offsets."""

        replay = self.replay_buffer
        if isinstance(replay, SingleStepCuePositionHerReplayBuffer):
            if (
                replay.probe_holdout_fraction
                != self.critic_probe_holdout_fraction
                or replay.probe_holdout_seed != self.critic_probe_holdout_seed
            ):
                raise RuntimeError(
                    "Policy and replay disagree on the Critic probe holdout split."
                )
            return replay.probe_holdout_mask(task_indices)
        indices = np.asarray(task_indices, dtype=np.uint64)
        if self.critic_probe_holdout_fraction == 0.0:
            return np.zeros(indices.shape, dtype=np.bool_)
        mixed = indices + np.uint64(self.critic_probe_holdout_seed + 1)
        mixed ^= mixed >> np.uint64(30)
        mixed *= np.uint64(0xBF58476D1CE4E5B9)
        mixed ^= mixed >> np.uint64(27)
        mixed *= np.uint64(0x94D049BB133111EB)
        mixed ^= mixed >> np.uint64(31)
        threshold = int(self.critic_probe_holdout_fraction * 1_000_000)
        return (mixed % np.uint64(1_000_000)) < np.uint64(threshold)

    def _critic_update(self, replay_data: ReplayBufferSamples) -> float:
        """Regress both critics directly to the immediate terminal reward."""

        target_q_values = replay_data.rewards
        current_q_values = self._critic_values(
            replay_data.observations,
            replay_data.actions,
        )
        if len(current_q_values) != 2:
            raise RuntimeError("SingleStepTD3BC requires exactly two critics.")
        base_loss = 0.5 * sum(
            F.smooth_l1_loss(current_q, target_q_values)
            for current_q in current_q_values
        )
        probe_delta_loss, probe_ranking_loss = self._local_probe_losses()
        critic_loss = (
            base_loss
            + self.critic_probe_delta_weight * probe_delta_loss
            + self.critic_probe_ranking_weight * probe_ranking_loss
        )
        self.critic.optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic.optimizer.step()
        self._last_critic_base_loss = float(base_loss.detach().item())
        self._last_critic_probe_delta_loss = float(
            probe_delta_loss.detach().item()
        )
        self._last_critic_probe_ranking_loss = float(
            probe_ranking_loss.detach().item()
        )
        return float(critic_loss.item())

    def _critic_candidate_targets(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose only candidates independently approved by both critics."""

        candidate_actions = self._candidate_action_grid(observations)
        batch_size, candidate_count, _ = candidate_actions.shape
        repeated_observations = observations[:, None, :].expand(
            -1,
            candidate_count,
            -1,
        )
        with torch.no_grad():
            q_value_tuple = self._critic_values(
                repeated_observations.reshape(-1, observations.shape[1]),
                candidate_actions.reshape(-1, candidate_actions.shape[2]),
            )
            if len(q_value_tuple) != 2:
                raise RuntimeError("Candidate ranking requires exactly two critics.")
            q1 = q_value_tuple[0].reshape(batch_size, candidate_count)
            q2 = q_value_tuple[1].reshape(batch_size, candidate_count)
            minimum_q = torch.minimum(q1, q2)
            disagreement = torch.abs(q1 - q2)
            offsets = torch.as_tensor(
                self.actor_candidate_offsets_mps,
                dtype=observations.dtype,
                device=observations.device,
            )
            center_index = int(
                np.flatnonzero(self.actor_candidate_offsets_mps == 0.0)[0]
            )
            center_q1 = q1[:, center_index : center_index + 1]
            center_q2 = q2[:, center_index : center_index + 1]
            center_safe = (
                torch.minimum(center_q1, center_q2)
                >= self.actor_candidate_min_safe_q
            )
            approved = (
                (q1 - center_q1
                 >= self.actor_candidate_min_q_improvement)
                & (q2 - center_q2
                   >= self.actor_candidate_min_q_improvement)
                & (minimum_q >= self.actor_candidate_min_safe_q)
                & (
                    disagreement
                    <= self.actor_candidate_max_critic_disagreement
                )
            )
            approved[:, center_index] = True
            improvement = torch.minimum(q1 - center_q1, q2 - center_q2)
            tie_penalty = 1.0e-5 * torch.abs(offsets)[None, :]
            scores = torch.where(
                approved,
                improvement - tie_penalty,
                torch.full_like(minimum_q, -torch.inf),
            )
            selected_indices = torch.argmax(scores, dim=1)
            selected_indices = torch.where(
                center_safe.reshape(-1),
                selected_indices,
                torch.full_like(selected_indices, center_index),
            )
            rows = torch.arange(batch_size, device=observations.device)
            selected_offsets = offsets[selected_indices]
            target_residual = (
                selected_offsets / self.max_speed_residual_mps
            )
            selected_disagreement = disagreement[rows, selected_indices]
            selected_minimum_q = minimum_q[rows, selected_indices]
            improved = selected_indices != center_index
        return (
            target_residual,
            improved,
            selected_disagreement,
            selected_minimum_q,
        )

    def _physical_probe_actor_loss(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Supervise the residual actor from measured per-state best probes."""

        replay = self.replay_buffer
        if not isinstance(replay, SingleStepCuePositionHerReplayBuffer):
            zero = torch.zeros((), dtype=torch.float32, device=self.device)
            return zero, zero, zero
        row_count = replay.buffer_size if replay.full else replay.pos
        anchors = np.flatnonzero(replay.local_probe_group_anchor[:row_count])
        if len(anchors):
            anchor_rows, anchor_envs = np.unravel_index(
                anchors,
                (row_count, replay.n_envs),
            )
            training_mask = ~self._probe_holdout_mask(
                replay.local_probe_task_index[anchor_rows, anchor_envs]
            )
            anchors = anchors[training_mask]
        if len(anchors) == 0:
            zero = torch.zeros((), dtype=torch.float32, device=self.device)
            return zero, zero, zero
        sample_count = min(self.critic_supervision_batch_size, len(anchors))
        sampled = np.random.choice(
            anchors,
            size=sample_count,
            replace=len(anchors) < sample_count,
        )
        rows, envs = np.unravel_index(
            sampled,
            (row_count, replay.n_envs),
        )
        observations = torch.as_tensor(
            replay.observations[rows, envs],
            dtype=torch.float32,
            device=self.device,
        )
        best_actions = torch.as_tensor(
            replay.local_probe_best_action[rows, envs],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            baseline = self._require_residual_policy().reference_actor(
                observations
            )
            target_residual = torch.clamp(
                (
                    best_actions[:, 1]
                    - baseline[:, 1]
                )
                / self.speed_residual_action_scale,
                -1.0,
                1.0,
            )
        predicted_residual = self.actor(observations)[:, 1]
        loss = F.smooth_l1_loss(predicted_residual, target_residual)
        safe_rate = torch.as_tensor(
            replay.local_probe_best_safe[rows, envs].mean(),
            dtype=torch.float32,
            device=self.device,
        )
        mean_abs_offset = torch.mean(
            torch.abs(target_residual) * self.max_speed_residual_mps
        )
        return loss, safe_rate, mean_abs_offset

    def _minimum_critic_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the pessimistic actor value and Q1/Q2 disagreement."""

        q_value_tuple = self._critic_values(observations, actions)
        if len(q_value_tuple) != 2:
            raise RuntimeError("SingleStepTD3BC requires exactly two critics.")
        q_values = torch.cat(q_value_tuple, dim=1)
        minimum, _ = torch.min(q_values, dim=1, keepdim=True)
        disagreement = torch.abs(q_values[:, 0] - q_values[:, 1])
        return minimum, disagreement

    def warmup_critic(
        self,
        gradient_steps: int,
        *,
        batch_size: int,
    ) -> dict[str, float | int]:
        """Fit critics to certified replay while keeping the BC actor frozen."""

        if gradient_steps <= 0 or batch_size <= 0:
            raise ValueError("Critic warmup steps and batch size must be positive.")
        if self.replay_buffer is None or self.replay_buffer.size() <= 0:
            raise RuntimeError("Critic warmup requires a populated replay buffer.")
        self.policy.set_training_mode(True)
        self._configure_optimizer_learning_rates()
        losses: list[float] = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )
            losses.append(self._critic_update(replay_data))
        self._critic_warmup_updates_completed += gradient_steps
        window = min(32, max(1, len(losses) // 4))
        return {
            "updates": gradient_steps,
            "initial_loss": float(np.mean(losses[:window])),
            "final_loss": float(np.mean(losses[-window:])),
            "final_base_loss": self._last_critic_base_loss,
            "final_probe_delta_loss": self._last_critic_probe_delta_loss,
            "final_probe_ranking_loss": (
                self._last_critic_probe_ranking_loss
            ),
        }

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self.gamma != 0.0:
            raise ValueError("SingleStepTD3BC requires gamma=0 for terminal transitions.")
        self.policy.set_training_mode(True)
        self._configure_optimizer_learning_rates()

        actor_losses: list[float] = []
        candidate_supervision_losses: list[float] = []
        physical_probe_supervision_losses: list[float] = []
        bc_losses: list[float] = []
        bc_weights: list[float] = []
        candidate_improvement_rates: list[float] = []
        candidate_disagreements: list[float] = []
        candidate_minimum_q_values: list[float] = []
        candidate_target_offsets_mps: list[float] = []
        physical_probe_safe_rates: list[float] = []
        physical_probe_target_offsets_mps: list[float] = []
        residual_penalties: list[float] = []
        speed_residuals_mps: list[float] = []
        critic_losses: list[float] = []
        actor_updates_this_call = 0
        actor_deferred = self.num_timesteps < self.actor_learning_starts
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size,
                env=self._vec_normalize_env,
            )
            critic_losses.append(self._critic_update(replay_data))

            update_index = self._n_updates + gradient_step + 1
            if (
                actor_deferred
                or update_index % self.actor_update_interval != 0
            ):
                continue
            if not self.candidate_ranking_enabled:
                raise RuntimeError(
                    "Actor updates require configured discrete candidate ranking."
                )
            residual_actions_pi = self.actor(replay_data.observations)
            (
                candidate_targets,
                candidate_improved,
                selected_disagreement,
                selected_minimum_q,
            ) = self._critic_candidate_targets(replay_data.observations)
            candidate_supervision_loss = F.smooth_l1_loss(
                residual_actions_pi[:, 1],
                candidate_targets,
            )
            if self.actor_physical_probe_supervision_weight > 0.0:
                (
                    physical_probe_supervision_loss,
                    physical_probe_safe_rate,
                    physical_probe_target_offset_mps,
                ) = self._physical_probe_actor_loss()
            else:
                physical_probe_supervision_loss = torch.zeros(
                    (), dtype=torch.float32, device=self.device
                )
                physical_probe_safe_rate = torch.zeros(
                    (), dtype=torch.float32, device=self.device
                )
                physical_probe_target_offset_mps = torch.zeros(
                    (), dtype=torch.float32, device=self.device
                )
            if self.residual_policy_enabled:
                residual_penalty = self.residual_l2_weight * torch.mean(
                    torch.square(residual_actions_pi[:, 1])
                )
            else:
                residual_penalty = torch.zeros((), device=self.device)
            bc_loss = self._behavior_cloning_regularization_loss()
            bc_weight = self._behavior_cloning_weight()
            actor_loss = (
                self.actor_candidate_supervision_weight
                * candidate_supervision_loss
                + self.actor_physical_probe_supervision_weight
                * physical_probe_supervision_loss
                + bc_weight * bc_loss
                + residual_penalty
            )
            self.actor.optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor.optimizer.step()

            actor_losses.append(float(actor_loss.item()))
            candidate_supervision_losses.append(
                float(candidate_supervision_loss.item())
            )
            physical_probe_supervision_losses.append(
                float(physical_probe_supervision_loss.item())
            )
            bc_losses.append(float(bc_loss.item()))
            bc_weights.append(bc_weight)
            candidate_improvement_rates.append(
                float(candidate_improved.float().mean().item())
            )
            candidate_disagreements.append(
                float(selected_disagreement.mean().item())
            )
            candidate_minimum_q_values.append(
                float(selected_minimum_q.mean().item())
            )
            candidate_target_offsets_mps.append(
                float(
                    torch.abs(candidate_targets).mean().item()
                    * self.max_speed_residual_mps
                )
            )
            physical_probe_safe_rates.append(
                float(physical_probe_safe_rate.item())
            )
            physical_probe_target_offsets_mps.append(
                float(physical_probe_target_offset_mps.item())
            )
            residual_penalties.append(float(residual_penalty.item()))
            if self.residual_policy_enabled:
                speed_residuals_mps.append(
                    float(
                        residual_actions_pi[:, 1]
                        .detach()
                        .abs()
                        .mean()
                        .item()
                        * self.max_speed_residual_mps
                    )
                )
            self._actor_updates += 1
            actor_updates_this_call += 1

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_updates", self._actor_updates)
        self.logger.record("train/actor_updates_this_rollout", actor_updates_this_call)
        self.logger.record("train/actor_deferred", float(actor_deferred))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record(
            "train/critic_base_loss",
            self._last_critic_base_loss,
        )
        self.logger.record(
            "train/critic_probe_delta_loss",
            self._last_critic_probe_delta_loss,
        )
        self.logger.record(
            "train/critic_probe_ranking_loss",
            self._last_critic_probe_ranking_loss,
        )
        self.logger.record("train/actor_learning_rate", self.actor_learning_rate)
        self.logger.record("train/critic_learning_rate", self.critic_learning_rate)
        self.logger.record(
            "train/rollout_exploration_std",
            self._last_rollout_exploration_std,
        )
        self.logger.record(
            "train/rollout_exploration_std_mps",
            self._last_rollout_exploration_std * self.max_speed_residual_mps,
        )
        self.logger.record(
            "train/next_exploration_std",
            self._current_exploration_std(),
        )
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record(
                "train/candidate_supervision_loss",
                np.mean(candidate_supervision_losses),
            )
            self.logger.record(
                "train/physical_probe_supervision_loss",
                np.mean(physical_probe_supervision_losses),
            )
            self.logger.record("train/bc_regularization_loss", np.mean(bc_losses))
            self.logger.record("train/bc_regularization_weight", np.mean(bc_weights))
            self.logger.record(
                "train/candidate_improvement_rate",
                np.mean(candidate_improvement_rates),
            )
            self.logger.record(
                "train/candidate_critic_disagreement",
                np.mean(candidate_disagreements),
            )
            self.logger.record(
                "train/candidate_minimum_q",
                np.mean(candidate_minimum_q_values),
            )
            self.logger.record(
                "train/candidate_target_abs_offset_mps",
                np.mean(candidate_target_offsets_mps),
            )
            self.logger.record(
                "train/physical_probe_safe_target_rate",
                np.mean(physical_probe_safe_rates),
            )
            self.logger.record(
                "train/physical_probe_target_abs_offset_mps",
                np.mean(physical_probe_target_offsets_mps),
            )
            self.logger.record(
                "train/residual_l2_penalty",
                np.mean(residual_penalties),
            )
            if speed_residuals_mps:
                self.logger.record(
                    "train/mean_abs_speed_residual_mps",
                    np.mean(speed_residuals_mps),
                )
        if self._post_update_hook is not None:
            self._post_update_hook(self)


class SingleStepCuePositionHerReplayBuffer(ReplayBuffer):
    """Relabel only the cue-stop goal of successful one-transition episodes.

    Standard HER relabels every desired goal from a future achieved state.  A
    one-shot pool episode has only one future state, and arbitrary object-ball
    endpoints are not valid target-pocket goals.  This buffer therefore keeps
    the requested pocket fixed and creates virtual transitions only from legal
    correct-pot outcomes.  Their desired cue stop is replaced with the actual
    final cue position, making the virtual transition an exact maximum reward.
    """

    def __init__(
        self,
        *args: Any,
        her_ratio: float = 0.25,
        success_ratio: float = 0.25,
        failure_ratio: float = 0.25,
        local_probe_ratio: float = 0.0,
        probe_holdout_fraction: float = 0.20,
        probe_holdout_seed: int = 20_000,
        **kwargs: Any,
    ) -> None:
        for name, value in (
            ("her_ratio", her_ratio),
            ("success_ratio", success_ratio),
            ("failure_ratio", failure_ratio),
            ("local_probe_ratio", local_probe_ratio),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
        if (
            her_ratio + success_ratio + failure_ratio + local_probe_ratio
            > 1.0 + 1.0e-12
        ):
            raise ValueError(
                "HER, success, failure, and local-probe replay ratios "
                "must sum to <= 1."
            )
        if not 0.0 <= probe_holdout_fraction < 0.5:
            raise ValueError("probe_holdout_fraction must be in [0, 0.5).")
        if probe_holdout_seed < 0:
            raise ValueError("probe_holdout_seed must be non-negative.")
        if bool(kwargs.get("optimize_memory_usage", False)):
            raise ValueError(
                "SingleStepCuePositionHerReplayBuffer requires explicit next observations."
            )
        super().__init__(*args, **kwargs)
        if self.obs_shape != (8,):
            raise ValueError(
                "Single-step cue-position HER requires the normalized 8-D observation."
            )
        self.her_ratio = float(her_ratio)
        self.success_ratio = float(success_ratio)
        self.failure_ratio = float(failure_ratio)
        self.local_probe_ratio = float(local_probe_ratio)
        self.probe_holdout_fraction = float(probe_holdout_fraction)
        self.probe_holdout_seed = int(probe_holdout_seed)
        self.her_version = SINGLE_STEP_HER_VERSION
        self.her_eligible = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.bool_,
        )
        self.achieved_stop_goals = np.zeros(
            (self.buffer_size, self.n_envs, 2),
            dtype=np.float32,
        )
        self.successful_transition = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.bool_,
        )
        self.failure_transition = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.bool_,
        )
        self.safe_transition = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.bool_,
        )
        self.joint_success_transition = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.bool_,
        )
        self.local_probe_transition = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.bool_,
        )
        self.local_probe_offset_mps = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32,
        )
        self.local_probe_task_index = np.full(
            (self.buffer_size, self.n_envs),
            -1,
            dtype=np.int64,
        )
        self.source_task_index = np.full(
            (self.buffer_size, self.n_envs),
            -1,
            dtype=np.int64,
        )
        self.local_probe_group_anchor = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.bool_,
        )
        self.local_probe_group_valid = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.bool_,
        )
        self.local_probe_best_action = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=np.float32,
        )
        self.local_probe_best_reward = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32,
        )
        self.local_probe_best_safe = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.bool_,
        )
        self.local_probe_center_action = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=np.float32,
        )
        self.local_probe_center_reward = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32,
        )
        self.critic_holdout_transition = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.bool_,
        )
        self.last_sample_composition: dict[str, int] = {
            "uniform": 0,
            "success": 0,
            "failure": 0,
            "local_probe": 0,
            "hindsight": 0,
        }

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Store the transition and its optional successful cue-stop goal."""

        storage_index = self.pos
        self.her_eligible[storage_index] = False
        self.achieved_stop_goals[storage_index] = 0.0
        self.successful_transition[storage_index] = False
        self.failure_transition[storage_index] = False
        self.safe_transition[storage_index] = False
        self.joint_success_transition[storage_index] = False
        self.local_probe_transition[storage_index] = False
        self.local_probe_offset_mps[storage_index] = 0.0
        self.local_probe_task_index[storage_index] = -1
        self.source_task_index[storage_index] = -1
        self.local_probe_group_anchor[storage_index] = False
        self.local_probe_group_valid[storage_index] = False
        self.local_probe_best_action[storage_index] = 0.0
        self.local_probe_best_reward[storage_index] = 0.0
        self.local_probe_best_safe[storage_index] = False
        self.local_probe_center_action[storage_index] = 0.0
        self.local_probe_center_reward[storage_index] = 0.0
        self.critic_holdout_transition[storage_index] = False
        for env_id, info in enumerate(infos):
            eligible = bool(
                done[env_id]
                and info.get("correct_pot", False)
                and not info.get("cue_scratch", False)
                and not info.get("wrong_pocket", False)
                and info.get("stopped", False)
                and not info.get("timed_out", False)
                and not info.get("numerical_failure", False)
            )
            final_position = np.asarray(
                info.get("cue_ball_final_position", (np.nan, np.nan)),
                dtype=np.float64,
            )
            if final_position.shape[0] < 2 or not np.all(
                np.isfinite(final_position[:2])
            ):
                eligible = False
            if eligible:
                normalized = np.array(
                    [
                        final_position[0] / OBSERVATION_X_SCALE,
                        final_position[1] / OBSERVATION_Y_SCALE,
                    ],
                    dtype=np.float64,
                )
                if np.any(np.abs(normalized) > 1.0 + 1e-6):
                    eligible = False
                else:
                    self.achieved_stop_goals[storage_index, env_id] = np.clip(
                        normalized,
                        -1.0,
                        1.0,
                    ).astype(np.float32)
            self.her_eligible[storage_index, env_id] = eligible
            self.successful_transition[storage_index, env_id] = eligible
            self.failure_transition[storage_index, env_id] = bool(
                done[env_id]
                and (
                    info.get("cue_scratch", False)
                    or info.get("wrong_pocket", False)
                    or info.get("timed_out", False)
                    or info.get("numerical_failure", False)
                )
            )
            self.safe_transition[storage_index, env_id] = bool(
                done[env_id]
                and info.get("correct_pot", False)
                and not info.get("cue_scratch", False)
                and not info.get("wrong_pocket", False)
                and info.get("stopped", False)
                and not info.get("timed_out", False)
                and not info.get("numerical_failure", False)
            )
            self.joint_success_transition[storage_index, env_id] = bool(
                done[env_id] and info.get("joint_success", False)
            )
            is_local_probe = bool(
                done[env_id] and info.get("local_speed_probe", False)
            )
            probe_offset = float(info.get("local_speed_probe_offset_mps", 0.0))
            if is_local_probe and not math.isfinite(probe_offset):
                raise ValueError(
                    "Local speed probes require a finite physical offset."
                )
            self.local_probe_transition[storage_index, env_id] = is_local_probe
            if is_local_probe:
                self.local_probe_offset_mps[storage_index, env_id] = probe_offset
                task_index = int(info.get("local_speed_probe_task_index", -1))
                if task_index < 0:
                    raise ValueError(
                        "Local speed probes require a non-negative task index."
                    )
                self.local_probe_task_index[storage_index, env_id] = task_index
            else:
                task_index = int(info.get("task_index", -1))
            if task_index >= 0:
                self.source_task_index[storage_index, env_id] = task_index
                self.critic_holdout_transition[storage_index, env_id] = bool(
                    self.probe_holdout_mask(
                        np.asarray([task_index], dtype=np.int64)
                    )[0]
                )
        super().add(obs, next_obs, action, reward, done, infos)

    def finalize_local_probe_group(
        self,
        rows_by_offset_mps: dict[float, int],
    ) -> None:
        """Attach each real probe to its safest, highest-reward candidate."""

        normalized_rows = {
            round(float(offset), 9): int(row)
            for offset, row in rows_by_offset_mps.items()
        }
        offsets = np.asarray(sorted(normalized_rows), dtype=np.float64)
        if (
            len(offsets) < 3
            or np.count_nonzero(offsets == 0.0) != 1
            or not np.allclose(offsets, -offsets[::-1], atol=1.0e-9)
        ):
            raise ValueError(
                "Local probe groups must be symmetric and contain a center row."
            )
        rows = np.asarray(
            [normalized_rows[round(float(offset), 9)] for offset in offsets],
            dtype=np.int64,
        )
        if np.any(rows < 0) or np.any(rows >= self.buffer_size):
            raise IndexError("A local probe group row is outside replay.")
        if not bool(np.all(self.local_probe_transition[rows])):
            raise ValueError(
                "Every local probe group row must contain probes in every world."
            )
        reference_tasks = self.local_probe_task_index[rows[0]]
        if any(
            not np.array_equal(self.local_probe_task_index[row], reference_tasks)
            for row in rows[1:]
        ):
            raise ValueError("Local probe group changed task/world identity.")
        for offset, row in zip(offsets, rows, strict=True):
            if not np.allclose(
                self.local_probe_offset_mps[row],
                offset,
                atol=1.0e-8,
                rtol=0.0,
            ):
                raise ValueError("A local probe group row has an incorrect offset.")

        rewards = self.rewards[rows]
        safe = self.safe_transition[rows]
        safe_rewards = np.where(safe, rewards, -np.inf)
        maximum_safe_reward = np.max(safe_rewards, axis=0)
        has_safe = np.isfinite(maximum_safe_reward)
        tied_best = safe & np.isclose(
            rewards,
            maximum_safe_reward[None, :],
            atol=1.0e-7,
            rtol=0.0,
        )
        tie_cost = np.where(tied_best, np.abs(offsets)[:, None], np.inf)
        best_candidate_indices = np.argmin(tie_cost, axis=0)
        center_index = int(np.flatnonzero(offsets == 0.0)[0])
        best_candidate_indices[~has_safe] = center_index
        env_indices = np.arange(self.n_envs, dtype=np.int64)
        best_rows = rows[best_candidate_indices]
        best_actions = self.actions[best_rows, env_indices]
        best_rewards = self.rewards[best_rows, env_indices]
        center_row = rows[center_index]
        center_actions = self.actions[center_row]
        center_rewards = self.rewards[center_row]
        for row in rows:
            self.local_probe_group_valid[row] = True
            self.local_probe_best_action[row] = best_actions
            self.local_probe_best_reward[row] = best_rewards
            self.local_probe_best_safe[row] = has_safe
            self.local_probe_center_action[row] = center_actions
            self.local_probe_center_reward[row] = center_rewards
        self.local_probe_group_anchor[rows[center_index]] = True
        holdout = self.probe_holdout_mask(reference_tasks)
        for row in rows:
            self.critic_holdout_transition[row] = holdout

    def probe_holdout_mask(self, task_indices: np.ndarray) -> np.ndarray:
        """Return the deterministic task-level Critic holdout split."""

        indices = np.asarray(task_indices, dtype=np.uint64)
        if self.probe_holdout_fraction == 0.0:
            return np.zeros(indices.shape, dtype=np.bool_)
        mixed = indices + np.uint64(self.probe_holdout_seed + 1)
        mixed ^= mixed >> np.uint64(30)
        mixed *= np.uint64(0xBF58476D1CE4E5B9)
        mixed ^= mixed >> np.uint64(27)
        mixed *= np.uint64(0x94D049BB133111EB)
        mixed ^= mixed >> np.uint64(31)
        threshold = int(self.probe_holdout_fraction * 1_000_000)
        return (mixed % np.uint64(1_000_000)) < np.uint64(threshold)

    @property
    def eligible_transition_count(self) -> int:
        """Return the number of currently stored transitions usable by HER."""

        row_count = self.buffer_size if self.full else self.pos
        return int(np.count_nonzero(self.her_eligible[:row_count]))

    @property
    def successful_transition_count(self) -> int:
        row_count = self.buffer_size if self.full else self.pos
        return int(np.count_nonzero(self.successful_transition[:row_count]))

    @property
    def failure_transition_count(self) -> int:
        row_count = self.buffer_size if self.full else self.pos
        return int(np.count_nonzero(self.failure_transition[:row_count]))

    @property
    def local_probe_transition_count(self) -> int:
        row_count = self.buffer_size if self.full else self.pos
        return int(np.count_nonzero(self.local_probe_transition[:row_count]))

    @property
    def local_probe_group_count(self) -> int:
        row_count = self.buffer_size if self.full else self.pos
        return int(np.count_nonzero(self.local_probe_group_valid[:row_count]))

    @property
    def local_probe_group_anchor_count(self) -> int:
        row_count = self.buffer_size if self.full else self.pos
        return int(np.count_nonzero(self.local_probe_group_anchor[:row_count]))

    @staticmethod
    def _draw_flat_indices(pool: np.ndarray, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        return np.random.choice(pool, size=count, replace=True).astype(np.int64)

    def sample(
        self,
        batch_size: int,
        env: VecNormalize | None = None,
    ) -> ReplayBufferSamples:
        """Mix ordinary replay with successful cue-stop hindsight samples."""

        if batch_size <= 0:
            raise ValueError("Replay batch_size must be positive.")
        row_count = self.buffer_size if self.full else self.pos
        if row_count <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        training_mask = ~self.critic_holdout_transition[:row_count]
        training_flat = np.flatnonzero(training_mask)
        if len(training_flat) == 0:
            raise RuntimeError("Replay contains no non-holdout training samples.")
        eligible_flat = np.flatnonzero(
            self.her_eligible[:row_count] & training_mask
        )
        success_flat = np.flatnonzero(
            self.successful_transition[:row_count] & training_mask
        )
        failure_flat = np.flatnonzero(
            self.failure_transition[:row_count] & training_mask
        )
        local_probe_flat = np.flatnonzero(
            self.local_probe_transition[:row_count] & training_mask
        )
        requested_hindsight = int(batch_size * self.her_ratio)
        requested_success = int(batch_size * self.success_ratio)
        requested_failure = int(batch_size * self.failure_ratio)
        requested_local_probe = int(batch_size * self.local_probe_ratio)
        hindsight_count = requested_hindsight if eligible_flat.size > 0 else 0
        success_count = requested_success if success_flat.size > 0 else 0
        failure_count = requested_failure if failure_flat.size > 0 else 0
        local_probe_count = (
            requested_local_probe if local_probe_flat.size > 0 else 0
        )
        uniform_count = (
            batch_size
            - hindsight_count
            - success_count
            - failure_count
            - local_probe_count
        )
        if uniform_count < 0:
            raise RuntimeError("Stratified replay counts exceeded the batch size.")

        uniform_flat = self._draw_flat_indices(
            training_flat,
            uniform_count,
        )
        success_sample_flat = self._draw_flat_indices(success_flat, success_count)
        failure_sample_flat = self._draw_flat_indices(failure_flat, failure_count)
        local_probe_sample_flat = self._draw_flat_indices(
            local_probe_flat,
            local_probe_count,
        )
        real_flat = np.concatenate(
            (
                uniform_flat,
                success_sample_flat,
                failure_sample_flat,
                local_probe_sample_flat,
            )
        )
        hindsight_flat = self._draw_flat_indices(eligible_flat, hindsight_count)
        flat_indices = np.concatenate((real_flat, hindsight_flat))
        real_count = len(real_flat)
        self.last_sample_composition = {
            "uniform": uniform_count,
            "success": success_count,
            "failure": failure_count,
            "local_probe": local_probe_count,
            "hindsight": hindsight_count,
        }
        batch_indices, env_indices = np.unravel_index(
            flat_indices,
            (row_count, self.n_envs),
        )

        observations = self.observations[batch_indices, env_indices].copy()
        next_observations = self.next_observations[
            batch_indices,
            env_indices,
        ].copy()
        rewards = self.rewards[batch_indices, env_indices].copy()
        if hindsight_count > 0:
            virtual_slice = slice(real_count, batch_size)
            hindsight_goals = self.achieved_stop_goals[
                batch_indices[virtual_slice],
                env_indices[virtual_slice],
            ]
            observations[virtual_slice, TARGET_STOP_OBSERVATION_SLICE] = hindsight_goals
            next_observations[
                virtual_slice,
                TARGET_STOP_OBSERVATION_SLICE,
            ] = hindsight_goals
            rewards[virtual_slice] = HINDSIGHT_SUCCESS_REWARD

        data = (
            self._normalize_obs(observations, env),
            self.actions[batch_indices, env_indices],
            self._normalize_obs(next_observations, env),
            (
                self.dones[batch_indices, env_indices]
                * (1 - self.timeouts[batch_indices, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(rewards.reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


def prefill_certified_replay_buffer(
    policy: SingleStepTD3BC,
    dataset: Any,
    observations: np.ndarray,
    actions: np.ndarray,
) -> dict[str, int]:
    """Insert canonical task-generation successes before online rollouts."""

    replay = policy.replay_buffer
    if not isinstance(replay, SingleStepCuePositionHerReplayBuffer):
        raise TypeError("Certified prefill requires the stratified HER replay buffer.")
    reference_observations = np.asarray(observations, dtype=np.float32)
    reference_actions = np.asarray(actions, dtype=np.float32)
    if reference_observations.shape != (len(dataset), 8):
        raise ValueError("Certified prefill observations do not match the task dataset.")
    if reference_actions.shape != (len(dataset), 2):
        raise ValueError("Certified prefill actions do not match the task dataset.")
    if len(dataset) == 0:
        raise ValueError("Cannot prefill replay from an empty task dataset.")

    stored_count = 0
    padded_count = 0
    for start in range(0, len(dataset), replay.n_envs):
        valid_count = min(replay.n_envs, len(dataset) - start)
        indices = np.arange(start, start + replay.n_envs, dtype=np.int64)
        if valid_count < replay.n_envs:
            indices[valid_count:] = indices[: replay.n_envs - valid_count] % len(dataset)
            padded_count += replay.n_envs - valid_count
        batch_observations = reference_observations[indices]
        batch_actions = reference_actions[indices]
        infos: list[dict[str, Any]] = []
        for task_index in indices:
            target_stop = np.asarray(
                dataset.target_stop_positions[int(task_index)],
                dtype=np.float64,
            )
            infos.append(
                {
                    "correct_pot": True,
                    "cue_scratch": False,
                    "wrong_pocket": False,
                    "stopped": True,
                    "timed_out": False,
                    "numerical_failure": False,
                    "cue_ball_final_position": np.array(
                        [target_stop[0], target_stop[1], 0.0],
                        dtype=np.float64,
                    ),
                    "TimeLimit.truncated": False,
                    "certified_task_prefill": True,
                    "task_index": int(task_index),
                }
            )
        replay.add(
            batch_observations,
            batch_observations.copy(),
            batch_actions,
            np.full(replay.n_envs, MAX_TERMINAL_REWARD, dtype=np.float32),
            np.ones(replay.n_envs, dtype=np.bool_),
            infos,
        )
        stored_count += replay.n_envs
    return {
        "task_count": len(dataset),
        "stored_count": stored_count,
        "padded_count": padded_count,
    }


def balanced_local_probe_task_indices(
    dataset: Any,
    task_count: int,
    *,
    seed: int,
) -> np.ndarray:
    """Select a deterministic pocket-balanced subset for local action probes."""

    if task_count <= 0 or task_count > len(dataset):
        raise ValueError("Local probe task count must be in [1, len(dataset)].")
    pocket_indices = np.asarray(dataset.pocket_indices, dtype=np.int64)
    represented_pockets = np.unique(pocket_indices)
    if represented_pockets.size == 0:
        raise ValueError("Local speed probes require represented pockets.")
    rng = np.random.default_rng(seed)
    base_count, remainder = divmod(task_count, len(represented_pockets))
    selected: list[np.ndarray] = []
    for rank, pocket_index in enumerate(represented_pockets):
        requested = base_count + int(rank < remainder)
        candidates = np.flatnonzero(pocket_indices == pocket_index)
        if requested > len(candidates):
            raise ValueError(
                "A pocket does not contain enough tasks for balanced local probes."
            )
        selected.append(rng.choice(candidates, size=requested, replace=False))
    task_indices = np.concatenate(selected).astype(np.int64)
    rng.shuffle(task_indices)
    return task_indices


def slot_aligned_local_probe_batch_starts(
    dataset: Any,
    task_count: int,
    *,
    num_worlds: int,
    seed: int,
) -> np.ndarray:
    """Choose balanced complete batches with stable counterfactual slots."""

    if num_worlds <= 0:
        raise ValueError("Canonical probe num_worlds must be positive.")
    if task_count <= 0 or task_count > len(dataset):
        raise ValueError("Local probe task count must be in [1, len(dataset)].")
    if task_count % num_worlds != 0:
        raise ValueError(
            "Slot-aligned local probe task count must be a multiple of num_worlds."
        )
    full_batch_count = len(dataset) // num_worlds
    requested_batch_count = task_count // num_worlds
    if requested_batch_count > full_batch_count:
        raise ValueError(
            "The task library does not contain enough complete execution batches."
        )
    pocket_indices = np.asarray(dataset.pocket_indices, dtype=np.int64)
    pocket_count = int(np.max(pocket_indices)) + 1
    batch_pocket_counts = np.stack(
        [
            np.bincount(
                pocket_indices[
                    batch_index * num_worlds : (batch_index + 1) * num_worlds
                ],
                minlength=pocket_count,
            )
            for batch_index in range(full_batch_count)
        ]
    ).astype(np.float64)
    population = np.sum(batch_pocket_counts, axis=0)
    target_proportions = population / np.sum(population)
    rng = np.random.default_rng(seed)
    tie_order = rng.permutation(full_batch_count)
    tie_rank = np.empty(full_batch_count, dtype=np.int64)
    tie_rank[tie_order] = np.arange(full_batch_count)
    selected: list[int] = []
    cumulative = np.zeros(pocket_count, dtype=np.float64)
    remaining = set(range(full_batch_count))
    for selection_index in range(requested_batch_count):
        target = (
            (selection_index + 1)
            * num_worlds
            * target_proportions
        )
        candidate_scores: list[tuple[float, int, int]] = []
        for batch_index in remaining:
            candidate_counts = cumulative + batch_pocket_counts[batch_index]
            normalized_error = (candidate_counts - target) / np.sqrt(target + 1.0)
            score = float(np.sum(np.square(normalized_error)))
            candidate_scores.append(
                (score, int(tie_rank[batch_index]), batch_index)
            )
        _, _, chosen = min(candidate_scores)
        selected.append(chosen)
        cumulative += batch_pocket_counts[chosen]
        remaining.remove(chosen)
    return np.asarray(selected, dtype=np.int64) * num_worlds


def collect_local_speed_probes(
    policy: SingleStepTD3BC,
    environment: VecEnv,
    dataset: Any,
    reference_observations: np.ndarray,
    reference_actions: np.ndarray,
    *,
    task_count: int,
    offsets_mps: tuple[float, ...],
    seed: int,
) -> dict[str, Any]:
    """Execute symmetric real-physics speed probes around the frozen BC action.

    Every selected canonical batch is kept intact.  Its tasks remain in the
    world slots used to publish their target stops, while speed offsets are
    executed in serial rollouts.  Thus paired rewards differ only in action,
    never in the hidden MJWarp world slot.
    """

    replay = policy.replay_buffer
    if not isinstance(replay, SingleStepCuePositionHerReplayBuffer):
        raise TypeError("Local speed probes require the stratified HER replay buffer.")
    observations = np.asarray(reference_observations, dtype=np.float32)
    actions = np.asarray(reference_actions, dtype=np.float32)
    if observations.shape != (len(dataset), 8):
        raise ValueError("Local probe observations do not match the task dataset.")
    if actions.shape != (len(dataset), 2):
        raise ValueError("Local probe actions do not match the task dataset.")
    offsets = np.asarray(offsets_mps, dtype=np.float64)
    if offsets.ndim != 1 or len(offsets) < 3:
        raise ValueError("Local speed probes require at least three offsets.")
    if (
        not np.all(np.isfinite(offsets))
        or len(np.unique(offsets)) != len(offsets)
        or not np.all(np.diff(offsets) > 0.0)
        or np.count_nonzero(offsets == 0.0) != 1
    ):
        raise ValueError(
            "Local speed probe offsets must be sorted, finite, unique, and include zero."
        )
    if not np.allclose(offsets, -offsets[::-1], atol=1.0e-12):
        raise ValueError("Local speed probe offsets must be symmetric around zero.")
    if np.max(np.abs(offsets)) > policy.max_speed_residual_mps + 1.0e-12:
        raise ValueError("Local speed probes exceed the configured residual bound.")
    batch_starts = slot_aligned_local_probe_batch_starts(
        dataset,
        task_count,
        num_worlds=environment.num_envs,
        seed=seed,
    )
    reference_policy = policy._require_residual_policy()
    baseline_batches: list[np.ndarray] = []
    reference_policy.reference_actor.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, len(observations), 4096):
            observation_tensor = torch.as_tensor(
                observations[start : start + 4096],
                dtype=torch.float32,
                device=policy.device,
            )
            baseline_batches.append(
                reference_policy.reference_actor(observation_tensor)
                .cpu()
                .numpy()
            )
    baseline_actions = np.concatenate(baseline_batches, axis=0).astype(
        np.float32
    )
    baseline_actions[:, 0] = 0.0
    speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    normalized_offsets = offsets / speed_half_range
    reward_sums = np.zeros(len(offsets), dtype=np.float64)
    correct_pot_counts = np.zeros(len(offsets), dtype=np.int64)
    joint_success_counts = np.zeros(len(offsets), dtype=np.int64)
    scratch_counts = np.zeros(len(offsets), dtype=np.int64)
    outcome_counts = np.zeros(len(offsets), dtype=np.int64)
    stored_count = 0
    for batch_start in batch_starts:
        batch_task_indices = np.arange(
            int(batch_start),
            int(batch_start) + environment.num_envs,
            dtype=np.int64,
        )
        rows_by_offset: dict[float, int] = {}
        for offset_index, (offset, normalized_offset) in enumerate(
            zip(offsets, normalized_offsets, strict=True)
        ):
            batch_actions = baseline_actions[batch_task_indices].copy()
            batch_actions[:, 0] = 0.0
            batch_actions[:, 1] += np.float32(normalized_offset)
            if np.any(batch_actions[:, 1] < -1.0) or np.any(
                batch_actions[:, 1] > 1.0
            ):
                raise ValueError(
                    "A local speed probe exceeds the physical action range."
                )
            environment.set_options(
                [
                    {"task_index": int(task_index)}
                    for task_index in batch_task_indices
                ]
            )
            batch_observations = np.asarray(
                environment.reset(),
                dtype=np.float32,
            )
            next_observations, rewards, dones, infos = environment.step(
                batch_actions
            )
            dones = np.asarray(dones, dtype=np.bool_)
            if not np.all(dones):
                raise RuntimeError(
                    "Every local speed probe must be a terminal shot."
                )
            terminal_observations: list[np.ndarray] = []
            for env_id, info in enumerate(infos):
                expected_task_index = int(batch_task_indices[env_id])
                if int(info.get("task_index", -1)) != expected_task_index:
                    raise RuntimeError(
                        "Local speed probe task/world identity changed."
                    )
                expected_world_slot = expected_task_index % environment.num_envs
                if env_id != expected_world_slot:
                    raise RuntimeError(
                        "Local speed probe left its canonical world slot."
                    )
                info["local_speed_probe"] = True
                info["local_speed_probe_offset_mps"] = float(offset)
                info["local_speed_probe_task_index"] = expected_task_index
                info["local_speed_probe_world_slot"] = env_id
                terminal_observations.append(
                    np.asarray(
                        info.get(
                            "terminal_observation",
                            next_observations[env_id],
                        ),
                        dtype=np.float32,
                    )
                )
                reward_sums[offset_index] += float(rewards[env_id])
                correct_pot_counts[offset_index] += int(
                    info.get("correct_pot", False)
                )
                joint_success_counts[offset_index] += int(
                    info.get("joint_success", False)
                )
                scratch_counts[offset_index] += int(
                    info.get("cue_scratch", False)
                )
                outcome_counts[offset_index] += 1
            storage_row = replay.pos
            replay.add(
                batch_observations,
                np.stack(terminal_observations).astype(np.float32),
                batch_actions,
                np.asarray(rewards, dtype=np.float32),
                dones,
                infos,
            )
            rows_by_offset[float(offset)] = storage_row
            stored_count += environment.num_envs
        replay.finalize_local_probe_group(rows_by_offset)

    offset_reports: list[dict[str, float | int]] = []
    for index, offset in enumerate(offsets):
        count = max(int(outcome_counts[index]), 1)
        offset_reports.append(
            {
                "offset_mps": float(offset),
                "sample_count": int(outcome_counts[index]),
                "mean_reward": float(reward_sums[index] / count),
                "correct_pot_rate": float(correct_pot_counts[index] / count),
                "joint_success_rate": float(joint_success_counts[index] / count),
                "scratch_rate": float(scratch_counts[index] / count),
            }
        )
    return {
        "probe_center": "frozen_bc_action",
        "world_slot_aligned": True,
        "offset_execution": "serial_same_world_slot",
        "num_worlds": int(environment.num_envs),
        "batch_starts": [int(value) for value in batch_starts],
        "task_count": int(task_count),
        "stored_transition_count": int(stored_count),
        "padded_task_count": 0,
        "offsets_mps": [float(value) for value in offsets],
        "per_offset": offset_reports,
    }


def critic_local_speed_diagnostics(
    policy: SingleStepTD3BC,
    dataset: Any,
    reference_observations: np.ndarray,
    reference_actions: np.ndarray,
    *,
    task_count: int,
    seed: int,
    minimum_physical_reward_difference: float = 0.05,
    batch_size: int = 4096,
    max_probe_samples: int = 16_384,
    **legacy_arguments: Any,
) -> dict[str, Any]:
    """Measure held-out candidate ranking and conservative selection quality."""

    del dataset, reference_observations, reference_actions, legacy_arguments
    if minimum_physical_reward_difference < 0.0:
        raise ValueError("Minimum physical reward difference must be non-negative.")
    if batch_size <= 0 or max_probe_samples <= 0:
        raise ValueError("Critic ranking diagnostic batch sizes must be positive.")
    replay = policy.replay_buffer
    if not isinstance(replay, SingleStepCuePositionHerReplayBuffer):
        raise TypeError("Critic ranking diagnostics require local-probe replay.")
    row_count = replay.buffer_size if replay.full else replay.pos
    all_anchors = np.flatnonzero(replay.local_probe_group_anchor[:row_count])
    if len(all_anchors) != task_count:
        raise ValueError(
            "Critic diagnostics require one center anchor per probed task: "
            f"expected {task_count}, found {len(all_anchors)}."
        )
    all_anchor_rows, all_anchor_envs = np.unravel_index(
        all_anchors,
        (row_count, replay.n_envs),
    )
    all_task_indices = replay.local_probe_task_index[
        all_anchor_rows,
        all_anchor_envs,
    ]
    held_out_mask = policy._probe_holdout_mask(all_task_indices)
    if not bool(np.any(held_out_mask)):
        raise ValueError(
            "Critic ranking diagnostics require a non-empty configured holdout."
        )
    anchor_rows = all_anchor_rows[held_out_mask]
    anchor_envs = all_anchor_envs[held_out_mask]
    task_indices = all_task_indices[held_out_mask]
    held_out_count = len(task_indices)
    split_rng = np.random.default_rng(seed)

    observations = torch.as_tensor(
        replay.observations[anchor_rows, anchor_envs],
        dtype=torch.float32,
        device=policy.device,
    )
    physical_best_actions = replay.local_probe_best_action[
        anchor_rows, anchor_envs
    ]
    physical_best_rewards = replay.local_probe_best_reward[
        anchor_rows, anchor_envs
    ]
    center_rewards = replay.rewards[
        anchor_rows, anchor_envs
    ]
    with torch.no_grad():
        candidate_actions = policy._candidate_action_grid(observations)
        batch_count, candidate_count, _ = candidate_actions.shape
        repeated_observations = observations[:, None, :].expand(
            -1,
            candidate_count,
            -1,
        )
        q_value_tuple = policy._critic_values(
            repeated_observations.reshape(-1, observations.shape[1]),
            candidate_actions.reshape(-1, candidate_actions.shape[2]),
        )
        q1 = q_value_tuple[0].reshape(batch_count, candidate_count)
        q2 = q_value_tuple[1].reshape(batch_count, candidate_count)
        minimum_q = torch.minimum(q1, q2)
        disagreement = torch.abs(q1 - q2)
        (
            selected_residual,
            selected_improved,
            selected_disagreement,
            selected_minimum_q,
        ) = policy._critic_candidate_targets(observations)
    offsets = policy.actor_candidate_offsets_mps.astype(np.float64)
    selected_offsets = (
        selected_residual.cpu().numpy().astype(np.float64)
        * policy.max_speed_residual_mps
    )
    speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    baseline_actions = candidate_actions[:, offsets == 0.0, :]
    baseline_actions = baseline_actions[:, 0, :].cpu().numpy()
    physical_best_offsets = (
        physical_best_actions[:, 1] - baseline_actions[:, 1]
    ) * speed_half_range
    exact_selection = np.isclose(
        selected_offsets,
        physical_best_offsets,
        atol=1.0e-6,
    )
    probe_lookup: dict[tuple[int, float], tuple[float, bool, bool, bool]] = {}
    all_probe_flat = np.flatnonzero(
        replay.local_probe_group_valid[:row_count]
    )
    all_probe_rows, all_probe_envs = np.unravel_index(
        all_probe_flat,
        (row_count, replay.n_envs),
    )
    for probe_row, probe_env in zip(
        all_probe_rows,
        all_probe_envs,
        strict=True,
    ):
        probe_lookup[
            (
                int(replay.local_probe_task_index[probe_row, probe_env]),
                round(
                    float(replay.local_probe_offset_mps[probe_row, probe_env]),
                    6,
                ),
            )
        ] = (
            float(replay.rewards[probe_row, probe_env]),
            bool(replay.safe_transition[probe_row, probe_env]),
            bool(replay.joint_success_transition[probe_row, probe_env]),
            bool(replay.failure_transition[probe_row, probe_env]),
        )
    selected_outcomes = [
        probe_lookup[(int(task_index), round(float(offset), 6))]
        for task_index, offset in zip(
            task_indices,
            selected_offsets,
            strict=True,
        )
    ]
    selected_physical_rewards = np.asarray(
        [outcome[0] for outcome in selected_outcomes],
        dtype=np.float64,
    )
    selected_physical_safe = np.asarray(
        [outcome[1] for outcome in selected_outcomes],
        dtype=np.bool_,
    )
    selected_physical_joint = np.asarray(
        [outcome[2] for outcome in selected_outcomes],
        dtype=np.bool_,
    )
    selected_physical_failure = np.asarray(
        [outcome[3] for outcome in selected_outcomes],
        dtype=np.bool_,
    )
    center_outcomes = [
        probe_lookup[(int(task_index), 0.0)] for task_index in task_indices
    ]
    center_physical_joint = np.asarray(
        [outcome[2] for outcome in center_outcomes],
        dtype=np.bool_,
    )
    center_physical_safe = np.asarray(
        [outcome[1] for outcome in center_outcomes],
        dtype=np.bool_,
    )
    center_physical_failure = np.asarray(
        [outcome[3] for outcome in center_outcomes],
        dtype=np.bool_,
    )
    candidate_physical_rewards = np.empty(
        (held_out_count, len(offsets)),
        dtype=np.float64,
    )
    candidate_physical_safe = np.empty_like(
        candidate_physical_rewards,
        dtype=np.bool_,
    )
    candidate_physical_joint = np.empty_like(
        candidate_physical_rewards,
        dtype=np.bool_,
    )
    candidate_physical_failure = np.empty_like(
        candidate_physical_rewards,
        dtype=np.bool_,
    )
    for task_row, task_index in enumerate(task_indices):
        for candidate_index, offset in enumerate(offsets):
            outcome = probe_lookup[
                (int(task_index), round(float(offset), 6))
            ]
            candidate_physical_rewards[task_row, candidate_index] = outcome[0]
            candidate_physical_safe[task_row, candidate_index] = outcome[1]
            candidate_physical_joint[task_row, candidate_index] = outcome[2]
            candidate_physical_failure[task_row, candidate_index] = outcome[3]
    selected_nonzero = np.abs(selected_offsets) > 1.0e-9
    selected_true_improvement = (
        selected_physical_safe
        & (
            selected_physical_rewards
            >= center_rewards + minimum_physical_reward_difference
        )
    )
    selected_reward_improvements = selected_physical_rewards - center_rewards
    selected_joint_improvements = (
        selected_physical_joint.astype(np.float64)
        - center_physical_joint.astype(np.float64)
    )
    selected_safe_improvements = (
        selected_physical_safe.astype(np.float64)
        - center_physical_safe.astype(np.float64)
    )
    selected_failure_increases = (
        selected_physical_failure.astype(np.float64)
        - center_physical_failure.astype(np.float64)
    )

    def paired_mean_and_standard_error(
        values: np.ndarray,
    ) -> tuple[float, float]:
        mean = float(np.mean(values))
        if len(values) <= 1:
            return mean, 0.0
        standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
        return mean, standard_error

    reward_improvement_mean, reward_improvement_se = (
        paired_mean_and_standard_error(selected_reward_improvements)
    )
    joint_improvement_mean, joint_improvement_se = (
        paired_mean_and_standard_error(selected_joint_improvements)
    )
    safe_improvement_mean, safe_improvement_se = (
        paired_mean_and_standard_error(selected_safe_improvements)
    )
    failure_increase_mean, failure_increase_se = (
        paired_mean_and_standard_error(selected_failure_increases)
    )
    true_improvement_count = int(
        np.count_nonzero(selected_true_improvement & selected_nonzero)
    )
    nonzero_count = int(np.count_nonzero(selected_nonzero))
    if nonzero_count:
        precision = true_improvement_count / nonzero_count
        z_value = 1.959963984540054
        denominator = 1.0 + z_value * z_value / nonzero_count
        center = (
            precision + z_value * z_value / (2.0 * nonzero_count)
        ) / denominator
        radius = z_value / denominator * math.sqrt(
            precision * (1.0 - precision) / nonzero_count
            + z_value * z_value / (4.0 * nonzero_count * nonzero_count)
        )
        precision_lower_bound = center - radius
    else:
        precision = float("nan")
        precision_lower_bound = float("nan")
    q1_numpy = q1.cpu().numpy().astype(np.float64)
    q2_numpy = q2.cpu().numpy().astype(np.float64)
    minimum_q_numpy = np.minimum(q1_numpy, q2_numpy)
    disagreement_numpy = np.abs(q1_numpy - q2_numpy)
    center_index = int(np.flatnonzero(offsets == 0.0)[0])
    q1_improvement = q1_numpy - q1_numpy[:, center_index : center_index + 1]
    q2_improvement = q2_numpy - q2_numpy[:, center_index : center_index + 1]
    pessimistic_improvement = np.minimum(q1_improvement, q2_improvement)
    threshold_sweep: list[dict[str, float | int]] = []
    sweep_rows = np.arange(held_out_count)
    for minimum_q_improvement in (0.02, 0.05, 0.10, 0.20):
        for maximum_disagreement in (0.05, 0.10, 0.25):
            for minimum_safe_q in (0.5, 1.0, 1.5):
                approved = (
                    (q1_improvement >= minimum_q_improvement)
                    & (q2_improvement >= minimum_q_improvement)
                    & (minimum_q_numpy >= minimum_safe_q)
                    & (disagreement_numpy <= maximum_disagreement)
                )
                approved[:, center_index] = True
                scores = np.where(
                    approved,
                    pessimistic_improvement
                    - 1.0e-5 * np.abs(offsets)[None, :],
                    -np.inf,
                )
                sweep_selection = np.argmax(scores, axis=1)
                center_safe = (
                    minimum_q_numpy[:, center_index] >= minimum_safe_q
                )
                sweep_selection[~center_safe] = center_index
                sweep_nonzero = sweep_selection != center_index
                sweep_rewards = candidate_physical_rewards[
                    sweep_rows,
                    sweep_selection,
                ]
                sweep_safe = candidate_physical_safe[
                    sweep_rows,
                    sweep_selection,
                ]
                sweep_joint = candidate_physical_joint[
                    sweep_rows,
                    sweep_selection,
                ]
                sweep_failure = candidate_physical_failure[
                    sweep_rows,
                    sweep_selection,
                ]
                sweep_true_improvement = (
                    sweep_safe
                    & (
                        sweep_rewards
                        >= center_rewards + minimum_physical_reward_difference
                    )
                )
                threshold_sweep.append(
                    {
                        "minimum_q_improvement": minimum_q_improvement,
                        "maximum_critic_disagreement": maximum_disagreement,
                        "minimum_safe_q": minimum_safe_q,
                        "nonzero_selection_count": int(
                            np.count_nonzero(sweep_nonzero)
                        ),
                        "nonzero_selection_rate": float(
                            np.mean(sweep_nonzero)
                        ),
                        "nonzero_true_improvement_precision": (
                            float(
                                np.mean(
                                    sweep_true_improvement[sweep_nonzero]
                                )
                            )
                            if np.any(sweep_nonzero)
                            else float("nan")
                        ),
                        "selected_reward_improvement_mean": float(
                            np.mean(sweep_rewards - center_rewards)
                        ),
                        "selected_joint_success_rate": float(
                            np.mean(sweep_joint)
                        ),
                        "selected_failure_rate": float(
                            np.mean(sweep_failure)
                        ),
                    }
                )

    probe_flat = np.flatnonzero(
        replay.local_probe_group_valid[:row_count]
        & (
            replay.local_probe_best_reward[:row_count]
            - replay.rewards[:row_count]
            >= minimum_physical_reward_difference
        )
    )
    if len(probe_flat):
        candidate_rows, candidate_envs = np.unravel_index(
            probe_flat,
            (row_count, replay.n_envs),
        )
        probe_flat = probe_flat[
            policy._probe_holdout_mask(
                replay.local_probe_task_index[
                    candidate_rows,
                    candidate_envs,
                ]
            )
        ]
    if len(probe_flat) > max_probe_samples:
        probe_flat = split_rng.choice(
            probe_flat,
            size=max_probe_samples,
            replace=False,
        )
    pair_agreements: list[np.ndarray] = []
    q1_agreements: list[np.ndarray] = []
    q2_agreements: list[np.ndarray] = []
    pessimistic_agreements: list[np.ndarray] = []
    mean_q_agreements: list[np.ndarray] = []
    probe_errors: list[np.ndarray] = []
    if len(probe_flat):
        probe_rows, probe_envs = np.unravel_index(
            probe_flat,
            (row_count, replay.n_envs),
        )
        with torch.no_grad():
            for start in range(0, len(probe_flat), batch_size):
                selection = slice(start, start + batch_size)
                batch_observations = torch.as_tensor(
                    replay.observations[
                        probe_rows[selection], probe_envs[selection]
                    ],
                    dtype=torch.float32,
                    device=policy.device,
                )
                probe_actions = torch.as_tensor(
                    replay.actions[
                        probe_rows[selection], probe_envs[selection]
                    ],
                    dtype=torch.float32,
                    device=policy.device,
                )
                best_actions = torch.as_tensor(
                    replay.local_probe_best_action[
                        probe_rows[selection], probe_envs[selection]
                    ],
                    dtype=torch.float32,
                    device=policy.device,
                )
                rewards = torch.as_tensor(
                    replay.rewards[
                        probe_rows[selection], probe_envs[selection]
                    ],
                    dtype=torch.float32,
                    device=policy.device,
                )
                probe_q = policy._critic_values(batch_observations, probe_actions)
                best_q = policy._critic_values(batch_observations, best_actions)
                q1_margin = (best_q[0] - probe_q[0]).reshape(-1)
                q2_margin = (best_q[1] - probe_q[1]).reshape(-1)
                q1_agreements.append((q1_margin > 0.0).cpu().numpy())
                q2_agreements.append((q2_margin > 0.0).cpu().numpy())
                pair_agreements.append(
                    ((q1_margin > 0.0) & (q2_margin > 0.0)).cpu().numpy()
                )
                pessimistic_agreements.append(
                    (
                        torch.minimum(best_q[0], best_q[1]).reshape(-1)
                        > torch.minimum(probe_q[0], probe_q[1]).reshape(-1)
                    )
                    .cpu()
                    .numpy()
                )
                mean_q_agreements.append(
                    (
                        (best_q[0] + best_q[1]).reshape(-1)
                        > (probe_q[0] + probe_q[1]).reshape(-1)
                    )
                    .cpu()
                    .numpy()
                )
                minimum_probe_q = torch.minimum(probe_q[0], probe_q[1]).reshape(-1)
                probe_errors.append(
                    torch.abs(minimum_probe_q - rewards).cpu().numpy()
                )
    pair_agreement = np.concatenate(pair_agreements)
    q1_agreement = np.concatenate(q1_agreements)
    q2_agreement = np.concatenate(q2_agreements)
    pessimistic_agreement = np.concatenate(pessimistic_agreements)
    mean_q_agreement = np.concatenate(mean_q_agreements)
    probe_error = np.concatenate(probe_errors)
    return {
        "held_out_task_count": int(held_out_count),
        "pairwise_eligible_count": int(len(pair_agreement)),
        "pairwise_both_critics_agreement": float(np.mean(pair_agreement)),
        "pairwise_q1_agreement": float(np.mean(q1_agreement)),
        "pairwise_q2_agreement": float(np.mean(q2_agreement)),
        "pairwise_pessimistic_q_agreement": float(
            np.mean(pessimistic_agreement)
        ),
        "pairwise_mean_q_agreement": float(np.mean(mean_q_agreement)),
        "probe_q_mae": float(np.mean(probe_error)),
        "candidate_exact_best_selection_rate": float(np.mean(exact_selection)),
        "candidate_nonzero_selection_rate": float(
            selected_improved.float().mean().item()
        ),
        "candidate_nonzero_selection_count": nonzero_count,
        "candidate_nonzero_true_improvement_count": true_improvement_count,
        "candidate_nonzero_true_improvement_precision": precision,
        "candidate_nonzero_true_improvement_precision_lower_95": (
            precision_lower_bound
        ),
        "candidate_selected_physical_safe_rate": float(
            np.mean(selected_physical_safe)
        ),
        "candidate_center_physical_safe_rate": float(
            np.mean(center_physical_safe)
        ),
        "candidate_selected_physical_safe_improvement_mean": (
            safe_improvement_mean
        ),
        "candidate_selected_physical_safe_improvement_standard_error": (
            safe_improvement_se
        ),
        "candidate_selected_physical_failure_rate": float(
            np.mean(selected_physical_failure)
        ),
        "candidate_center_physical_failure_rate": float(
            np.mean(center_physical_failure)
        ),
        "candidate_selected_physical_failure_increase_mean": (
            failure_increase_mean
        ),
        "candidate_selected_physical_failure_increase_standard_error": (
            failure_increase_se
        ),
        "candidate_selected_physical_joint_success_rate": float(
            np.mean(selected_physical_joint)
        ),
        "candidate_center_physical_joint_success_rate": float(
            np.mean(center_physical_joint)
        ),
        "candidate_selected_physical_joint_success_improvement_mean": (
            joint_improvement_mean
        ),
        "candidate_selected_physical_joint_success_improvement_standard_error": (
            joint_improvement_se
        ),
        "candidate_selected_physical_reward_mean": float(
            np.mean(selected_physical_rewards)
        ),
        "candidate_center_physical_reward_mean": float(
            np.mean(center_rewards)
        ),
        "candidate_selected_physical_reward_improvement_mean": (
            reward_improvement_mean
        ),
        "candidate_selected_physical_reward_improvement_standard_error": (
            reward_improvement_se
        ),
        "candidate_threshold_sweep": threshold_sweep,
        "candidate_selected_disagreement_mean": float(
            selected_disagreement.mean().item()
        ),
        "candidate_selected_minimum_q_mean": float(
            selected_minimum_q.mean().item()
        ),
        "physical_improvement_available_rate": float(
            np.mean(
                physical_best_rewards
                >= center_rewards + minimum_physical_reward_difference
            )
        ),
        "minimum_physical_reward_difference": float(
            minimum_physical_reward_difference
        ),
    }


def _predict_td3_actions(
    policy: TD3Policy,
    observations: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            tensor = torch.as_tensor(
                observations[start : start + batch_size],
                dtype=torch.float32,
                device=policy.device,
            )
            predictions.append(
                policy.actor(tensor).cpu().numpy()
            )
    return np.concatenate(predictions, axis=0).astype(np.float64)


def _td3_bc_metrics(
    policy: TD3Policy,
    observations: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    angle_weight: float,
    speed_weight: float,
) -> tuple[float, float, float, float, float]:
    predictions = _predict_td3_actions(policy, observations, batch_size=batch_size)
    errors = predictions - np.asarray(targets, dtype=np.float64)
    weighted_squared_error = angle_weight * np.square(
        errors[:, 0]
    ) + speed_weight * np.square(errors[:, 1])
    angle_error_deg = np.abs(errors[:, 0]) * math.degrees(MAX_ANGLE_RESIDUAL)
    speed_error = np.abs(errors[:, 1]) * 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    metrics = (
        float(np.mean(weighted_squared_error)),
        float(np.mean(angle_error_deg)),
        float(np.mean(speed_error)),
        float(np.percentile(angle_error_deg, 95)),
        float(np.percentile(speed_error, 95)),
    )
    if not np.all(np.isfinite(metrics)):
        raise FloatingPointError("TD3 behavior-cloning metrics are non-finite.")
    return metrics


def td3_behavior_cloning_metrics(
    policy: TD3Policy,
    dataset: Any,
    *,
    batch_size: int,
    angle_weight: float,
    speed_weight: float,
) -> dict[str, float | int]:
    """Return action-space BC accuracy on an independent task library."""

    observations, targets = generated_behavior_cloning_data(dataset)
    metrics = _td3_bc_metrics(
        policy,
        observations,
        targets,
        batch_size=min(batch_size, len(dataset)),
        angle_weight=angle_weight,
        speed_weight=speed_weight,
    )
    return {
        "sample_count": int(len(dataset)),
        "loss": metrics[0],
        "angle_mae_deg": metrics[1],
        "speed_mae_mps": metrics[2],
        "angle_p95_deg": metrics[3],
        "speed_p95_mps": metrics[4],
    }


def behavior_clone_td3_policy(
    policy: TD3Policy,
    dataset: Any,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    final_learning_rate: float,
    angle_weight: float = 4.0,
    speed_weight: float = 4.0,
    seed: int = 0,
    max_grad_norm: float = 1.0,
) -> BehaviorCloningReport:
    """Fit the deterministic TD3 actor to exact feasible task actions."""

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("TD3 behavior-cloning epochs and batch size must be positive.")
    for name, value in (
        ("learning_rate", learning_rate),
        ("final_learning_rate", final_learning_rate),
        ("angle_weight", angle_weight),
        ("speed_weight", speed_weight),
        ("max_grad_norm", max_grad_norm),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"TD3 behavior-cloning {name} must be positive and finite.")
    observations, targets = generated_behavior_cloning_data(dataset)
    effective_batch_size = min(int(batch_size), len(dataset))
    initial = _td3_bc_metrics(
        policy,
        observations,
        targets,
        batch_size=effective_batch_size,
        angle_weight=angle_weight,
        speed_weight=speed_weight,
    )
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=learning_rate)
    weights = torch.as_tensor(
        [angle_weight, speed_weight],
        dtype=torch.float32,
        device=policy.device,
    )
    rng = np.random.default_rng(seed)
    policy.set_training_mode(True)
    for epoch in range(epochs):
        progress = epoch / max(epochs - 1, 1)
        epoch_learning_rate = learning_rate * (
            final_learning_rate / learning_rate
        ) ** progress
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = epoch_learning_rate
        indices = rng.permutation(len(dataset))
        for start in range(0, len(dataset), effective_batch_size):
            batch_indices = indices[start : start + effective_batch_size]
            observation_tensor = torch.as_tensor(
                observations[batch_indices],
                dtype=torch.float32,
                device=policy.device,
            )
            target_tensor = torch.as_tensor(
                targets[batch_indices],
                dtype=torch.float32,
                device=policy.device,
            )
            predictions = policy.actor(observation_tensor)
            loss = torch.mean(
                torch.sum(weights * torch.square(predictions - target_tensor), dim=1)
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("TD3 behavior-cloning loss became non-finite.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.actor.parameters(), max_grad_norm)
            optimizer.step()
    final = _td3_bc_metrics(
        policy,
        observations,
        targets,
        batch_size=effective_batch_size,
        angle_weight=angle_weight,
        speed_weight=speed_weight,
    )
    return BehaviorCloningReport(
        sample_count=len(dataset),
        epochs=epochs,
        initial_loss=initial[0],
        final_loss=final[0],
        initial_angle_mae_deg=initial[1],
        final_angle_mae_deg=final[1],
        initial_speed_mae_mps=initial[2],
        final_speed_mae_mps=final[2],
        final_angle_p95_deg=final[3],
        final_speed_p95_mps=final[4],
    )


class TD3CheckpointMidLevelPolicy(CheckpointMidLevelPolicy):
    """Adapt a TD3+BC checkpoint to the mid-level policy contracts."""

    model_class = SingleStepTD3BC
    algorithm_label = "td3_bc"


def record_post_update_replay_diagnostics(policy: SingleStepTD3BC) -> None:
    """Record replay state after the current vector transition and updates."""

    replay = policy.replay_buffer
    if not isinstance(replay, SingleStepCuePositionHerReplayBuffer):
        return
    stored_count = replay.size() * replay.n_envs
    eligible_count = replay.eligible_transition_count
    policy.logger.record("replay/stored_transitions", stored_count)
    policy.logger.record("replay/her_eligible_transitions", eligible_count)
    policy.logger.record(
        "replay/her_eligible_fraction",
        eligible_count / max(stored_count, 1),
    )
    policy.logger.record("replay/her_sample_ratio", replay.her_ratio)
    policy.logger.record("replay/success_sample_ratio", replay.success_ratio)
    policy.logger.record("replay/failure_sample_ratio", replay.failure_ratio)
    policy.logger.record(
        "replay/local_probe_sample_ratio",
        replay.local_probe_ratio,
    )
    policy.logger.record(
        "replay/successful_transitions",
        replay.successful_transition_count,
    )
    policy.logger.record(
        "replay/failure_transitions",
        replay.failure_transition_count,
    )
    policy.logger.record(
        "replay/local_probe_transitions",
        replay.local_probe_transition_count,
    )
    policy.logger.record(
        "replay/local_probe_grouped_transitions",
        replay.local_probe_group_count,
    )
    policy.logger.record(
        "replay/local_probe_group_anchors",
        replay.local_probe_group_anchor_count,
    )
    for name, count in replay.last_sample_composition.items():
        policy.logger.record(f"replay/last_sample_{name}", count)


class PostUpdateTrainingHook:
    """Flush current metrics and save periodic state after gradient updates."""

    def __init__(
        self,
        checkpoint_base: Path,
        *,
        checkpoint_every: int,
        initial_timesteps: int,
    ) -> None:
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive.")
        if initial_timesteps < 0:
            raise ValueError("initial_timesteps must be non-negative.")
        base = Path(checkpoint_base)
        self.checkpoint_base = (
            base.with_suffix("") if base.suffix == ".zip" else base
        )
        self.checkpoint_every = int(checkpoint_every)
        self.next_checkpoint_timestep = (
            initial_timesteps // self.checkpoint_every + 1
        ) * self.checkpoint_every

    def _checkpoint_paths(self, timestep: int) -> tuple[Path, Path]:
        parent = self.checkpoint_base.parent
        name = self.checkpoint_base.name
        checkpoint = parent / f"{name}_step_{timestep}_steps"
        replay = parent / f"{name}_step_replay_buffer_{timestep}_steps.pkl"
        return checkpoint, replay

    def __call__(self, policy: SingleStepTD3BC) -> None:
        record_post_update_replay_diagnostics(policy)
        # SB3 normally dumps inside collect_rollouts(), before train().  Dumping
        # here keeps rollout, replay, and optimizer metrics on the same row and
        # guarantees that the final update is emitted.
        policy.dump_logs()

        timestep = int(policy.num_timesteps)
        if timestep < self.next_checkpoint_timestep:
            return
        while self.next_checkpoint_timestep <= timestep:
            self.next_checkpoint_timestep += self.checkpoint_every
        checkpoint, replay = self._checkpoint_paths(timestep)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        policy.save(str(checkpoint))
        policy.save_replay_buffer(replay)
        stored_count = (
            policy.replay_buffer.size() * policy.replay_buffer.n_envs
            if policy.replay_buffer is not None
            else 0
        )
        print(
            f"post_update_checkpoint={checkpoint}.zip "
            f"timesteps={timestep} replay_transitions={stored_count} "
            f"critic_updates={policy._n_updates} "
            f"actor_updates={policy._actor_updates}",
            flush=True,
        )


def replay_buffer_path(checkpoint: Path) -> Path:
    """Return the canonical replay-buffer companion path for a checkpoint."""

    base = checkpoint.with_suffix("") if checkpoint.suffix == ".zip" else checkpoint
    return Path(f"{base}.replay_buffer.pkl")


def resolve_replay_buffer_path(checkpoint: Path) -> Path:
    """Resolve final or SB3 periodic-checkpoint replay-buffer naming."""

    canonical = replay_buffer_path(checkpoint)
    if canonical.exists():
        return canonical
    stem = checkpoint.with_suffix("").name
    marker = "_steps"
    if stem.endswith(marker):
        prefix_and_step = stem[: -len(marker)]
        prefix, separator, step = prefix_and_step.rpartition("_")
        if separator and step.isdigit():
            periodic = checkpoint.with_name(
                f"{prefix}_replay_buffer_{step}_steps.pkl"
            )
            if periodic.exists():
                return periodic
    return canonical
