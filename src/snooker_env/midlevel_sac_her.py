"""Single-step deterministic TD3+BC and cue-position HER support."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.td3.policies import Actor, TD3Policy
import torch
from torch import nn
from torch.nn import functional as F

from snooker_env.midlevel_ppo import (
    BehaviorCloningReport,
    CheckpointMidLevelPolicy,
    generated_behavior_cloning_data,
)
from snooker_env.midlevel_offline_curves import (
    OfflineActorBatch,
    OfflineSpeedCurveDataset,
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
SINGLE_STEP_TD3_VERSION = "offline-curve-her-structured-td3-bc-v8"
MIDLEVEL_GEOMETRIC_FEATURE_VERSION = "relative-shot-geometry-v1"
MIDLEVEL_GEOMETRIC_FEATURE_DIM = 47
HINDSIGHT_SUCCESS_REWARD = MAX_TERMINAL_REWARD
TARGET_STOP_OBSERVATION_SLICE = slice(6, 8)
OFFLINE_HARD_TASK_METRICS = (
    "canonical_speed_error",
    "success_interval_distance",
)
SAFE_CANDIDATE_CLASSIFIER_VERSION = "bc-relative-safe-set-ensemble-v1"
STRUCTURED_SPEED_BC_VERSION = "pocket-head-supervised-hindsight-bc-v2"
STRUCTURED_SPEED_ANGLE_MODES = ("reference", "zero")


@dataclass(frozen=True)
class OfflineCurveActorReport:
    """Reconstruction and physical-surrogate metrics for offline Actor fitting."""

    sample_count: int
    exact_hindsight_count: int
    epochs: int
    gradient_updates: int
    initial_loss: float
    final_loss: float
    initial_angle_mae_deg: float
    final_angle_mae_deg: float
    initial_speed_mae_mps: float
    final_speed_mae_mps: float
    final_angle_p95_deg: float
    final_speed_p95_mps: float
    initial_estimated_stop_mae_m: float
    final_estimated_stop_mae_m: float
    final_estimated_stop_p90_m: float
    hindsight_fraction: float
    physical_loss_weight: float
    sensitivity_minimum: float
    sensitivity_maximum: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "sample_count": self.sample_count,
            "exact_hindsight_count": self.exact_hindsight_count,
            "epochs": self.epochs,
            "gradient_updates": self.gradient_updates,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "initial_angle_mae_deg": self.initial_angle_mae_deg,
            "final_angle_mae_deg": self.final_angle_mae_deg,
            "initial_speed_mae_mps": self.initial_speed_mae_mps,
            "final_speed_mae_mps": self.final_speed_mae_mps,
            "final_angle_p95_deg": self.final_angle_p95_deg,
            "final_speed_p95_mps": self.final_speed_p95_mps,
            "initial_estimated_stop_mae_m": self.initial_estimated_stop_mae_m,
            "final_estimated_stop_mae_m": self.final_estimated_stop_mae_m,
            "final_estimated_stop_p90_m": self.final_estimated_stop_p90_m,
            "hindsight_fraction": self.hindsight_fraction,
            "physical_loss_weight": self.physical_loss_weight,
            "sensitivity_minimum": self.sensitivity_minimum,
            "sensitivity_maximum": self.sensitivity_maximum,
        }


@dataclass(frozen=True)
class StructuredSpeedBCReport:
    """Audit trail for pure supervised, speed-focused hindsight BC."""

    task_count: int
    legal_curve_sample_count: int
    legal_curve_samples_per_pocket: tuple[int, ...]
    canonical_anchor_count: int
    total_supervised_sample_count: int
    epochs: int
    gradient_updates: int
    angle_mode: str
    speed_trunk_frozen: bool
    sensitivity_estimator: str
    initial_canonical_speed_mae_mps: float
    final_canonical_speed_mae_mps: float
    final_canonical_speed_p95_mps: float
    final_canonical_angle_mae_deg: float
    final_canonical_angle_p95_deg: float
    initial_hindsight_speed_mae_mps: float
    final_hindsight_speed_mae_mps: float
    final_hindsight_speed_p95_mps: float
    final_hindsight_local_stop_proxy_mae_m: float
    final_hindsight_local_stop_proxy_p90_m: float
    canonical_anchor_weight: float
    middle_pocket_weight: float
    sensitivity_weight_minimum: float
    sensitivity_weight_maximum: float
    speed_error_scale_mps: float
    sensitivity_loss_weight: float
    sensitivity_distance_scale_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": STRUCTURED_SPEED_BC_VERSION,
            "task_count": self.task_count,
            "legal_curve_sample_count": self.legal_curve_sample_count,
            "legal_curve_samples_per_pocket": list(
                self.legal_curve_samples_per_pocket
            ),
            "pocket_head_count": len(POCKET_POSITIONS),
            "critic_gradient_updates": 0,
            "canonical_anchor_count": self.canonical_anchor_count,
            "total_supervised_sample_count": self.total_supervised_sample_count,
            "epochs": self.epochs,
            "gradient_updates": self.gradient_updates,
            "angle_mode": self.angle_mode,
            "speed_trunk_frozen": self.speed_trunk_frozen,
            "sensitivity_estimator": self.sensitivity_estimator,
            "initial_canonical_speed_mae_mps": (
                self.initial_canonical_speed_mae_mps
            ),
            "final_canonical_speed_mae_mps": (
                self.final_canonical_speed_mae_mps
            ),
            "final_canonical_speed_p95_mps": (
                self.final_canonical_speed_p95_mps
            ),
            "final_canonical_angle_mae_deg": (
                self.final_canonical_angle_mae_deg
            ),
            "final_canonical_angle_p95_deg": (
                self.final_canonical_angle_p95_deg
            ),
            "initial_hindsight_speed_mae_mps": (
                self.initial_hindsight_speed_mae_mps
            ),
            "final_hindsight_speed_mae_mps": (
                self.final_hindsight_speed_mae_mps
            ),
            "final_hindsight_speed_p95_mps": (
                self.final_hindsight_speed_p95_mps
            ),
            "final_hindsight_local_stop_proxy_mae_m": (
                self.final_hindsight_local_stop_proxy_mae_m
            ),
            "final_hindsight_local_stop_proxy_p90_m": (
                self.final_hindsight_local_stop_proxy_p90_m
            ),
            "canonical_anchor_weight": self.canonical_anchor_weight,
            "middle_pocket_weight": self.middle_pocket_weight,
            "sensitivity_weight_minimum": self.sensitivity_weight_minimum,
            "sensitivity_weight_maximum": self.sensitivity_weight_maximum,
            "speed_error_scale_mps": self.speed_error_scale_mps,
            "sensitivity_loss_weight": self.sensitivity_loss_weight,
            "sensitivity_distance_scale_m": (
                self.sensitivity_distance_scale_m
            ),
        }


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


class StructuredSpeedActor(BasePolicy):
    """Frozen angle behavior plus six pocket-specific inverse-speed heads.

    The reference angle network is architecturally identical to the previous
    canonical TD3 Actor and is never optimized here.  This makes the learned
    angle and exact-zero variants a controlled A/B: both train precisely the
    same speed model, and only their deployed angle differs.
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        *,
        angle_mode: str = "reference",
        pocket_head_count: int = len(POCKET_POSITIONS),
        angle_reference_net_arch: tuple[int, ...] = (512, 512, 256),
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )
        if angle_mode not in STRUCTURED_SPEED_ANGLE_MODES:
            raise ValueError(
                f"Structured speed angle mode must be one of "
                f"{STRUCTURED_SPEED_ANGLE_MODES}."
            )
        if pocket_head_count != len(POCKET_POSITIONS):
            raise ValueError("Structured speed Actor requires six pocket heads.")
        if not net_arch or any(size <= 0 for size in net_arch):
            raise ValueError("Structured speed hidden sizes must be positive.")
        if not angle_reference_net_arch or any(
            size <= 0 for size in angle_reference_net_arch
        ):
            raise ValueError("Angle-reference hidden sizes must be positive.")
        self.net_arch = list(net_arch)
        self.features_dim = int(features_dim)
        self.activation_fn = activation_fn
        self.angle_mode = str(angle_mode)
        self.pocket_head_count = int(pocket_head_count)
        self.angle_reference_net_arch = tuple(angle_reference_net_arch)

        # Instantiate exactly the architecture used by the v10 canonical BC.
        # The trained state is copied explicitly by the supervised trainer.
        self.angle_reference = Actor(
            observation_space,
            action_space,
            list(self.angle_reference_net_arch),
            MidLevelGeometricFeatures(observation_space),
            MIDLEVEL_GEOMETRIC_FEATURE_DIM,
            activation_fn,
            normalize_images,
        )
        for parameter in self.angle_reference.parameters():
            parameter.requires_grad_(False)
        self.angle_reference.set_training_mode(False)

        layers: list[nn.Module] = []
        input_size = self.features_dim
        for hidden_size in self.net_arch:
            layers.extend((nn.Linear(input_size, hidden_size), activation_fn()))
            input_size = hidden_size
        self.speed_trunk = nn.Sequential(*layers)
        self.speed_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(input_size, 1), nn.Tanh())
                for _ in range(self.pocket_head_count)
            ]
        )
        normalized_pockets = np.stack(
            [POCKET_POSITIONS[name][:2] for name in sorted(POCKET_POSITIONS)]
        ).astype(np.float32)
        normalized_pockets /= np.asarray(
            (OBSERVATION_X_SCALE, OBSERVATION_Y_SCALE),
            dtype=np.float32,
        )
        self.register_buffer(
            "normalized_pocket_positions",
            torch.as_tensor(normalized_pockets),
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "net_arch": self.net_arch,
                "features_dim": self.features_dim,
                "activation_fn": self.activation_fn,
                "features_extractor": self.features_extractor,
                "angle_mode": self.angle_mode,
                "pocket_head_count": self.pocket_head_count,
                "angle_reference_net_arch": self.angle_reference_net_arch,
            }
        )
        return data

    def load_angle_reference_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Install and freeze a canonical-BC Actor as the angle controller."""

        self.angle_reference.load_state_dict(state_dict, strict=True)
        for parameter in self.angle_reference.parameters():
            parameter.requires_grad_(False)
        self.angle_reference.set_training_mode(False)

    def initialize_speed_from_actor(self, source_actor: Actor) -> None:
        """Copy one canonical BC speed mapping into every pocket head.

        The shared feature extractor and hidden layers have the same shapes as
        the previous Actor.  Copying the speed row of its final layer into all
        six heads makes the structured Actor reproduce the source BC speed to
        float32 evaluation precision before hindsight fine-tuning begins.
        """

        source_linears = [
            layer
            for layer in source_actor.mu
            if isinstance(layer, nn.Linear)
        ]
        target_linears = [
            layer for layer in self.speed_trunk if isinstance(layer, nn.Linear)
        ]
        if len(source_linears) != len(target_linears) + 1:
            raise ValueError(
                "Reference and structured speed hidden architectures differ."
            )
        self.features_extractor.load_state_dict(
            source_actor.features_extractor.state_dict(),
            strict=True,
        )
        with torch.no_grad():
            for source, target in zip(
                source_linears[:-1],
                target_linears,
                strict=True,
            ):
                target.weight.copy_(source.weight)
                target.bias.copy_(source.bias)
            source_output = source_linears[-1]
            if source_output.out_features != 2:
                raise ValueError("Reference Actor does not emit angle and speed.")
            for head in self.speed_heads:
                output = head[0]
                if not isinstance(output, nn.Linear):
                    raise TypeError("Structured speed head output is not linear.")
                output.weight.copy_(source_output.weight[1:2])
                output.bias.copy_(source_output.bias[1:2])

    def initialize_speed_from_reference(self) -> None:
        """Use the frozen angle-reference Actor as the speed initializer."""

        self.initialize_speed_from_actor(self.angle_reference)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.angle_mode == "reference":
            with torch.no_grad():
                angle = self.angle_reference(observations)[:, :1]
        else:
            angle = torch.zeros(
                (len(observations), 1),
                dtype=observations.dtype,
                device=observations.device,
            )
        features = self.extract_features(observations, self.features_extractor)
        latent = self.speed_trunk(features)
        speeds = torch.stack(
            [head(latent) for head in self.speed_heads],
            dim=1,
        )
        pocket_xy = observations[:, 4:6]
        squared_distance = torch.sum(
            torch.square(
                pocket_xy[:, None, :]
                - self.normalized_pocket_positions[None, :, :]
            ),
            dim=2,
        )
        pocket_indices = torch.argmin(squared_distance, dim=1)
        rows = torch.arange(len(observations), device=observations.device)
        speed = speeds[rows, pocket_indices]
        return torch.cat((angle, speed), dim=1)

    def _predict(
        self,
        observation: Any,
        deterministic: bool = False,
    ) -> torch.Tensor:
        del deterministic
        return self(observation)

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)
        self.angle_reference.set_training_mode(False)


class StructuredSpeedTD3Policy(ConservativeResidualTD3Policy):
    """TD3 checkpoint container whose deployed Actor is pure supervised BC."""

    def __init__(
        self,
        *args: Any,
        angle_mode: str = "reference",
        pocket_head_count: int = len(POCKET_POSITIONS),
        angle_reference_net_arch: tuple[int, ...] = (512, 512, 256),
        **kwargs: Any,
    ) -> None:
        self.structured_angle_mode = str(angle_mode)
        self.structured_pocket_head_count = int(pocket_head_count)
        self.structured_angle_reference_net_arch = tuple(
            angle_reference_net_arch
        )
        super().__init__(*args, **kwargs)

    def make_actor(
        self,
        features_extractor: BaseFeaturesExtractor | None = None,
    ) -> StructuredSpeedActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs,
            features_extractor,
        )
        return StructuredSpeedActor(
            **actor_kwargs,
            angle_mode=self.structured_angle_mode,
            pocket_head_count=self.structured_pocket_head_count,
            angle_reference_net_arch=(
                self.structured_angle_reference_net_arch
            ),
        ).to(self.device)

    def install_angle_reference(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Copy one frozen angle reference into actor, target, and snapshot."""

        actors = (self.actor, self.actor_target, self.reference_actor)
        for actor in actors:
            if not isinstance(actor, StructuredSpeedActor):
                raise TypeError("Structured policy constructed a non-structured Actor.")
            actor.load_angle_reference_state_dict(state_dict)
            actor.initialize_speed_from_reference()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.snapshot_reference_actor()

    def install_speed_reference(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Install a separate canonical BC speed initializer.

        The frozen angle Actor remains untouched, so choosing an expanded-data
        speed initializer cannot silently change the controlled angle A/B.
        """

        source_actor = Actor(
            self.observation_space,
            self.action_space,
            list(self.structured_angle_reference_net_arch),
            MidLevelGeometricFeatures(self.observation_space),
            MIDLEVEL_GEOMETRIC_FEATURE_DIM,
            self.activation_fn,
            self.normalize_images,
        ).to(self.device)
        source_actor.load_state_dict(state_dict, strict=True)
        source_actor.set_training_mode(False)
        actors = (self.actor, self.actor_target, self.reference_actor)
        for actor in actors:
            if not isinstance(actor, StructuredSpeedActor):
                raise TypeError("Structured policy constructed a non-structured Actor.")
            actor.initialize_speed_from_actor(source_actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.snapshot_reference_actor()


STRUCTURED_CURVE_EVENT_NAMES = (
    "correct_pot",
    "safe",
    "cue_scratch",
    "stopped",
    "timed_out",
    "wrong_pocket",
    "numerical_failure",
    "joint_success",
)


class StructuredCurveCritic(nn.Module):
    """Predict reward, cue endpoint, and terminal events for one shot."""

    def __init__(
        self,
        observation_space: Any,
        *,
        hidden_sizes: tuple[int, ...] = (512, 512, 256),
    ) -> None:
        super().__init__()
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("Structured Critic hidden sizes must be positive.")
        self.features = MidLevelGeometricFeatures(observation_space)
        layers: list[nn.Module] = []
        input_size = MIDLEVEL_GEOMETRIC_FEATURE_DIM + 2
        for hidden_size in hidden_sizes:
            layers.extend(
                (
                    nn.Linear(input_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.SiLU(),
                )
            )
            input_size = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.reward_head = nn.Linear(input_size, 1)
        self.cue_delta_head = nn.Linear(input_size, 2)
        self.event_head = nn.Linear(input_size, len(STRUCTURED_CURVE_EVENT_NAMES))

    def forward(
        self,
        observations: torch.Tensor,
        action_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(observations)
        latent = self.trunk(torch.cat((features, action_features), dim=1))
        return (
            self.reward_head(latent),
            self.cue_delta_head(latent),
            self.event_head(latent),
        )


class SafeCandidateClassifier(nn.Module):
    """Classify a BC-relative speed grid with optional pocket-specific heads.

    A scalar regression target averages incompatible safe speeds when the
    one-shot dynamics are discontinuous.  This module instead emits one logit
    for every residual-speed candidate.  Six heads can isolate the distinct
    corner/middle-pocket geometries while retaining a shared geometric trunk.
    """

    def __init__(
        self,
        observation_space: Any,
        *,
        candidate_count: int,
        pocket_head_count: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        if candidate_count <= 1:
            raise ValueError("Safe-candidate classifier needs multiple candidates.")
        if pocket_head_count not in (1, len(POCKET_POSITIONS)):
            raise ValueError("Safe-candidate pocket heads must be one or six.")
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("Safe-candidate hidden sizes must be positive.")
        self.candidate_count = int(candidate_count)
        self.pocket_head_count = int(pocket_head_count)
        self.features = MidLevelGeometricFeatures(observation_space)
        layers: list[nn.Module] = []
        input_size = MIDLEVEL_GEOMETRIC_FEATURE_DIM
        for hidden_size in hidden_sizes:
            layers.extend(
                (
                    nn.Linear(input_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.SiLU(),
                )
            )
            input_size = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(
            input_size,
            self.pocket_head_count * self.candidate_count,
        )
        normalized_pockets = np.stack(
            [POCKET_POSITIONS[name][:2] for name in sorted(POCKET_POSITIONS)]
        ).astype(np.float32)
        normalized_pockets /= np.asarray(
            (OBSERVATION_X_SCALE, OBSERVATION_Y_SCALE),
            dtype=np.float32,
        )
        self.register_buffer(
            "normalized_pocket_positions",
            torch.as_tensor(normalized_pockets),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        latent = self.trunk(self.features(observations))
        logits = self.head(latent).reshape(
            len(observations),
            self.pocket_head_count,
            self.candidate_count,
        )
        if self.pocket_head_count == 1:
            return logits[:, 0]
        pocket_xy = observations[:, 4:6]
        squared_distance = torch.sum(
            torch.square(
                pocket_xy[:, None, :]
                - self.normalized_pocket_positions[None, :, :]
            ),
            dim=2,
        )
        pocket_indices = torch.argmin(squared_distance, dim=1)
        batch_indices = torch.arange(len(observations), device=observations.device)
        return logits[batch_indices, pocket_indices]


class SafeCandidateEnsemble(nn.Module):
    """Independent safe-set classifiers used for conservative abstention."""

    def __init__(
        self,
        observation_space: Any,
        *,
        candidate_count: int,
        ensemble_size: int,
        pocket_head_count: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        if ensemble_size <= 0:
            raise ValueError("Safe-candidate ensemble size must be positive.")
        self.members = nn.ModuleList(
            [
                SafeCandidateClassifier(
                    observation_space,
                    candidate_count=candidate_count,
                    pocket_head_count=pocket_head_count,
                    hidden_sizes=hidden_sizes,
                )
                for _ in range(ensemble_size)
            ]
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [member(observations) for member in self.members],
            dim=0,
        )


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
        self.offline_speed_curves: OfflineSpeedCurveDataset | None = None
        self.offline_actor_supervision_weight = 0.0
        self.offline_actor_hindsight_fraction = 0.5
        self.offline_actor_batch_size = 0
        self.offline_actor_angle_weight = 1.0
        self.offline_actor_speed_weight = 8.0
        self.offline_actor_physical_loss_weight = 1.0
        self.offline_actor_margin_loss_weight = 0.0
        self.offline_actor_success_margin_m = 0.05
        self.offline_actor_success_interval_loss_weight = 0.0
        self.offline_actor_success_interval_scale_mps = 0.01
        self.offline_actor_physical_distance_scale_m = 0.05
        self.offline_actor_sensitivity_minimum = 0.25
        self.offline_actor_sensitivity_maximum = 4.0
        self.offline_actor_middle_pocket_weight = 1.0
        self.offline_actor_hard_task_weight = 1.0
        self.offline_actor_hard_task_quantile = 0.75
        self.offline_actor_hard_task_metric = "canonical_speed_error"
        self.offline_actor_hard_speed_error_threshold_mps = 0.0
        self.offline_actor_hard_task_threshold_mps = 0.0
        self.offline_actor_holdout_fraction = 0.20
        self.offline_actor_holdout_seed = 20_000
        self.offline_actor_training_mask: np.ndarray | None = None
        self.offline_actor_task_sampling_weights: np.ndarray | None = None
        self.offline_actor_hard_task_mask: np.ndarray | None = None
        self.offline_continuous_residual_inference = False
        self._offline_actor_rng = np.random.default_rng(0)
        self.safe_candidate_classifier_enabled = False
        self.safe_candidate_classifier: SafeCandidateEnsemble | None = None
        self.safe_candidate_optimizer: torch.optim.Optimizer | None = None
        self.safe_candidate_offsets_mps = np.zeros(1, dtype=np.float32)
        self.safe_candidate_ensemble_size = 1
        self.safe_candidate_pocket_head_count = 1
        self.safe_candidate_hidden_sizes = (512, 512, 256)
        self.safe_candidate_min_probability = 0.80
        self.safe_candidate_min_improvement = 0.10
        self.safe_candidate_max_disagreement = 0.10
        self.safe_candidate_uncertainty_scale = 1.0
        self.safe_candidate_residual_penalty = 0.05
        self.safe_candidate_label_tolerance_mps = 0.0025
        self.safe_candidate_unknown_weight = 0.25
        self.safe_candidate_positive_weight = 1.0
        self.safe_candidate_selection_loss_weight = 0.5
        self.safe_candidate_selection_target = "set"
        self.safe_candidate_batch_size = 2048
        self.safe_candidate_learning_rate = 3.0e-4
        self.safe_candidate_weight_decay = 1.0e-5
        self.safe_candidate_targets: np.ndarray | None = None
        self.safe_candidate_known: np.ndarray | None = None
        self.safe_candidate_reference_offsets_mps: np.ndarray | None = None
        self.safe_candidate_training_pool: np.ndarray | None = None
        self._safe_candidate_rng = np.random.default_rng(0)
        self._last_safe_candidate_approval_rate = 0.0
        self.certified_rollout_baseline_enabled = False
        self.certified_rollout_selection_version = "measured-safe-curve-v1"
        self.certified_rollout_observations: np.ndarray | None = None
        self.certified_rollout_actions: np.ndarray | None = None
        self.certified_rollout_curve_actions: np.ndarray | None = None
        self.certified_rollout_safe: np.ndarray | None = None
        self.certified_rollout_seed = 0
        self._certified_rollout_lookup: dict[bytes, int] = {}
        self._certified_rollout_rng = np.random.default_rng(0)
        self.structured_curve_critic_enabled = False
        self.structured_critic1: StructuredCurveCritic | None = None
        self.structured_critic2: StructuredCurveCritic | None = None
        self.structured_critic1_optimizer: torch.optim.Optimizer | None = None
        self.structured_critic2_optimizer: torch.optim.Optimizer | None = None
        self.structured_critic_learning_rate = 3.0e-4
        self.structured_critic_task_batch_size = 256
        self.structured_reward_weight = 1.0
        self.structured_reward_delta_weight = 1.0
        self.structured_ranking_weight = 1.0
        self.structured_ranking_temperature = 0.25
        self.structured_cue_delta_weight = 1.0
        self.structured_cue_delta_scale_m = 0.10
        self.structured_event_weight = 1.0
        self.structured_event_balance_clip = 10.0
        self.structured_critic_holdout_fraction = 0.20
        self.structured_critic_holdout_seed = 20_000
        self.structured_critic_training_indices: np.ndarray | None = None
        self.structured_event_positive_rates: np.ndarray | None = None
        self.structured_actor_gate_enabled = False
        self.structured_gate_min_safe_probability = 0.98
        self.structured_gate_max_scratch_probability = 0.02
        self.structured_gate_min_reward_improvement = 0.01
        self.structured_gate_max_reward_disagreement = 0.25
        self._structured_rng1 = np.random.default_rng(1)
        self._structured_rng2 = np.random.default_rng(2)
        self._structured_critic_updates = 0
        self._last_structured_critic_loss = 0.0
        self._last_structured_actor_gate_approval_rate = 1.0
        self._actor_updates = 0
        self._critic_warmup_updates_completed = 0
        self._post_update_hook: Callable[[SingleStepTD3BC], None] | None = None
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        """Keep process-local orchestration hooks out of model archives."""

        return [
            *super()._excluded_save_params(),
            "_post_update_hook",
            "offline_speed_curves",
            "offline_actor_training_mask",
            "offline_actor_task_sampling_weights",
            "offline_actor_hard_task_mask",
            "_offline_actor_rng",
            "safe_candidate_classifier",
            "safe_candidate_optimizer",
            "safe_candidate_targets",
            "safe_candidate_known",
            "safe_candidate_reference_offsets_mps",
            "safe_candidate_training_pool",
            "_safe_candidate_rng",
            "certified_rollout_observations",
            "certified_rollout_actions",
            "certified_rollout_curve_actions",
            "certified_rollout_safe",
            "_certified_rollout_lookup",
            "_certified_rollout_rng",
            "structured_critic1",
            "structured_critic2",
            "structured_critic1_optimizer",
            "structured_critic2_optimizer",
            "structured_critic_training_indices",
            "structured_event_positive_rates",
            "_structured_rng1",
            "_structured_rng2",
        ]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, torch_variables = super()._get_torch_save_params()
        if self.structured_curve_critic_enabled:
            state_dicts = [
                *state_dicts,
                "structured_critic1",
                "structured_critic2",
                "structured_critic1_optimizer",
                "structured_critic2_optimizer",
            ]
        if getattr(self, "safe_candidate_classifier_enabled", False):
            state_dicts = [
                *state_dicts,
                "safe_candidate_classifier",
                "safe_candidate_optimizer",
            ]
        return state_dicts, torch_variables

    def set_post_update_hook(
        self,
        hook: Callable[[SingleStepTD3BC], None] | None,
    ) -> None:
        """Run ``hook`` after one collected rollout has been fully trained."""

        self._post_update_hook = hook

    def _setup_model(self) -> None:
        super()._setup_model()
        if self.structured_curve_critic_enabled:
            self._build_structured_curve_critics()
        if getattr(self, "safe_candidate_classifier_enabled", False):
            self._build_safe_candidate_classifier()
        if self.residual_policy_enabled:
            self._apply_residual_runtime_constraints()

    def _build_safe_candidate_classifier(self) -> None:
        """Create the persisted safe-set ensemble and its optimizer."""

        self.safe_candidate_classifier = SafeCandidateEnsemble(
            self.observation_space,
            candidate_count=len(self.safe_candidate_offsets_mps),
            ensemble_size=self.safe_candidate_ensemble_size,
            pocket_head_count=self.safe_candidate_pocket_head_count,
            hidden_sizes=tuple(self.safe_candidate_hidden_sizes),
        ).to(self.device)
        self.safe_candidate_optimizer = torch.optim.AdamW(
            self.safe_candidate_classifier.parameters(),
            lr=self.safe_candidate_learning_rate,
            weight_decay=self.safe_candidate_weight_decay,
        )

    def _build_structured_curve_critics(self) -> None:
        """Create both independent structured Critic ensembles and optimizers."""

        self.structured_critic1 = StructuredCurveCritic(
            self.observation_space
        ).to(self.device)
        self.structured_critic2 = StructuredCurveCritic(
            self.observation_space
        ).to(self.device)
        self.structured_critic1_optimizer = torch.optim.AdamW(
            self.structured_critic1.parameters(),
            lr=self.structured_critic_learning_rate,
            weight_decay=1.0e-5,
        )
        self.structured_critic2_optimizer = torch.optim.AdamW(
            self.structured_critic2.parameters(),
            lr=self.structured_critic_learning_rate,
            weight_decay=1.0e-5,
        )

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
        *,
        baseline_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.residual_policy_enabled:
            return residual_actions
        if baseline_actions is None:
            policy = self._require_residual_policy()
            with torch.no_grad():
                baseline = policy.reference_actor(observations)
        else:
            baseline = baseline_actions
            if baseline.shape != residual_actions.shape:
                raise ValueError("Residual baseline action shape is inconsistent.")
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
            certified_baseline: torch.Tensor | None = None
            classifier_approved: torch.Tensor | None = None
            certified_rollout = (
                self.certified_rollout_baseline_enabled and not deterministic
            )
            if certified_rollout:
                certified_baseline = self._certified_rollout_baseline_actions(
                    observation_tensor
                )
                residual = torch.zeros_like(certified_baseline)
            elif deterministic and getattr(
                self,
                "safe_candidate_classifier_enabled",
                False,
            ):
                residual, classifier_approved, _ = (
                    self._safe_candidate_residual_tensor(observation_tensor)
                )
            else:
                residual = self.actor(observation_tensor)
            if deterministic:
                if classifier_approved is not None:
                    pass
                elif not self.offline_continuous_residual_inference:
                    residual = self._quantize_candidate_residual(residual)
                else:
                    # Keep sub-grid speed corrections from offline curve
                    # supervision, but never inject rollout exploration into
                    # deterministic validation.
                    residual = torch.cat(
                        (
                            torch.zeros_like(residual[:, 1:2]),
                            torch.clamp(residual[:, 1:2], -1.0, 1.0),
                        ),
                        dim=1,
                    )
            else:
                if certified_rollout:
                    # The selected baseline is itself one of this task's
                    # independently replayed safe curve actions.  Do not add
                    # unmeasured noise around it.
                    self._last_rollout_exploration_std = 0.0
                else:
                    exploration_std = self._current_exploration_std()
                    self._last_rollout_exploration_std = exploration_std
                    speed = torch.clamp(
                        residual[:, 1:2]
                        + exploration_std
                        * torch.randn_like(residual[:, 1:2]),
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
                baseline_actions=certified_baseline,
            )
            if certified_rollout:
                approved = torch.ones(
                    len(observation_tensor),
                    dtype=torch.bool,
                    device=self.device,
                )
            else:
                effective, approved = self._apply_structured_actor_gate(
                    observation_tensor,
                    effective,
                    baseline_actions=certified_baseline,
                    require_reward_improvement=True,
                )
            self._last_structured_actor_gate_approval_rate = float(
                approved.float().mean().item()
            )
            if classifier_approved is not None:
                self._last_safe_candidate_approval_rate = float(
                    classifier_approved.float().mean().item()
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

    def configure_offline_curve_actor_supervision(
        self,
        curves: OfflineSpeedCurveDataset,
        *,
        supervision_weight: float,
        hindsight_fraction: float,
        batch_size: int,
        angle_weight: float,
        speed_weight: float,
        physical_loss_weight: float,
        physical_distance_scale_m: float,
        sensitivity_minimum: float,
        sensitivity_maximum: float,
        holdout_fraction: float,
        holdout_seed: int,
        seed: int,
        margin_loss_weight: float = 0.0,
        success_margin_m: float = 0.05,
        success_interval_loss_weight: float = 0.0,
        success_interval_scale_mps: float = 0.01,
        middle_pocket_weight: float = 1.0,
        hard_task_weight: float = 1.0,
        hard_task_quantile: float = 0.75,
        hard_task_metric: str = "canonical_speed_error",
        continuous_residual_inference: bool = True,
    ) -> None:
        """Attach immutable exact-action/HER supervision for residual updates."""

        if not self.residual_policy_enabled:
            raise RuntimeError(
                "Configure the frozen BC residual before offline Actor supervision."
            )
        for name, value in (
            ("supervision_weight", supervision_weight),
            ("angle_weight", angle_weight),
            ("speed_weight", speed_weight),
            ("success_margin_m", success_margin_m),
            ("physical_distance_scale_m", physical_distance_scale_m),
            ("success_interval_scale_mps", success_interval_scale_mps),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Offline residual Actor {name} must be positive and finite."
                )
        for name, value in (
            ("physical_loss_weight", physical_loss_weight),
            ("margin_loss_weight", margin_loss_weight),
            ("success_interval_loss_weight", success_interval_loss_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Offline residual Actor {name} must be finite and "
                    "non-negative."
                )
        for name, value in (
            ("middle_pocket_weight", middle_pocket_weight),
            ("hard_task_weight", hard_task_weight),
        ):
            if not math.isfinite(value) or value < 1.0:
                raise ValueError(
                    f"Offline residual Actor {name} must be finite and at least one."
                )
        if not 0.0 < hard_task_quantile < 1.0:
            raise ValueError(
                "Offline residual Actor hard-task quantile must be in (0, 1)."
            )
        if hard_task_metric not in OFFLINE_HARD_TASK_METRICS:
            raise ValueError(
                "Offline residual Actor hard-task metric must be one of "
                f"{OFFLINE_HARD_TASK_METRICS}."
            )
        if not 0.0 <= hindsight_fraction <= 1.0:
            raise ValueError("Offline residual hindsight fraction must be in [0, 1].")
        if batch_size <= 0:
            raise ValueError("Offline residual Actor batch size must be positive.")
        if not 0.0 < sensitivity_minimum <= 1.0 <= sensitivity_maximum:
            raise ValueError("Offline residual sensitivity bounds must bracket one.")
        training_mask = ~curves.holdout_mask(
            fraction=holdout_fraction,
            seed=holdout_seed,
        )
        if not np.any(training_mask):
            raise ValueError("Offline residual Actor has no training tasks.")
        self.offline_speed_curves = curves
        self.offline_actor_supervision_weight = float(supervision_weight)
        self.offline_actor_hindsight_fraction = float(hindsight_fraction)
        self.offline_actor_batch_size = int(batch_size)
        self.offline_actor_angle_weight = float(angle_weight)
        self.offline_actor_speed_weight = float(speed_weight)
        self.offline_actor_physical_loss_weight = float(physical_loss_weight)
        self.offline_actor_margin_loss_weight = float(margin_loss_weight)
        self.offline_actor_success_margin_m = float(success_margin_m)
        self.offline_actor_success_interval_loss_weight = float(
            success_interval_loss_weight
        )
        self.offline_actor_success_interval_scale_mps = float(
            success_interval_scale_mps
        )
        self.offline_actor_physical_distance_scale_m = float(
            physical_distance_scale_m
        )
        self.offline_actor_sensitivity_minimum = float(sensitivity_minimum)
        self.offline_actor_sensitivity_maximum = float(sensitivity_maximum)
        self.offline_actor_middle_pocket_weight = float(middle_pocket_weight)
        self.offline_actor_hard_task_weight = float(hard_task_weight)
        self.offline_actor_hard_task_quantile = float(hard_task_quantile)
        self.offline_actor_hard_task_metric = str(hard_task_metric)
        self.offline_actor_holdout_fraction = float(holdout_fraction)
        self.offline_actor_holdout_seed = int(holdout_seed)
        self.offline_actor_training_mask = training_mask
        self._configure_offline_actor_task_sampling_weights(curves)
        self.offline_continuous_residual_inference = bool(
            continuous_residual_inference
        )
        self._offline_actor_rng = np.random.default_rng(seed)

    def _configure_offline_actor_task_sampling_weights(
        self,
        curves: OfflineSpeedCurveDataset,
    ) -> None:
        """Weight middle pockets and large frozen-BC errors without leakage."""

        training_mask = self.offline_actor_training_mask
        if training_mask is None or not np.any(training_mask):
            raise RuntimeError("Offline residual Actor training split is missing.")
        weights = np.ones(curves.task_count, dtype=np.float64)
        if self.offline_actor_middle_pocket_weight > 1.0:
            weights[curves.middle_pocket_mask] *= (
                self.offline_actor_middle_pocket_weight
            )
        reference = self._require_residual_policy().reference_actor
        predictions: list[np.ndarray] = []
        reference.set_training_mode(False)
        with torch.no_grad():
            for start in range(0, curves.task_count, 8192):
                observations = torch.as_tensor(
                    curves.observation[start : start + 8192],
                    dtype=torch.float32,
                    device=self.device,
                )
                predictions.append(reference(observations).cpu().numpy())
        reference_actions = np.concatenate(predictions, axis=0)
        speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
        predicted_offsets = (
            reference_actions[:, 1] - curves.center_action[:, 1]
        ) * speed_half_range
        hard_task_metric = getattr(
            self,
            "offline_actor_hard_task_metric",
            "canonical_speed_error",
        )
        if hard_task_metric == "canonical_speed_error":
            hard_task_score = np.abs(predicted_offsets)
        elif hard_task_metric == "success_interval_distance":
            hard_task_score = curves.success_interval_distances_mps(
                predicted_offsets,
                success_margin_m=self.offline_actor_success_margin_m,
            )
        else:
            raise RuntimeError(
                f"Unsupported offline hard-task metric: {hard_task_metric!r}."
            )
        threshold = float(
            np.quantile(
                hard_task_score[training_mask],
                self.offline_actor_hard_task_quantile,
            )
        )
        hard_task_mask = hard_task_score >= threshold
        if self.offline_actor_hard_task_weight > 1.0:
            weights[hard_task_mask] *= self.offline_actor_hard_task_weight
        self.offline_actor_hard_speed_error_threshold_mps = threshold
        self.offline_actor_hard_task_threshold_mps = threshold
        self.offline_actor_task_sampling_weights = weights.astype(np.float32)
        self.offline_actor_hard_task_mask = hard_task_mask

    def offline_actor_sampling_report(self) -> dict[str, float | int]:
        """Summarize the effective task distribution used by residual fitting."""

        curves = self.offline_speed_curves
        training_mask = self.offline_actor_training_mask
        weights = self.offline_actor_task_sampling_weights
        hard_task_mask = self.offline_actor_hard_task_mask
        if (
            curves is None
            or training_mask is None
            or weights is None
            or hard_task_mask is None
        ):
            raise RuntimeError("Offline residual Actor sampling is not configured.")
        selected_weights = weights[training_mask].astype(np.float64)
        middle = curves.middle_pocket_mask & training_mask
        total_weight = float(np.sum(selected_weights))
        squared_weight = float(np.sum(np.square(selected_weights)))
        hard = hard_task_mask & training_mask
        return {
            "training_task_count": int(np.count_nonzero(training_mask)),
            "middle_task_count": int(np.count_nonzero(middle)),
            "hard_task_count": int(np.count_nonzero(hard)),
            "middle_pocket_weight": self.offline_actor_middle_pocket_weight,
            "hard_task_weight": self.offline_actor_hard_task_weight,
            "hard_task_quantile": self.offline_actor_hard_task_quantile,
            "hard_task_metric": self.offline_actor_hard_task_metric,
            "hard_task_threshold_mps": (
                self.offline_actor_hard_task_threshold_mps
            ),
            "hard_speed_error_threshold_mps": (
                self.offline_actor_hard_speed_error_threshold_mps
            ),
            "expected_middle_sample_fraction": float(
                np.sum(weights[middle], dtype=np.float64) / total_weight
            ),
            "effective_task_count": float(
                total_weight * total_weight / squared_weight
            ),
        }

    def configure_safe_candidate_classifier(
        self,
        curves: OfflineSpeedCurveDataset,
        *,
        offsets_mps: tuple[float, ...],
        ensemble_size: int,
        pocket_head_count: int,
        hidden_sizes: tuple[int, ...],
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        positive_weight: float,
        selection_loss_weight: float,
        selection_target: str,
        unknown_weight: float,
        label_tolerance_mps: float,
        min_probability: float,
        min_improvement: float,
        max_disagreement: float,
        uncertainty_scale: float,
        residual_penalty: float,
        seed: int,
    ) -> dict[str, float | int | list[float] | str]:
        """Attach BC-relative set-valued supervision from measured curves."""

        if not self.residual_policy_enabled:
            raise RuntimeError(
                "Safe-candidate classification requires a frozen BC residual."
            )
        training_mask = self.offline_actor_training_mask
        if self.offline_speed_curves is not curves or training_mask is None:
            raise RuntimeError(
                "Configure offline curve supervision before the classifier."
            )
        offsets = np.asarray(offsets_mps, dtype=np.float64)
        if (
            offsets.ndim != 1
            or len(offsets) <= 1
            or not np.all(np.isfinite(offsets))
            or not np.all(np.diff(offsets) > 0.0)
            or np.count_nonzero(np.isclose(offsets, 0.0, atol=1.0e-10)) != 1
        ):
            raise ValueError(
                "Safe-candidate offsets must be sorted, unique, and include zero."
            )
        if float(np.max(np.abs(offsets))) > self.max_speed_residual_mps + 1.0e-8:
            raise ValueError(
                "Safe-candidate offsets exceed the configured residual bound."
            )
        if ensemble_size <= 0 or pocket_head_count not in (1, 6):
            raise ValueError("Safe-candidate ensemble/head counts are invalid.")
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("Safe-candidate hidden sizes must be positive.")
        if batch_size <= 0 or seed < 0:
            raise ValueError("Safe-candidate batch size and seed are invalid.")
        for name, value in (
            ("learning_rate", learning_rate),
            ("positive_weight", positive_weight),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Safe-candidate {name} must be positive.")
        for name, value in (
            ("weight_decay", weight_decay),
            ("selection_loss_weight", selection_loss_weight),
            ("label_tolerance_mps", label_tolerance_mps),
            ("min_probability", min_probability),
            ("min_improvement", min_improvement),
            ("max_disagreement", max_disagreement),
            ("uncertainty_scale", uncertainty_scale),
            ("residual_penalty", residual_penalty),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Safe-candidate {name} must be finite and non-negative."
                )
        if not 0.0 <= unknown_weight <= 1.0:
            raise ValueError("Safe-candidate unknown weight must be in [0, 1].")
        if selection_target not in ("set", "nearest"):
            raise ValueError(
                "Safe-candidate selection target must be 'set' or 'nearest'."
            )
        if min_probability > 1.0 or min_improvement > 1.0:
            raise ValueError("Safe-candidate probability gates cannot exceed one.")

        self.safe_candidate_offsets_mps = offsets.astype(np.float32)
        self.safe_candidate_ensemble_size = int(ensemble_size)
        self.safe_candidate_pocket_head_count = int(pocket_head_count)
        self.safe_candidate_hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.safe_candidate_batch_size = int(batch_size)
        self.safe_candidate_learning_rate = float(learning_rate)
        self.safe_candidate_weight_decay = float(weight_decay)
        self.safe_candidate_positive_weight = float(positive_weight)
        self.safe_candidate_selection_loss_weight = float(
            selection_loss_weight
        )
        self.safe_candidate_selection_target = str(selection_target)
        self.safe_candidate_unknown_weight = float(unknown_weight)
        self.safe_candidate_label_tolerance_mps = float(label_tolerance_mps)
        self.safe_candidate_min_probability = float(min_probability)
        self.safe_candidate_min_improvement = float(min_improvement)
        self.safe_candidate_max_disagreement = float(max_disagreement)
        self.safe_candidate_uncertainty_scale = float(uncertainty_scale)
        self.safe_candidate_residual_penalty = float(residual_penalty)
        self.safe_candidate_classifier_enabled = True
        self._safe_candidate_rng = np.random.default_rng(seed)
        self.safe_candidate_training_pool = np.flatnonzero(training_mask)
        self._build_safe_candidate_classifier()

        reference_offsets, targets, known = (
            self._prepare_safe_candidate_targets(curves)
        )
        train = training_mask
        zero_index = int(np.argmin(np.abs(offsets)))
        return {
            "version": SAFE_CANDIDATE_CLASSIFIER_VERSION,
            "candidate_count": int(len(offsets)),
            "offsets_mps": [float(value) for value in offsets],
            "ensemble_size": self.safe_candidate_ensemble_size,
            "pocket_head_count": self.safe_candidate_pocket_head_count,
            "training_task_count": int(np.count_nonzero(train)),
            "known_label_rate": float(np.mean(known[train])),
            "positive_label_rate": float(np.mean(targets[train])),
            "positive_rate_given_known": float(
                np.sum(targets[train]) / max(np.sum(known[train]), 1)
            ),
            "task_with_positive_candidate_rate": float(
                np.mean(np.any(targets[train], axis=1))
            ),
            "reference_safe_rate": float(np.mean(targets[train, zero_index])),
            "label_tolerance_mps": self.safe_candidate_label_tolerance_mps,
            "positive_weight": self.safe_candidate_positive_weight,
            "selection_loss_weight": (
                self.safe_candidate_selection_loss_weight
            ),
            "selection_target": self.safe_candidate_selection_target,
            "unknown_weight": self.safe_candidate_unknown_weight,
        }

    def _prepare_safe_candidate_targets(
        self,
        curves: OfflineSpeedCurveDataset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Materialize BC-relative safe labels without changing the network."""

        reference = self._require_residual_policy().reference_actor
        reference_parts: list[np.ndarray] = []
        reference.set_training_mode(False)
        with torch.no_grad():
            for start in range(0, curves.task_count, 8192):
                observations = torch.as_tensor(
                    curves.observation[start : start + 8192],
                    dtype=torch.float32,
                    device=self.device,
                )
                reference_parts.append(reference(observations).cpu().numpy())
        reference_actions = np.concatenate(reference_parts, axis=0)
        speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
        reference_offsets = (
            reference_actions[:, 1] - curves.center_action[:, 1]
        ) * speed_half_range
        offsets = self.safe_candidate_offsets_mps.astype(np.float64)
        targets = np.zeros(
            (curves.task_count, len(offsets)),
            dtype=np.bool_,
        )
        known = np.zeros_like(targets)
        minimum_offset = float(curves.offsets_mps[0])
        maximum_offset = float(curves.offsets_mps[-1])
        tolerance = self.safe_candidate_label_tolerance_mps
        for candidate_index, residual_offset in enumerate(offsets):
            absolute_offsets = reference_offsets + residual_offset
            known[:, candidate_index] = (
                (absolute_offsets >= minimum_offset - 1.0e-8)
                & (absolute_offsets <= maximum_offset + 1.0e-8)
            )
            distances = curves.success_interval_distances_mps(
                absolute_offsets,
                success_margin_m=self.offline_actor_success_margin_m,
            )
            targets[:, candidate_index] = known[:, candidate_index] & (
                distances <= tolerance + 1.0e-8
            )
        self.safe_candidate_reference_offsets_mps = reference_offsets.astype(
            np.float32
        )
        self.safe_candidate_targets = targets
        self.safe_candidate_known = known
        return reference_offsets, targets, known

    def attach_safe_candidate_curves_for_audit(
        self,
        curves: OfflineSpeedCurveDataset,
    ) -> None:
        """Restore excluded curve labels after loading a classifier checkpoint."""

        self._require_safe_candidate_classifier()
        self.offline_speed_curves = curves
        holdout = curves.holdout_mask(
            fraction=self.offline_actor_holdout_fraction,
            seed=self.offline_actor_holdout_seed,
        )
        self.offline_actor_training_mask = ~holdout
        self.safe_candidate_training_pool = np.flatnonzero(~holdout)
        self._prepare_safe_candidate_targets(curves)

    def _require_safe_candidate_classifier(self) -> SafeCandidateEnsemble:
        classifier = self.safe_candidate_classifier
        if not self.safe_candidate_classifier_enabled or classifier is None:
            raise RuntimeError("Safe-candidate classifier is not configured.")
        return classifier

    def _safe_candidate_residual_tensor(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select a candidate only when its ensemble lower bound is better."""

        classifier = self._require_safe_candidate_classifier()
        probabilities = torch.sigmoid(classifier(observations))
        means = torch.mean(probabilities, dim=0)
        disagreements = torch.std(probabilities, dim=0, unbiased=False)
        scale = self.safe_candidate_uncertainty_scale
        lower = means - scale * disagreements
        upper = means + scale * disagreements
        offsets = torch.as_tensor(
            self.safe_candidate_offsets_mps,
            dtype=observations.dtype,
            device=observations.device,
        )
        maximum = max(float(np.max(np.abs(self.safe_candidate_offsets_mps))), 1.0e-8)
        scores = lower - self.safe_candidate_residual_penalty * (
            torch.abs(offsets)[None, :] / maximum
        )
        best_indices = torch.argmax(scores, dim=1)
        batch_indices = torch.arange(len(observations), device=observations.device)
        zero_index = int(np.argmin(np.abs(self.safe_candidate_offsets_mps)))
        best_lower = lower[batch_indices, best_indices]
        best_disagreement = disagreements[batch_indices, best_indices]
        baseline_upper = upper[:, zero_index]
        approved = (
            (best_indices != zero_index)
            & (best_lower >= self.safe_candidate_min_probability)
            & (
                best_lower - baseline_upper
                >= self.safe_candidate_min_improvement
            )
            & (
                best_disagreement
                <= self.safe_candidate_max_disagreement
            )
        )
        selected_indices = torch.where(
            approved,
            best_indices,
            torch.full_like(best_indices, zero_index),
        )
        selected_offsets = offsets[selected_indices]
        normalized_speed = selected_offsets / self.max_speed_residual_mps
        residual = torch.stack(
            (torch.zeros_like(normalized_speed), normalized_speed),
            dim=1,
        )
        return residual, approved, selected_indices

    def _safe_candidate_training_loss(self) -> tuple[torch.Tensor, dict[str, float]]:
        classifier = self._require_safe_candidate_classifier()
        targets = self.safe_candidate_targets
        known = self.safe_candidate_known
        pool = self.safe_candidate_training_pool
        if targets is None or known is None or pool is None:
            raise RuntimeError("Safe-candidate training labels are unavailable.")
        losses: list[torch.Tensor] = []
        positive_probabilities: list[torch.Tensor] = []
        negative_probabilities: list[torch.Tensor] = []
        binary_losses: list[torch.Tensor] = []
        selection_losses: list[torch.Tensor] = []
        zero_index = int(np.argmin(np.abs(self.safe_candidate_offsets_mps)))
        for member in classifier.members:
            indices = self._safe_candidate_rng.choice(
                pool,
                size=self.safe_candidate_batch_size,
                replace=len(pool) < self.safe_candidate_batch_size,
            )
            observations = torch.as_tensor(
                self.offline_speed_curves.observation[indices],
                dtype=torch.float32,
                device=self.device,
            )
            labels = torch.as_tensor(
                targets[indices],
                dtype=torch.float32,
                device=self.device,
            )
            known_mask = torch.as_tensor(
                known[indices],
                dtype=torch.bool,
                device=self.device,
            )
            logits = member(observations)
            element_loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
            )
            weights = torch.where(
                known_mask,
                torch.ones_like(element_loss),
                torch.full_like(element_loss, self.safe_candidate_unknown_weight),
            )
            weights = torch.where(
                labels > 0.5,
                weights * self.safe_candidate_positive_weight,
                weights,
            )
            binary_loss = torch.sum(weights * element_loss) / torch.sum(weights)
            desired = labels > 0.5
            baseline_safe = desired[:, zero_index]
            desired = desired.clone()
            desired[baseline_safe] = False
            desired[baseline_safe, zero_index] = True
            no_safe_candidate = ~torch.any(desired, dim=1)
            desired[no_safe_candidate, zero_index] = True
            if self.safe_candidate_selection_target == "nearest":
                offsets = torch.as_tensor(
                    np.abs(self.safe_candidate_offsets_mps),
                    dtype=logits.dtype,
                    device=self.device,
                )
                distances = torch.where(
                    desired,
                    offsets[None, :],
                    torch.full_like(logits, torch.inf),
                )
                nearest = torch.argmin(distances, dim=1)
                desired = F.one_hot(
                    nearest,
                    num_classes=len(self.safe_candidate_offsets_mps),
                ).to(torch.bool)
            log_probabilities = F.log_softmax(logits, dim=1)
            masked_log_probabilities = torch.where(
                desired,
                log_probabilities,
                torch.full_like(log_probabilities, -torch.inf),
            )
            selection_loss = -torch.mean(
                torch.logsumexp(masked_log_probabilities, dim=1)
            )
            losses.append(
                binary_loss
                + self.safe_candidate_selection_loss_weight * selection_loss
            )
            binary_losses.append(binary_loss.detach())
            selection_losses.append(selection_loss.detach())
            probabilities = torch.sigmoid(logits.detach())
            if bool(torch.any(labels > 0.5)):
                positive_probabilities.append(probabilities[labels > 0.5])
            if bool(torch.any(known_mask & (labels < 0.5))):
                negative_probabilities.append(
                    probabilities[known_mask & (labels < 0.5)]
                )
        loss = torch.mean(torch.stack(losses))
        return loss, {
            "positive_probability": float(
                torch.mean(torch.cat(positive_probabilities)).item()
            ),
            "known_negative_probability": float(
                torch.mean(torch.cat(negative_probabilities)).item()
            ),
            "binary_loss": float(torch.mean(torch.stack(binary_losses)).item()),
            "selection_loss": float(
                torch.mean(torch.stack(selection_losses)).item()
            ),
        }

    def warmup_safe_candidate_classifier(
        self,
        gradient_steps: int,
        *,
        max_grad_norm: float,
    ) -> dict[str, float | int]:
        """Fit all ensemble members to the set of measured-safe candidates."""

        if gradient_steps <= 0 or max_grad_norm <= 0.0:
            raise ValueError("Safe-candidate warmup settings must be positive.")
        classifier = self._require_safe_candidate_classifier()
        optimizer = self.safe_candidate_optimizer
        if optimizer is None:
            raise RuntimeError("Safe-candidate optimizer is unavailable.")
        classifier.train()
        losses: list[float] = []
        positives: list[float] = []
        negatives: list[float] = []
        binary_losses: list[float] = []
        selection_losses: list[float] = []
        for _ in range(gradient_steps):
            loss, metrics = self._safe_candidate_training_loss()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            positives.append(metrics["positive_probability"])
            negatives.append(metrics["known_negative_probability"])
            binary_losses.append(metrics["binary_loss"])
            selection_losses.append(metrics["selection_loss"])
        classifier.eval()
        window = min(128, gradient_steps)
        return {
            "version": SAFE_CANDIDATE_CLASSIFIER_VERSION,
            "updates": int(gradient_steps),
            "learning_rate": self.safe_candidate_learning_rate,
            "initial_loss": float(np.mean(losses[:window])),
            "final_loss": float(np.mean(losses[-window:])),
            "final_positive_probability": float(np.mean(positives[-window:])),
            "final_known_negative_probability": float(
                np.mean(negatives[-window:])
            ),
            "final_binary_loss": float(np.mean(binary_losses[-window:])),
            "final_selection_loss": float(
                np.mean(selection_losses[-window:])
            ),
        }

    def safe_candidate_classifier_diagnostics(
        self,
        *,
        holdout: bool,
        batch_size: int = 4096,
    ) -> dict[str, Any]:
        """Measure paired safe-set gains/losses on a task-disjoint split."""

        curves = self.offline_speed_curves
        targets = self.safe_candidate_targets
        known = self.safe_candidate_known
        if curves is None or targets is None or known is None or batch_size <= 0:
            raise RuntimeError("Safe-candidate diagnostics are unavailable.")
        holdout_mask = curves.holdout_mask(
            fraction=self.offline_actor_holdout_fraction,
            seed=self.offline_actor_holdout_seed,
        )
        selected_mask = holdout_mask if holdout else ~holdout_mask
        tasks = np.flatnonzero(selected_mask)
        selected_parts: list[np.ndarray] = []
        approval_parts: list[np.ndarray] = []
        disagreement_parts: list[np.ndarray] = []
        classifier = self._require_safe_candidate_classifier()
        classifier.eval()
        with torch.no_grad():
            for start in range(0, len(tasks), batch_size):
                observations = torch.as_tensor(
                    curves.observation[tasks[start : start + batch_size]],
                    dtype=torch.float32,
                    device=self.device,
                )
                probabilities = torch.sigmoid(classifier(observations))
                disagreements = torch.std(
                    probabilities,
                    dim=0,
                    unbiased=False,
                )
                _, approved, selected = self._safe_candidate_residual_tensor(
                    observations
                )
                batch_indices = torch.arange(
                    len(observations),
                    device=self.device,
                )
                selected_parts.append(selected.cpu().numpy())
                approval_parts.append(approved.cpu().numpy())
                disagreement_parts.append(
                    disagreements[batch_indices, selected].cpu().numpy()
                )
        selected_indices = np.concatenate(selected_parts).astype(np.int64)
        approved = np.concatenate(approval_parts).astype(np.bool_)
        selected_disagreement = np.concatenate(disagreement_parts)
        zero_index = int(np.argmin(np.abs(self.safe_candidate_offsets_mps)))
        task_rows = np.arange(len(tasks), dtype=np.int64)
        split_targets = targets[tasks]
        split_known = known[tasks]
        baseline_safe = split_targets[:, zero_index]
        selected_safe = split_targets[task_rows, selected_indices]
        selected_known = split_known[task_rows, selected_indices]
        gains = ~baseline_safe & selected_safe
        losses = baseline_safe & ~selected_safe
        residuals = self.safe_candidate_offsets_mps[selected_indices]
        pocket_indices = curves.pocket_indices[tasks]
        per_pocket: dict[str, dict[str, int | float]] = {}
        pocket_names = np.asarray(sorted(POCKET_POSITIONS))
        for pocket_index, pocket_name in enumerate(pocket_names):
            mask = pocket_indices == pocket_index
            pocket_task_count = int(np.count_nonzero(mask))
            per_pocket[str(pocket_name)] = {
                "task_count": pocket_task_count,
                "baseline_safe_rate": float(
                    np.mean(baseline_safe[mask]) if pocket_task_count else 0.0
                ),
                "selected_safe_rate": float(
                    np.mean(selected_safe[mask]) if pocket_task_count else 0.0
                ),
                "gain_count": int(np.count_nonzero(gains & mask)),
                "loss_count": int(np.count_nonzero(losses & mask)),
            }
        changed_count = int(np.count_nonzero(approved))
        gain_count = int(np.count_nonzero(gains))
        loss_count = int(np.count_nonzero(losses))
        discordant_count = gain_count + loss_count
        return {
            "version": SAFE_CANDIDATE_CLASSIFIER_VERSION,
            "split": "holdout" if holdout else "training",
            "task_count": int(len(tasks)),
            "baseline_safe_rate": float(np.mean(baseline_safe)),
            "selected_safe_rate": float(np.mean(selected_safe)),
            "safe_rate_improvement": float(
                np.mean(selected_safe) - np.mean(baseline_safe)
            ),
            "gain_count": gain_count,
            "loss_count": loss_count,
            "net_gain_count": gain_count - loss_count,
            "paired_improvement_precision": float(
                gain_count / discordant_count if discordant_count else 1.0
            ),
            "baseline_regression_rate": float(
                loss_count / max(int(np.count_nonzero(baseline_safe)), 1)
            ),
            "approval_rate": float(np.mean(approved)),
            "changed_count": changed_count,
            "changed_safe_precision": float(
                np.mean(selected_safe[approved]) if changed_count else 1.0
            ),
            "changed_known_rate": float(
                np.mean(selected_known[approved]) if changed_count else 1.0
            ),
            "oracle_candidate_safe_rate": float(
                np.mean(np.any(split_targets, axis=1))
            ),
            "mean_abs_residual_mps": float(np.mean(np.abs(residuals))),
            "p95_abs_residual_mps": float(
                np.percentile(np.abs(residuals), 95)
            ),
            "mean_selected_disagreement": float(
                np.mean(selected_disagreement)
            ),
            "per_pocket": per_pocket,
        }

    def configure_certified_rollout_baseline(
        self,
        curves: OfflineSpeedCurveDataset,
        *,
        enabled: bool,
        seed: int = 0,
    ) -> None:
        """Use measured-safe curve actions only for stochastic rollouts.

        This is deliberately not used by deterministic inference.  It keeps
        physical exploration on independently replayed task/action pairs while
        the deployable Actor remains a pure observation-to-action map.
        """

        self.certified_rollout_baseline_enabled = bool(enabled)
        self.certified_rollout_selection_version = "measured-safe-curve-v1"
        if not enabled:
            self.certified_rollout_observations = None
            self.certified_rollout_actions = None
            self.certified_rollout_curve_actions = None
            self.certified_rollout_safe = None
            self._certified_rollout_lookup = {}
            return
        if seed < 0:
            raise ValueError("Certified rollout seed must be non-negative.")
        observations = np.ascontiguousarray(
            curves.observation,
            dtype=np.float32,
        )
        actions = np.ascontiguousarray(
            curves.center_action,
            dtype=np.float32,
        )
        lookup: dict[bytes, int] = {}
        for index, observation in enumerate(observations):
            key = observation.tobytes()
            if key in lookup:
                raise ValueError(
                    "Certified rollout observations are not uniquely keyed."
                )
            lookup[key] = index
        self.certified_rollout_observations = observations
        self.certified_rollout_actions = actions
        self.certified_rollout_curve_actions = np.ascontiguousarray(
            curves.action,
            dtype=np.float32,
        )
        self.certified_rollout_safe = np.ascontiguousarray(
            curves.safe,
            dtype=np.bool_,
        )
        self.certified_rollout_seed = int(seed)
        self._certified_rollout_lookup = lookup
        self._certified_rollout_rng = np.random.default_rng(seed)

    def _certified_rollout_baseline_actions(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        actions = self.certified_rollout_actions
        curve_actions = self.certified_rollout_curve_actions
        safe = self.certified_rollout_safe
        if (
            not self.certified_rollout_baseline_enabled
            or actions is None
            or curve_actions is None
            or safe is None
            or not self._certified_rollout_lookup
        ):
            raise RuntimeError("Certified rollout baseline is not attached.")
        host = np.ascontiguousarray(
            observations.detach().cpu().numpy(),
            dtype=np.float32,
        )
        indices: list[int] = []
        offset_indices: list[int] = []
        for observation in host:
            index = self._certified_rollout_lookup.get(observation.tobytes())
            if index is None:
                raise RuntimeError(
                    "Online rollout observation is absent from the certified "
                    "task library; refusing an uncentered exploratory action."
                )
            indices.append(index)
            safe_offsets = np.flatnonzero(safe[:, index])
            if len(safe_offsets) == 0:
                raise RuntimeError(
                    "Certified task has no measured safe rollout action."
                )
            offset_indices.append(
                int(self._certified_rollout_rng.choice(safe_offsets))
            )
        return torch.as_tensor(
            curve_actions[
                np.asarray(offset_indices, dtype=np.int64),
                np.asarray(indices, dtype=np.int64),
            ],
            dtype=observations.dtype,
            device=observations.device,
        )

    def _offline_curve_actor_supervision_loss(
        self,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        curves = self.offline_speed_curves
        if (
            curves is None
            or self.offline_actor_training_mask is None
            or self.offline_actor_batch_size <= 0
        ):
            zero = torch.zeros((), dtype=torch.float32, device=self.device)
            return zero, {
                "angle_loss": 0.0,
                "speed_loss": 0.0,
                "physical_stop_loss": 0.0,
                "physical_margin_loss": 0.0,
                "physical_range_loss": 0.0,
                "success_interval_loss": 0.0,
                "success_interval_coverage_rate": 0.0,
                "success_interval_distance_mps": 0.0,
            }
        batch = curves.sample_actor_batch(
            self._offline_actor_rng,
            batch_size=self.offline_actor_batch_size,
            hindsight_fraction=self.offline_actor_hindsight_fraction,
            task_mask=self.offline_actor_training_mask,
            task_sampling_weights=self.offline_actor_task_sampling_weights,
            sensitivity_minimum=self.offline_actor_sensitivity_minimum,
            sensitivity_maximum=self.offline_actor_sensitivity_maximum,
        )
        observations = torch.as_tensor(
            batch.observations,
            dtype=torch.float32,
            device=self.device,
        )
        targets = torch.as_tensor(
            batch.actions,
            dtype=torch.float32,
            device=self.device,
        )
        sensitivity = torch.as_tensor(
            batch.sensitivity_weights,
            dtype=torch.float32,
            device=self.device,
        )
        residual_predictions = self.actor(observations)
        predictions = self._effective_action_tensor(
            observations,
            residual_predictions,
        )
        angle_loss = torch.mean(torch.square(predictions[:, 0] - targets[:, 0]))
        speed_loss = torch.mean(
            sensitivity
            * F.smooth_l1_loss(
                predictions[:, 1],
                targets[:, 1],
                reduction="none",
            )
        )
        stop_loss, margin_loss, range_loss = _interpolated_curve_stop_loss(
            predictions,
            batch,
            curves.offsets_mps,
            distance_scale_m=self.offline_actor_physical_distance_scale_m,
            success_margin_m=self.offline_actor_success_margin_m,
        )
        (
            success_interval_loss,
            success_interval_coverage,
            success_interval_distance,
        ) = _success_interval_speed_loss(
            predictions,
            batch,
            curves.offsets_mps,
            success_margin_m=self.offline_actor_success_margin_m,
            distance_scale_mps=(
                self.offline_actor_success_interval_scale_mps
            ),
        )
        total = (
            self.offline_actor_angle_weight * angle_loss
            + self.offline_actor_speed_weight * speed_loss
            + self.offline_actor_physical_loss_weight * (stop_loss + range_loss)
            + self.offline_actor_margin_loss_weight * margin_loss
            + self.offline_actor_success_interval_loss_weight
            * success_interval_loss
        )
        return total, {
            "angle_loss": float(angle_loss.detach().item()),
            "speed_loss": float(speed_loss.detach().item()),
            "physical_stop_loss": float(stop_loss.detach().item()),
            "physical_margin_loss": float(margin_loss.detach().item()),
            "physical_range_loss": float(range_loss.detach().item()),
            "success_interval_loss": float(
                success_interval_loss.detach().item()
            ),
            "success_interval_coverage_rate": float(
                success_interval_coverage.detach().item()
            ),
            "success_interval_distance_mps": float(
                success_interval_distance.detach().item()
            ),
        }

    def warmup_offline_residual_actor(
        self,
        gradient_steps: int,
        *,
        max_grad_norm: float,
        learning_rate: float | None = None,
    ) -> dict[str, float | int]:
        """Fit the bounded residual directly to exact canonical/HER actions."""

        if gradient_steps <= 0 or max_grad_norm <= 0.0:
            raise ValueError("Offline residual warmup settings must be positive.")
        effective_learning_rate = (
            self.actor_learning_rate
            if learning_rate is None
            else float(learning_rate)
        )
        if (
            not math.isfinite(effective_learning_rate)
            or effective_learning_rate <= 0.0
        ):
            raise ValueError(
                "Offline residual Actor learning rate must be positive and finite."
            )
        if self.offline_speed_curves is None:
            raise RuntimeError("Offline residual warmup requires configured curves.")
        self.policy.set_training_mode(True)
        self._set_optimizer_learning_rate(
            self.actor.optimizer,
            effective_learning_rate,
        )
        losses: list[float] = []
        components: dict[str, list[float]] = {
            "angle_loss": [],
            "speed_loss": [],
            "physical_stop_loss": [],
            "physical_margin_loss": [],
            "physical_range_loss": [],
            "success_interval_loss": [],
            "success_interval_coverage_rate": [],
            "success_interval_distance_mps": [],
        }
        for _ in range(gradient_steps):
            loss, values = self._offline_curve_actor_supervision_loss()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    "Offline residual Actor warmup loss became non-finite."
                )
            self.actor.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
            self.actor.optimizer.step()
            losses.append(float(loss.detach().item()))
            for name, value in values.items():
                components[name].append(value)
            self._actor_updates += 1
        window = min(32, max(1, len(losses) // 4))
        report: dict[str, float | int] = {
            "updates": gradient_steps,
            "learning_rate": effective_learning_rate,
            "initial_loss": float(np.mean(losses[:window])),
            "final_loss": float(np.mean(losses[-window:])),
        }
        report.update(
            {
                f"final_{name}": float(np.mean(values[-window:]))
                for name, values in components.items()
            }
        )
        return report

    def effective_actor_predictions(
        self,
        observations: np.ndarray,
        *,
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Return deterministic BC-plus-residual actions without rollout noise."""

        if not self.residual_policy_enabled:
            raise RuntimeError("Effective Actor predictions require residual mode.")
        values = np.asarray(observations, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 8 or batch_size <= 0:
            raise ValueError("Effective Actor observations or batch size are invalid.")
        predictions: list[np.ndarray] = []
        self.policy.set_training_mode(False)
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                observation_tensor = torch.as_tensor(
                    values[start : start + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                if getattr(
                    self,
                    "safe_candidate_classifier_enabled",
                    False,
                ):
                    residual, _, _ = self._safe_candidate_residual_tensor(
                        observation_tensor
                    )
                else:
                    residual = self.actor(observation_tensor)
                    residual = torch.cat(
                        (
                            torch.zeros_like(residual[:, 1:2]),
                            torch.clamp(residual[:, 1:2], -1.0, 1.0),
                        ),
                        dim=1,
                    )
                effective = self._effective_action_tensor(
                    observation_tensor,
                    residual,
                )
                predictions.append(effective.cpu().numpy())
        return np.concatenate(predictions).astype(np.float32)

    def effective_actor_action_metrics(
        self,
        observations: np.ndarray,
        targets: np.ndarray,
        *,
        batch_size: int = 4096,
    ) -> dict[str, float | int]:
        """Measure effective continuous actions against certified actions."""

        expected = np.asarray(targets, dtype=np.float32)
        if expected.shape != (len(observations), 2):
            raise ValueError("Effective Actor targets must have shape (N, 2).")
        predictions = self.effective_actor_predictions(
            observations,
            batch_size=batch_size,
        )
        errors = predictions.astype(np.float64) - expected.astype(np.float64)
        angle_error_deg = np.abs(errors[:, 0]) * math.degrees(
            MAX_ANGLE_RESIDUAL
        )
        speed_error_mps = np.abs(errors[:, 1]) * 0.5 * (
            MAX_CUE_SPEED - MIN_CUE_SPEED
        )
        return {
            "sample_count": int(len(expected)),
            "angle_mae_deg": float(np.mean(angle_error_deg)),
            "angle_p95_deg": float(np.percentile(angle_error_deg, 95)),
            "speed_mae_mps": float(np.mean(speed_error_mps)),
            "speed_p95_mps": float(np.percentile(speed_error_mps, 95)),
        }

    def offline_residual_actor_diagnostics(
        self,
        *,
        batch_size: int = 4096,
    ) -> dict[str, float | int]:
        """Audit effective actions on the task-disjoint physical holdout."""

        curves = self.offline_speed_curves
        if curves is None:
            raise RuntimeError("Offline residual diagnostics require curves.")
        holdout = curves.holdout_mask(
            fraction=self.offline_actor_holdout_fraction,
            seed=self.offline_actor_holdout_seed,
        )
        tasks = np.flatnonzero(holdout)
        if len(tasks) == 0:
            raise RuntimeError("Offline residual Actor holdout is empty.")
        predictions = self.effective_actor_predictions(
            curves.observation[tasks],
            batch_size=batch_size,
        )
        metrics = self.effective_actor_action_metrics(
            curves.observation[tasks],
            curves.center_action[tasks],
            batch_size=batch_size,
        )
        stop_errors = _estimated_canonical_stop_errors(
            predictions,
            curves,
            tasks,
        )
        reference_policy = self._require_residual_policy()
        reference_parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(tasks), batch_size):
                observation_tensor = torch.as_tensor(
                    curves.observation[tasks[start : start + batch_size]],
                    dtype=torch.float32,
                    device=self.device,
                )
                reference_parts.append(
                    reference_policy.reference_actor(observation_tensor)
                    .cpu()
                    .numpy()
                )
        reference_predictions = np.concatenate(reference_parts)
        speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
        applied_residual_mps = np.abs(
            predictions[:, 1] - reference_predictions[:, 1]
        ) * speed_half_range
        metrics.update(
            {
                "estimated_stop_mae_m": float(np.mean(stop_errors)),
                "estimated_stop_p90_m": float(
                    np.percentile(stop_errors, 90)
                ),
                "estimated_stop_p95_m": float(
                    np.percentile(stop_errors, 95)
                ),
                "applied_residual_mae_mps": float(
                    np.mean(applied_residual_mps)
                ),
                "applied_residual_p95_mps": float(
                    np.percentile(applied_residual_mps, 95)
                ),
                "residual_saturation_rate": float(
                    np.mean(
                        applied_residual_mps
                        >= self.max_speed_residual_mps - 1.0e-5
                    )
                ),
            }
        )
        return metrics

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

    def configure_structured_curve_critics(
        self,
        curves: OfflineSpeedCurveDataset,
        *,
        learning_rate: float,
        task_batch_size: int,
        reward_weight: float,
        reward_delta_weight: float,
        ranking_weight: float,
        ranking_temperature: float,
        cue_delta_weight: float,
        cue_delta_scale_m: float,
        event_weight: float,
        event_balance_clip: float,
        holdout_fraction: float,
        holdout_seed: int,
        seed: int,
    ) -> None:
        """Configure grouped physical supervision for two independent Critics."""

        if not self.residual_policy_enabled:
            raise RuntimeError(
                "Structured Critics require a frozen BC-centered action feature."
            )
        for name, value in (
            ("learning_rate", learning_rate),
            ("reward_weight", reward_weight),
            ("reward_delta_weight", reward_delta_weight),
            ("ranking_weight", ranking_weight),
            ("ranking_temperature", ranking_temperature),
            ("cue_delta_weight", cue_delta_weight),
            ("cue_delta_scale_m", cue_delta_scale_m),
            ("event_weight", event_weight),
            ("event_balance_clip", event_balance_clip),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Structured Critic {name} must be positive and finite."
                )
        if task_batch_size <= 0:
            raise ValueError("Structured Critic task batch size must be positive.")
        holdout = curves.holdout_mask(
            fraction=holdout_fraction,
            seed=holdout_seed,
        )
        training_indices = np.flatnonzero(~holdout)
        if len(training_indices) == 0 or not np.any(holdout):
            raise ValueError(
                "Structured Critic requires non-empty task-level train and holdout sets."
            )
        if self.offline_speed_curves is not None and (
            self.offline_speed_curves.path.resolve() != curves.path.resolve()
        ):
            raise ValueError("Actor and Critic offline curve datasets disagree.")
        self.offline_speed_curves = curves
        self.structured_critic_learning_rate = float(learning_rate)
        self.structured_critic_task_batch_size = int(task_batch_size)
        self.structured_reward_weight = float(reward_weight)
        self.structured_reward_delta_weight = float(reward_delta_weight)
        self.structured_ranking_weight = float(ranking_weight)
        self.structured_ranking_temperature = float(ranking_temperature)
        self.structured_cue_delta_weight = float(cue_delta_weight)
        self.structured_cue_delta_scale_m = float(cue_delta_scale_m)
        self.structured_event_weight = float(event_weight)
        self.structured_event_balance_clip = float(event_balance_clip)
        self.structured_critic_holdout_fraction = float(holdout_fraction)
        self.structured_critic_holdout_seed = int(holdout_seed)
        self.structured_critic_training_indices = training_indices
        self.structured_event_positive_rates = np.mean(
            curves.event_targets[:, training_indices, :],
            axis=(0, 1),
        ).astype(np.float32)
        self._structured_rng1 = np.random.default_rng(seed + 1)
        self._structured_rng2 = np.random.default_rng(seed + 2)
        self.structured_curve_critic_enabled = True
        self._build_structured_curve_critics()

    def attach_offline_speed_curves_for_resume(
        self,
        curves: OfflineSpeedCurveDataset,
        *,
        seed: int,
    ) -> None:
        """Restore excluded immutable arrays without rebuilding loaded networks."""

        if not self.structured_curve_critic_enabled:
            raise RuntimeError("Resume checkpoint has no structured Critics.")
        self._require_structured_curve_critics()
        actor_holdout = curves.holdout_mask(
            fraction=self.offline_actor_holdout_fraction,
            seed=self.offline_actor_holdout_seed,
        )
        critic_holdout = curves.holdout_mask(
            fraction=self.structured_critic_holdout_fraction,
            seed=self.structured_critic_holdout_seed,
        )
        self.offline_speed_curves = curves
        self.offline_actor_training_mask = ~actor_holdout
        self._configure_offline_actor_task_sampling_weights(curves)
        self.structured_critic_training_indices = np.flatnonzero(~critic_holdout)
        self.structured_event_positive_rates = np.mean(
            curves.event_targets[:, self.structured_critic_training_indices, :],
            axis=(0, 1),
        ).astype(np.float32)
        self._offline_actor_rng = np.random.default_rng(seed)
        self._structured_rng1 = np.random.default_rng(seed + 1)
        self._structured_rng2 = np.random.default_rng(seed + 2)
        if self.certified_rollout_baseline_enabled:
            self.configure_certified_rollout_baseline(
                curves,
                enabled=True,
                seed=self.certified_rollout_seed,
            )

    def _require_structured_curve_critics(
        self,
    ) -> tuple[
        StructuredCurveCritic,
        StructuredCurveCritic,
        torch.optim.Optimizer,
        torch.optim.Optimizer,
    ]:
        values = (
            self.structured_critic1,
            self.structured_critic2,
            self.structured_critic1_optimizer,
            self.structured_critic2_optimizer,
        )
        if not self.structured_curve_critic_enabled or any(
            value is None for value in values
        ):
            raise RuntimeError("Structured curve Critics are not configured.")
        return values  # type: ignore[return-value]

    def _structured_critic_predictions(
        self,
        critic: StructuredCurveCritic,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return critic(
            observations,
            self._critic_action_features(observations, actions),
        )

    def _structured_curve_critic_loss(
        self,
        critic: StructuredCurveCritic,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        curves = self.offline_speed_curves
        training_indices = self.structured_critic_training_indices
        positive_rates = self.structured_event_positive_rates
        if curves is None or training_indices is None or positive_rates is None:
            raise RuntimeError("Structured Critic has no offline training data.")
        tasks = rng.choice(
            training_indices,
            size=self.structured_critic_task_batch_size,
            replace=self.structured_critic_task_batch_size > len(training_indices),
        ).astype(np.int64)
        task_count = len(tasks)
        offset_count = curves.offset_count
        observations = torch.as_tensor(
            np.repeat(curves.observation[tasks], offset_count, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            np.transpose(curves.action[:, tasks, :], (1, 0, 2)).reshape(-1, 2),
            dtype=torch.float32,
            device=self.device,
        )
        reward_targets = torch.as_tensor(
            curves.reward[:, tasks].T.reshape(-1, 1),
            dtype=torch.float32,
            device=self.device,
        )
        cue_targets = torch.as_tensor(
            np.transpose(
                curves.cue_final_delta_xy_m[:, tasks, :],
                (1, 0, 2),
            ).reshape(-1, 2)
            / self.structured_cue_delta_scale_m,
            dtype=torch.float32,
            device=self.device,
        )
        event_targets = torch.as_tensor(
            np.transpose(curves.event_targets[:, tasks, :], (1, 0, 2)).reshape(
                -1,
                len(STRUCTURED_CURVE_EVENT_NAMES),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        predicted_reward, predicted_cue_delta, event_logits = (
            self._structured_critic_predictions(critic, observations, actions)
        )
        reward_loss = F.smooth_l1_loss(predicted_reward, reward_targets)
        predicted_grouped = predicted_reward.reshape(task_count, offset_count)
        target_grouped = reward_targets.reshape(task_count, offset_count)
        center = curves.center_index
        reward_delta_loss = F.smooth_l1_loss(
            predicted_grouped - predicted_grouped[:, center : center + 1],
            target_grouped - target_grouped[:, center : center + 1],
        )
        target_distribution = torch.softmax(
            target_grouped / self.structured_ranking_temperature,
            dim=1,
        )
        ranking_loss = torch.mean(
            torch.sum(
                -target_distribution
                * torch.log_softmax(
                    predicted_grouped / self.structured_ranking_temperature,
                    dim=1,
                ),
                dim=1,
            )
        )
        safe_index = STRUCTURED_CURVE_EVENT_NAMES.index("safe")
        safe_mask = event_targets[:, safe_index] > 0.5
        if bool(torch.any(safe_mask)):
            cue_delta_loss = F.smooth_l1_loss(
                predicted_cue_delta[safe_mask],
                cue_targets[safe_mask],
            )
        else:
            cue_delta_loss = torch.zeros(
                (), dtype=torch.float32, device=self.device
            )
        rates = torch.as_tensor(
            positive_rates,
            dtype=torch.float32,
            device=self.device,
        )
        positive_weights = 0.5 / torch.clamp(rates, min=1.0e-6)
        negative_weights = 0.5 / torch.clamp(1.0 - rates, min=1.0e-6)
        event_weights = torch.where(
            event_targets > 0.5,
            positive_weights[None, :],
            negative_weights[None, :],
        )
        event_weights = torch.clamp(
            event_weights,
            max=self.structured_event_balance_clip,
        )
        event_loss = torch.mean(
            event_weights
            * F.binary_cross_entropy_with_logits(
                event_logits,
                event_targets,
                reduction="none",
            )
        )
        total = (
            self.structured_reward_weight * reward_loss
            + self.structured_reward_delta_weight * reward_delta_loss
            + self.structured_ranking_weight * ranking_loss
            + self.structured_cue_delta_weight * cue_delta_loss
            + self.structured_event_weight * event_loss
        )
        return total, {
            "reward_loss": float(reward_loss.detach().item()),
            "reward_delta_loss": float(reward_delta_loss.detach().item()),
            "ranking_loss": float(ranking_loss.detach().item()),
            "cue_delta_loss": float(cue_delta_loss.detach().item()),
            "event_loss": float(event_loss.detach().item()),
        }

    def _structured_curve_critic_update(self) -> dict[str, float]:
        critic1, critic2, optimizer1, optimizer2 = (
            self._require_structured_curve_critics()
        )
        reports: list[dict[str, float]] = []
        losses: list[float] = []
        for critic, optimizer, rng in (
            (critic1, optimizer1, self._structured_rng1),
            (critic2, optimizer2, self._structured_rng2),
        ):
            loss, components = self._structured_curve_critic_loss(critic, rng)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Structured Critic loss became non-finite.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            reports.append(components)
        self._structured_critic_updates += 1
        self._last_structured_critic_loss = float(np.mean(losses))
        result = {"loss": self._last_structured_critic_loss}
        for name in reports[0]:
            result[name] = float(np.mean([report[name] for report in reports]))
        return result

    def warmup_structured_curve_critics(
        self,
        gradient_steps: int,
    ) -> dict[str, float | int]:
        """Train both structured Critics only on grouped offline physics."""

        if gradient_steps <= 0:
            raise ValueError("Structured Critic warmup updates must be positive.")
        critic1, critic2, _, _ = self._require_structured_curve_critics()
        critic1.train(True)
        critic2.train(True)
        reports = [
            self._structured_curve_critic_update()
            for _ in range(gradient_steps)
        ]
        window = min(32, max(1, len(reports) // 4))
        result: dict[str, float | int] = {
            "updates": gradient_steps,
            "initial_loss": float(
                np.mean([report["loss"] for report in reports[:window]])
            ),
            "final_loss": float(
                np.mean([report["loss"] for report in reports[-window:]])
            ),
        }
        for name in reports[0]:
            if name == "loss":
                continue
            result[f"final_{name}"] = float(
                np.mean([report[name] for report in reports[-window:]])
            )
        return result

    def structured_curve_critic_diagnostics(
        self,
        *,
        minimum_reward_difference: float = 0.05,
        batch_tasks: int = 1024,
        max_tasks: int | None = None,
        seed: int = 30_000,
    ) -> dict[str, float | int | dict[str, float]]:
        """Evaluate both Critics on task-disjoint complete seven-point curves."""

        if minimum_reward_difference < 0.0 or batch_tasks <= 0:
            raise ValueError("Structured Critic diagnostic settings are invalid.")
        curves = self.offline_speed_curves
        if curves is None:
            raise RuntimeError("Structured Critic diagnostics require offline curves.")
        holdout = curves.holdout_mask(
            fraction=self.structured_critic_holdout_fraction,
            seed=self.structured_critic_holdout_seed,
        )
        task_indices = np.flatnonzero(holdout)
        if max_tasks is not None and len(task_indices) > max_tasks:
            rng = np.random.default_rng(seed)
            task_indices = np.sort(
                rng.choice(task_indices, size=max_tasks, replace=False)
            )
        if len(task_indices) == 0:
            raise RuntimeError("Structured Critic holdout is empty.")
        critic1, critic2, _, _ = self._require_structured_curve_critics()
        critic1.train(False)
        critic2.train(False)
        reward1_parts: list[np.ndarray] = []
        reward2_parts: list[np.ndarray] = []
        cue1_parts: list[np.ndarray] = []
        cue2_parts: list[np.ndarray] = []
        event1_parts: list[np.ndarray] = []
        event2_parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(task_indices), batch_tasks):
                tasks = task_indices[start : start + batch_tasks]
                count = len(tasks)
                observations = torch.as_tensor(
                    np.repeat(curves.observation[tasks], curves.offset_count, axis=0),
                    dtype=torch.float32,
                    device=self.device,
                )
                actions = torch.as_tensor(
                    np.transpose(
                        curves.action[:, tasks, :],
                        (1, 0, 2),
                    ).reshape(-1, 2),
                    dtype=torch.float32,
                    device=self.device,
                )
                output1 = self._structured_critic_predictions(
                    critic1, observations, actions
                )
                output2 = self._structured_critic_predictions(
                    critic2, observations, actions
                )
                reward1_parts.append(
                    output1[0].reshape(count, curves.offset_count).cpu().numpy()
                )
                reward2_parts.append(
                    output2[0].reshape(count, curves.offset_count).cpu().numpy()
                )
                cue1_parts.append(
                    output1[1]
                    .reshape(count, curves.offset_count, 2)
                    .cpu()
                    .numpy()
                    * self.structured_cue_delta_scale_m
                )
                cue2_parts.append(
                    output2[1]
                    .reshape(count, curves.offset_count, 2)
                    .cpu()
                    .numpy()
                    * self.structured_cue_delta_scale_m
                )
                event1_parts.append(
                    torch.sigmoid(output1[2])
                    .reshape(
                        count,
                        curves.offset_count,
                        len(STRUCTURED_CURVE_EVENT_NAMES),
                    )
                    .cpu()
                    .numpy()
                )
                event2_parts.append(
                    torch.sigmoid(output2[2])
                    .reshape(
                        count,
                        curves.offset_count,
                        len(STRUCTURED_CURVE_EVENT_NAMES),
                    )
                    .cpu()
                    .numpy()
                )
        reward1 = np.concatenate(reward1_parts)
        reward2 = np.concatenate(reward2_parts)
        cue1 = np.concatenate(cue1_parts)
        cue2 = np.concatenate(cue2_parts)
        event1 = np.concatenate(event1_parts)
        event2 = np.concatenate(event2_parts)
        reward_targets = curves.reward[:, task_indices].T
        cue_targets = np.transpose(
            curves.cue_final_delta_xy_m[:, task_indices, :],
            (1, 0, 2),
        )
        event_targets = np.transpose(
            curves.event_targets[:, task_indices, :],
            (1, 0, 2),
        )
        pessimistic_reward = np.minimum(reward1, reward2)
        pairwise_eligible = 0
        pairwise_q1_correct = 0
        pairwise_q2_correct = 0
        pairwise_both_correct = 0
        for left in range(curves.offset_count):
            for right in range(left + 1, curves.offset_count):
                physical_difference = (
                    reward_targets[:, right] - reward_targets[:, left]
                )
                eligible = np.abs(physical_difference) >= minimum_reward_difference
                if not np.any(eligible):
                    continue
                physical_sign = np.sign(physical_difference[eligible])
                q1_correct = (
                    np.sign(reward1[eligible, right] - reward1[eligible, left])
                    == physical_sign
                )
                q2_correct = (
                    np.sign(reward2[eligible, right] - reward2[eligible, left])
                    == physical_sign
                )
                pairwise_eligible += int(np.count_nonzero(eligible))
                pairwise_q1_correct += int(np.count_nonzero(q1_correct))
                pairwise_q2_correct += int(np.count_nonzero(q2_correct))
                pairwise_both_correct += int(
                    np.count_nonzero(q1_correct & q2_correct)
                )
        safe_index = STRUCTURED_CURVE_EVENT_NAMES.index("safe")
        scratch_index = STRUCTURED_CURVE_EVENT_NAMES.index("cue_scratch")
        safe_targets = event_targets[:, :, safe_index]
        conservative_safe_probability = np.minimum(
            event1[:, :, safe_index],
            event2[:, :, safe_index],
        )
        conservative_scratch_probability = np.maximum(
            event1[:, :, scratch_index],
            event2[:, :, scratch_index],
        )
        cue_prediction = 0.5 * (cue1 + cue2)
        safe_mask = safe_targets > 0.5
        cue_errors = np.linalg.norm(cue_prediction - cue_targets, axis=2)
        selected = np.argmax(pessimistic_reward, axis=1)
        event_brier: dict[str, float] = {}
        averaged_events = 0.5 * (event1 + event2)
        for index, name in enumerate(STRUCTURED_CURVE_EVENT_NAMES):
            event_brier[name] = float(
                np.mean(np.square(averaged_events[:, :, index] - event_targets[:, :, index]))
            )
        denominator = max(pairwise_eligible, 1)
        return {
            "held_out_task_count": int(len(task_indices)),
            "held_out_transition_count": int(
                len(task_indices) * curves.offset_count
            ),
            "reward_q1_mae": float(np.mean(np.abs(reward1 - reward_targets))),
            "reward_q2_mae": float(np.mean(np.abs(reward2 - reward_targets))),
            "reward_pessimistic_mae": float(
                np.mean(np.abs(pessimistic_reward - reward_targets))
            ),
            "reward_critic_disagreement_mean": float(
                np.mean(np.abs(reward1 - reward2))
            ),
            "pairwise_eligible_count": int(pairwise_eligible),
            "pairwise_q1_agreement": float(pairwise_q1_correct / denominator),
            "pairwise_q2_agreement": float(pairwise_q2_correct / denominator),
            "pairwise_both_critics_agreement": float(
                pairwise_both_correct / denominator
            ),
            "canonical_selection_rate": float(
                np.mean(selected == curves.center_index)
            ),
            "cue_endpoint_mae_m_on_safe": float(np.mean(cue_errors[safe_mask])),
            "cue_endpoint_p90_m_on_safe": float(
                np.percentile(cue_errors[safe_mask], 90)
            ),
            "safe_brier": float(
                np.mean(
                    np.square(conservative_safe_probability - safe_targets)
                )
            ),
            "safe_accuracy": float(
                np.mean(
                    (conservative_safe_probability >= 0.5)
                    == (safe_targets > 0.5)
                )
            ),
            "scratch_brier": float(
                np.mean(
                    np.square(
                        conservative_scratch_probability
                        - event_targets[:, :, scratch_index]
                    )
                )
            ),
            "event_brier": event_brier,
        }

    def configure_structured_actor_gate(
        self,
        *,
        enabled: bool,
        minimum_safe_probability: float,
        maximum_scratch_probability: float,
        minimum_reward_improvement: float,
        maximum_reward_disagreement: float,
    ) -> None:
        """Allow residual actions only when both structured Critics approve."""

        if enabled:
            self._require_structured_curve_critics()
        for name, value in (
            ("minimum_safe_probability", minimum_safe_probability),
            ("maximum_scratch_probability", maximum_scratch_probability),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Structured Actor gate {name} must be in [0, 1].")
        for name, value in (
            ("minimum_reward_improvement", minimum_reward_improvement),
            ("maximum_reward_disagreement", maximum_reward_disagreement),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Structured Actor gate {name} must be finite and non-negative."
                )
        self.structured_actor_gate_enabled = bool(enabled)
        self.structured_gate_min_safe_probability = float(
            minimum_safe_probability
        )
        self.structured_gate_max_scratch_probability = float(
            maximum_scratch_probability
        )
        self.structured_gate_min_reward_improvement = float(
            minimum_reward_improvement
        )
        self.structured_gate_max_reward_disagreement = float(
            maximum_reward_disagreement
        )

    def _apply_structured_actor_gate(
        self,
        observations: torch.Tensor,
        proposed_actions: torch.Tensor,
        *,
        baseline_actions: torch.Tensor | None = None,
        require_reward_improvement: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.structured_actor_gate_enabled:
            approved = torch.ones(
                len(observations), dtype=torch.bool, device=self.device
            )
            return proposed_actions, approved
        critic1, critic2, _, _ = self._require_structured_curve_critics()
        if baseline_actions is None:
            reference = self._require_residual_policy().reference_actor(
                observations
            )
            baseline_actions = torch.cat(
                (torch.zeros_like(reference[:, :1]), reference[:, 1:2]),
                dim=1,
            )
        else:
            baseline_actions = torch.cat(
                (
                    torch.zeros_like(baseline_actions[:, :1]),
                    baseline_actions[:, 1:2],
                ),
                dim=1,
            )
        proposed1 = self._structured_critic_predictions(
            critic1, observations, proposed_actions
        )
        proposed2 = self._structured_critic_predictions(
            critic2, observations, proposed_actions
        )
        baseline1 = self._structured_critic_predictions(
            critic1, observations, baseline_actions
        )
        baseline2 = self._structured_critic_predictions(
            critic2, observations, baseline_actions
        )
        safe_index = STRUCTURED_CURVE_EVENT_NAMES.index("safe")
        scratch_index = STRUCTURED_CURVE_EVENT_NAMES.index("cue_scratch")
        safe_probability = torch.minimum(
            torch.sigmoid(proposed1[2][:, safe_index]),
            torch.sigmoid(proposed2[2][:, safe_index]),
        )
        scratch_probability = torch.maximum(
            torch.sigmoid(proposed1[2][:, scratch_index]),
            torch.sigmoid(proposed2[2][:, scratch_index]),
        )
        approved = (
            (
                proposed1[0].reshape(-1) - baseline1[0].reshape(-1)
                >= self.structured_gate_min_reward_improvement
            )
            & (
                proposed2[0].reshape(-1) - baseline2[0].reshape(-1)
                >= self.structured_gate_min_reward_improvement
            )
            & (
                torch.abs(
                    proposed1[0].reshape(-1) - proposed2[0].reshape(-1)
                )
                <= self.structured_gate_max_reward_disagreement
            )
            & (safe_probability >= self.structured_gate_min_safe_probability)
            & (scratch_probability <= self.structured_gate_max_scratch_probability)
        )
        if not require_reward_improvement:
            approved = (
                (
                    torch.abs(
                        proposed1[0].reshape(-1)
                        - proposed2[0].reshape(-1)
                    )
                    <= self.structured_gate_max_reward_disagreement
                )
                & (
                    safe_probability
                    >= self.structured_gate_min_safe_probability
                )
                & (
                    scratch_probability
                    <= self.structured_gate_max_scratch_probability
                )
            )
        return torch.where(
            approved[:, None],
            proposed_actions,
            baseline_actions,
        ), approved

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
        offline_actor_supervision_losses: list[float] = []
        offline_actor_stop_losses: list[float] = []
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
            residual_actions_pi = self.actor(replay_data.observations)
            if self.actor_candidate_supervision_weight > 0.0:
                if not self.candidate_ranking_enabled:
                    raise RuntimeError(
                        "Critic candidate supervision requires a configured grid."
                    )
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
            else:
                candidate_supervision_loss = torch.zeros(
                    (), dtype=torch.float32, device=self.device
                )
                candidate_targets = torch.zeros(
                    len(replay_data.observations),
                    dtype=torch.float32,
                    device=self.device,
                )
                candidate_improved = torch.zeros_like(
                    candidate_targets, dtype=torch.bool
                )
                selected_disagreement = torch.zeros_like(candidate_targets)
                selected_minimum_q = torch.zeros_like(candidate_targets)
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
            if self.offline_actor_supervision_weight > 0.0:
                offline_actor_supervision_loss, offline_components = (
                    self._offline_curve_actor_supervision_loss()
                )
            else:
                offline_actor_supervision_loss = torch.zeros(
                    (), dtype=torch.float32, device=self.device
                )
                offline_components = {"physical_stop_loss": 0.0}
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
                + self.offline_actor_supervision_weight
                * offline_actor_supervision_loss
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
            offline_actor_supervision_losses.append(
                float(offline_actor_supervision_loss.item())
            )
            offline_actor_stop_losses.append(
                float(offline_components["physical_stop_loss"])
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
        self.logger.record(
            "train/structured_actor_gate_approval_rate",
            self._last_structured_actor_gate_approval_rate,
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
            self.logger.record(
                "train/offline_actor_supervision_loss",
                np.mean(offline_actor_supervision_losses),
            )
            self.logger.record(
                "train/offline_actor_physical_stop_loss",
                np.mean(offline_actor_stop_losses),
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


def _interpolated_curve_stop_loss(
    predictions: torch.Tensor,
    batch: OfflineActorBatch,
    offsets_mps: np.ndarray,
    *,
    distance_scale_m: float,
    success_margin_m: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate through each task's measured piecewise-linear speed curve."""

    if distance_scale_m <= 0.0:
        raise ValueError("Physical Actor distance scale must be positive.")
    if success_margin_m <= 0.0:
        raise ValueError("Physical Actor success margin must be positive.")
    offsets = torch.as_tensor(
        offsets_mps,
        dtype=predictions.dtype,
        device=predictions.device,
    )
    canonical_actions = torch.as_tensor(
        batch.canonical_actions,
        dtype=predictions.dtype,
        device=predictions.device,
    )
    speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    predicted_offsets = (
        predictions[:, 1] - canonical_actions[:, 1]
    ) * speed_half_range
    clamped_offsets = torch.clamp(
        predicted_offsets,
        float(offsets[0]),
        float(offsets[-1]),
    )
    upper = torch.bucketize(clamped_offsets, offsets, right=False)
    upper = torch.clamp(upper, 1, len(offsets) - 1)
    lower = upper - 1
    rows = torch.arange(len(predictions), device=predictions.device)
    curve_points = torch.as_tensor(
        batch.curve_cue_final_xy,
        dtype=predictions.dtype,
        device=predictions.device,
    )
    lower_points = curve_points[rows, lower]
    upper_points = curve_points[rows, upper]
    lower_offsets = offsets[lower]
    upper_offsets = offsets[upper]
    interpolation = (
        (clamped_offsets - lower_offsets)
        / torch.clamp(upper_offsets - lower_offsets, min=1.0e-8)
    ).unsqueeze(1)
    predicted_stop = lower_points + interpolation * (upper_points - lower_points)
    desired_stop = torch.as_tensor(
        batch.desired_stop_xy,
        dtype=predictions.dtype,
        device=predictions.device,
    )
    stop_error = torch.linalg.vector_norm(predicted_stop - desired_stop, dim=1)
    stop_loss = F.smooth_l1_loss(
        stop_error / distance_scale_m,
        torch.zeros_like(stop_error),
    )
    margin_excess = F.relu(stop_error - success_margin_m)
    margin_loss = F.smooth_l1_loss(
        margin_excess / distance_scale_m,
        torch.zeros_like(margin_excess),
    )
    maximum_offset = float(max(abs(offsets_mps[0]), abs(offsets_mps[-1])))
    range_excess = F.relu(torch.abs(predicted_offsets) - maximum_offset)
    range_loss = F.smooth_l1_loss(
        range_excess / maximum_offset,
        torch.zeros_like(range_excess),
    )
    return stop_loss, margin_loss, range_loss


def _success_interval_speed_loss(
    predictions: torch.Tensor,
    batch: OfflineActorBatch,
    offsets_mps: np.ndarray,
    *,
    success_margin_m: float,
    distance_scale_mps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize only distance outside measured-safe 5 cm speed intervals."""

    if success_margin_m <= 0.0 or distance_scale_mps <= 0.0:
        raise ValueError("Success interval scales must be positive.")
    offsets = torch.as_tensor(
        offsets_mps,
        dtype=predictions.dtype,
        device=predictions.device,
    )
    canonical_actions = torch.as_tensor(
        batch.canonical_actions,
        dtype=predictions.dtype,
        device=predictions.device,
    )
    speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    predicted_offsets = (
        predictions[:, 1] - canonical_actions[:, 1]
    ) * speed_half_range
    curve_points = torch.as_tensor(
        batch.curve_cue_final_xy,
        dtype=predictions.dtype,
        device=predictions.device,
    )
    curve_safe = torch.as_tensor(
        batch.curve_safe,
        dtype=torch.bool,
        device=predictions.device,
    )
    desired_stop = torch.as_tensor(
        batch.desired_stop_xy,
        dtype=predictions.dtype,
        device=predictions.device,
    )
    acceptable = curve_safe & (
        torch.linalg.vector_norm(
            curve_points - desired_stop[:, None, :],
            dim=2,
        )
        <= success_margin_m
    )
    if not bool(torch.all(torch.any(acceptable, dim=1))):
        raise RuntimeError("An offline Actor sample has no safe success speed.")
    point_distance = torch.abs(
        predicted_offsets[:, None] - offsets[None, :]
    )
    infinity = torch.full_like(point_distance, torch.inf)
    point_distance = torch.where(acceptable, point_distance, infinity)
    nearest_distance = torch.min(point_distance, dim=1).values
    if len(offsets_mps) > 1:
        pair_valid = acceptable[:, :-1] & acceptable[:, 1:]
        interval_distance = F.relu(
            offsets[None, :-1] - predicted_offsets[:, None]
        ) + F.relu(
            predicted_offsets[:, None] - offsets[None, 1:]
        )
        interval_distance = torch.where(
            pair_valid,
            interval_distance,
            torch.full_like(interval_distance, torch.inf),
        )
        nearest_distance = torch.minimum(
            nearest_distance,
            torch.min(interval_distance, dim=1).values,
        )
    normalized_distance = nearest_distance / distance_scale_mps
    loss = F.smooth_l1_loss(
        normalized_distance,
        torch.zeros_like(normalized_distance),
    )
    coverage = torch.mean((nearest_distance <= 1.0e-7).to(predictions.dtype))
    return loss, coverage, torch.mean(nearest_distance)


def _offline_actor_batch_loss(
    policy: TD3Policy,
    curves: OfflineSpeedCurveDataset,
    batch: OfflineActorBatch,
    *,
    angle_weight: float,
    speed_weight: float,
    physical_loss_weight: float,
    physical_distance_scale_m: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    observations = torch.as_tensor(
        batch.observations,
        dtype=torch.float32,
        device=policy.device,
    )
    targets = torch.as_tensor(
        batch.actions,
        dtype=torch.float32,
        device=policy.device,
    )
    sensitivity = torch.as_tensor(
        batch.sensitivity_weights,
        dtype=torch.float32,
        device=policy.device,
    )
    predictions = policy.actor(observations)
    angle_loss = torch.mean(torch.square(predictions[:, 0] - targets[:, 0]))
    speed_error_loss = F.smooth_l1_loss(
        predictions[:, 1],
        targets[:, 1],
        reduction="none",
    )
    speed_loss = torch.mean(sensitivity * speed_error_loss)
    stop_loss, _, range_loss = _interpolated_curve_stop_loss(
        predictions,
        batch,
        curves.offsets_mps,
        distance_scale_m=physical_distance_scale_m,
    )
    total = (
        angle_weight * angle_loss
        + speed_weight * speed_loss
        + physical_loss_weight * (stop_loss + range_loss)
    )
    return total, {
        "angle_loss": float(angle_loss.detach().item()),
        "speed_loss": float(speed_loss.detach().item()),
        "physical_stop_loss": float(stop_loss.detach().item()),
        "physical_range_loss": float(range_loss.detach().item()),
    }


def _estimated_canonical_stop_errors(
    predictions: np.ndarray,
    curves: OfflineSpeedCurveDataset,
    task_indices: np.ndarray,
) -> np.ndarray:
    """Interpolate measured endpoints at predicted speeds for diagnostics."""

    selected = np.asarray(task_indices, dtype=np.int64)
    offsets = curves.offsets_mps.astype(np.float64)
    speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    predicted_offsets = (
        np.asarray(predictions, dtype=np.float64)[:, 1]
        - curves.center_action[selected, 1].astype(np.float64)
    ) * speed_half_range
    clamped = np.clip(predicted_offsets, offsets[0], offsets[-1])
    upper = np.searchsorted(offsets, clamped, side="left")
    upper = np.clip(upper, 1, len(offsets) - 1)
    lower = upper - 1
    fraction = (
        (clamped - offsets[lower])
        / np.maximum(offsets[upper] - offsets[lower], 1.0e-12)
    )
    lower_points = curves.cue_final[lower, selected, :2].astype(np.float64)
    upper_points = curves.cue_final[upper, selected, :2].astype(np.float64)
    predicted_stop = lower_points + fraction[:, None] * (
        upper_points - lower_points
    )
    errors = np.linalg.norm(
        predicted_stop - curves.target_stop_position[selected],
        axis=1,
    )
    excess = np.maximum(
        np.abs(predicted_offsets) - max(abs(offsets[0]), abs(offsets[-1])),
        0.0,
    )
    errors += excess * curves.sensitivity_m_per_mps()[selected]
    return errors


def _structured_hindsight_speed_metrics(
    policy: TD3Policy,
    curves: OfflineSpeedCurveDataset,
    flat_indices: np.ndarray,
    *,
    batch_size: int,
) -> tuple[float, float, float, float]:
    """Measure inverse-speed error on every legal achieved-stop relabel."""

    selected = np.asarray(flat_indices, dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("Structured hindsight metric set is empty or malformed.")
    speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    xy_scale = np.asarray(
        (OBSERVATION_X_SCALE, OBSERVATION_Y_SCALE),
        dtype=np.float32,
    )
    sensitivity = curves.legal_local_sensitivity_m_per_mps().astype(np.float64)
    speed_errors: list[np.ndarray] = []
    local_stop_proxies: list[np.ndarray] = []
    for start in range(0, len(selected), batch_size):
        batch_flat = selected[start : start + batch_size]
        offset_indices = batch_flat // curves.task_count
        task_indices = batch_flat % curves.task_count
        observations = curves.observation[task_indices].copy()
        observations[:, TARGET_STOP_OBSERVATION_SLICE] = (
            curves.cue_final[offset_indices, task_indices, :2] / xy_scale
        )
        predictions = _predict_td3_actions(
            policy,
            observations,
            batch_size=len(observations),
        )
        targets = curves.action[offset_indices, task_indices, 1]
        errors_mps = (
            np.abs(predictions[:, 1] - targets).astype(np.float64)
            * speed_half_range
        )
        speed_errors.append(errors_mps)
        local_stop_proxies.append(errors_mps * sensitivity[task_indices])
    errors = np.concatenate(speed_errors)
    proxies = np.concatenate(local_stop_proxies)
    return (
        float(np.mean(errors)),
        float(np.percentile(errors, 95)),
        float(np.mean(proxies)),
        float(np.percentile(proxies, 90)),
    )


def behavior_clone_structured_speed_policy(
    policy: TD3Policy,
    curves: OfflineSpeedCurveDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    final_learning_rate: float,
    speed_weight: float = 1.0,
    speed_error_scale_mps: float = 0.005,
    canonical_anchor_weight: float = 64.0,
    middle_pocket_weight: float = 2.0,
    sensitivity_weight_minimum: float = 0.5,
    sensitivity_weight_maximum: float = 4.0,
    sensitivity_loss_weight: float = 0.2,
    sensitivity_distance_scale_m: float = 0.05,
    freeze_speed_trunk: bool = False,
    seed: int = 0,
    max_grad_norm: float = 1.0,
) -> StructuredSpeedBCReport:
    """Train pure supervised inverse-speed BC without a learned Critic.

    Every legal point in every seven-speed physical curve is visited exactly
    once per epoch.  Its achieved cue endpoint replaces only the desired
    cue-stop goal and its measured action speed is the label.  One canonical
    anchor per task is added so deployment-time generated goals remain directly
    represented.  The angle branch is frozen throughout.
    """

    actor = policy.actor
    if not isinstance(actor, StructuredSpeedActor):
        raise TypeError("Structured speed BC requires StructuredSpeedActor.")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("Structured speed BC epochs and batch size must be positive.")
    for name, value in (
        ("learning_rate", learning_rate),
        ("final_learning_rate", final_learning_rate),
        ("speed_weight", speed_weight),
        ("speed_error_scale_mps", speed_error_scale_mps),
        ("canonical_anchor_weight", canonical_anchor_weight),
        ("middle_pocket_weight", middle_pocket_weight),
        ("sensitivity_loss_weight", sensitivity_loss_weight),
        ("sensitivity_distance_scale_m", sensitivity_distance_scale_m),
        ("max_grad_norm", max_grad_norm),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Structured speed BC {name} must be positive and finite.")
    if final_learning_rate > learning_rate:
        raise ValueError("Structured speed BC final learning rate cannot increase.")
    if middle_pocket_weight < 1.0:
        raise ValueError("Middle-pocket weight must be at least one.")
    if canonical_anchor_weight < 1.0:
        raise ValueError("Canonical anchor weight must be at least one.")
    if not (
        0.0 < sensitivity_weight_minimum <= 1.0
        <= sensitivity_weight_maximum
    ):
        raise ValueError("Sensitivity weights must bracket one.")

    legal_flat = np.flatnonzero(curves.safe).astype(np.int64)
    if len(legal_flat) == 0:
        raise RuntimeError("Offline curves contain no legal supervised outcomes.")
    total_sample_count = len(legal_flat) + curves.task_count
    legal_tasks = legal_flat % curves.task_count
    legal_per_pocket = np.bincount(
        curves.pocket_indices[legal_tasks],
        minlength=len(POCKET_POSITIONS),
    )
    canonical_initial = _td3_bc_metrics(
        policy,
        curves.observation,
        curves.center_action,
        batch_size=min(batch_size, curves.task_count),
        angle_weight=1.0,
        speed_weight=speed_weight,
    )
    hindsight_initial = _structured_hindsight_speed_metrics(
        policy,
        curves,
        legal_flat,
        batch_size=batch_size,
    )

    sensitivity = curves.legal_local_sensitivity_m_per_mps().astype(np.float64)
    positive_sensitivity = sensitivity[sensitivity > 0.0]
    sensitivity_scale = (
        float(np.median(positive_sensitivity))
        if positive_sensitivity.size
        else 1.0
    )
    sensitivity_ratio = np.clip(
        sensitivity / max(sensitivity_scale, 1.0e-12),
        sensitivity_weight_minimum,
        sensitivity_weight_maximum,
    )
    effective_sensitivity = sensitivity_scale * sensitivity_ratio
    task_weights = sensitivity_ratio.copy()
    task_weights[curves.middle_pocket_mask] *= middle_pocket_weight
    task_weights /= float(np.mean(task_weights))

    if freeze_speed_trunk:
        for module in (actor.features_extractor, actor.speed_trunk):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    parameters = [
        parameter for parameter in actor.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("Structured speed Actor has no trainable parameters.")
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    rng = np.random.default_rng(seed)
    speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    xy_scale = np.asarray(
        (OBSERVATION_X_SCALE, OBSERVATION_Y_SCALE),
        dtype=np.float32,
    )
    gradient_updates = 0
    policy.set_training_mode(True)
    for epoch in range(epochs):
        epoch_losses: list[float] = []
        progress = epoch / max(epochs - 1, 1)
        epoch_learning_rate = learning_rate * (
            final_learning_rate / learning_rate
        ) ** progress
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = epoch_learning_rate
        order = rng.permutation(total_sample_count)
        for start in range(0, total_sample_count, batch_size):
            sample_ids = order[start : start + batch_size]
            curve_mask = sample_ids < len(legal_flat)
            batch_count = len(sample_ids)
            observations = np.empty((batch_count, 8), dtype=np.float32)
            target_speeds = np.empty(batch_count, dtype=np.float32)
            task_indices = np.empty(batch_count, dtype=np.int64)

            if np.any(curve_mask):
                curve_flat = legal_flat[sample_ids[curve_mask]]
                offset_indices = curve_flat // curves.task_count
                selected_tasks = curve_flat % curves.task_count
                task_indices[curve_mask] = selected_tasks
                observations[curve_mask] = curves.observation[selected_tasks]
                observations[
                    curve_mask,
                    TARGET_STOP_OBSERVATION_SLICE,
                ] = (
                    curves.cue_final[
                        offset_indices,
                        selected_tasks,
                        :2,
                    ]
                    / xy_scale
                )
                target_speeds[curve_mask] = curves.action[
                    offset_indices,
                    selected_tasks,
                    1,
                ]
            anchor_mask = ~curve_mask
            if np.any(anchor_mask):
                selected_tasks = sample_ids[anchor_mask] - len(legal_flat)
                task_indices[anchor_mask] = selected_tasks
                observations[anchor_mask] = curves.observation[selected_tasks]
                target_speeds[anchor_mask] = curves.center_action[
                    selected_tasks,
                    1,
                ]

            observation_tensor = torch.as_tensor(
                observations,
                dtype=torch.float32,
                device=policy.device,
            )
            target_tensor = torch.as_tensor(
                target_speeds,
                dtype=torch.float32,
                device=policy.device,
            )
            sample_weights = task_weights[task_indices].copy()
            sample_weights[anchor_mask] *= canonical_anchor_weight
            weight_tensor = torch.as_tensor(
                sample_weights,
                dtype=torch.float32,
                device=policy.device,
            )
            sensitivity_tensor = torch.as_tensor(
                effective_sensitivity[task_indices],
                dtype=torch.float32,
                device=policy.device,
            )
            predictions = actor(observation_tensor)[:, 1]
            normalized_error = predictions - target_tensor
            speed_error_mps = normalized_error * speed_half_range
            direct_loss = F.smooth_l1_loss(
                speed_error_mps / speed_error_scale_mps,
                torch.zeros_like(speed_error_mps),
                reduction="none",
            )
            local_stop_error = (
                speed_error_mps
                * sensitivity_tensor
                / sensitivity_distance_scale_m
            )
            sensitivity_loss = F.smooth_l1_loss(
                local_stop_error,
                torch.zeros_like(local_stop_error),
                reduction="none",
            )
            loss = torch.sum(
                weight_tensor
                * (
                    speed_weight * direct_loss
                    + sensitivity_loss_weight * sensitivity_loss
                )
            ) / torch.sum(weight_tensor)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    "Structured speed BC loss became non-finite."
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            optimizer.step()
            gradient_updates += 1
            epoch_losses.append(float(loss.detach().item()))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                "structured_speed_epoch="
                + str(epoch + 1)
                + f"/{epochs} learning_rate={epoch_learning_rate:.8g} "
                + f"mean_loss={float(np.mean(epoch_losses)):.8g}",
                flush=True,
            )

    canonical_final = _td3_bc_metrics(
        policy,
        curves.observation,
        curves.center_action,
        batch_size=min(batch_size, curves.task_count),
        angle_weight=1.0,
        speed_weight=speed_weight,
    )
    hindsight_final = _structured_hindsight_speed_metrics(
        policy,
        curves,
        legal_flat,
        batch_size=batch_size,
    )
    policy.set_training_mode(False)
    return StructuredSpeedBCReport(
        task_count=curves.task_count,
        legal_curve_sample_count=len(legal_flat),
        legal_curve_samples_per_pocket=tuple(
            int(value) for value in legal_per_pocket
        ),
        canonical_anchor_count=curves.task_count,
        total_supervised_sample_count=total_sample_count,
        epochs=epochs,
        gradient_updates=gradient_updates,
        angle_mode=actor.angle_mode,
        speed_trunk_frozen=freeze_speed_trunk,
        sensitivity_estimator="nearest_legal_center_slope_v1",
        initial_canonical_speed_mae_mps=canonical_initial[2],
        final_canonical_speed_mae_mps=canonical_final[2],
        final_canonical_speed_p95_mps=canonical_final[4],
        final_canonical_angle_mae_deg=canonical_final[1],
        final_canonical_angle_p95_deg=canonical_final[3],
        initial_hindsight_speed_mae_mps=hindsight_initial[0],
        final_hindsight_speed_mae_mps=hindsight_final[0],
        final_hindsight_speed_p95_mps=hindsight_final[1],
        final_hindsight_local_stop_proxy_mae_m=hindsight_final[2],
        final_hindsight_local_stop_proxy_p90_m=hindsight_final[3],
        canonical_anchor_weight=canonical_anchor_weight,
        middle_pocket_weight=middle_pocket_weight,
        sensitivity_weight_minimum=sensitivity_weight_minimum,
        sensitivity_weight_maximum=sensitivity_weight_maximum,
        speed_error_scale_mps=speed_error_scale_mps,
        sensitivity_loss_weight=sensitivity_loss_weight,
        sensitivity_distance_scale_m=sensitivity_distance_scale_m,
    )


def behavior_clone_td3_policy_with_offline_curves(
    policy: TD3Policy,
    curves: OfflineSpeedCurveDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    final_learning_rate: float,
    angle_weight: float = 1.0,
    speed_weight: float = 8.0,
    hindsight_fraction: float = 0.5,
    physical_loss_weight: float = 1.0,
    physical_distance_scale_m: float = 0.05,
    sensitivity_minimum: float = 0.25,
    sensitivity_maximum: float = 4.0,
    holdout_fraction: float = 0.0,
    holdout_seed: int = 20_000,
    seed: int = 0,
    max_grad_norm: float = 1.0,
) -> OfflineCurveActorReport:
    """Fit a deterministic inverse policy from canonical and legal HER goals."""

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("Offline curve Actor epochs and batch size must be positive.")
    for name, value in (
        ("learning_rate", learning_rate),
        ("final_learning_rate", final_learning_rate),
        ("angle_weight", angle_weight),
        ("speed_weight", speed_weight),
        ("physical_distance_scale_m", physical_distance_scale_m),
        ("max_grad_norm", max_grad_norm),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Offline curve Actor {name} must be positive and finite.")
    if not math.isfinite(physical_loss_weight) or physical_loss_weight < 0.0:
        raise ValueError(
            "Offline curve Actor physical loss weight must be finite and "
            "non-negative."
        )
    if final_learning_rate > learning_rate:
        raise ValueError("Offline curve Actor final learning rate cannot increase.")
    if not 0.0 <= hindsight_fraction <= 1.0:
        raise ValueError("Offline Actor hindsight fraction must be in [0, 1].")
    if not 0.0 < sensitivity_minimum <= 1.0 <= sensitivity_maximum:
        raise ValueError("Offline Actor sensitivity bounds must bracket one.")
    holdout = curves.holdout_mask(fraction=holdout_fraction, seed=holdout_seed)
    training_mask = ~holdout
    evaluation_indices = np.flatnonzero(holdout)
    if len(evaluation_indices) == 0:
        evaluation_indices = np.flatnonzero(training_mask)
    initial = _td3_bc_metrics(
        policy,
        curves.observation[evaluation_indices],
        curves.center_action[evaluation_indices],
        batch_size=min(batch_size, len(evaluation_indices)),
        angle_weight=angle_weight,
        speed_weight=speed_weight,
    )
    initial_predictions = _predict_td3_actions(
        policy,
        curves.observation[evaluation_indices],
        batch_size=min(batch_size, len(evaluation_indices)),
    )
    initial_stop_errors = _estimated_canonical_stop_errors(
        initial_predictions,
        curves,
        evaluation_indices,
    )
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    canonical_per_batch = max(
        int(round(batch_size * (1.0 - hindsight_fraction))),
        1,
    )
    updates_per_epoch = math.ceil(
        int(np.count_nonzero(training_mask)) / canonical_per_batch
    )
    gradient_updates = 0
    policy.set_training_mode(True)
    for epoch in range(epochs):
        progress = epoch / max(epochs - 1, 1)
        epoch_learning_rate = learning_rate * (
            final_learning_rate / learning_rate
        ) ** progress
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = epoch_learning_rate
        for _ in range(updates_per_epoch):
            actor_batch = curves.sample_actor_batch(
                rng,
                batch_size=batch_size,
                hindsight_fraction=hindsight_fraction,
                task_mask=training_mask,
                sensitivity_minimum=sensitivity_minimum,
                sensitivity_maximum=sensitivity_maximum,
            )
            loss, _ = _offline_actor_batch_loss(
                policy,
                curves,
                actor_batch,
                angle_weight=angle_weight,
                speed_weight=speed_weight,
                physical_loss_weight=physical_loss_weight,
                physical_distance_scale_m=physical_distance_scale_m,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Offline curve Actor loss became non-finite.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.actor.parameters(), max_grad_norm)
            optimizer.step()
            gradient_updates += 1
    final = _td3_bc_metrics(
        policy,
        curves.observation[evaluation_indices],
        curves.center_action[evaluation_indices],
        batch_size=min(batch_size, len(evaluation_indices)),
        angle_weight=angle_weight,
        speed_weight=speed_weight,
    )
    final_predictions = _predict_td3_actions(
        policy,
        curves.observation[evaluation_indices],
        batch_size=min(batch_size, len(evaluation_indices)),
    )
    final_stop_errors = _estimated_canonical_stop_errors(
        final_predictions,
        curves,
        evaluation_indices,
    )
    return OfflineCurveActorReport(
        sample_count=int(np.count_nonzero(training_mask)),
        exact_hindsight_count=len(
            curves.her_eligible_flat_indices(task_mask=training_mask)
        ),
        epochs=epochs,
        gradient_updates=gradient_updates,
        initial_loss=initial[0],
        final_loss=final[0],
        initial_angle_mae_deg=initial[1],
        final_angle_mae_deg=final[1],
        initial_speed_mae_mps=initial[2],
        final_speed_mae_mps=final[2],
        final_angle_p95_deg=final[3],
        final_speed_p95_mps=final[4],
        initial_estimated_stop_mae_m=float(np.mean(initial_stop_errors)),
        final_estimated_stop_mae_m=float(np.mean(final_stop_errors)),
        final_estimated_stop_p90_m=float(np.percentile(final_stop_errors, 90)),
        hindsight_fraction=float(hindsight_fraction),
        physical_loss_weight=float(physical_loss_weight),
        sensitivity_minimum=float(sensitivity_minimum),
        sensitivity_maximum=float(sensitivity_maximum),
    )


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
