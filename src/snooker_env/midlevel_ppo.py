"""Stable-Baselines3 checkpoint adapter for the mid-level policy pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution,
    get_action_dim,
)
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from torch import nn

from snooker_env.midlevel_ppo_env import shot_observation, task_observation
from snooker_env.midlevel_rl import ImpactParameters
from snooker_env.midlevel_tasks import TwoBallTaskDataset
from snooker_env.midlevel_two_ball import (
    CUE_FOLLOW_THROUGH,
    CUE_START_BACKOFF,
    CUE_TIP_LOCAL_X,
    MAX_ANGLE_RESIDUAL,
    MAX_CUE_SPEED,
    MIN_CUE_SPEED,
    POCKET_POSITIONS,
    decode_action,
    encode_speed_action,
    ghost_ball_direction,
)
from snooker_env.pipeline_types import CueCommand, Pose3D, SceneState, SkillCommand, SkillId
from snooker_env.table_geometry import BALL_RADIUS


MIDLEVEL_TRAINING_MANIFEST_VERSION = 2


class BoundedActorCriticPolicy(ActorCriticPolicy):
    """Gaussian latent policy transformed by tanh into the exact action Box."""

    def _build(self, lr_schedule: Any) -> None:
        self.action_dist = SquashedDiagGaussianDistribution(
            get_action_dim(self.action_space)
        )
        super()._build(lr_schedule)


@dataclass(frozen=True)
class BehaviorCloningReport:
    """Finite reconstruction metrics for generated feasible actions."""

    sample_count: int
    epochs: int
    initial_loss: float
    final_loss: float
    initial_angle_mae_deg: float
    final_angle_mae_deg: float
    initial_speed_mae_mps: float
    final_speed_mae_mps: float
    final_angle_p95_deg: float
    final_speed_p95_mps: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "sample_count": self.sample_count,
            "epochs": self.epochs,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "initial_angle_mae_deg": self.initial_angle_mae_deg,
            "final_angle_mae_deg": self.final_angle_mae_deg,
            "initial_speed_mae_mps": self.initial_speed_mae_mps,
            "final_speed_mae_mps": self.final_speed_mae_mps,
            "final_angle_p95_deg": self.final_angle_p95_deg,
            "final_speed_p95_mps": self.final_speed_p95_mps,
        }


def set_independent_action_std(
    policy: ActorCriticPolicy,
    action_std: Sequence[float],
) -> None:
    """Set one positive latent Gaussian standard deviation per action axis."""

    values = np.asarray(action_std, dtype=np.float64)
    expected_shape = (get_action_dim(policy.action_space),)
    if values.shape != expected_shape:
        raise ValueError(
            f"action_std has shape {values.shape}, expected {expected_shape}."
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Every action standard deviation must be positive and finite.")
    if policy.log_std is None:
        raise TypeError("The policy has no state-independent Gaussian log_std.")
    with torch.no_grad():
        policy.log_std.copy_(
            torch.as_tensor(
                np.log(values),
                dtype=policy.log_std.dtype,
                device=policy.log_std.device,
            )
        )


def generated_behavior_cloning_data(
    dataset: TwoBallTaskDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized observations and exact actions stored by generation."""

    observations = np.stack(
        [task_observation(dataset[index]) for index in range(len(dataset))]
    ).astype(np.float32)
    actions = np.zeros((len(dataset), 2), dtype=np.float32)
    for index in range(len(dataset)):
        task = dataset[index]
        baseline = ghost_ball_direction(
            task.cue_position,
            task.object_position,
            task.pocket_position,
        )
        direction = np.asarray(task.generated_direction, dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        signed_residual = math.atan2(
            float(baseline[0] * direction[1] - baseline[1] * direction[0]),
            float(np.dot(baseline, direction)),
        )
        normalized_residual = signed_residual / MAX_ANGLE_RESIDUAL
        if abs(normalized_residual) > 1.0 + 1e-6:
            raise ValueError(
                f"Task {index} generated direction is outside the action range: "
                f"{math.degrees(signed_residual):.6g} degrees."
            )
        actions[index, 0] = np.float32(
            np.clip(normalized_residual, -1.0, 1.0)
        )
        actions[index, 1] = np.float32(
            encode_speed_action(task.generated_speed)
        )
    return observations, actions


def _predict_deterministic_actions(
    policy: ActorCriticPolicy,
    observations: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            observation_tensor = torch.as_tensor(
                observations[start : start + batch_size],
                dtype=torch.float32,
                device=policy.device,
            )
            actions = policy.get_distribution(observation_tensor).mode()
            predictions.append(actions.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float64)


def _behavior_cloning_metrics(
    policy: ActorCriticPolicy,
    observations: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    angle_weight: float,
) -> tuple[float, float, float, float, float]:
    predictions = _predict_deterministic_actions(
        policy,
        observations,
        batch_size=batch_size,
    )
    errors = predictions - np.asarray(targets, dtype=np.float64)
    weighted_squared_error = (
        angle_weight * np.square(errors[:, 0]) + np.square(errors[:, 1])
    )
    angle_error_deg = np.abs(errors[:, 0]) * math.degrees(MAX_ANGLE_RESIDUAL)
    speed_error = (
        np.abs(errors[:, 1]) * 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    )
    metrics = (
        float(np.mean(weighted_squared_error)),
        float(np.mean(angle_error_deg)),
        float(np.mean(speed_error)),
        float(np.percentile(angle_error_deg, 95)),
        float(np.percentile(speed_error, 95)),
    )
    if not np.all(np.isfinite(metrics)):
        raise FloatingPointError("Behavior-cloning metrics are non-finite.")
    return metrics


def behavior_clone_policy(
    policy: ActorCriticPolicy,
    dataset: TwoBallTaskDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    angle_weight: float = 4.0,
    seed: int = 0,
    max_grad_norm: float = 1.0,
) -> BehaviorCloningReport:
    """Fit the bounded deterministic policy mean to feasible generated actions."""

    if epochs <= 0:
        raise ValueError("Behavior-cloning epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("Behavior-cloning batch size must be positive.")
    for name, value in (
        ("learning_rate", learning_rate),
        ("angle_weight", angle_weight),
        ("max_grad_norm", max_grad_norm),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Behavior-cloning {name} must be positive and finite.")

    observations, targets = generated_behavior_cloning_data(dataset)
    effective_batch_size = min(int(batch_size), len(dataset))
    initial = _behavior_cloning_metrics(
        policy,
        observations,
        targets,
        batch_size=effective_batch_size,
        angle_weight=angle_weight,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    weights = torch.as_tensor(
        [angle_weight, 1.0],
        dtype=torch.float32,
        device=policy.device,
    )
    rng = np.random.default_rng(seed)
    policy.set_training_mode(True)
    for _ in range(epochs):
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
            predictions = policy.get_distribution(observation_tensor).mode()
            loss = torch.mean(
                torch.sum(
                    weights * torch.square(predictions - target_tensor),
                    dim=1,
                )
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Behavior-cloning loss became non-finite.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
    final = _behavior_cloning_metrics(
        policy,
        observations,
        targets,
        batch_size=effective_batch_size,
        angle_weight=angle_weight,
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


def manifest_differences(
    expected: object,
    actual: object,
    *,
    path: str = "manifest",
) -> tuple[str, ...]:
    """Return stable, human-readable differences between JSON-like values."""

    differences: list[str] = []
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys):
            differences.append(f"{path}.{key}: missing")
        for key in sorted(actual_keys - expected_keys):
            differences.append(f"{path}.{key}: unexpected")
        for key in sorted(expected_keys & actual_keys):
            differences.extend(
                manifest_differences(
                    expected[key],
                    actual[key],
                    path=f"{path}.{key}",
                )
            )
        return tuple(differences)
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return (f"{path}: expected length {len(expected)}, got {len(actual)}",)
        for index, (expected_value, actual_value) in enumerate(
            zip(expected, actual, strict=True)
        ):
            differences.extend(
                manifest_differences(
                    expected_value,
                    actual_value,
                    path=f"{path}[{index}]",
                )
            )
        return tuple(differences)
    if expected != actual:
        return (f"{path}: expected {expected!r}, got {actual!r}",)
    return ()


def require_checkpoint_manifest(
    model: PPO,
    expected: Mapping[str, object],
    *,
    context: str,
) -> None:
    """Reject legacy or incompatible checkpoints before rollout/training."""

    actual = getattr(model, "midlevel_training_manifest", None)
    if not isinstance(actual, Mapping):
        raise ValueError(
            f"{context} checkpoint has no mid-level training manifest; "
            "legacy checkpoints cannot be resumed safely."
        )
    differences = manifest_differences(expected, actual)
    if differences:
        details = "\n".join(f"- {difference}" for difference in differences[:20])
        raise ValueError(f"{context} checkpoint manifest mismatch:\n{details}")


def require_checkpoint_manifest_subset(
    model: PPO,
    expected: Mapping[str, object],
    *,
    context: str,
) -> None:
    """Validate selected manifest fields while allowing unrelated metadata."""

    actual = getattr(model, "midlevel_training_manifest", None)
    if not isinstance(actual, Mapping):
        raise ValueError(
            f"{context} checkpoint has no mid-level training manifest."
        )

    def select(
        expected_values: Mapping[str, object],
        actual_values: Mapping[str, object],
    ) -> dict[str, object]:
        selected: dict[str, object] = {}
        for key, expected_value in expected_values.items():
            actual_value = actual_values.get(key)
            if isinstance(expected_value, Mapping) and isinstance(
                actual_value, Mapping
            ):
                selected[key] = select(expected_value, actual_value)
            elif key in actual_values:
                selected[key] = actual_value
        return selected

    differences = manifest_differences(expected, select(expected, actual))
    if differences:
        details = "\n".join(f"- {difference}" for difference in differences[:20])
        raise ValueError(f"{context} checkpoint manifest mismatch:\n{details}")


@dataclass(frozen=True)
class PredictedShot:
    """World-frame output of a checkpoint-backed impact policy."""

    direction: np.ndarray
    speed: float
    normalized_action: np.ndarray


def _horizontal_cue_pose(
    cue_ball_position: np.ndarray,
    direction: np.ndarray,
    backoff: float,
) -> Pose3D:
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    rear_contact = np.asarray(cue_ball_position, dtype=np.float64).copy() - direction * BALL_RADIUS
    cue_tip = rear_contact - direction * backoff
    cue_body_position = cue_tip - direction * CUE_TIP_LOCAL_X
    yaw = float(np.arctan2(direction[1], direction[0]))
    quaternion = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)
    return Pose3D(position=cue_body_position, quat_wxyz=quaternion)


class PPOCheckpointMidLevelPolicy:
    """Expose a trained contextual PPO policy through existing interfaces."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        skill_id: SkillId = SkillId.POSITION_SHOT,
        device: str = "cpu",
        deterministic: bool = True,
    ) -> None:
        self.skill_id = skill_id
        self.deterministic = bool(deterministic)
        self.model = PPO.load(str(checkpoint), device=device)

    def predict(
        self,
        cue_position: np.ndarray,
        object_position: np.ndarray,
        pocket_position: np.ndarray,
        target_stop_position: np.ndarray,
    ) -> PredictedShot:
        observation = shot_observation(
            cue_position,
            object_position,
            pocket_position,
            target_stop_position,
        )
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        normalized_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        direction_xy, speed = decode_action(
            normalized_action,
            cue_position,
            object_position,
            pocket_position,
        )
        return PredictedShot(
            direction=np.array([direction_xy[0], direction_xy[1], 0.0], dtype=np.float64),
            speed=speed,
            normalized_action=normalized_action,
        )

    def infer_impact(self, command: SkillCommand, state: SceneState) -> ImpactParameters:
        """Return ``ImpactParameters`` for the curriculum policy contract."""

        cue_name = command.intent.cue_ball_name
        object_name = command.intent.object_ball_name
        pocket_name = command.intent.target_pocket
        if cue_name not in state.balls:
            raise KeyError(f"Missing cue ball state: {cue_name}")
        if object_name is None or object_name not in state.balls:
            raise KeyError(f"Missing object ball state: {object_name}")
        if pocket_name not in POCKET_POSITIONS:
            raise KeyError(f"Unknown named pocket: {pocket_name}")
        target_stop = command.intent.desired_cue_ball_position
        if target_stop is None:
            raise ValueError("PPO position-shot policy requires desired_cue_ball_position.")
        prediction = self.predict(
            state.balls[cue_name].position,
            state.balls[object_name].position,
            POCKET_POSITIONS[pocket_name],
            target_stop,
        )
        return ImpactParameters(cue_direction=prediction.direction, cue_speed=prediction.speed)

    def rollout(self, command: SkillCommand, state: SceneState) -> tuple[CueCommand, ...]:
        """Return horizontal center-hit commands for the high/mid/low pipeline."""

        impact = self.infer_impact(command, state)
        cue_ball_position = state.balls[command.intent.cue_ball_name].position
        direction = impact.cue_direction / max(float(np.linalg.norm(impact.cue_direction)), 1e-12)
        start_pose = _horizontal_cue_pose(cue_ball_position, direction, CUE_START_BACKOFF)
        follow_pose = _horizontal_cue_pose(cue_ball_position, direction, -CUE_FOLLOW_THROUGH)
        zero = np.zeros(3, dtype=np.float64)
        return (
            CueCommand(
                pose=start_pose,
                linear_velocity=zero.copy(),
                angular_velocity=zero.copy(),
                duration=0.05,
                debug_label=f"{self.skill_id.value}:ppo_setup",
            ),
            CueCommand(
                pose=start_pose,
                linear_velocity=direction * impact.cue_speed,
                angular_velocity=zero.copy(),
                duration=(CUE_START_BACKOFF + CUE_FOLLOW_THROUGH) / impact.cue_speed,
                debug_label=f"{self.skill_id.value}:ppo_stroke",
            ),
            CueCommand(
                pose=follow_pose,
                linear_velocity=zero.copy(),
                angular_velocity=zero.copy(),
                duration=0.05,
                debug_label=f"{self.skill_id.value}:ppo_follow_through",
            ),
        )
