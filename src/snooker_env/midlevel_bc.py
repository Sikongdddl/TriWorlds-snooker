"""Direct behavior cloning for the robot-free two-ball mid-level policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from snooker_env.midlevel_difficulty import (
    DIFFICULTY_LEVEL_NAMES,
    TASK_DIFFICULTY_CELLS,
    TASK_DIFFICULTY_VERSION,
)
from snooker_env.midlevel_two_ball_env import (
    OBSERVATION_X_SCALE,
    OBSERVATION_Y_SCALE,
    shot_observation,
)
from snooker_env.midlevel_tasks import TwoBallTaskDataset
from snooker_env.midlevel_two_ball import (
    CUE_FOLLOW_THROUGH,
    CUE_START_BACKOFF,
    CUE_TIP_LOCAL_X,
    MAX_ANGLE_RESIDUAL,
    MAX_CUE_SPEED,
    MIN_CUE_SPEED,
    POCKET_NAMES,
    POCKET_POSITIONS,
    decode_action,
)
from snooker_env.pipeline_types import (
    CueCommand,
    Pose3D,
    SceneState,
    SkillCommand,
    SkillId,
)
from snooker_env.table_geometry import BALL_RADIUS


DIRECT_BC_CHECKPOINT_VERSION = 1
DIRECT_BC_ALGORITHM_VERSION = "direct-midlevel-bc-v1"
MIDLEVEL_GEOMETRIC_FEATURE_VERSION = "midlevel-shot-geometry-v1"
MIDLEVEL_GEOMETRIC_FEATURE_DIM = 47
DEFAULT_HIDDEN_SIZES = (512, 512, 256)


@dataclass(frozen=True)
class BehaviorCloningMetrics:
    """Reconstruction metrics in normalized and physical action units."""

    loss: float
    angle_mae_deg: float
    speed_mae_mps: float
    angle_p95_deg: float
    speed_p95_mps: float

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "angle_mae_deg": self.angle_mae_deg,
            "speed_mae_mps": self.speed_mae_mps,
            "angle_p95_deg": self.angle_p95_deg,
            "speed_p95_mps": self.speed_p95_mps,
        }


@dataclass(frozen=True)
class BehaviorCloningReport:
    """Complete audit record for one direct BC run."""

    sample_count: int
    epochs: int
    gradient_updates: int
    seed: int
    initial: BehaviorCloningMetrics
    final: BehaviorCloningMetrics

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "epochs": self.epochs,
            "gradient_updates": self.gradient_updates,
            "seed": self.seed,
            "initial": self.initial.as_dict(),
            "final": self.final.as_dict(),
        }


class MidLevelGeometricFeatures(nn.Module):
    """Expand four absolute XY positions into deterministic shot geometry."""

    def __init__(self) -> None:
        super().__init__()
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
        if observations.ndim != 2 or observations.shape[1] != 8:
            raise ValueError(
                "Mid-level observations must have shape (batch, 8), got "
                f"{tuple(observations.shape)}."
            )
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
            torch.linalg.vector_norm(object_to_pocket, dim=1, keepdim=True),
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
        pocket_one_hot = nn.functional.one_hot(
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


class DirectBCActor(nn.Module):
    """Deterministic bounded Actor trained only from generated actions."""

    def __init__(
        self,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    ) -> None:
        super().__init__()
        normalized_sizes = tuple(int(size) for size in hidden_sizes)
        if not normalized_sizes or any(size <= 0 for size in normalized_sizes):
            raise ValueError("Every hidden layer size must be positive.")
        self.hidden_sizes = normalized_sizes
        self.features = MidLevelGeometricFeatures()
        layers: list[nn.Module] = []
        input_size = MIDLEVEL_GEOMETRIC_FEATURE_DIM
        for hidden_size in normalized_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.ReLU()))
            input_size = hidden_size
        layers.extend((nn.Linear(input_size, 2), nn.Tanh()))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(self.features(observations))


class DirectBCPolicy:
    """Inference and checkpoint wrapper for the direct BC Actor."""

    def __init__(
        self,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.actor = DirectBCActor(hidden_sizes).to(device)
        self.manifest: dict[str, object] = {}
        self.training_report: dict[str, object] = {}

    @property
    def device(self) -> torch.device:
        return next(self.actor.parameters()).device

    def predict(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        single = values.ndim == 1
        if single:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != 8:
            raise ValueError(
                "Mid-level observation must have shape (8,) or (batch, 8)."
            )
        self.actor.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(values, device=self.device)
            action = self.actor(tensor).cpu().numpy().astype(np.float32)
        return action[0] if single else action

    def save(self, path: Path) -> None:
        checkpoint = Path(path)
        if checkpoint.suffix != ".pt":
            raise ValueError("Direct BC checkpoints must use the .pt extension.")
        if not self.manifest:
            raise ValueError("Cannot save a direct BC checkpoint without a manifest.")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(json.dumps(self.manifest))
        training_report = json.loads(json.dumps(self.training_report))
        payload = {
            "checkpoint_version": DIRECT_BC_CHECKPOINT_VERSION,
            "algorithm_version": DIRECT_BC_ALGORITHM_VERSION,
            "hidden_sizes": list(self.actor.hidden_sizes),
            "actor_state_dict": self.actor.state_dict(),
            "manifest": manifest,
            "training_report": training_report,
        }
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=checkpoint.parent,
                prefix=f".{checkpoint.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            torch.save(payload, temporary_path)
            os.replace(temporary_path, checkpoint)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "DirectBCPolicy":
        checkpoint = Path(path)
        payload = torch.load(
            checkpoint,
            map_location=device,
            weights_only=True,
        )
        if not isinstance(payload, Mapping):
            raise ValueError("Direct BC checkpoint payload is malformed.")
        if payload.get("checkpoint_version") != DIRECT_BC_CHECKPOINT_VERSION:
            raise ValueError("Unsupported direct BC checkpoint version.")
        if payload.get("algorithm_version") != DIRECT_BC_ALGORITHM_VERSION:
            raise ValueError("Unsupported direct BC algorithm version.")
        hidden_sizes = payload.get("hidden_sizes")
        if not isinstance(hidden_sizes, (list, tuple)):
            raise ValueError("Direct BC checkpoint has no valid architecture.")
        model = cls(hidden_sizes, device=device)
        state_dict = payload.get("actor_state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("Direct BC checkpoint has no Actor state dictionary.")
        model.actor.load_state_dict(state_dict, strict=True)
        manifest = payload.get("manifest")
        report = payload.get("training_report")
        if not isinstance(manifest, dict) or not isinstance(report, dict):
            raise ValueError("Direct BC checkpoint metadata is malformed.")
        model.manifest = manifest
        model.training_report = report
        model.actor.eval()
        return model


def generated_behavior_cloning_data(
    dataset: TwoBallTaskDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorize stored tasks into direct BC observations and action labels."""

    pocket_array = np.stack(
        [POCKET_POSITIONS[name][:2] for name in POCKET_NAMES]
    ).astype(np.float64)
    pockets = pocket_array[dataset.pocket_indices.astype(np.int64)]
    raw_observations = np.concatenate(
        (
            dataset.cue_positions,
            dataset.object_positions,
            pockets,
            dataset.target_stop_positions,
        ),
        axis=1,
    )
    scales = np.asarray(
        [
            OBSERVATION_X_SCALE,
            OBSERVATION_Y_SCALE,
            OBSERVATION_X_SCALE,
            OBSERVATION_Y_SCALE,
            OBSERVATION_X_SCALE,
            OBSERVATION_Y_SCALE,
            OBSERVATION_X_SCALE,
            OBSERVATION_Y_SCALE,
        ],
        dtype=np.float64,
    )
    observations = raw_observations / scales
    if np.any(np.abs(observations) > 1.0 + 1.0e-8):
        raise ValueError("Task coordinates are outside the BC observation bounds.")

    object_to_pocket = pockets - dataset.object_positions
    object_to_pocket /= np.linalg.norm(
        object_to_pocket,
        axis=1,
        keepdims=True,
    )
    ghost_positions = (
        dataset.object_positions - 2.0 * BALL_RADIUS * object_to_pocket
    )
    baseline = ghost_positions - dataset.cue_positions
    baseline /= np.linalg.norm(baseline, axis=1, keepdims=True)
    # Normalization is label preprocessing only.  Keep the certified dataset
    # arrays immutable so repeated metric passes cannot change its content hash.
    directions = np.asarray(
        dataset.generated_directions,
        dtype=np.float64,
    ).copy()
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    cross = baseline[:, 0] * directions[:, 1] - baseline[:, 1] * directions[:, 0]
    dot = np.sum(baseline * directions, axis=1)
    angle_residual = np.arctan2(cross, dot) / MAX_ANGLE_RESIDUAL
    if np.any(np.abs(angle_residual) > 1.0 + 1.0e-6):
        raise ValueError("A generated direction is outside the action range.")
    speed_fraction = (
        dataset.generated_speeds - MIN_CUE_SPEED
    ) / (MAX_CUE_SPEED - MIN_CUE_SPEED)
    actions = np.stack(
        (angle_residual, 2.0 * speed_fraction - 1.0),
        axis=1,
    )
    if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(actions)):
        raise FloatingPointError("Direct BC data contains non-finite values.")
    return (
        np.clip(observations, -1.0, 1.0).astype(np.float32),
        np.clip(actions, -1.0, 1.0).astype(np.float32),
    )


def behavior_cloning_metrics(
    policy: DirectBCPolicy,
    dataset: TwoBallTaskDataset,
    *,
    batch_size: int,
    angle_weight: float,
    speed_weight: float,
) -> BehaviorCloningMetrics:
    """Evaluate deterministic action reconstruction on a fixed dataset."""

    observations, targets = generated_behavior_cloning_data(dataset)
    predicted = _batched_policy_predictions(
        policy,
        observations,
        batch_size=batch_size,
    )
    return _behavior_cloning_metrics_from_errors(
        predicted.astype(np.float64) - targets.astype(np.float64),
        angle_weight=angle_weight,
        speed_weight=speed_weight,
    )


def _batched_policy_predictions(
    policy: DirectBCPolicy,
    observations: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("Behavior-cloning metric batch size must be positive.")
    predictions: list[np.ndarray] = []
    for start in range(0, len(observations), batch_size):
        predictions.append(policy.predict(observations[start : start + batch_size]))
    return np.concatenate(predictions, axis=0).astype(np.float64)


def _behavior_cloning_metrics_from_errors(
    error: np.ndarray,
    *,
    angle_weight: float,
    speed_weight: float,
) -> BehaviorCloningMetrics:
    if error.ndim != 2 or error.shape[1] != 2 or len(error) == 0:
        raise ValueError("Behavior-cloning errors must have shape (batch, 2).")
    loss = np.mean(
        angle_weight * np.square(error[:, 0])
        + speed_weight * np.square(error[:, 1])
    )
    angle_error_deg = np.abs(error[:, 0]) * math.degrees(MAX_ANGLE_RESIDUAL)
    speed_error_mps = (
        np.abs(error[:, 1]) * 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    )
    values = np.asarray(
        (
            loss,
            np.mean(angle_error_deg),
            np.mean(speed_error_mps),
            np.percentile(angle_error_deg, 95),
            np.percentile(speed_error_mps, 95),
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Direct BC metrics became non-finite.")
    return BehaviorCloningMetrics(*map(float, values))


def behavior_cloning_metrics_by_difficulty(
    policy: DirectBCPolicy,
    dataset: TwoBallTaskDataset,
    *,
    batch_size: int,
    angle_weight: float,
    speed_weight: float,
) -> dict[str, object]:
    """Report reconstruction separately for every represented distance cell."""

    observations, targets = generated_behavior_cloning_data(dataset)
    predictions = _batched_policy_predictions(
        policy,
        observations,
        batch_size=batch_size,
    )
    errors = predictions - targets.astype(np.float64)
    indices = dataset.difficulty_indices().astype(np.int64)

    cells: dict[str, object] = {}
    levels: dict[str, object] = {}
    for cell in TASK_DIFFICULTY_CELLS:
        selected = indices == cell.index
        if not bool(np.any(selected)):
            continue
        cells[cell.name] = {
            "sample_count": int(np.sum(selected)),
            **_behavior_cloning_metrics_from_errors(
                errors[selected],
                angle_weight=angle_weight,
                speed_weight=speed_weight,
            ).as_dict(),
        }
    for level, level_name in enumerate(DIFFICULTY_LEVEL_NAMES):
        cell_indices = [
            cell.index for cell in TASK_DIFFICULTY_CELLS if cell.level == level
        ]
        selected = np.isin(indices, cell_indices)
        if not bool(np.any(selected)):
            continue
        levels[level_name] = {
            "sample_count": int(np.sum(selected)),
            **_behavior_cloning_metrics_from_errors(
                errors[selected],
                angle_weight=angle_weight,
                speed_weight=speed_weight,
            ).as_dict(),
        }
    return {
        "difficulty_version": TASK_DIFFICULTY_VERSION,
        "cells": cells,
        "difficulty_levels": levels,
    }


def train_direct_behavior_cloning(
    policy: DirectBCPolicy,
    dataset: TwoBallTaskDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    final_learning_rate: float,
    angle_weight: float,
    speed_weight: float,
    seed: int,
    max_grad_norm: float = 1.0,
    progress: Callable[[int, int, float, float], None] | None = None,
) -> BehaviorCloningReport:
    """Train both action dimensions once from certified task actions."""

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("Direct BC epochs and batch size must be positive.")
    for name, value in (
        ("learning_rate", learning_rate),
        ("final_learning_rate", final_learning_rate),
        ("angle_weight", angle_weight),
        ("speed_weight", speed_weight),
        ("max_grad_norm", max_grad_norm),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Direct BC {name} must be positive and finite.")
    if final_learning_rate > learning_rate:
        raise ValueError("Direct BC final learning rate cannot increase.")
    if seed < 0:
        raise ValueError("Direct BC seed must be non-negative.")

    observations, targets = generated_behavior_cloning_data(dataset)
    effective_batch_size = min(batch_size, len(dataset))
    initial = behavior_cloning_metrics(
        policy,
        dataset,
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
    gradient_updates = 0
    policy.actor.train()
    for epoch in range(epochs):
        epoch_fraction = epoch / max(epochs - 1, 1)
        epoch_learning_rate = learning_rate * (
            final_learning_rate / learning_rate
        ) ** epoch_fraction
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = epoch_learning_rate
        order = rng.permutation(len(dataset))
        epoch_losses: list[float] = []
        for start in range(0, len(dataset), effective_batch_size):
            indices = order[start : start + effective_batch_size]
            observation_tensor = torch.as_tensor(
                observations[indices],
                dtype=torch.float32,
                device=policy.device,
            )
            target_tensor = torch.as_tensor(
                targets[indices],
                dtype=torch.float32,
                device=policy.device,
            )
            predictions = policy.actor(observation_tensor)
            loss = torch.mean(
                torch.sum(
                    weights * torch.square(predictions - target_tensor),
                    dim=1,
                )
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Direct BC loss became non-finite.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.actor.parameters(), max_grad_norm)
            optimizer.step()
            epoch_losses.append(float(loss.detach().item()))
            gradient_updates += 1
        if progress is not None:
            progress(
                epoch + 1,
                epochs,
                epoch_learning_rate,
                float(np.mean(epoch_losses)),
            )
    policy.actor.eval()
    final = behavior_cloning_metrics(
        policy,
        dataset,
        batch_size=effective_batch_size,
        angle_weight=angle_weight,
        speed_weight=speed_weight,
    )
    return BehaviorCloningReport(
        sample_count=len(dataset),
        epochs=epochs,
        gradient_updates=gradient_updates,
        seed=seed,
        initial=initial,
        final=final,
    )


def task_physics_manifest(dataset: TwoBallTaskDataset) -> dict[str, object]:
    """Return task metadata that must match at evaluation time."""

    return {
        "xml_sha256": dataset.xml_hash,
        "model_sha256": dataset.model_hash,
        "backend": dataset.physics_backend,
        "backend_sha256": dataset.backend_hash,
        "execution_max_time": dataset.execution_max_time,
        "stop_speed": dataset.stop_speed,
        "stop_hold_time": dataset.stop_hold_time,
    }


def require_compatible_task_physics(
    reference: TwoBallTaskDataset,
    candidate: TwoBallTaskDataset,
    *,
    context: str,
) -> None:
    """Reject a validation/test split produced by different physics."""

    expected = task_physics_manifest(reference)
    actual = task_physics_manifest(candidate)
    differences = [
        name for name in expected if expected[name] != actual.get(name)
    ]
    if differences:
        raise ValueError(
            f"{context} task physics mismatch: " + ", ".join(differences)
        )


def validate_policy_for_dataset(
    policy: DirectBCPolicy,
    dataset: TwoBallTaskDataset,
) -> None:
    """Validate checkpoint semantics and physics before executing a shot."""

    algorithm = policy.manifest.get("algorithm")
    physics = policy.manifest.get("physics")
    if not isinstance(algorithm, Mapping) or not isinstance(physics, Mapping):
        raise ValueError("Direct BC checkpoint manifest is incomplete.")
    if algorithm.get("version") != DIRECT_BC_ALGORITHM_VERSION:
        raise ValueError("Direct BC checkpoint algorithm version is unsupported.")
    expected = task_physics_manifest(dataset)
    differences = [
        name for name in expected if physics.get(name) != expected[name]
    ]
    if differences:
        raise ValueError(
            "Direct BC checkpoint physics mismatch: " + ", ".join(differences)
        )


def write_json(path: Path, values: Mapping[str, object]) -> None:
    """Write a small JSON artifact atomically."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(values, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


@dataclass(frozen=True)
class PredictedShot:
    """World-frame output of a direct BC checkpoint."""

    direction: np.ndarray
    speed: float
    normalized_action: np.ndarray


def _horizontal_cue_pose(
    cue_ball_position: np.ndarray,
    direction: np.ndarray,
    backoff: float,
) -> Pose3D:
    normalized_direction = np.asarray(direction, dtype=np.float64)
    normalized_direction /= max(
        float(np.linalg.norm(normalized_direction)),
        1.0e-12,
    )
    rear_contact = (
        np.asarray(cue_ball_position, dtype=np.float64).copy()
        - normalized_direction * BALL_RADIUS
    )
    cue_tip = rear_contact - normalized_direction * backoff
    cue_body_position = cue_tip - normalized_direction * CUE_TIP_LOCAL_X
    yaw = float(np.arctan2(normalized_direction[1], normalized_direction[0]))
    quaternion = np.array(
        [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)],
        dtype=np.float64,
    )
    return Pose3D(position=cue_body_position, quat_wxyz=quaternion)


class BCCheckpointMidLevelPolicy:
    """Expose a direct BC checkpoint through the mid-level pipeline contract."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        skill_id: SkillId = SkillId.POSITION_SHOT,
        device: str = "cpu",
    ) -> None:
        self.skill_id = skill_id
        self.policy = DirectBCPolicy.load(checkpoint, device=device)

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
        normalized_action = np.clip(
            np.asarray(self.policy.predict(observation), dtype=np.float64),
            -1.0,
            1.0,
        )
        direction_xy, speed = decode_action(
            normalized_action,
            cue_position,
            object_position,
            pocket_position,
        )
        return PredictedShot(
            direction=np.array(
                [direction_xy[0], direction_xy[1], 0.0],
                dtype=np.float64,
            ),
            speed=speed,
            normalized_action=normalized_action,
        )

    def infer_impact(
        self,
        command: SkillCommand,
        state: SceneState,
    ) -> PredictedShot:
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
            raise ValueError(
                "Position-shot policy requires desired_cue_ball_position."
            )
        return self.predict(
            state.balls[cue_name].position,
            state.balls[object_name].position,
            POCKET_POSITIONS[pocket_name],
            target_stop,
        )

    def rollout(
        self,
        command: SkillCommand,
        state: SceneState,
    ) -> tuple[CueCommand, ...]:
        prediction = self.infer_impact(command, state)
        cue_ball_position = state.balls[
            command.intent.cue_ball_name
        ].position
        direction = prediction.direction / max(
            float(np.linalg.norm(prediction.direction)),
            1.0e-12,
        )
        start_pose = _horizontal_cue_pose(
            cue_ball_position,
            direction,
            CUE_START_BACKOFF,
        )
        follow_pose = _horizontal_cue_pose(
            cue_ball_position,
            direction,
            -CUE_FOLLOW_THROUGH,
        )
        zero = np.zeros(3, dtype=np.float64)
        return (
            CueCommand(
                pose=start_pose,
                linear_velocity=zero.copy(),
                angular_velocity=zero.copy(),
                duration=0.05,
                debug_label=f"{self.skill_id.value}:direct_bc_setup",
            ),
            CueCommand(
                pose=start_pose,
                linear_velocity=direction * prediction.speed,
                angular_velocity=zero.copy(),
                duration=(
                    CUE_START_BACKOFF + CUE_FOLLOW_THROUGH
                )
                / prediction.speed,
                debug_label=f"{self.skill_id.value}:direct_bc_stroke",
            ),
            CueCommand(
                pose=follow_pose,
                linear_velocity=zero.copy(),
                angular_velocity=zero.copy(),
                duration=0.05,
                debug_label=f"{self.skill_id.value}:direct_bc_follow_through",
            ),
        )
