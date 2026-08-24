"""Validated grouped offline speed curves for single-shot mid-level learning."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from snooker_env.midlevel_ppo_env import (
    MAX_TERMINAL_REWARD,
    OBSERVATION_X_SCALE,
    OBSERVATION_Y_SCALE,
)
from snooker_env.midlevel_two_ball import (
    MAX_CUE_SPEED,
    MIN_CUE_SPEED,
    POCKET_POSITIONS,
)


OFFLINE_SPEED_CURVE_VERSION = "canonical-generated-speed-perturbations-v1"
OFFLINE_CURVE_TRAINING_VERSION = "grouped-offline-curve-her-v1"
TARGET_STOP_OBSERVATION_SLICE = slice(6, 8)


def deterministic_task_holdout_mask(
    task_indices: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Return a stable task-level split shared by every action for a task."""

    if not 0.0 <= fraction < 0.5:
        raise ValueError("Task holdout fraction must be in [0, 0.5).")
    if seed < 0:
        raise ValueError("Task holdout seed must be non-negative.")
    indices = np.asarray(task_indices, dtype=np.uint64)
    if fraction == 0.0:
        return np.zeros(indices.shape, dtype=np.bool_)
    mixed = indices + np.uint64(seed + 1)
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    threshold = int(fraction * 1_000_000)
    return (mixed % np.uint64(1_000_000)) < np.uint64(threshold)


@dataclass(frozen=True)
class OfflineActorBatch:
    """Exact inverse-action supervision with grouped physical curve context."""

    observations: np.ndarray
    actions: np.ndarray
    task_indices: np.ndarray
    desired_stop_xy: np.ndarray
    curve_cue_final_xy: np.ndarray
    curve_safe: np.ndarray
    canonical_actions: np.ndarray
    sensitivity_weights: np.ndarray
    hindsight: np.ndarray


@dataclass(frozen=True)
class OfflineSpeedCurveDataset:
    """One immutable seven-point speed curve for every canonical task.

    The first array dimension is the physical speed offset and the second is
    task identity.  Keeping that grouping intact is essential: sampling the
    points independently lets a scalar reward regressor explain state effects
    while ignoring the much smaller local action effect.
    """

    path: Path
    metadata: dict[str, Any]
    offsets_mps: np.ndarray
    task_indices: np.ndarray
    observation: np.ndarray
    action: np.ndarray
    center_action: np.ndarray
    target_stop_position: np.ndarray
    cue_final: np.ndarray
    cue_final_delta_xy_m: np.ndarray
    object_final: np.ndarray
    reward: np.ndarray
    reward_object_ball: np.ndarray
    reward_cue_position: np.ndarray
    reward_joint_success_bonus: np.ndarray
    correct_pot: np.ndarray
    cue_scratch: np.ndarray
    wrong_pocket: np.ndarray
    stopped: np.ndarray
    timed_out: np.ndarray
    numerical_failure: np.ndarray
    joint_success: np.ndarray
    object_pocket_error: np.ndarray

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        task_dataset: Any | None = None,
        reference_observations: np.ndarray | None = None,
        reference_actions: np.ndarray | None = None,
        center_stop_tolerance_m: float = 5.0e-3,
    ) -> "OfflineSpeedCurveDataset":
        """Load and fully validate an offline curve archive."""

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        required = {
            "metadata",
            "offsets_mps",
            "task_indices",
            "observation",
            "action",
            "center_action",
            "target_stop_position",
            "cue_final",
            "cue_final_delta_xy_m",
            "object_final",
            "reward",
            "reward_object_ball",
            "reward_cue_position",
            "reward_joint_success_bonus",
            "correct_pot",
            "cue_scratch",
            "wrong_pocket",
            "stopped",
            "timed_out",
            "numerical_failure",
            "joint_success",
            "object_pocket_error",
        }
        with np.load(source, allow_pickle=False) as archive:
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(
                    "Offline speed curve archive is missing fields: "
                    + ", ".join(missing)
                )
            metadata = json.loads(str(archive["metadata"].item()))
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in required
                if name != "metadata"
            }
        dataset = cls(path=source, metadata=metadata, **arrays)
        dataset.validate(
            task_dataset=task_dataset,
            reference_observations=reference_observations,
            reference_actions=reference_actions,
            center_stop_tolerance_m=center_stop_tolerance_m,
        )
        return dataset

    @property
    def task_count(self) -> int:
        return int(len(self.task_indices))

    @property
    def offset_count(self) -> int:
        return int(len(self.offsets_mps))

    @property
    def center_index(self) -> int:
        return int(np.flatnonzero(self.offsets_mps == 0.0)[0])

    @property
    def pocket_indices(self) -> np.ndarray:
        """Recover the stable pocket index encoded in every observation."""

        pocket_xy = self.observation[:, 4:6].astype(np.float64) * np.asarray(
            (OBSERVATION_X_SCALE, OBSERVATION_Y_SCALE),
            dtype=np.float64,
        )
        positions = np.stack(
            [POCKET_POSITIONS[name][:2] for name in sorted(POCKET_POSITIONS)]
        ).astype(np.float64)
        squared_distance = np.sum(
            np.square(pocket_xy[:, None, :] - positions[None, :, :]),
            axis=2,
        )
        indices = np.argmin(squared_distance, axis=1).astype(np.int64)
        if float(np.max(np.min(squared_distance, axis=1))) > 1.0e-8:
            raise ValueError("Offline observations contain an unknown pocket position.")
        return indices

    @property
    def middle_pocket_mask(self) -> np.ndarray:
        names = np.asarray(sorted(POCKET_POSITIONS))
        return np.char.startswith(names[self.pocket_indices], "pocket_middle_")

    @property
    def safe(self) -> np.ndarray:
        return (
            self.correct_pot
            & ~self.cue_scratch
            & ~self.wrong_pocket
            & self.stopped
            & ~self.timed_out
            & ~self.numerical_failure
        )

    @property
    def failure(self) -> np.ndarray:
        return (
            self.cue_scratch
            | self.wrong_pocket
            | self.timed_out
            | self.numerical_failure
        )

    @property
    def event_targets(self) -> np.ndarray:
        """Return eight structured event labels in ``(offset, task, event)``."""

        return np.stack(
            (
                self.correct_pot,
                self.safe,
                self.cue_scratch,
                self.stopped,
                self.timed_out,
                self.wrong_pocket,
                self.numerical_failure,
                self.joint_success,
            ),
            axis=2,
        ).astype(np.float32)

    @property
    def event_names(self) -> tuple[str, ...]:
        return (
            "correct_pot",
            "safe",
            "cue_scratch",
            "stopped",
            "timed_out",
            "wrong_pocket",
            "numerical_failure",
            "joint_success",
        )

    def validate(
        self,
        *,
        task_dataset: Any | None,
        reference_observations: np.ndarray | None,
        reference_actions: np.ndarray | None,
        center_stop_tolerance_m: float,
    ) -> None:
        """Reject partial, reordered, mismatched, or non-canonical archives."""

        if self.metadata.get("format_version") != OFFLINE_SPEED_CURVE_VERSION:
            raise ValueError("Unsupported offline speed curve format version.")
        if self.metadata.get("center_action_source") != "canonical_generated_action":
            raise ValueError("Offline speed curves are not canonical-action centered.")
        if not bool(self.metadata.get("world_slot_aligned", False)):
            raise ValueError("Offline speed curves were not world/slot aligned.")
        offsets = np.asarray(self.offsets_mps, dtype=np.float64)
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
                "Offline speed offsets must be finite, sorted, unique, "
                "symmetric, and contain zero."
            )
        task_count = self.task_count
        offset_count = self.offset_count
        if int(self.metadata.get("task_count", -1)) != task_count:
            raise ValueError("Offline curve metadata task count is inconsistent.")
        if int(self.metadata.get("offset_count", -1)) != offset_count:
            raise ValueError("Offline curve metadata offset count is inconsistent.")
        if int(self.metadata.get("record_count", -1)) != task_count * offset_count:
            raise ValueError("Offline curve metadata record count is inconsistent.")
        if not np.array_equal(self.task_indices, np.arange(task_count)):
            raise ValueError("Offline curve task indices are incomplete or reordered.")

        task_fields = {
            "observation": (task_count, 8),
            "center_action": (task_count, 2),
            "target_stop_position": (task_count, 2),
        }
        curve_fields = {
            "action": (offset_count, task_count, 2),
            "cue_final": (offset_count, task_count, 3),
            "cue_final_delta_xy_m": (offset_count, task_count, 2),
            "object_final": (offset_count, task_count, 3),
            "reward": (offset_count, task_count),
            "reward_object_ball": (offset_count, task_count),
            "reward_cue_position": (offset_count, task_count),
            "reward_joint_success_bonus": (offset_count, task_count),
            "correct_pot": (offset_count, task_count),
            "cue_scratch": (offset_count, task_count),
            "wrong_pocket": (offset_count, task_count),
            "stopped": (offset_count, task_count),
            "timed_out": (offset_count, task_count),
            "numerical_failure": (offset_count, task_count),
            "joint_success": (offset_count, task_count),
            "object_pocket_error": (offset_count, task_count),
        }
        for name, shape in {**task_fields, **curve_fields}.items():
            values = np.asarray(getattr(self, name))
            if values.shape != shape:
                raise ValueError(
                    f"Offline curve field {name!r} has shape {values.shape}, "
                    f"expected {shape}."
                )
            if (
                values.dtype.kind in "fc"
                and name != "object_pocket_error"
                and not np.all(np.isfinite(values))
            ):
                raise FloatingPointError(
                    f"Offline curve field {name!r} contains non-finite values."
                )
        if np.any(np.isnan(self.object_pocket_error)) or np.any(
            np.isneginf(self.object_pocket_error)
        ):
            raise FloatingPointError(
                "Offline object-pocket errors contain NaN or negative infinity."
            )

        center = self.center_index
        if not np.array_equal(self.action[center], self.center_action):
            raise ValueError("Zero-offset actions do not equal canonical actions.")
        speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
        measured_offsets = (
            self.action[:, :, 1] - self.center_action[None, :, 1]
        ) * speed_half_range
        if not np.allclose(
            measured_offsets,
            offsets[:, None],
            atol=2.0e-6,
            rtol=0.0,
        ):
            raise ValueError("Stored action speeds do not match declared offsets.")
        if not np.allclose(
            self.action[:, :, 0],
            self.center_action[None, :, 0],
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError("Offline speed probes changed the angle action.")
        if not bool(np.all(self.safe[center])):
            raise ValueError("A canonical zero-offset outcome is not a safe pot.")
        if not bool(np.all(self.joint_success[center])):
            raise ValueError("A canonical zero-offset outcome is not a joint success.")
        if not np.allclose(
            self.reward[center],
            MAX_TERMINAL_REWARD,
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise ValueError("Canonical zero-offset rewards are not maximal.")
        center_stop_error = np.linalg.norm(
            self.cue_final[center, :, :2] - self.target_stop_position,
            axis=1,
        )
        if float(np.max(center_stop_error)) > center_stop_tolerance_m:
            raise ValueError(
                "Canonical zero-offset stop error exceeds tolerance: "
                f"{float(np.max(center_stop_error)):.6g} m."
            )
        if not np.allclose(
            self.cue_final_delta_xy_m,
            self.cue_final[:, :, :2] - self.target_stop_position[None, :, :],
            atol=2.0e-6,
            rtol=0.0,
        ):
            raise ValueError("Stored cue endpoint deltas are inconsistent.")

        if task_dataset is not None:
            if len(task_dataset) != task_count:
                raise ValueError("Offline curves do not cover the full task library.")
            expected_hash = task_dataset.content_sha256()
            if self.metadata.get("task_library_content_sha256") != expected_hash:
                raise ValueError("Offline curves do not match the task-library hash.")
            if not np.allclose(
                self.target_stop_position,
                np.asarray(task_dataset.target_stop_positions),
                atol=2.0e-7,
                rtol=0.0,
            ):
                raise ValueError("Offline curve goals changed from the task library.")
        if reference_observations is not None:
            expected = np.asarray(reference_observations, dtype=np.float32)
            if not np.array_equal(self.observation, expected):
                raise ValueError("Offline curve observations do not match tasks.")
        if reference_actions is not None:
            expected = np.asarray(reference_actions, dtype=np.float32)
            if not np.allclose(
                self.center_action,
                expected,
                atol=2.0e-7,
                rtol=0.0,
            ):
                raise ValueError("Offline canonical actions do not match tasks.")

        safe_goal = self.cue_final[:, :, :2][self.safe]
        normalized_goal = safe_goal / np.array(
            [OBSERVATION_X_SCALE, OBSERVATION_Y_SCALE],
            dtype=np.float32,
        )
        if np.any(np.abs(normalized_goal) > 1.0 + 1.0e-5):
            raise ValueError("A HER-eligible cue endpoint is outside observation bounds.")

    def holdout_mask(self, *, fraction: float, seed: int) -> np.ndarray:
        return deterministic_task_holdout_mask(
            self.task_indices,
            fraction=fraction,
            seed=seed,
        )

    def sensitivity_m_per_mps(self) -> np.ndarray:
        """Estimate local endpoint sensitivity from the closest symmetric pair."""

        positive = self.offsets_mps[self.offsets_mps > 0.0]
        negative = self.offsets_mps[self.offsets_mps < 0.0]
        delta = float(min(positive[0], abs(negative[-1])))
        plus = int(np.flatnonzero(np.isclose(self.offsets_mps, delta))[0])
        minus = int(np.flatnonzero(np.isclose(self.offsets_mps, -delta))[0])
        displacement = np.linalg.norm(
            self.cue_final[plus, :, :2] - self.cue_final[minus, :, :2],
            axis=1,
        )
        return (displacement / (2.0 * delta)).astype(np.float32)

    def legal_local_sensitivity_m_per_mps(self) -> np.ndarray:
        """Estimate the nearest successful one-/two-sided endpoint slope.

        Supervised hindsight excludes failed curve points, so its sensitivity
        weighting must not be dominated by endpoints from scratches, missed
        pots, or timeouts.  Starting at the smallest nonzero speed magnitude,
        use every legal side at the first magnitude available for each task and
        average their center-to-point slope magnitudes.  Tasks with no legal
        noncenter point receive the dataset median rather than a discontinuous
        failure-derived value.
        """

        center_xy = self.cue_final[self.center_index, :, :2].astype(np.float64)
        sensitivity = np.full(self.task_count, np.nan, dtype=np.float64)
        magnitudes = np.unique(
            np.abs(self.offsets_mps[self.offsets_mps != 0.0])
        )
        for magnitude in np.sort(magnitudes):
            offset_indices = np.flatnonzero(
                np.isclose(np.abs(self.offsets_mps), magnitude)
            )
            legal = self.safe[offset_indices]
            available = np.any(legal, axis=0) & ~np.isfinite(sensitivity)
            if not np.any(available):
                continue
            slopes = np.linalg.norm(
                self.cue_final[offset_indices, :, :2] - center_xy[None, :, :],
                axis=2,
            ) / float(magnitude)
            legal_count = np.sum(legal, axis=0)
            mean_legal_slope = np.sum(
                np.where(legal, slopes, 0.0),
                axis=0,
            ) / np.maximum(legal_count, 1)
            sensitivity[available] = mean_legal_slope[available]
        finite = np.isfinite(sensitivity)
        if not np.any(finite):
            raise RuntimeError("Offline curves contain no legal local speed probe.")
        sensitivity[~finite] = float(np.median(sensitivity[finite]))
        if np.any(~np.isfinite(sensitivity)) or np.any(sensitivity < 0.0):
            raise FloatingPointError("Legal local sensitivities are malformed.")
        return sensitivity.astype(np.float32)

    def sensitivity_weights(
        self,
        *,
        minimum: float = 0.25,
        maximum: float = 4.0,
    ) -> np.ndarray:
        if not 0.0 < minimum <= 1.0 <= maximum:
            raise ValueError("Sensitivity weights must bracket one.")
        sensitivity = self.sensitivity_m_per_mps().astype(np.float64)
        positive = sensitivity[sensitivity > 0.0]
        scale = float(np.median(positive)) if positive.size else 1.0
        weights = np.clip(sensitivity / max(scale, 1.0e-12), minimum, maximum)
        return weights.astype(np.float32)

    def success_interval_distances_mps(
        self,
        predicted_offsets_mps: np.ndarray,
        *,
        task_indices: np.ndarray | None = None,
        desired_stop_xy: np.ndarray | None = None,
        success_margin_m: float = 0.05,
    ) -> np.ndarray:
        """Distance to the nearest measured-safe continuous success interval."""

        if not np.isfinite(success_margin_m) or success_margin_m <= 0.0:
            raise ValueError("Success interval margin must be positive and finite.")
        selected = (
            np.arange(self.task_count, dtype=np.int64)
            if task_indices is None
            else np.asarray(task_indices, dtype=np.int64)
        )
        predictions = np.asarray(predicted_offsets_mps, dtype=np.float64)
        if predictions.shape != selected.shape or not np.all(
            np.isfinite(predictions)
        ):
            raise ValueError("Predicted speed offsets are malformed.")
        if np.any(selected < 0) or np.any(selected >= self.task_count):
            raise IndexError("Success interval task index is out of range.")
        desired = (
            self.target_stop_position[selected]
            if desired_stop_xy is None
            else np.asarray(desired_stop_xy, dtype=np.float64)
        )
        if desired.shape != (len(selected), 2) or not np.all(np.isfinite(desired)):
            raise ValueError("Success interval desired stops are malformed.")
        curve_points = np.transpose(
            self.cue_final[:, selected, :2],
            (1, 0, 2),
        ).astype(np.float64)
        curve_safe = np.transpose(self.safe[:, selected], (1, 0))
        acceptable = curve_safe & (
            np.linalg.norm(curve_points - desired[:, None, :], axis=2)
            <= success_margin_m
        )
        if not bool(np.all(np.any(acceptable, axis=1))):
            raise RuntimeError("A task has no measured-safe success speed.")
        offsets = self.offsets_mps.astype(np.float64)
        point_distance = np.abs(predictions[:, None] - offsets[None, :])
        point_distance[~acceptable] = np.inf
        best = np.min(point_distance, axis=1)
        if self.offset_count > 1:
            pair_valid = acceptable[:, :-1] & acceptable[:, 1:]
            interval_distance = np.maximum(
                offsets[None, :-1] - predictions[:, None],
                0.0,
            ) + np.maximum(
                predictions[:, None] - offsets[None, 1:],
                0.0,
            )
            interval_distance[~pair_valid] = np.inf
            best = np.minimum(best, np.min(interval_distance, axis=1))
        return best.astype(np.float32)

    def her_eligible_flat_indices(
        self,
        *,
        task_mask: np.ndarray | None = None,
        include_center: bool = False,
    ) -> np.ndarray:
        eligible = self.safe.copy()
        if not include_center:
            eligible[self.center_index] = False
        if task_mask is not None:
            selected = np.asarray(task_mask, dtype=np.bool_)
            if selected.shape != (self.task_count,):
                raise ValueError("HER task mask has an incorrect shape.")
            eligible &= selected[None, :]
        return np.flatnonzero(eligible)

    def sample_actor_batch(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
        hindsight_fraction: float,
        task_mask: np.ndarray | None = None,
        task_sampling_weights: np.ndarray | None = None,
        sensitivity_minimum: float = 0.25,
        sensitivity_maximum: float = 4.0,
    ) -> OfflineActorBatch:
        """Mix canonical actions and exact successful cue-goal relabels."""

        if batch_size <= 0:
            raise ValueError("Offline actor batch size must be positive.")
        if not 0.0 <= hindsight_fraction <= 1.0:
            raise ValueError("Offline actor hindsight fraction must be in [0, 1].")
        selected_tasks = (
            np.ones(self.task_count, dtype=np.bool_)
            if task_mask is None
            else np.asarray(task_mask, dtype=np.bool_)
        )
        if selected_tasks.shape != (self.task_count,) or not np.any(selected_tasks):
            raise ValueError("Offline actor task mask is empty or malformed.")
        task_pool = np.flatnonzero(selected_tasks)
        her_pool = self.her_eligible_flat_indices(task_mask=selected_tasks)
        sampling_weights: np.ndarray | None = None
        if task_sampling_weights is not None:
            sampling_weights = np.asarray(
                task_sampling_weights,
                dtype=np.float64,
            )
            if (
                sampling_weights.shape != (self.task_count,)
                or not np.all(np.isfinite(sampling_weights))
                or np.any(sampling_weights[selected_tasks] <= 0.0)
            ):
                raise ValueError(
                    "Offline Actor task sampling weights are malformed."
                )

        def probabilities(task_indices: np.ndarray) -> np.ndarray | None:
            if sampling_weights is None:
                return None
            values = sampling_weights[task_indices]
            if np.allclose(values, values[0], atol=0.0, rtol=0.0):
                return None
            return values / np.sum(values)

        hindsight_count = int(round(batch_size * hindsight_fraction))
        if her_pool.size == 0:
            hindsight_count = 0
        canonical_count = batch_size - hindsight_count
        canonical_tasks = rng.choice(
            task_pool,
            size=canonical_count,
            replace=canonical_count > len(task_pool),
            p=probabilities(task_pool),
        ).astype(np.int64)
        canonical_offsets = np.full(
            canonical_count,
            self.center_index,
            dtype=np.int64,
        )
        if hindsight_count:
            her_tasks = her_pool % self.task_count
            hindsight_flat = rng.choice(
                her_pool,
                size=hindsight_count,
                replace=hindsight_count > len(her_pool),
                p=probabilities(her_tasks),
            )
            hindsight_offsets, hindsight_tasks = np.unravel_index(
                hindsight_flat,
                (self.offset_count, self.task_count),
            )
            task_indices = np.concatenate(
                (canonical_tasks, hindsight_tasks.astype(np.int64))
            )
            offset_indices = np.concatenate(
                (canonical_offsets, hindsight_offsets.astype(np.int64))
            )
        else:
            task_indices = canonical_tasks
            offset_indices = canonical_offsets
        hindsight = offset_indices != self.center_index
        observations = self.observation[task_indices].copy()
        desired_stop_xy = self.target_stop_position[task_indices].copy()
        if np.any(hindsight):
            desired_stop_xy[hindsight] = self.cue_final[
                offset_indices[hindsight],
                task_indices[hindsight],
                :2,
            ]
            observations[hindsight, TARGET_STOP_OBSERVATION_SLICE] = (
                desired_stop_xy[hindsight]
                / np.array(
                    [OBSERVATION_X_SCALE, OBSERVATION_Y_SCALE],
                    dtype=np.float32,
                )
            )
        actions = self.action[offset_indices, task_indices]
        curve_points = np.transpose(
            self.cue_final[:, task_indices, :2],
            (1, 0, 2),
        )
        curve_safe = np.transpose(
            self.safe[:, task_indices],
            (1, 0),
        )
        sensitivity = self.sensitivity_weights(
            minimum=sensitivity_minimum,
            maximum=sensitivity_maximum,
        )[task_indices]
        permutation = rng.permutation(batch_size)
        return OfflineActorBatch(
            observations=observations[permutation].astype(np.float32),
            actions=actions[permutation].astype(np.float32),
            task_indices=task_indices[permutation],
            desired_stop_xy=desired_stop_xy[permutation].astype(np.float32),
            curve_cue_final_xy=curve_points[permutation].astype(np.float32),
            curve_safe=curve_safe[permutation],
            canonical_actions=self.center_action[task_indices][permutation].astype(
                np.float32
            ),
            sensitivity_weights=sensitivity[permutation],
            hindsight=hindsight[permutation],
        )

    def report(self) -> dict[str, Any]:
        sensitivity = self.sensitivity_m_per_mps().astype(np.float64)
        legal_sensitivity = self.legal_local_sensitivity_m_per_mps().astype(
            np.float64
        )
        safe = self.safe
        return {
            "version": OFFLINE_CURVE_TRAINING_VERSION,
            "path": str(self.path),
            "task_count": self.task_count,
            "offset_count": self.offset_count,
            "record_count": self.task_count * self.offset_count,
            "offsets_mps": [float(value) for value in self.offsets_mps],
            "safe_count": int(np.count_nonzero(safe)),
            "safe_rate": float(np.mean(safe)),
            "noncenter_her_count": int(
                len(self.her_eligible_flat_indices(include_center=False))
            ),
            "sensitivity_m_per_mps": {
                "mean": float(np.mean(sensitivity)),
                "p50": float(np.percentile(sensitivity, 50)),
                "p90": float(np.percentile(sensitivity, 90)),
                "p99": float(np.percentile(sensitivity, 99)),
            },
            "legal_local_sensitivity_m_per_mps": {
                "estimator": "nearest_legal_center_slope_v1",
                "mean": float(np.mean(legal_sensitivity)),
                "p50": float(np.percentile(legal_sensitivity, 50)),
                "p90": float(np.percentile(legal_sensitivity, 90)),
                "p99": float(np.percentile(legal_sensitivity, 99)),
            },
            "task_library_content_sha256": self.metadata.get(
                "task_library_content_sha256"
            ),
        }
