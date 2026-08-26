"""Targeted local-speed augmentation for the mid-level two-ball task library."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Callable, Mapping, Sequence

import numpy as np

from snooker_env.midlevel_difficulty import TASK_DIFFICULTY_CELLS
from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL
from snooker_env.midlevel_mujoco_warp_tasks import (
    GenerationStatus,
    _backend_dataset,
    _canonical_replay_task,
    _generated_actions,
    _require_slot_identity,
    _validate_mujoco_warp_dataset_compatibility,
)
from snooker_env.midlevel_mujoco_warp_vec_env import (
    MJWarpMidLevelVecEnv,
    active_mujoco_warp_backend_sha256,
)
from snooker_env.midlevel_tasks import (
    EVENT_FLAG_NAMES,
    TwoBallTask,
    TwoBallTaskDataset,
)
from snooker_env.midlevel_two_ball import (
    POCKET_NAMES,
    TwoBallShotSimulator,
    quantize_cue_speed,
)
from snooker_env.mujoco_warp_sdf import MUJOCO_WARP_NCONMAX, MUJOCO_WARP_NJMAX


LOCAL_SPEED_AUGMENTATION_VERSION = "targeted-local-speed-v1"
LOCAL_SPEED_OFFSETS_MPS = (-0.02, -0.01, 0.01, 0.02)
LOCAL_SPEED_TASKS_PER_GROUP = len(LOCAL_SPEED_OFFSETS_MPS)
LOCAL_SPEED_PROVENANCE_SPEED_ATOL_MPS = 1e-7
CORNER_POCKET_FACTOR = 2
MIDDLE_POCKET_FACTOR = 1
LONG_CUE_DISTANCE_FACTOR = 3
OTHER_CUE_DISTANCE_FACTOR = 1
LocalSpeedKey = tuple[int, int]


@dataclass(frozen=True)
class LocalSpeedGroupPlan:
    """One source geometry assigned to one global augmentation group."""

    global_group_index: int
    source_task_index: int
    pocket_index: int
    difficulty_cell_index: int


@dataclass(frozen=True)
class LocalSpeedTaskGroup:
    """Four stable speed perturbations sharing an exact source geometry."""

    plan: LocalSpeedGroupPlan
    tasks: tuple[TwoBallTask, ...]
    actual_speed_offsets: tuple[float, ...]


@dataclass(frozen=True)
class LocalSpeedAugmentation:
    """Generated task arrays plus row-level source provenance."""

    dataset: TwoBallTaskDataset
    source_task_indices: np.ndarray
    global_group_indices: np.ndarray
    requested_speed_offsets: np.ndarray
    actual_speed_offsets: np.ndarray
    source_speeds: np.ndarray
    source_content_sha256: str
    global_seed: int
    shard_index: int
    shard_count: int
    global_task_count: int


def local_speed_key_weights() -> dict[LocalSpeedKey, int]:
    """Return the declared corner/long-distance sampling weights."""

    weights: dict[LocalSpeedKey, int] = {}
    for pocket_index in range(len(POCKET_NAMES)):
        pocket_factor = (
            CORNER_POCKET_FACTOR
            if pocket_index < 4
            else MIDDLE_POCKET_FACTOR
        )
        for cell in TASK_DIFFICULTY_CELLS:
            cue_factor = (
                LONG_CUE_DISTANCE_FACTOR
                if cell.cue_object_band == 2
                else OTHER_CUE_DISTANCE_FACTOR
            )
            weights[(pocket_index, cell.index)] = pocket_factor * cue_factor
    return weights


def local_speed_group_quotas(global_task_count: int) -> dict[LocalSpeedKey, int]:
    """Allocate exact integer group quotas by deterministic largest remainder."""

    if global_task_count <= 0:
        raise ValueError("global_task_count must be positive.")
    if global_task_count % LOCAL_SPEED_TASKS_PER_GROUP != 0:
        raise ValueError(
            "Local-speed task count must be divisible by "
            f"{LOCAL_SPEED_TASKS_PER_GROUP}."
        )
    group_count = global_task_count // LOCAL_SPEED_TASKS_PER_GROUP
    weights = local_speed_key_weights()
    total_weight = sum(weights.values())
    exact = {
        key: group_count * weight / total_weight
        for key, weight in weights.items()
    }
    quotas = {key: int(np.floor(value)) for key, value in exact.items()}
    remainder = group_count - sum(quotas.values())
    ranked = sorted(
        weights,
        key=lambda key: (-(exact[key] - quotas[key]), key),
    )
    for key in ranked[:remainder]:
        quotas[key] += 1
    if sum(quotas.values()) != group_count:
        raise AssertionError("Local-speed quota allocation lost a group.")
    return quotas


def _source_indices_by_key(
    source: TwoBallTaskDataset,
) -> dict[LocalSpeedKey, np.ndarray]:
    cells = source.difficulty_indices().astype(np.int64)
    return {
        (pocket_index, cell.index): np.flatnonzero(
            (source.pocket_indices.astype(np.int64) == pocket_index)
            & (cells == cell.index)
        )
        for pocket_index in range(len(POCKET_NAMES))
        for cell in TASK_DIFFICULTY_CELLS
    }


def plan_local_speed_shard(
    source: TwoBallTaskDataset,
    *,
    global_task_count: int,
    global_seed: int,
    shard_index: int,
    shard_count: int,
) -> tuple[list[LocalSpeedGroupPlan], dict[LocalSpeedKey, list[int]]]:
    """Build one deterministic shard and a disjoint source fallback pool."""

    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid local-speed shard index/count.")
    group_count = global_task_count // LOCAL_SPEED_TASKS_PER_GROUP
    if group_count % shard_count != 0:
        raise ValueError(
            "Local-speed group count must be divisible by shard_count."
        )
    quotas = local_speed_group_quotas(global_task_count)
    source_by_key = _source_indices_by_key(source)
    rng = np.random.default_rng(global_seed)
    permuted: dict[LocalSpeedKey, np.ndarray] = {}
    for key in sorted(source_by_key):
        values = source_by_key[key].copy()
        rng.shuffle(values)
        if len(values) < quotas[key]:
            raise ValueError(
                f"Source library has {len(values)} geometries for {key}, "
                f"but the augmentation requires {quotas[key]}."
            )
        permuted[key] = values

    keys: list[LocalSpeedKey] = []
    for key in sorted(quotas):
        keys.extend([key] * quotas[key])
    rng.shuffle(keys)
    cursors = {key: 0 for key in quotas}
    global_plans: list[LocalSpeedGroupPlan] = []
    for global_group_index, key in enumerate(keys):
        source_task_index = int(permuted[key][cursors[key]])
        cursors[key] += 1
        global_plans.append(
            LocalSpeedGroupPlan(
                global_group_index=global_group_index,
                source_task_index=source_task_index,
                pocket_index=key[0],
                difficulty_cell_index=key[1],
            )
        )

    groups_per_shard = group_count // shard_count
    start = shard_index * groups_per_shard
    plans = global_plans[start : start + groups_per_shard]
    fallbacks: dict[LocalSpeedKey, list[int]] = {}
    for key in sorted(permuted):
        unused = permuted[key][quotas[key] :]
        fallbacks[key] = [
            int(value)
            for offset, value in enumerate(unused)
            if offset % shard_count == shard_index
        ]
    return plans, fallbacks


def _candidate_seed(
    source_content_sha256: str,
    global_seed: int,
    source_task_index: int,
    offset_index: int,
) -> int:
    digest = hashlib.sha256(
        (
            f"{LOCAL_SPEED_AUGMENTATION_VERSION}\0{source_content_sha256}\0"
            f"{global_seed}\0{source_task_index}\0{offset_index}"
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _candidate_group(
    source: TwoBallTaskDataset,
    plan: LocalSpeedGroupPlan,
    *,
    source_content_sha256: str,
    global_seed: int,
) -> LocalSpeedTaskGroup:
    source_task = source[plan.source_task_index]
    tasks: list[TwoBallTask] = []
    actual_offsets: list[float] = []
    for offset_index, requested_offset in enumerate(LOCAL_SPEED_OFFSETS_MPS):
        speed = quantize_cue_speed(source_task.generated_speed + requested_offset)
        actual_offsets.append(speed - source_task.generated_speed)
        tasks.append(
            TwoBallTask(
                cue_position=source_task.cue_position.copy(),
                object_position=source_task.object_position.copy(),
                pocket_name=source_task.pocket_name,
                pocket_position=source_task.pocket_position.copy(),
                target_stop_position=source_task.cue_position.copy(),
                generated_direction=source_task.generated_direction.copy(),
                generated_speed=speed,
                candidate_seed=_candidate_seed(
                    source_content_sha256,
                    global_seed,
                    plan.source_task_index,
                    offset_index,
                ),
                elapsed_time=0.0,
                min_object_pocket_distance=source_task.min_object_pocket_distance,
                event_metrics={name: False for name in EVENT_FLAG_NAMES},
            )
        )
    return LocalSpeedTaskGroup(
        plan=plan,
        tasks=tuple(tasks),
        actual_speed_offsets=tuple(actual_offsets),
    )


class _DoubleReplayRunner:
    """Two independent reusable MJWarp environments for stable task screening."""

    def __init__(
        self,
        *,
        model_path: Path,
        num_worlds: int,
        seed: int,
        device: str,
        chunk_steps: int,
        check_interval_steps: int,
        nconmax: int,
        njmax: int,
        max_time: float,
    ) -> None:
        self.model_path = model_path
        self.num_worlds = num_worlds
        self.seed = seed
        self.device = device
        self.chunk_steps = chunk_steps
        self.check_interval_steps = check_interval_steps
        self.nconmax = nconmax
        self.njmax = njmax
        self.max_time = max_time
        self.first: MJWarpMidLevelVecEnv | None = None
        self.second: MJWarpMidLevelVecEnv | None = None

    def run(
        self,
        dataset: TwoBallTaskDataset,
        indices: Sequence[int],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        if len(indices) != self.num_worlds:
            raise ValueError("Double replay requires exactly num_worlds indices.")
        tasks = [dataset[int(index)] for index in indices]
        if self.first is None:
            kwargs = dict(
                num_envs=self.num_worlds,
                device=self.device,
                chunk_steps=self.chunk_steps,
                check_interval_steps=self.check_interval_steps,
                nconmax=self.nconmax,
                njmax=self.njmax,
                max_time=self.max_time,
            )
            self.first = MJWarpMidLevelVecEnv(
                dataset, self.model_path, seed=self.seed, **kwargs
            )
            self.second = MJWarpMidLevelVecEnv(
                dataset, self.model_path, seed=self.seed + 1, **kwargs
            )
        else:
            self.first.replace_task_dataset(dataset)
            assert self.second is not None
            self.second.replace_task_dataset(dataset)

        options = [{"task_index": int(index)} for index in indices]
        self.first.set_options(options)
        self.first.reset()
        _, _, _, first_infos = self.first.step(_generated_actions(tasks))
        _require_slot_identity(tasks, first_infos, context="Local-speed first replay")

        assert self.second is not None
        self.second.set_options(options)
        self.second.reset()
        _, _, _, second_infos = self.second.step(_generated_actions(tasks))
        _require_slot_identity(
            tasks,
            second_infos,
            context="Local-speed independent replay",
        )
        return first_infos, second_infos

    def close(self) -> None:
        if self.first is not None:
            self.first.close()
        if self.second is not None:
            self.second.close()
        self.first = None
        self.second = None


def _flatten_groups(groups: Sequence[LocalSpeedTaskGroup]) -> list[TwoBallTask]:
    return [task for group in groups for task in group.tasks]


def _canonical_groups(
    groups: Sequence[LocalSpeedTaskGroup],
    first_infos: Sequence[dict[str, object]],
    second_infos: Sequence[dict[str, object]],
    *,
    stop_tolerance: float,
) -> list[LocalSpeedTaskGroup | None]:
    canonical_groups: list[LocalSpeedTaskGroup | None] = []
    for group_offset, group in enumerate(groups):
        tasks: list[TwoBallTask] = []
        for task_offset, candidate in enumerate(group.tasks):
            flat_index = (
                group_offset * LOCAL_SPEED_TASKS_PER_GROUP + task_offset
            )
            canonical = _canonical_replay_task(
                candidate,
                first_infos[flat_index],
                second_infos[flat_index],
                stop_tolerance=stop_tolerance,
            )
            if canonical is None:
                tasks = []
                break
            tasks.append(canonical)
        canonical_groups.append(
            LocalSpeedTaskGroup(
                plan=group.plan,
                tasks=tuple(tasks),
                actual_speed_offsets=group.actual_speed_offsets,
            )
            if len(tasks) == LOCAL_SPEED_TASKS_PER_GROUP
            else None
        )
    return canonical_groups


def _screen_groups(
    groups: Sequence[LocalSpeedTaskGroup],
    *,
    simulator: TwoBallShotSimulator,
    backend_hash: str,
    runner: _DoubleReplayRunner,
    generation_seed: int,
    stop_tolerance: float,
) -> list[LocalSpeedTaskGroup | None]:
    if not groups:
        return []
    tasks = _flatten_groups(groups)
    valid_count = len(tasks)
    while len(tasks) < runner.num_worlds:
        tasks.append(tasks[-1])
    dataset = _backend_dataset(
        tasks,
        simulator,
        generation_seed,
        backend_hash,
    )
    indices = list(range(runner.num_worlds))
    first_infos, second_infos = runner.run(dataset, indices)
    return _canonical_groups(
        groups,
        first_infos[:valid_count],
        second_infos[:valid_count],
        stop_tolerance=stop_tolerance,
    )


def generate_local_speed_augmentation(
    source: TwoBallTaskDataset,
    *,
    global_task_count: int,
    global_seed: int,
    shard_index: int,
    shard_count: int,
    model_path: Path = DEFAULT_MIDLEVEL_MODEL,
    num_worlds: int = 4096,
    device: str = "cuda:0",
    chunk_steps: int = 64,
    check_interval_steps: int = 8192,
    nconmax: int = MUJOCO_WARP_NCONMAX,
    njmax: int = MUJOCO_WARP_NJMAX,
    max_time: float = 8.0,
    stop_tolerance: float = 5e-3,
    max_fixed_layout_rounds: int = 8,
    status: GenerationStatus | None = None,
) -> LocalSpeedAugmentation:
    """Generate one shard with group-atomic screening and final-slot replay."""

    if num_worlds <= 0 or num_worlds % LOCAL_SPEED_TASKS_PER_GROUP != 0:
        raise ValueError(
            "num_worlds must be positive and divisible by the group size."
        )
    if max_fixed_layout_rounds <= 0:
        raise ValueError("max_fixed_layout_rounds must be positive.")
    plans, fallback_indices = plan_local_speed_shard(
        source,
        global_task_count=global_task_count,
        global_seed=global_seed,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    if not plans:
        raise ValueError("The requested shard contains no local-speed groups.")
    source_hash = source.content_sha256()
    simulator = TwoBallShotSimulator(model_path, max_time=max_time)
    xml_hash, model_hash, backend_hash = active_mujoco_warp_backend_sha256(
        model_path
    )
    if xml_hash != simulator.xml_hash or model_hash != simulator.model_hash:
        raise RuntimeError("MJWarp fingerprint does not match augmentation model.")
    _validate_mujoco_warp_dataset_compatibility(
        source,
        simulator,
        backend_hash,
    )

    used_sources = {plan.source_task_index for plan in plans}
    fallback_cursors = {key: 0 for key in fallback_indices}

    def replacement_plan(original: LocalSpeedGroupPlan) -> LocalSpeedGroupPlan:
        key = (original.pocket_index, original.difficulty_cell_index)
        pool = fallback_indices[key]
        cursor = fallback_cursors[key]
        while cursor < len(pool) and pool[cursor] in used_sources:
            cursor += 1
        if cursor >= len(pool):
            raise RuntimeError(
                "Local-speed fallback source pool exhausted for "
                f"pocket={POCKET_NAMES[key[0]]} cell={key[1]}."
            )
        source_index = pool[cursor]
        fallback_cursors[key] = cursor + 1
        used_sources.add(source_index)
        return LocalSpeedGroupPlan(
            global_group_index=original.global_group_index,
            source_task_index=source_index,
            pocket_index=original.pocket_index,
            difficulty_cell_index=original.difficulty_cell_index,
        )

    runner = _DoubleReplayRunner(
        model_path=model_path,
        num_worlds=num_worlds,
        seed=global_seed + shard_index * 2,
        device=device,
        chunk_steps=chunk_steps,
        check_interval_steps=check_interval_steps,
        nconmax=nconmax,
        njmax=njmax,
        max_time=max_time,
    )
    groups_per_batch = num_worlds // LOCAL_SPEED_TASKS_PER_GROUP
    accepted: list[LocalSpeedTaskGroup | None] = [None] * len(plans)
    pending = list(range(len(plans)))
    current_plans = list(plans)
    attempts = 0
    try:
        while pending:
            slots = pending[:groups_per_batch]
            del pending[: len(slots)]
            candidates = [
                _candidate_group(
                    source,
                    current_plans[slot],
                    source_content_sha256=source_hash,
                    global_seed=global_seed,
                )
                for slot in slots
            ]
            attempts += len(candidates)
            screened = _screen_groups(
                candidates,
                simulator=simulator,
                backend_hash=backend_hash,
                runner=runner,
                generation_seed=global_seed,
                stop_tolerance=stop_tolerance,
            )
            passed = 0
            for slot, candidate, canonical in zip(
                slots, candidates, screened, strict=True
            ):
                if canonical is None:
                    current_plans[slot] = replacement_plan(candidate.plan)
                    pending.append(slot)
                else:
                    accepted[slot] = canonical
                    passed += 1
            if status is not None:
                status(
                    f"group screening passed={passed}/{len(candidates)} "
                    f"complete={len(plans) - len(pending)}/{len(plans)} "
                    f"source_attempts={attempts}"
                )

        canonical_groups = [group for group in accepted if group is not None]
        if len(canonical_groups) != len(plans):
            raise AssertionError("Local-speed group screening lost a slot.")

        for batch_start in range(0, len(canonical_groups), groups_per_batch):
            batch_end = min(batch_start + groups_per_batch, len(canonical_groups))
            for round_index in range(1, max_fixed_layout_rounds + 1):
                flat_tasks = _flatten_groups(canonical_groups)
                dataset = _backend_dataset(
                    flat_tasks,
                    simulator,
                    global_seed,
                    backend_hash,
                )
                task_start = batch_start * LOCAL_SPEED_TASKS_PER_GROUP
                valid_tasks = (batch_end - batch_start) * LOCAL_SPEED_TASKS_PER_GROUP
                indices = [task_start + offset for offset in range(valid_tasks)]
                while len(indices) < num_worlds:
                    indices.append(indices[-1])
                first_infos, second_infos = runner.run(dataset, indices)
                replayed = _canonical_groups(
                    canonical_groups[batch_start:batch_end],
                    first_infos[:valid_tasks],
                    second_infos[:valid_tasks],
                    stop_tolerance=stop_tolerance,
                )
                failed_local = [
                    offset
                    for offset, group in enumerate(replayed)
                    if group is None
                ]
                for offset, group in enumerate(replayed):
                    if group is not None:
                        canonical_groups[batch_start + offset] = group
                if not failed_local:
                    if status is not None:
                        status(
                            "fixed-layout group batch "
                            f"{batch_start // groups_per_batch + 1}/"
                            f"{int(np.ceil(len(canonical_groups) / groups_per_batch))} "
                            f"passed round={round_index} groups={batch_end - batch_start}"
                        )
                    break
                if round_index == max_fixed_layout_rounds:
                    raise RuntimeError(
                        "Local-speed fixed-layout replay did not converge; "
                        f"batch_group={batch_start} failures={failed_local[:16]}."
                    )
                if status is not None:
                    status(
                        f"fixed-layout replacing groups={len(failed_local)} "
                        f"round={round_index} batch_group={batch_start}"
                    )
                replacement_slots = [batch_start + value for value in failed_local]
                replacement_groups: list[LocalSpeedTaskGroup | None] = [
                    None
                ] * len(replacement_slots)
                retry = list(range(len(replacement_slots)))
                while retry:
                    retry_batch = retry[:groups_per_batch]
                    del retry[: len(retry_batch)]
                    candidates = []
                    for retry_index in retry_batch:
                        slot = replacement_slots[retry_index]
                        plan = replacement_plan(canonical_groups[slot].plan)
                        candidates.append(
                            _candidate_group(
                                source,
                                plan,
                                source_content_sha256=source_hash,
                                global_seed=global_seed,
                            )
                        )
                    screened = _screen_groups(
                        candidates,
                        simulator=simulator,
                        backend_hash=backend_hash,
                        runner=runner,
                        generation_seed=global_seed,
                        stop_tolerance=stop_tolerance,
                    )
                    for retry_index, canonical in zip(
                        retry_batch, screened, strict=True
                    ):
                        if canonical is None:
                            retry.append(retry_index)
                        else:
                            replacement_groups[retry_index] = canonical
                for slot, group in zip(
                    replacement_slots, replacement_groups, strict=True
                ):
                    assert group is not None
                    canonical_groups[slot] = group
            else:  # pragma: no cover
                raise AssertionError("Unreachable local-speed replay state.")
    finally:
        runner.close()

    final_tasks = _flatten_groups(canonical_groups)
    dataset = _backend_dataset(
        final_tasks,
        simulator,
        global_seed,
        backend_hash,
    )
    source_indices = np.repeat(
        np.asarray(
            [group.plan.source_task_index for group in canonical_groups],
            dtype=np.int64,
        ),
        LOCAL_SPEED_TASKS_PER_GROUP,
    )
    group_indices = np.repeat(
        np.asarray(
            [group.plan.global_group_index for group in canonical_groups],
            dtype=np.int64,
        ),
        LOCAL_SPEED_TASKS_PER_GROUP,
    )
    requested_offsets = np.tile(
        np.asarray(LOCAL_SPEED_OFFSETS_MPS, dtype=np.float64),
        len(canonical_groups),
    )
    actual_offsets = np.asarray(
        [
            value
            for group in canonical_groups
            for value in group.actual_speed_offsets
        ],
        dtype=np.float64,
    )
    source_speeds = source.generated_speeds[source_indices]
    return LocalSpeedAugmentation(
        dataset=dataset,
        source_task_indices=source_indices,
        global_group_indices=group_indices,
        requested_speed_offsets=requested_offsets,
        actual_speed_offsets=actual_offsets,
        source_speeds=np.asarray(source_speeds, dtype=np.float64),
        source_content_sha256=source_hash,
        global_seed=global_seed,
        shard_index=shard_index,
        shard_count=shard_count,
        global_task_count=global_task_count,
    )


def _provenance_arrays(augmentation: LocalSpeedAugmentation) -> dict[str, np.ndarray]:
    return {
        "source_task_indices": np.asarray(
            augmentation.source_task_indices, dtype=np.int64
        ),
        "global_group_indices": np.asarray(
            augmentation.global_group_indices, dtype=np.int64
        ),
        "requested_speed_offsets": np.asarray(
            augmentation.requested_speed_offsets, dtype=np.float64
        ),
        "actual_speed_offsets": np.asarray(
            augmentation.actual_speed_offsets, dtype=np.float64
        ),
        "source_speeds": np.asarray(augmentation.source_speeds, dtype=np.float64),
    }


def provenance_content_sha256(
    metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256()
    for key in sorted(metadata):
        if key == "provenance_content_sha256":
            continue
        digest.update(key.encode("utf-8"))
        digest.update(json.dumps(metadata[key], sort_keys=True).encode("utf-8"))
        digest.update(b"\0")
    for name in sorted(arrays):
        values = np.asarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(repr(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def save_local_speed_provenance(
    augmentation: LocalSpeedAugmentation,
    path: Path,
) -> None:
    arrays = _provenance_arrays(augmentation)
    metadata: dict[str, object] = {
        "version": LOCAL_SPEED_AUGMENTATION_VERSION,
        "source_content_sha256": augmentation.source_content_sha256,
        "augmentation_content_sha256": augmentation.dataset.content_sha256(),
        "task_count": len(augmentation.dataset),
        "tasks_per_group": LOCAL_SPEED_TASKS_PER_GROUP,
        "requested_speed_offsets_mps": list(LOCAL_SPEED_OFFSETS_MPS),
        "global_seed": augmentation.global_seed,
        "shard_index": augmentation.shard_index,
        "shard_count": augmentation.shard_count,
        "global_task_count": augmentation.global_task_count,
        "weighting": {
            "corner_pocket": CORNER_POCKET_FACTOR,
            "middle_pocket": MIDDLE_POCKET_FACTOR,
            "long_cue_distance": LONG_CUE_DISTANCE_FACTOR,
            "other_cue_distance": OTHER_CUE_DISTANCE_FACTOR,
        },
    }
    metadata["provenance_content_sha256"] = provenance_content_sha256(
        metadata, arrays
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.stem}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary_path,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            **arrays,
        )
        temporary_path.replace(output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_local_speed_provenance(
    path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {
            name: np.asarray(archive[name])
            for name in (
                "source_task_indices",
                "global_group_indices",
                "requested_speed_offsets",
                "actual_speed_offsets",
                "source_speeds",
            )
        }
    if metadata.get("version") != LOCAL_SPEED_AUGMENTATION_VERSION:
        raise ValueError(f"Unsupported local-speed provenance version: {path}")
    expected = metadata.get("provenance_content_sha256")
    if expected != provenance_content_sha256(metadata, arrays):
        raise ValueError(f"Local-speed provenance hash mismatch: {path}")
    return metadata, arrays


def require_local_speed_provenance(
    dataset: TwoBallTaskDataset,
    source: TwoBallTaskDataset,
    metadata: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Audit exact geometry grouping, speed offsets, hashes, and quotas."""

    count = len(dataset)
    if count % LOCAL_SPEED_TASKS_PER_GROUP != 0:
        raise ValueError("Local-speed dataset does not contain complete groups.")
    if metadata.get("source_content_sha256") != source.content_sha256():
        raise ValueError("Local-speed provenance names the wrong source library.")
    if metadata.get("augmentation_content_sha256") != dataset.content_sha256():
        raise ValueError("Local-speed provenance names the wrong augmentation.")
    if int(metadata.get("task_count", -1)) != count:
        raise ValueError("Local-speed provenance task count mismatch.")
    for name, values in arrays.items():
        if np.asarray(values).shape != (count,):
            raise ValueError(f"Local-speed provenance {name} has the wrong shape.")
    source_indices = np.asarray(arrays["source_task_indices"], dtype=np.int64)
    group_indices = np.asarray(arrays["global_group_indices"], dtype=np.int64)
    if np.any(source_indices < 0) or np.any(source_indices >= len(source)):
        raise ValueError("Local-speed source index is out of range.")
    expected_requested = np.tile(
        np.asarray(LOCAL_SPEED_OFFSETS_MPS, dtype=np.float64),
        count // LOCAL_SPEED_TASKS_PER_GROUP,
    )
    if not np.array_equal(arrays["requested_speed_offsets"], expected_requested):
        raise ValueError("Local-speed requested offsets are not canonical groups.")
    for start in range(0, count, LOCAL_SPEED_TASKS_PER_GROUP):
        source_index = int(source_indices[start])
        stop = start + LOCAL_SPEED_TASKS_PER_GROUP
        if not np.all(source_indices[start:stop] == source_index):
            raise ValueError("A local-speed group mixes source geometries.")
        if not np.all(group_indices[start:stop] == group_indices[start]):
            raise ValueError("A local-speed group mixes global group identities.")
        if not np.allclose(
            dataset.cue_positions[start:stop],
            source.cue_positions[source_index],
            rtol=0.0,
            atol=0.0,
        ) or not np.allclose(
            dataset.object_positions[start:stop],
            source.object_positions[source_index],
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("A local-speed task changed its source geometry.")
        if not np.all(
            dataset.pocket_indices[start:stop]
            == source.pocket_indices[source_index]
        ):
            raise ValueError("A local-speed task changed its source pocket.")
        if not np.allclose(
            dataset.generated_directions[start:stop],
            source.generated_directions[source_index],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("A local-speed task changed its source direction.")
    grouped_sources = source_indices[::LOCAL_SPEED_TASKS_PER_GROUP]
    if len(set(map(int, grouped_sources))) != len(grouped_sources):
        raise ValueError("Local-speed groups reuse a source geometry.")
    expected_actual = dataset.generated_speeds - source.generated_speeds[source_indices]
    if not np.allclose(
        arrays["actual_speed_offsets"],
        expected_actual,
        rtol=0.0,
        atol=LOCAL_SPEED_PROVENANCE_SPEED_ATOL_MPS,
    ):
        raise ValueError("Local-speed actual offsets do not match task speeds.")
    if not np.allclose(
        arrays["source_speeds"],
        source.generated_speeds[source_indices],
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Local-speed provenance source speeds are incorrect.")
