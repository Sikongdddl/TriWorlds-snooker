#!/usr/bin/env python3
"""Audit and merge targeted local-speed shards with the balanced core library."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_local_speed_tasks import (  # noqa: E402
    LOCAL_SPEED_TASKS_PER_GROUP,
    LocalSpeedAugmentation,
    load_local_speed_provenance,
    local_speed_group_quotas,
    require_local_speed_provenance,
    save_local_speed_provenance,
)
from snooker_env.midlevel_tasks import (  # noqa: E402
    DEFAULT_BALANCED_CORE_TASKS,
    DEFAULT_LOCAL_SPEED_AUGMENTATION_TASKS,
    DEFAULT_TRAIN_TASKS,
    TwoBallTaskDataset,
    require_balanced_task_difficulty,
    require_complete_task_difficulty,
)


ARRAY_FIELDS = (
    "cue_positions",
    "object_positions",
    "pocket_indices",
    "target_stop_positions",
    "generated_directions",
    "generated_speeds",
    "candidate_seeds",
    "elapsed_times",
    "min_object_pocket_distances",
    "event_flags",
)
PHYSICS_FIELDS = (
    "xml_hash",
    "model_hash",
    "physics_backend",
    "backend_hash",
    "execution_max_time",
    "stop_speed",
    "stop_hold_time",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--provenance-shards", nargs="+", type=Path, required=True)
    parser.add_argument("--augmentation-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--augmentation-count",
        type=int,
        default=DEFAULT_LOCAL_SPEED_AUGMENTATION_TASKS,
    )
    parser.add_argument("--training-count", type=int, default=DEFAULT_TRAIN_TASKS)
    return parser.parse_args()


def _require_compatible(
    reference: TwoBallTaskDataset,
    candidate: TwoBallTaskDataset,
    *,
    context: str,
) -> None:
    differences = [
        name
        for name in PHYSICS_FIELDS
        if getattr(reference, name) != getattr(candidate, name)
    ]
    if differences:
        raise ValueError(f"{context} physics mismatch: {differences}")


def _concatenate(
    datasets: Sequence[TwoBallTaskDataset],
    *,
    generation_seed: int,
) -> TwoBallTaskDataset:
    reference = datasets[0]
    return TwoBallTaskDataset(
        **{
            name: np.concatenate(
                [np.asarray(getattr(dataset, name)) for dataset in datasets],
                axis=0,
            )
            for name in ARRAY_FIELDS
        },
        xml_hash=reference.xml_hash,
        model_hash=reference.model_hash,
        physics_backend=reference.physics_backend,
        backend_hash=reference.backend_hash,
        generation_seed=generation_seed,
        execution_max_time=reference.execution_max_time,
        stop_speed=reference.stop_speed,
        stop_hold_time=reference.stop_hold_time,
    )


def _write_json(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(values, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if len(args.shards) != len(args.provenance_shards):
        raise ValueError("Every task shard requires one provenance shard.")
    if args.augmentation_count % LOCAL_SPEED_TASKS_PER_GROUP != 0:
        raise ValueError("Augmentation count must contain complete speed groups.")
    if args.training_count != DEFAULT_BALANCED_CORE_TASKS + args.augmentation_count:
        raise ValueError("Training count does not equal core plus augmentation.")
    outputs = (
        args.augmentation_output,
        args.provenance_output,
        args.training_output,
        args.manifest_output,
    )
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Merged local-speed output exists: " + ", ".join(map(str, existing))
        )

    core = TwoBallTaskDataset.load(args.core, validate_model=False)
    if len(core) != DEFAULT_BALANCED_CORE_TASKS:
        raise ValueError(
            f"Balanced core has {len(core)} tasks, expected "
            f"{DEFAULT_BALANCED_CORE_TASKS}."
        )
    require_balanced_task_difficulty(core, context="Balanced core")
    shards: list[TwoBallTaskDataset] = []
    provenance_metadata: list[dict[str, object]] = []
    provenance_arrays: list[dict[str, np.ndarray]] = []
    for task_path, provenance_path in zip(
        args.shards, args.provenance_shards, strict=True
    ):
        shard = TwoBallTaskDataset.load(task_path, validate_model=False)
        _require_compatible(core, shard, context=str(task_path))
        metadata, arrays = load_local_speed_provenance(provenance_path)
        require_local_speed_provenance(shard, core, metadata, arrays)
        shards.append(shard)
        provenance_metadata.append(metadata)
        provenance_arrays.append(arrays)

    augmentation = _concatenate(shards, generation_seed=shards[0].generation_seed)
    if len(augmentation) != args.augmentation_count:
        raise ValueError(
            f"Merged augmentation has {len(augmentation)} tasks, expected "
            f"{args.augmentation_count}."
        )
    merged_arrays = {
        name: np.concatenate([values[name] for values in provenance_arrays])
        for name in provenance_arrays[0]
    }
    group_indices = merged_arrays["global_group_indices"][::LOCAL_SPEED_TASKS_PER_GROUP]
    expected_group_indices = np.arange(
        args.augmentation_count // LOCAL_SPEED_TASKS_PER_GROUP,
        dtype=np.int64,
    )
    if not np.array_equal(group_indices, expected_group_indices):
        raise ValueError("Shard merge does not cover global groups exactly once in order.")
    if len(set(map(int, augmentation.candidate_seeds))) != len(augmentation):
        raise ValueError("Merged augmentation contains duplicate candidate seeds.")

    combined = LocalSpeedAugmentation(
        dataset=augmentation,
        source_task_indices=merged_arrays["source_task_indices"],
        global_group_indices=merged_arrays["global_group_indices"],
        requested_speed_offsets=merged_arrays["requested_speed_offsets"],
        actual_speed_offsets=merged_arrays["actual_speed_offsets"],
        source_speeds=merged_arrays["source_speeds"],
        source_content_sha256=core.content_sha256(),
        global_seed=int(provenance_metadata[0]["global_seed"]),
        shard_index=0,
        shard_count=1,
        global_task_count=args.augmentation_count,
    )
    require_local_speed_provenance(
        augmentation,
        core,
        {
            **provenance_metadata[0],
            "augmentation_content_sha256": augmentation.content_sha256(),
            "task_count": len(augmentation),
        },
        merged_arrays,
    )

    actual_group_counts = np.zeros((6, 9), dtype=np.int64)
    cells = augmentation.difficulty_indices()
    np.add.at(
        actual_group_counts,
        (
            augmentation.pocket_indices[::LOCAL_SPEED_TASKS_PER_GROUP].astype(np.int64),
            cells[::LOCAL_SPEED_TASKS_PER_GROUP].astype(np.int64),
        ),
        1,
    )
    expected_quotas = local_speed_group_quotas(args.augmentation_count)
    expected_group_counts = np.asarray(
        [
            [expected_quotas[(pocket, cell)] for cell in range(9)]
            for pocket in range(6)
        ],
        dtype=np.int64,
    )
    if not np.array_equal(actual_group_counts, expected_group_counts):
        raise ValueError("Merged augmentation does not match targeted group quotas.")

    group_stops = augmentation.target_stop_positions.reshape(
        -1, LOCAL_SPEED_TASKS_PER_GROUP, 2
    )
    outer_stop_displacement = np.linalg.norm(
        group_stops[:, -1] - group_stops[:, 0], axis=1
    )
    adjacent_stop_displacement = np.linalg.norm(
        np.diff(group_stops, axis=1), axis=2
    )

    training = _concatenate(
        [core, augmentation], generation_seed=core.generation_seed
    )
    if len(training) != args.training_count:
        raise ValueError(
            f"Merged training has {len(training)} tasks, expected {args.training_count}."
        )
    require_complete_task_difficulty(training, context="Targeted training")
    for name in ARRAY_FIELDS:
        if not np.array_equal(
            np.asarray(getattr(training, name))[: len(core)],
            np.asarray(getattr(core, name)),
        ):
            raise RuntimeError(f"Merged training changed core array {name}.")

    augmentation.save(args.augmentation_output)
    save_local_speed_provenance(combined, args.provenance_output)
    training.save(args.training_output)
    manifest = {
        "version": "targeted-local-speed-v1",
        "core": {
            "path": str(args.core),
            "task_count": len(core),
            "content_sha256": core.content_sha256(),
        },
        "augmentation": {
            "path": str(args.augmentation_output),
            "provenance_path": str(args.provenance_output),
            "task_count": len(augmentation),
            "group_count": len(augmentation) // LOCAL_SPEED_TASKS_PER_GROUP,
            "content_sha256": augmentation.content_sha256(),
            "pocket_cell_group_counts": actual_group_counts.tolist(),
            "corner_task_count": int(
                np.sum(actual_group_counts[:4]) * LOCAL_SPEED_TASKS_PER_GROUP
            ),
            "middle_task_count": int(
                np.sum(actual_group_counts[4:]) * LOCAL_SPEED_TASKS_PER_GROUP
            ),
            "long_cue_distance_task_count": int(
                np.sum(actual_group_counts[:, 6:9])
                * LOCAL_SPEED_TASKS_PER_GROUP
            ),
            "local_physics_response_m": {
                "outer_speed_stop_displacement_percentiles": {
                    name: float(value)
                    for name, value in zip(
                        ("min", "p05", "p25", "p50", "p75", "p95", "max"),
                        np.percentile(
                            outer_stop_displacement,
                            (0, 5, 25, 50, 75, 95, 100),
                        ),
                        strict=True,
                    )
                },
                "adjacent_speed_stop_displacement_percentiles": {
                    name: float(value)
                    for name, value in zip(
                        ("min", "p05", "p25", "p50", "p75", "p95", "max"),
                        np.percentile(
                            adjacent_stop_displacement,
                            (0, 5, 25, 50, 75, 95, 100),
                        ),
                        strict=True,
                    )
                },
            },
            "shards": [
                {
                    "path": str(path),
                    "provenance_path": str(provenance_path),
                    "task_count": len(shard),
                    "content_sha256": shard.content_sha256(),
                }
                for path, provenance_path, shard in zip(
                    args.shards,
                    args.provenance_shards,
                    shards,
                    strict=True,
                )
            ],
        },
        "training": {
            "path": str(args.training_output),
            "task_count": len(training),
            "content_sha256": training.content_sha256(),
            "difficulty_profile": training.difficulty_profile(),
            "core_prefix_preserved": True,
        },
    }
    _write_json(args.manifest_output, manifest)
    print(
        f"merged local-speed augmentation={len(augmentation)} "
        f"training={len(training)} manifest={args.manifest_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
