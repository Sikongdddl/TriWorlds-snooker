"""Merge independently replayed train/validation task shards safely."""

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

from snooker_env.midlevel_tasks import (  # noqa: E402
    DEFAULT_BALANCED_CORE_TASKS,
    DEFAULT_VALIDATION_TASKS,
    TwoBallTaskDataset,
    require_balanced_task_difficulty,
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
    parser.add_argument("--train-shards", nargs="+", type=Path, required=True)
    parser.add_argument("--validation-shards", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--train-output",
        type=Path,
        default=(
            ROOT
            / "outputs/tasks/midlevel_two_ball_train.balanced_unvalidated.npz"
        ),
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=(
            ROOT
            / "outputs/tasks/midlevel_two_ball_validation.balanced_unvalidated.npz"
        ),
    )
    parser.add_argument(
        "--train-count", type=int, default=DEFAULT_BALANCED_CORE_TASKS
    )
    parser.add_argument(
        "--validation-count",
        type=int,
        default=DEFAULT_VALIDATION_TASKS,
    )
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _require_compatible(
    reference: TwoBallTaskDataset,
    candidate: TwoBallTaskDataset,
    *,
    context: str,
) -> None:
    differences = [
        field
        for field in PHYSICS_FIELDS
        if getattr(reference, field) != getattr(candidate, field)
    ]
    if differences:
        raise ValueError(
            f"{context} physics mismatch: " + ", ".join(differences)
        )


def _task_identities(dataset: TwoBallTaskDataset) -> list[tuple[int, int, int]]:
    return list(
        zip(
            map(int, dataset.pocket_indices),
            map(int, dataset.difficulty_indices()),
            map(int, dataset.candidate_seeds),
            strict=True,
        )
    )


def _load_shards(
    paths: Sequence[Path],
    *,
    context: str,
) -> list[TwoBallTaskDataset]:
    if not paths:
        raise ValueError(f"{context} requires at least one shard.")
    datasets: list[TwoBallTaskDataset] = []
    occupied: set[tuple[int, int, int]] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        dataset = TwoBallTaskDataset.load(path, validate_model=False)
        if datasets:
            _require_compatible(datasets[0], dataset, context=context)
        require_balanced_task_difficulty(dataset, context=f"{context} shard {path}")
        identities = _task_identities(dataset)
        if len(set(identities)) != len(identities):
            raise ValueError(f"{context} shard contains duplicate tasks: {path}")
        overlap = occupied.intersection(identities)
        if overlap:
            raise ValueError(
                f"{context} shards contain {len(overlap)} duplicate task identities."
            )
        occupied.update(identities)
        datasets.append(dataset)
    return datasets


def _concatenate(
    datasets: Sequence[TwoBallTaskDataset],
    *,
    generation_seed: int,
) -> TwoBallTaskDataset:
    reference = datasets[0]
    arrays = {
        field: np.concatenate(
            [np.asarray(getattr(dataset, field)) for dataset in datasets],
            axis=0,
        )
        for field in ARRAY_FIELDS
    }
    return TwoBallTaskDataset(
        **arrays,
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
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _shard_manifest(
    paths: Sequence[Path],
    datasets: Sequence[TwoBallTaskDataset],
) -> list[dict[str, object]]:
    return [
        {
            "path": str(path),
            "task_count": len(dataset),
            "generation_seed": dataset.generation_seed,
            "content_sha256": dataset.content_sha256(),
        }
        for path, dataset in zip(paths, datasets, strict=True)
    ]


def main() -> None:
    args = parse_args()
    if args.train_count <= 0 or args.validation_count <= 0:
        raise ValueError("Expected split counts must be positive.")
    if args.train_output.resolve() == args.validation_output.resolve():
        raise ValueError("Train and validation outputs must be distinct.")
    manifest_output = args.manifest_output or Path(
        f"{args.train_output}.merge_manifest.json"
    )
    outputs = (args.train_output, args.validation_output, manifest_output)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Merged output exists; pass --overwrite to replace: "
            + ", ".join(map(str, existing))
        )

    train_shards = _load_shards(args.train_shards, context="Training")
    validation_shards = _load_shards(
        args.validation_shards,
        context="Validation",
    )
    _require_compatible(
        train_shards[0],
        validation_shards[0],
        context="Train/validation",
    )
    training = _concatenate(train_shards, generation_seed=0)
    validation = _concatenate(validation_shards, generation_seed=1)
    if len(training) != args.train_count:
        raise ValueError(
            f"Merged training count is {len(training)}, expected {args.train_count}."
        )
    if len(validation) != args.validation_count:
        raise ValueError(
            "Merged validation count is "
            f"{len(validation)}, expected {args.validation_count}."
        )
    require_balanced_task_difficulty(training, context="Merged training")
    require_balanced_task_difficulty(validation, context="Merged validation")
    overlap = set(_task_identities(training)).intersection(
        _task_identities(validation)
    )
    if overlap:
        raise ValueError(
            f"Train and validation contain {len(overlap)} overlapping tasks."
        )

    training.save(args.train_output)
    validation.save(args.validation_output)
    manifest: dict[str, object] = {
        "train": {
            "output": str(args.train_output),
            "task_count": len(training),
            "content_sha256": training.content_sha256(),
            "difficulty_profile": training.difficulty_profile(),
            "shards": _shard_manifest(args.train_shards, train_shards),
        },
        "validation": {
            "output": str(args.validation_output),
            "task_count": len(validation),
            "content_sha256": validation.content_sha256(),
            "difficulty_profile": validation.difficulty_profile(),
            "shards": _shard_manifest(
                args.validation_shards,
                validation_shards,
            ),
        },
        "train_validation_overlap_count": 0,
    }
    _write_json(manifest_output, manifest)
    print(
        f"merged train={len(training)} validation={len(validation)} "
        f"manifest={manifest_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
