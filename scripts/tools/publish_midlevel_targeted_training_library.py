#!/usr/bin/env python3
"""Publish the 393,216-task targeted training set after complete GPU replay."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

import numpy as np

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_bc import write_json  # noqa: E402
from snooker_env.midlevel_local_speed_tasks import (  # noqa: E402
    load_local_speed_provenance,
    require_local_speed_provenance,
)
from snooker_env.midlevel_tasks import (  # noqa: E402
    DEFAULT_BALANCED_CORE_TASKS,
    DEFAULT_LOCAL_SPEED_AUGMENTATION_TASKS,
    DEFAULT_TRAIN_TASKS,
    DEFAULT_VALIDATION_TASKS,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-staged", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--augmentation", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--merge-manifest", type=Path, required=True)
    parser.add_argument("--replay-reports", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--validation",
        type=Path,
        default=ROOT / "outputs/tasks/midlevel_two_ball_validation.npz",
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=ROOT / "outputs/tasks/midlevel_two_ball_train.npz",
    )
    parser.add_argument(
        "--backup-output",
        type=Path,
        default=(
            ROOT
            / "outputs/tasks/archive/pre_targeted_local_speed_v1/"
            "midlevel_two_ball_train.npz"
        ),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=(
            ROOT / "outputs/tasks/midlevel_two_ball_targeted_local_speed_v1.json"
        ),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"JSON artifact is malformed: {path}")
    return values


def _verify_reports(
    dataset: TwoBallTaskDataset,
    paths: Sequence[Path],
) -> dict[str, object]:
    ranges: list[tuple[int, int]] = []
    maximum_error = 0.0
    expected_hash = dataset.content_sha256()
    for path in paths:
        report = _read_json(path)
        if report.get("dataset_content_sha256") != expected_hash:
            raise ValueError(f"Replay report hash mismatch: {path}")
        if int(report.get("dataset_task_count", -1)) != len(dataset):
            raise ValueError(f"Replay report task count mismatch: {path}")
        start = int(report.get("start_task", -1))
        end = int(report.get("end_task_exclusive", -1))
        checked = int(report.get("checked_count", -1))
        passed = int(report.get("passed_count", -1))
        if end - start != checked or passed != checked or report.get("failures") != []:
            raise ValueError(f"Replay report did not pass fully: {path}")
        error = float(report.get("max_stop_replay_error_m", float("inf")))
        if not math.isfinite(error) or error < 0.0:
            raise ValueError(f"Replay report error is malformed: {path}")
        maximum_error = max(maximum_error, error)
        ranges.append((start, end))
    ranges.sort()
    cursor = 0
    for start, end in ranges:
        if start != cursor or end <= start:
            raise ValueError(f"Replay ranges have a gap or overlap at {cursor}.")
        cursor = end
    if cursor != len(dataset):
        raise ValueError(f"Replay covers {cursor}/{len(dataset)} training tasks.")
    return {
        "report_paths": [str(path) for path in paths],
        "checked_count": cursor,
        "passed_count": cursor,
        "max_stop_replay_error_m": maximum_error,
        "ranges": [list(value) for value in ranges],
    }


def _identity_set(dataset: TwoBallTaskDataset) -> set[tuple[int, int, int]]:
    return set(
        zip(
            map(int, dataset.pocket_indices),
            map(int, dataset.difficulty_indices()),
            map(int, dataset.candidate_seeds),
            strict=True,
        )
    )


def main() -> None:
    args = parse_args()
    training = TwoBallTaskDataset.load(args.training_staged, validate_model=False)
    core = TwoBallTaskDataset.load(args.core, validate_model=False)
    augmentation = TwoBallTaskDataset.load(args.augmentation, validate_model=False)
    validation = TwoBallTaskDataset.load(args.validation, validate_model=False)
    if len(training) != DEFAULT_TRAIN_TASKS:
        raise ValueError(f"Training count is {len(training)}, expected {DEFAULT_TRAIN_TASKS}.")
    if len(core) != DEFAULT_BALANCED_CORE_TASKS:
        raise ValueError("Balanced core task count is incorrect.")
    if len(augmentation) != DEFAULT_LOCAL_SPEED_AUGMENTATION_TASKS:
        raise ValueError("Local-speed augmentation task count is incorrect.")
    if len(validation) != DEFAULT_VALIDATION_TASKS:
        raise ValueError("Validation task count is incorrect.")
    require_complete_task_difficulty(training, context="Published training")
    require_balanced_task_difficulty(core, context="Published core")
    require_balanced_task_difficulty(validation, context="Published validation")
    for name in ARRAY_FIELDS:
        training_values = np.asarray(getattr(training, name))
        if not np.array_equal(training_values[: len(core)], getattr(core, name)):
            raise ValueError(f"Training does not preserve core field {name}.")
        if not np.array_equal(training_values[len(core) :], getattr(augmentation, name)):
            raise ValueError(f"Training does not preserve augmentation field {name}.")

    metadata, arrays = load_local_speed_provenance(args.provenance)
    require_local_speed_provenance(augmentation, core, metadata, arrays)
    merge_manifest = _read_json(args.merge_manifest)
    if (
        merge_manifest.get("training", {}).get("content_sha256")
        != training.content_sha256()
    ):
        raise ValueError("Merge manifest does not name the staged training set.")
    overlap = _identity_set(training).intersection(_identity_set(validation))
    if overlap:
        raise ValueError(f"Training and validation overlap by {len(overlap)} tasks.")
    replay = _verify_reports(training, args.replay_reports)

    if args.training_output.exists():
        current = TwoBallTaskDataset.load(args.training_output, validate_model=False)
        if current.content_sha256() != core.content_sha256():
            raise ValueError(
                "Current published training library is not the declared balanced core."
            )
        if args.backup_output.exists():
            backup = TwoBallTaskDataset.load(args.backup_output, validate_model=False)
            if backup.content_sha256() != core.content_sha256():
                raise FileExistsError("Existing core backup has different content.")
        else:
            args.backup_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.training_output, args.backup_output)

    args.training_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=args.training_output.parent,
        prefix=f".{args.training_output.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copy2(args.training_staged, temporary_path)
        copied = TwoBallTaskDataset.load(temporary_path, validate_model=False)
        if copied.content_sha256() != training.content_sha256():
            raise RuntimeError("Prepared training copy changed its content hash.")
        os.replace(temporary_path, args.training_output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    manifest = {
        "version": "targeted-local-speed-v1",
        "training": {
            "path": str(args.training_output),
            "task_count": len(training),
            "content_sha256": training.content_sha256(),
            "difficulty_profile": training.difficulty_profile(),
            "replay": replay,
        },
        "balanced_core": {
            "path": str(args.backup_output),
            "task_count": len(core),
            "content_sha256": core.content_sha256(),
        },
        "local_speed_augmentation": {
            "path": str(args.augmentation),
            "provenance_path": str(args.provenance),
            "task_count": len(augmentation),
            "content_sha256": augmentation.content_sha256(),
        },
        "validation": {
            "path": str(args.validation),
            "task_count": len(validation),
            "content_sha256": validation.content_sha256(),
            "unchanged": True,
        },
        "merge_manifest": str(args.merge_manifest),
        "train_validation_overlap_count": 0,
    }
    write_json(args.manifest_output, manifest)
    print(
        f"published targeted training={len(training)} "
        f"validation_unchanged={len(validation)} manifest={args.manifest_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
