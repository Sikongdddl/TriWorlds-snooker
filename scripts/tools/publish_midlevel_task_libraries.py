"""Publish merged task libraries only after complete range-replay coverage."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_bc import write_json  # noqa: E402
from snooker_env.midlevel_tasks import (  # noqa: E402
    TwoBallTaskDataset,
    require_balanced_task_difficulty,
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
    parser.add_argument("--train-staged", type=Path, required=True)
    parser.add_argument("--validation-staged", type=Path, required=True)
    parser.add_argument("--train-reports", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--validation-reports",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=ROOT / "outputs/tasks/midlevel_two_ball_train.npz",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=ROOT / "outputs/tasks/midlevel_two_ball_validation.npz",
    )
    parser.add_argument(
        "--backup-directory",
        type=Path,
        default=ROOT / "outputs/tasks/archive/pre_difficulty_grid_v1",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=(
            ROOT / "outputs/tasks/midlevel_two_ball_difficulty_grid_v1.json"
        ),
    )
    return parser.parse_args()


def _read_report(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"Replay report is malformed: {path}")
    return values


def _verify_reports(
    dataset: TwoBallTaskDataset,
    paths: Sequence[Path],
    *,
    context: str,
) -> dict[str, object]:
    expected_hash = dataset.content_sha256()
    ranges: list[tuple[int, int]] = []
    maximum_error = 0.0
    for path in paths:
        report = _read_report(path)
        if report.get("dataset_content_sha256") != expected_hash:
            raise ValueError(f"{context} replay hash mismatch: {path}")
        if int(report.get("dataset_task_count", -1)) != len(dataset):
            raise ValueError(f"{context} replay task count mismatch: {path}")
        start = int(report.get("start_task", -1))
        end = int(report.get("end_task_exclusive", -1))
        checked = int(report.get("checked_count", -1))
        passed = int(report.get("passed_count", -1))
        failures = report.get("failures")
        if end - start != checked or passed != checked or failures != []:
            raise ValueError(f"{context} replay did not pass completely: {path}")
        ranges.append((start, end))
        reported_error = float(
            report.get("max_stop_replay_error_m", float("inf"))
        )
        if not math.isfinite(reported_error) or reported_error < 0.0:
            raise ValueError(f"{context} replay error is malformed: {path}")
        maximum_error = max(maximum_error, reported_error)
    ranges.sort()
    cursor = 0
    for start, end in ranges:
        if start != cursor or end <= start:
            raise ValueError(
                f"{context} replay ranges have a gap or overlap at task {cursor}."
            )
        cursor = end
    if cursor != len(dataset):
        raise ValueError(
            f"{context} replay covers {cursor}/{len(dataset)} tasks."
        )
    return {
        "report_paths": [str(path) for path in paths],
        "checked_count": cursor,
        "passed_count": cursor,
        "max_stop_replay_error_m": maximum_error,
        "ranges": [list(values) for values in ranges],
    }


def _task_identities(dataset: TwoBallTaskDataset) -> set[tuple[int, int, int]]:
    return set(
        zip(
            map(int, dataset.pocket_indices),
            map(int, dataset.difficulty_indices()),
            map(int, dataset.candidate_seeds),
            strict=True,
        )
    )


def _backup_existing(source: Path, destination: Path) -> dict[str, object] | None:
    if not source.exists():
        return None
    current = TwoBallTaskDataset.load(source, validate_model=False)
    current_hash = current.content_sha256()
    if destination.exists():
        backup = TwoBallTaskDataset.load(destination, validate_model=False)
        if backup.content_sha256() != current_hash:
            raise FileExistsError(
                f"Existing backup does not match the library being replaced: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return {
        "path": str(destination),
        "task_count": len(current),
        "content_sha256": current_hash,
    }


def _prepare_published_copy(
    source: Path,
    destination: Path,
    expected_hash: str,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copy2(source, temporary_path)
        copied = TwoBallTaskDataset.load(temporary_path, validate_model=False)
        if copied.content_sha256() != expected_hash:
            raise RuntimeError("Prepared publication copy changed content hash.")
        return temporary_path
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    training = TwoBallTaskDataset.load(args.train_staged, validate_model=False)
    validation = TwoBallTaskDataset.load(
        args.validation_staged,
        validate_model=False,
    )
    require_balanced_task_difficulty(training, context="Published training")
    require_balanced_task_difficulty(validation, context="Published validation")
    physics_differences = [
        field
        for field in PHYSICS_FIELDS
        if getattr(training, field) != getattr(validation, field)
    ]
    if physics_differences:
        raise ValueError(
            "Published train/validation physics mismatch: "
            + ", ".join(physics_differences)
        )
    overlap = _task_identities(training).intersection(
        _task_identities(validation)
    )
    if overlap:
        raise ValueError(
            f"Train and validation contain {len(overlap)} overlapping tasks."
        )
    train_replay = _verify_reports(
        training,
        args.train_reports,
        context="Training",
    )
    validation_replay = _verify_reports(
        validation,
        args.validation_reports,
        context="Validation",
    )

    train_backup = _backup_existing(
        args.train_output,
        args.backup_directory / args.train_output.name,
    )
    validation_backup = _backup_existing(
        args.validation_output,
        args.backup_directory / args.validation_output.name,
    )
    train_hash = training.content_sha256()
    validation_hash = validation.content_sha256()
    train_temporary = _prepare_published_copy(
        args.train_staged,
        args.train_output,
        train_hash,
    )
    validation_temporary = _prepare_published_copy(
        args.validation_staged,
        args.validation_output,
        validation_hash,
    )
    try:
        os.replace(train_temporary, args.train_output)
        train_temporary = None
        os.replace(validation_temporary, args.validation_output)
        validation_temporary = None
    finally:
        if train_temporary is not None:
            train_temporary.unlink(missing_ok=True)
        if validation_temporary is not None:
            validation_temporary.unlink(missing_ok=True)

    manifest: dict[str, object] = {
        "version": "two-distance-grid-v1",
        "train": {
            "path": str(args.train_output),
            "task_count": len(training),
            "content_sha256": train_hash,
            "difficulty_profile": training.difficulty_profile(),
            "replay": train_replay,
            "previous_library": train_backup,
        },
        "validation": {
            "path": str(args.validation_output),
            "task_count": len(validation),
            "content_sha256": validation_hash,
            "difficulty_profile": validation.difficulty_profile(),
            "replay": validation_replay,
            "previous_library": validation_backup,
        },
        "train_validation_overlap_count": 0,
    }
    write_json(args.manifest_output, manifest)
    print(
        f"published train={len(training)} validation={len(validation)} "
        f"manifest={args.manifest_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
