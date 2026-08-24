"""Audit split disjointness and seal the seed-7 library for final-only use."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_tasks import TwoBallTaskDataset  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_identities(dataset: TwoBallTaskDataset) -> set[tuple[int, int]]:
    """Return the deterministic generation identity for every task.

    Candidate seeds are 32-bit and may legitimately collide at this scale.
    The pocket is an input to candidate generation, so only the composite key
    identifies a repeated task.
    """

    identities = set(
        zip(
            map(int, dataset.pocket_indices),
            map(int, dataset.candidate_seeds),
            strict=True,
        )
    )
    if len(identities) != len(dataset):
        raise ValueError("A split contains duplicate task generation identities.")
    return identities


def exact_task_payloads(dataset: TwoBallTaskDataset) -> np.ndarray:
    """Return exact row keys for policy inputs and generated shot labels."""

    payload = np.ascontiguousarray(
        np.column_stack(
            (
                dataset.cue_positions,
                dataset.object_positions,
                dataset.pocket_indices.astype(np.float64),
                dataset.target_stop_positions,
                dataset.generated_directions,
                dataset.generated_speeds,
            )
        ),
        dtype=np.float64,
    )
    row_dtype = np.dtype((np.void, payload.shape[1] * payload.dtype.itemsize))
    rows = payload.view(row_dtype).reshape(-1)
    if len(np.unique(rows)) != len(dataset):
        raise ValueError("A split contains duplicate exact task payloads.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    train = TwoBallTaskDataset.load(args.train, validate_model=False)
    development = TwoBallTaskDataset.load(
        args.development,
        validate_model=False,
    )
    test = TwoBallTaskDataset.load(args.test, validate_model=False)
    if len(train) != 208_896:
        raise ValueError(f"Expected 208896 training tasks, got {len(train)}.")
    if len(development) != 12_288:
        raise ValueError(
            f"Expected 12288 development tasks, got {len(development)}."
        )
    if len(test) != 12_288 or test.generation_seed != 7:
        raise ValueError(
            "Fresh test must contain 12288 tasks generated with seed 7."
        )
    for candidate, label in (
        (development, "development"),
        (test, "test"),
    ):
        for field in (
            "xml_hash",
            "model_hash",
            "physics_backend",
            "backend_hash",
            "execution_max_time",
            "stop_speed",
            "stop_hold_time",
        ):
            if getattr(candidate, field) != getattr(train, field):
                raise ValueError(f"{label} physics field {field!r} differs.")
    raw_seed_overlap = {
        "train_development": int(
            len(np.intersect1d(train.candidate_seeds, development.candidate_seeds))
        ),
        "train_test": int(
            len(np.intersect1d(train.candidate_seeds, test.candidate_seeds))
        ),
        "development_test": int(
            len(
                np.intersect1d(
                    development.candidate_seeds,
                    test.candidate_seeds,
                )
            )
        ),
    }
    identities = {
        "train": task_identities(train),
        "development": task_identities(development),
        "test": task_identities(test),
    }
    identity_overlap = {
        "train_development": len(
            identities["train"].intersection(identities["development"])
        ),
        "train_test": len(
            identities["train"].intersection(identities["test"])
        ),
        "development_test": len(
            identities["development"].intersection(identities["test"])
        ),
    }
    if any(identity_overlap.values()):
        raise RuntimeError(
            "Task generation identities overlap across splits: "
            f"{identity_overlap}"
        )
    payloads = {
        "train": exact_task_payloads(train),
        "development": exact_task_payloads(development),
        "test": exact_task_payloads(test),
    }
    payload_overlap = {
        "train_development": int(
            len(np.intersect1d(payloads["train"], payloads["development"]))
        ),
        "train_test": int(
            len(np.intersect1d(payloads["train"], payloads["test"]))
        ),
        "development_test": int(
            len(np.intersect1d(payloads["development"], payloads["test"]))
        ),
    }
    if any(payload_overlap.values()):
        raise RuntimeError(
            f"Exact task payloads overlap across splits: {payload_overlap}"
        )
    pocket_counts = np.bincount(test.pocket_indices, minlength=6).tolist()
    if pocket_counts != [2_048] * 6:
        raise ValueError(f"Fresh test pocket counts are unbalanced: {pocket_counts}")

    report = {
        "version": "midlevel-fresh-test-seal-v2",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_before_formal_development_selection",
        "policy": {
            "allowed_use": (
                "one deterministic final evaluation after development-only "
                "model selection is locked"
            ),
            "forbidden_uses": [
                "training",
                "early_stopping",
                "hyperparameter_selection",
                "seed_selection",
                "model_selection",
            ],
        },
        "train": {
            "path": str(args.train),
            "task_count": len(train),
            "content_sha256": train.content_sha256(),
            "archive_sha256": file_sha256(args.train),
        },
        "development": {
            "path": str(args.development),
            "task_count": len(development),
            "content_sha256": development.content_sha256(),
            "archive_sha256": file_sha256(args.development),
        },
        "test": {
            "path": str(args.test),
            "task_count": len(test),
            "generation_seed": test.generation_seed,
            "pocket_counts": pocket_counts,
            "content_sha256": test.content_sha256(),
            "archive_sha256": file_sha256(args.test),
        },
        "task_identity_overlap_counts": identity_overlap,
        "exact_task_payload_overlap_counts": payload_overlap,
        "raw_candidate_seed_overlap_counts_informational": raw_seed_overlap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
