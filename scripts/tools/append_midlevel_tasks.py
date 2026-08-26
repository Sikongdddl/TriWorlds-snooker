"""Append a fully validated task library and republish after complete replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.midlevel_mujoco_warp_tasks import (  # noqa: E402
    validate_mujoco_warp_task_dataset,
)
from snooker_env.midlevel_tasks import (  # noqa: E402
    EVENT_FLAG_NAMES,
    MUJOCO_WARP_PHYSICS_BACKEND,
    POCKET_NAMES,
    TwoBallTaskDataset,
)
from snooker_env.midlevel_two_ball import TwoBallShotSimulator  # noqa: E402


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


def _require_compatible(
    base: TwoBallTaskDataset,
    addition: TwoBallTaskDataset,
) -> None:
    fields = (
        "xml_hash",
        "model_hash",
        "physics_backend",
        "backend_hash",
        "execution_max_time",
        "stop_speed",
        "stop_hold_time",
    )
    for field in fields:
        if getattr(base, field) != getattr(addition, field):
            raise ValueError(f"Task libraries disagree on {field}.")
    # ``candidate_seed`` is interpreted together with the requested pocket and
    # distance cell by ``_sample_candidate``. Reject only collisions of that
    # complete deterministic generation identity.
    base_identities = set(
        zip(
            map(int, base.pocket_indices),
            map(int, base.difficulty_indices()),
            map(int, base.candidate_seeds),
            strict=True,
        )
    )
    addition_identities = set(
        zip(
            map(int, addition.pocket_indices),
            map(int, addition.difficulty_indices()),
            map(int, addition.candidate_seeds),
            strict=True,
        )
    )
    overlapping_identities = base_identities.intersection(addition_identities)
    if overlapping_identities:
        raise ValueError(
            "Task libraries contain overlapping "
            "(pocket, difficulty_cell, candidate_seed) "
            "identities; refusing to append "
            f"{len(overlapping_identities)} duplicate tasks."
        )


def _pocket_counts(dataset: TwoBallTaskDataset) -> list[int]:
    return np.bincount(
        dataset.pocket_indices,
        minlength=len(POCKET_NAMES),
    ).astype(int).tolist()


def concatenate_datasets(
    base: TwoBallTaskDataset,
    addition: TwoBallTaskDataset,
) -> TwoBallTaskDataset:
    """Concatenate compatible datasets without changing task ordering."""

    _require_compatible(base, addition)
    arrays = {
        field: np.concatenate(
            [np.asarray(getattr(base, field)), np.asarray(getattr(addition, field))],
            axis=0,
        )
        for field in ARRAY_FIELDS
    }
    return TwoBallTaskDataset(
        **arrays,
        xml_hash=base.xml_hash,
        model_hash=base.model_hash,
        physics_backend=base.physics_backend,
        backend_hash=base.backend_hash,
        # Dataset version 4 stores one primary generation seed.  Preserve the
        # original seed for compatibility and record full provenance in the
        # append manifest written beside the published archive.
        generation_seed=base.generation_seed,
        execution_max_time=base.execution_max_time,
        stop_speed=base.stop_speed,
        stop_hold_time=base.stop_hold_time,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("addition", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--physics-device", default="cuda:0")
    parser.add_argument("--num-worlds", type=int, default=4096)
    parser.add_argument("--chunk-steps", type=int, default=64)
    parser.add_argument("--check-interval-steps", type=int, default=8192)
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=1024)
    parser.add_argument("--max-shot-time", type=float, default=8.0)
    parser.add_argument("--stop-tolerance", type=float, default=5e-3)
    args = parser.parse_args()

    if args.num_worlds <= 0:
        raise ValueError("--num-worlds must be positive.")
    if args.output.resolve() != args.base.resolve():
        raise ValueError(
            "Safe append requires --output to name the existing base library."
        )
    if args.backup.resolve() in (args.base.resolve(), args.addition.resolve()):
        raise ValueError("--backup must be distinct from both input libraries.")

    simulator = TwoBallShotSimulator(args.model, max_time=args.max_shot_time)
    base = TwoBallTaskDataset.load(
        args.base,
        simulator=simulator,
        expected_backend=MUJOCO_WARP_PHYSICS_BACKEND,
    )
    addition = TwoBallTaskDataset.load(
        args.addition,
        simulator=simulator,
        expected_backend=MUJOCO_WARP_PHYSICS_BACKEND,
    )
    merged = concatenate_datasets(base, addition)

    addition_pockets = _pocket_counts(addition)
    if len(set(addition_pockets)) != 1:
        raise ValueError(
            "Addition is not balanced across all pockets: "
            f"{dict(zip(POCKET_NAMES, addition_pockets))}"
        )
    merged_pockets = _pocket_counts(merged)
    if len(set(merged_pockets)) != 1:
        raise ValueError(
            "Merged library is not balanced across all pockets: "
            f"{dict(zip(POCKET_NAMES, merged_pockets))}"
        )

    staged = args.output.with_name(
        f"{args.output.stem}.append_unvalidated{args.output.suffix}"
    )
    merged.save(staged)
    print(
        f"append: staged={staged} base={len(base)} addition={len(addition)} "
        f"merged={len(merged)}",
        flush=True,
    )
    report = validate_mujoco_warp_task_dataset(
        merged,
        model_path=simulator.model_path,
        max_tasks=None,
        num_worlds=args.num_worlds,
        device=args.physics_device,
        chunk_steps=args.chunk_steps,
        check_interval_steps=args.check_interval_steps,
        nconmax=args.nconmax,
        njmax=args.njmax,
        max_time=args.max_shot_time,
        stop_tolerance=args.stop_tolerance,
    )
    print(
        f"append: replay={report.passed_count}/{report.checked_count} "
        f"max_stop_error={report.max_stop_replay_error:.6g}m",
        flush=True,
    )
    if report.failures:
        raise RuntimeError("; ".join(report.failures[:10]))

    base_hash = base.content_sha256()
    addition_hash = addition.content_sha256()
    if args.backup.exists():
        backup = TwoBallTaskDataset.load(args.backup, validate_model=False)
        if backup.content_sha256() != base_hash:
            raise FileExistsError(
                f"Existing backup does not match the base library: {args.backup}"
            )
    else:
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.base, args.backup)

    staged.replace(args.output)
    manifest = {
        "base": {
            "path": str(args.backup),
            "task_count": len(base),
            "generation_seed": base.generation_seed,
            "content_sha256": base_hash,
        },
        "addition": {
            "path": str(args.addition),
            "task_count": len(addition),
            "generation_seed": addition.generation_seed,
            "content_sha256": addition_hash,
        },
        "published": {
            "path": str(args.output),
            "task_count": len(merged),
            "content_sha256": merged.content_sha256(),
            "pocket_counts": dict(zip(POCKET_NAMES, merged_pockets)),
        },
        "replay": {
            "checked_count": report.checked_count,
            "passed_count": report.passed_count,
            "max_stop_replay_error_m": report.max_stop_replay_error,
        },
        "event_flag_names": EVENT_FLAG_NAMES,
    }
    manifest_path = Path(f"{args.output}.append_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"append: output={args.output} tasks={len(merged)} "
        f"backup={args.backup} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
