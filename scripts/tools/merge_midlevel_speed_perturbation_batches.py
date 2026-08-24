"""Validate and merge complete canonical speed-perturbation batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_ppo import generated_behavior_cloning_data  # noqa: E402
from snooker_env.midlevel_ppo_env import MAX_TERMINAL_REWARD  # noqa: E402
from snooker_env.midlevel_tasks import TwoBallTaskDataset  # noqa: E402


FORMAT_VERSION = "canonical-generated-speed-perturbations-v1"
NON_MERGED_FIELDS = {"metadata", "offsets_mps"}


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_batch(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if "metadata" not in arrays:
        raise ValueError(f"{path} has no metadata.")
    metadata = json.loads(str(arrays["metadata"].item()))
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path} has an incompatible format version.")
    if metadata.get("center_action_source") != "canonical_generated_action":
        raise ValueError(f"{path} is not centered on canonical generated actions.")
    return metadata, arrays


def _distribution(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", type=Path, nargs="+")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--expected-task-count", type=int, required=True)
    parser.add_argument("--center-stop-tolerance", type=float, default=5e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = TwoBallTaskDataset.load(args.tasks, validate_model=False)
    if len(dataset) != args.expected_task_count:
        raise ValueError(
            f"Final task library has {len(dataset)} tasks; expected "
            f"{args.expected_task_count}."
        )
    reference_observations, reference_actions = generated_behavior_cloning_data(
        dataset
    )
    loaded = []
    for path in args.batches:
        metadata, arrays = _load_batch(path)
        task_indices = arrays["task_indices"].astype(np.int64)
        if len(task_indices) != int(metadata["task_count"]):
            raise ValueError(f"{path} task count disagrees with metadata.")
        if not np.array_equal(
            task_indices,
            np.arange(task_indices[0], task_indices[0] + len(task_indices)),
        ):
            raise ValueError(f"{path} task indices are not contiguous.")
        if not np.array_equal(
            arrays["observation"], reference_observations[task_indices]
        ):
            raise ValueError(f"{path} observations do not match the final library.")
        expected_actions = reference_actions[task_indices].copy()
        expected_actions[:, 0] = 0.0
        if not np.array_equal(arrays["center_action"], expected_actions):
            raise ValueError(f"{path} center actions do not match the final library.")
        expected_targets = dataset.target_stop_positions[task_indices].astype(
            np.float32
        )
        if not np.array_equal(
            arrays["target_stop_position"], expected_targets
        ):
            raise ValueError(f"{path} target stops do not match the final library.")
        loaded.append((path, metadata, arrays))

    loaded.sort(key=lambda item: int(item[2]["task_indices"][0]))
    offsets = loaded[0][2]["offsets_mps"].astype(np.float64)
    field_names = set(loaded[0][2])
    for path, _, arrays in loaded[1:]:
        if set(arrays) != field_names:
            raise ValueError(f"{path} has different fields from the first batch.")
        if not np.array_equal(arrays["offsets_mps"], offsets):
            raise ValueError(f"{path} has different speed offsets.")
    task_indices = np.concatenate(
        [item[2]["task_indices"] for item in loaded], axis=0
    )
    expected_indices = np.arange(args.expected_task_count, dtype=np.int64)
    if not np.array_equal(task_indices, expected_indices):
        missing = np.setdiff1d(expected_indices, task_indices)[:10].tolist()
        duplicates = len(task_indices) - len(np.unique(task_indices))
        raise ValueError(
            "Batches do not cover the final task library exactly: "
            f"records={len(task_indices)} missing_head={missing} "
            f"duplicate_count={duplicates}."
        )

    merged: dict[str, np.ndarray] = {}
    for name in sorted(field_names - NON_MERGED_FIELDS):
        sample = loaded[0][2][name]
        batch_count = len(loaded[0][2]["task_indices"])
        if sample.ndim >= 2 and sample.shape[:2] == (len(offsets), batch_count):
            axis = 1
        elif sample.ndim >= 1 and sample.shape[0] == batch_count:
            axis = 0
        else:
            raise ValueError(
                f"Cannot identify the task axis for field {name}: {sample.shape}."
            )
        merged[name] = np.concatenate(
            [item[2][name] for item in loaded], axis=axis
        )

    center_index = int(np.flatnonzero(offsets == 0.0)[0])
    center_valid = (
        merged["correct_pot"][center_index]
        & ~merged["cue_scratch"][center_index]
        & ~merged["wrong_pocket"][center_index]
        & merged["stopped"][center_index]
        & ~merged["timed_out"][center_index]
        & ~merged["numerical_failure"][center_index]
    )
    center_error = merged["cue_target_error_m"][center_index]
    if not np.all(center_valid):
        raise RuntimeError("Merged canonical center contains infeasible outcomes.")
    if np.any(center_error > args.center_stop_tolerance):
        raise RuntimeError(
            "Merged canonical center exceeds the stop tolerance: "
            f"{np.max(center_error):.6g}m."
        )
    if not np.allclose(
        merged["reward"][center_index],
        MAX_TERMINAL_REWARD,
        atol=1e-6,
        rtol=0.0,
    ):
        raise RuntimeError("Merged canonical center reward is not maximal.")

    source_hashes = sorted(
        {
            str(metadata["source_task_library_content_sha256"])
            for _, metadata, _ in loaded
        }
    )
    final_metadata = {
        "format_version": FORMAT_VERSION,
        "center_action_source": "canonical_generated_action",
        "task_library": str(args.tasks),
        "task_library_content_sha256": dataset.content_sha256(),
        "task_count": len(dataset),
        "offsets_mps": [float(value) for value in offsets],
        "offset_count": len(offsets),
        "record_count": int(len(dataset) * len(offsets)),
        "batch_count": len(loaded),
        "source_task_library_content_sha256": source_hashes,
        "world_slot_aligned": True,
    }
    _atomic_savez(
        args.output,
        metadata=np.asarray(json.dumps(final_metadata, sort_keys=True)),
        offsets_mps=offsets,
        **merged,
    )

    per_offset = []
    for index, offset in enumerate(offsets):
        displacement = np.linalg.norm(
            merged["cue_final_delta_xy_m"][index], axis=1
        )
        per_offset.append(
            {
                "offset_mps": float(offset),
                "reward_mean": float(np.mean(merged["reward"][index])),
                "correct_pot_rate": float(
                    np.mean(merged["correct_pot"][index])
                ),
                "joint_success_rate": float(
                    np.mean(merged["joint_success"][index])
                ),
                "scratch_rate": float(np.mean(merged["cue_scratch"][index])),
                "cue_displacement_from_center_m": _distribution(displacement),
                "cue_target_error_m": _distribution(
                    merged["cue_target_error_m"][index]
                ),
            }
        )
    report = {
        **final_metadata,
        "canonical_center_max_stop_error_m": float(np.max(center_error)),
        "canonical_center_passed_count": int(np.count_nonzero(center_valid)),
        "per_offset": per_offset,
        "output": str(args.output),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"speed_perturbation_merge=PASS batches={len(loaded)} "
        f"tasks={len(dataset)} offsets={len(offsets)} "
        f"records={len(dataset) * len(offsets)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
