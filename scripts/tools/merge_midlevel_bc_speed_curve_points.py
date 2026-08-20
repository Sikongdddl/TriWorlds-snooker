"""Validate and merge persisted real-physics BC speed-curve points."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from collect_midlevel_bc_speed_curves import _curve_report  # noqa: E402
from snooker_env.midlevel_two_ball import (  # noqa: E402
    MAX_CUE_SPEED,
    MIN_CUE_SPEED,
)


METADATA_FIELDS = {
    "checkpoint",
    "task_library",
    "batch_start",
    "num_worlds",
    "offset_mps",
    "task_indices",
    "bc_actions",
    "generated_actions",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("points", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loaded: list[dict[str, np.ndarray]] = []
    for path in args.points:
        with np.load(path, allow_pickle=False) as archive:
            loaded.append({name: np.asarray(archive[name]) for name in archive.files})
    loaded.sort(key=lambda point: float(point["offset_mps"]))
    offsets = np.asarray(
        [float(point["offset_mps"]) for point in loaded],
        dtype=np.float64,
    )
    if (
        len(np.unique(offsets)) != len(offsets)
        or not np.all(np.diff(offsets) > 0.0)
        or not np.any(offsets == 0.0)
        or not np.allclose(offsets, -offsets[::-1], atol=1.0e-12)
    ):
        raise ValueError("Curve points must be unique, sorted, symmetric, and include zero.")
    reference = loaded[0]
    for point in loaded[1:]:
        for field in (
            "checkpoint",
            "task_library",
            "batch_start",
            "num_worlds",
            "task_indices",
            "bc_actions",
            "generated_actions",
            "observation",
        ):
            if not np.array_equal(point[field], reference[field]):
                raise ValueError(f"Curve point metadata disagrees on {field}.")
    data_fields = sorted(set(reference) - METADATA_FIELDS)
    stacked = {
        field: np.stack([point[field] for point in loaded], axis=0)
        for field in data_fields
    }
    speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    correction = (
        reference["generated_actions"][:, 1].astype(np.float64)
        - reference["bc_actions"][:, 1].astype(np.float64)
    ) * speed_half_range
    speed_error = np.abs(correction)
    report = {
        "curve_version": "frozen-bc-real-physics-speed-curve-v1",
        "checkpoint": str(reference["checkpoint"]),
        "task_library": str(reference["task_library"]),
        "batch_start": int(reference["batch_start"]),
        "task_count": int(reference["num_worlds"]),
        "num_worlds": int(reference["num_worlds"]),
        "offsets_mps": [float(value) for value in offsets],
        "world_slot_aligned": True,
        "offset_execution": "independent_same_task_slot",
        "bc_speed_error_mps": {
            "mean": float(np.mean(speed_error)),
            "p50": float(np.percentile(speed_error, 50)),
            "p95": float(np.percentile(speed_error, 95)),
            "max": float(np.max(speed_error)),
        },
        **_curve_report(offsets, stacked, correction),
    }
    if not all(
        math.isfinite(value)
        for value in (
            report["bc_speed_error_mps"]["mean"],
            report["curve_roughness"]["adjacent_reward_change_mean"],
        )
    ):
        raise FloatingPointError("Merged curve report contains non-finite metrics.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        task_indices=reference["task_indices"],
        offsets_mps=offsets,
        bc_actions=reference["bc_actions"],
        generated_actions=reference["generated_actions"],
        generated_speed_correction_mps=correction,
        **stacked,
    )
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
