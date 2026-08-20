"""Merge task-disjoint speed-curve batches on a common offset subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


TASK_FIELDS = {
    "task_indices",
    "bc_actions",
    "generated_actions",
    "generated_speed_correction_mps",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("curves", type=Path, nargs="+")
    parser.add_argument(
        "--offsets-mps",
        type=float,
        nargs="+",
        default=(-0.03, -0.01, 0.0, 0.01, 0.03),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requested_offsets = np.asarray(args.offsets_mps, dtype=np.float64)
    loaded: list[dict[str, np.ndarray]] = []
    for path in args.curves:
        with np.load(path, allow_pickle=False) as archive:
            curve = {name: np.asarray(archive[name]) for name in archive.files}
        indices: list[int] = []
        for offset in requested_offsets:
            matches = np.flatnonzero(
                np.isclose(curve["offsets_mps"], offset, atol=1.0e-9, rtol=0.0)
            )
            if len(matches) != 1:
                raise ValueError(f"{path} does not contain offset {offset:+.3f}.")
            indices.append(int(matches[0]))
        selected = np.asarray(indices, dtype=np.int64)
        curve["offsets_mps"] = requested_offsets.copy()
        for name, values in list(curve.items()):
            if name not in TASK_FIELDS and name != "offsets_mps":
                curve[name] = values[selected]
        loaded.append(curve)
    loaded.sort(key=lambda curve: int(np.min(curve["task_indices"])))
    field_names = set(loaded[0])
    if any(set(curve) != field_names for curve in loaded[1:]):
        raise ValueError("Curve batches have different fields.")
    task_indices = np.concatenate(
        [curve["task_indices"] for curve in loaded],
        axis=0,
    )
    if len(np.unique(task_indices)) != len(task_indices):
        raise ValueError("Curve batches contain duplicate tasks.")
    merged: dict[str, np.ndarray] = {"offsets_mps": requested_offsets}
    for name in sorted(field_names - {"offsets_mps"}):
        axis = 0 if name in TASK_FIELDS else 1
        merged[name] = np.concatenate(
            [curve[name] for curve in loaded],
            axis=axis,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **merged)
    print(
        f"curve_batch_count={len(loaded)} task_count={len(task_indices)} "
        f"offset_count={len(requested_offsets)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
