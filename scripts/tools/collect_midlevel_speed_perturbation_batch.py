"""Collect canonical-action speed perturbations for one fixed world batch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from collect_midlevel_bc_speed_curves import _run_curve_point  # noqa: E402
from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.midlevel_mujoco_warp_vec_env import (  # noqa: E402
    MJWarpMidLevelVecEnv,
)
from snooker_env.midlevel_ppo import generated_behavior_cloning_data  # noqa: E402
from snooker_env.midlevel_ppo_env import MAX_TERMINAL_REWARD  # noqa: E402
from snooker_env.midlevel_tasks import TwoBallTaskDataset  # noqa: E402
from snooker_env.midlevel_two_ball import (  # noqa: E402
    MAX_CUE_SPEED,
    MIN_CUE_SPEED,
)


FORMAT_VERSION = "canonical-generated-speed-perturbations-v1"
DEFAULT_OFFSETS_MPS = (-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03)


def _distribution(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


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


def _validate_offsets(offsets: np.ndarray) -> int:
    if (
        offsets.ndim != 1
        or len(offsets) < 3
        or not np.all(np.isfinite(offsets))
        or len(np.unique(offsets)) != len(offsets)
        or not np.all(np.diff(offsets) > 0.0)
        or np.count_nonzero(offsets == 0.0) != 1
        or not np.allclose(offsets, -offsets[::-1], atol=1.0e-12)
    ):
        raise ValueError(
            "Offsets must be sorted, unique, symmetric, finite, and include zero."
        )
    return int(np.flatnonzero(offsets == 0.0)[0])


def _batch_report(
    *,
    offsets: np.ndarray,
    results: dict[str, np.ndarray],
    cue_delta_xy: np.ndarray,
    target_error: np.ndarray,
    batch_start: int,
    task_count: int,
    source_task_count: int,
    source_content_sha256: str,
) -> dict[str, Any]:
    per_offset: list[dict[str, Any]] = []
    for index, offset in enumerate(offsets):
        per_offset.append(
            {
                "offset_mps": float(offset),
                "reward_mean": float(np.mean(results["reward"][index])),
                "correct_pot_rate": float(
                    np.mean(results["correct_pot"][index])
                ),
                "joint_success_rate": float(
                    np.mean(results["joint_success"][index])
                ),
                "scratch_rate": float(np.mean(results["cue_scratch"][index])),
                "stopped_rate": float(np.mean(results["stopped"][index])),
                "cue_displacement_from_center_m": _distribution(
                    np.linalg.norm(cue_delta_xy[index], axis=1)
                ),
                "cue_target_error_m": _distribution(target_error[index]),
            }
        )

    event_fields = (
        "correct_pot",
        "cue_scratch",
        "wrong_pocket",
        "stopped",
        "timed_out",
        "numerical_failure",
        "joint_success",
    )
    event_vectors = np.stack(
        [results[field] for field in event_fields],
        axis=2,
    )
    adjacent: list[dict[str, Any]] = []
    for index in range(len(offsets) - 1):
        endpoint_change = np.linalg.norm(
            results["cue_final"][index + 1, :, :2]
            - results["cue_final"][index, :, :2],
            axis=1,
        )
        reward_change = np.abs(
            results["reward"][index + 1] - results["reward"][index]
        )
        outcome_flip = np.any(
            event_vectors[index + 1] != event_vectors[index],
            axis=1,
        )
        adjacent.append(
            {
                "lower_offset_mps": float(offsets[index]),
                "upper_offset_mps": float(offsets[index + 1]),
                "cue_endpoint_change_m": _distribution(endpoint_change),
                "reward_absolute_change": _distribution(reward_change),
                "any_outcome_flip_rate": float(np.mean(outcome_flip)),
            }
        )
    return {
        "format_version": FORMAT_VERSION,
        "center_action_source": "canonical_generated_action",
        "batch_start": batch_start,
        "task_count": task_count,
        "source_task_library_count": source_task_count,
        "source_task_library_content_sha256": source_content_sha256,
        "offsets_mps": [float(value) for value in offsets],
        "per_offset": per_offset,
        "adjacent_offsets": adjacent,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--batch-start", type=int, required=True)
    parser.add_argument("--num-worlds", type=int, default=4096)
    parser.add_argument(
        "--offsets-mps",
        type=float,
        nargs="+",
        default=DEFAULT_OFFSETS_MPS,
    )
    parser.add_argument("--physics-device", default="cuda:0")
    parser.add_argument("--chunk-steps", type=int, default=64)
    parser.add_argument("--check-interval-steps", type=int, default=8192)
    parser.add_argument("--max-shot-time", type=float, default=8.0)
    parser.add_argument("--center-stop-tolerance", type=float, default=5e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.num_worlds <= 0:
        raise ValueError("--num-worlds must be positive.")
    if args.batch_start < 0 or args.batch_start % args.num_worlds != 0:
        raise ValueError("--batch-start must be a fixed-world batch boundary.")
    if (
        not math.isfinite(args.center_stop_tolerance)
        or args.center_stop_tolerance < 0.0
    ):
        raise ValueError("--center-stop-tolerance must be finite and non-negative.")
    offsets = np.asarray(args.offsets_mps, dtype=np.float64)
    center_index = _validate_offsets(offsets)

    dataset = TwoBallTaskDataset.load(args.tasks, validate_model=False)
    batch_stop = args.batch_start + args.num_worlds
    if batch_stop > len(dataset):
        raise ValueError("Perturbation collection requires a complete world batch.")
    task_indices = np.arange(args.batch_start, batch_stop, dtype=np.int64)
    all_observations, all_generated_actions = generated_behavior_cloning_data(
        dataset
    )
    observations = all_observations[task_indices]
    center_actions = all_generated_actions[task_indices].copy()
    center_actions[:, 0] = 0.0
    target_stop_positions = dataset.target_stop_positions[task_indices].astype(
        np.float32
    )
    speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)

    actions_by_offset = np.repeat(
        center_actions[None, :, :],
        len(offsets),
        axis=0,
    )
    actions_by_offset[:, :, 1] += (
        offsets[:, None] / speed_half_range
    ).astype(np.float32)
    if np.any(actions_by_offset < -1.0) or np.any(actions_by_offset > 1.0):
        raise ValueError("A perturbed canonical action exceeds the action range.")

    environment = MJWarpMidLevelVecEnv(
        dataset,
        args.model,
        num_envs=args.num_worlds,
        device=args.physics_device,
        chunk_steps=args.chunk_steps,
        check_interval_steps=args.check_interval_steps,
        max_time=args.max_shot_time,
    )
    points: list[dict[str, np.ndarray]] = []
    try:
        for offset, actions in zip(offsets, actions_by_offset, strict=True):
            print(f"speed_perturbation_offset_mps={offset:+.3f}", flush=True)
            points.append(_run_curve_point(environment, task_indices, actions))
    finally:
        environment.close()

    for point in points:
        if not np.array_equal(point["observation"], observations):
            raise RuntimeError("A perturbation rollout changed its task observation.")
    results = {
        name: np.stack([point[name] for point in points], axis=0)
        for name in points[0]
        if name != "observation"
    }
    center_cue_xy = results["cue_final"][center_index, :, :2]
    center_error = np.linalg.norm(
        center_cue_xy - target_stop_positions,
        axis=1,
    )
    center_valid = (
        results["correct_pot"][center_index]
        & ~results["cue_scratch"][center_index]
        & ~results["wrong_pocket"][center_index]
        & results["stopped"][center_index]
        & ~results["timed_out"][center_index]
        & ~results["numerical_failure"][center_index]
    )
    if not bool(np.all(center_valid)):
        failures = np.flatnonzero(~center_valid)[:10].tolist()
        raise RuntimeError(
            "Canonical center action failed feasibility for batch-local rows "
            f"{failures}."
        )
    if np.any(center_error > args.center_stop_tolerance):
        failures = np.flatnonzero(
            center_error > args.center_stop_tolerance
        )[:10].tolist()
        raise RuntimeError(
            "Canonical center action did not reproduce its target stop for "
            f"batch-local rows {failures}; max error={np.max(center_error):.6g}m."
        )
    if not np.allclose(
        results["reward"][center_index],
        MAX_TERMINAL_REWARD,
        atol=1e-6,
        rtol=0.0,
    ):
        raise RuntimeError("Canonical center action did not receive maximum reward.")

    cue_delta_xy = (
        results["cue_final"][:, :, :2] - center_cue_xy[None, :, :]
    ).astype(np.float32)
    target_error = np.linalg.norm(
        results["cue_final"][:, :, :2]
        - target_stop_positions[None, :, :],
        axis=2,
    ).astype(np.float32)
    report = _batch_report(
        offsets=offsets,
        results=results,
        cue_delta_xy=cue_delta_xy,
        target_error=target_error,
        batch_start=args.batch_start,
        task_count=args.num_worlds,
        source_task_count=len(dataset),
        source_content_sha256=dataset.content_sha256(),
    )
    metadata = {
        "format_version": FORMAT_VERSION,
        "center_action_source": "canonical_generated_action",
        "task_library": str(args.tasks),
        "source_task_library_count": len(dataset),
        "source_task_library_content_sha256": dataset.content_sha256(),
        "batch_start": args.batch_start,
        "task_count": args.num_worlds,
        "record_count": int(args.num_worlds * len(offsets)),
        "offsets_mps": [float(value) for value in offsets],
        "world_slot_aligned": True,
        "offset_execution": "serial_same_world_slot",
    }
    _atomic_savez(
        args.output,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        task_indices=task_indices,
        offsets_mps=offsets,
        observation=observations,
        target_stop_position=target_stop_positions,
        center_action=center_actions,
        cue_final_delta_xy_m=cue_delta_xy,
        cue_target_error_m=target_error,
        **results,
    )
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"speed_perturbation_batch=PASS start={args.batch_start} "
        f"tasks={args.num_worlds} records={args.num_worlds * len(offsets)} "
        f"center_max_stop_error_m={np.max(center_error):.6g} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
