"""Audit same-slot repeatability and cross-slot bias in MJWarp rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.midlevel_mujoco_warp_vec_env import (  # noqa: E402
    MJWarpMidLevelVecEnv,
)
from snooker_env.midlevel_sac_her import (  # noqa: E402
    slot_aligned_local_probe_batch_starts,
)
from snooker_env.midlevel_tasks import (  # noqa: E402
    MUJOCO_WARP_PHYSICS_BACKEND,
    TwoBallTaskDataset,
)
from snooker_env.midlevel_two_ball import encode_speed_action  # noqa: E402


BOOLEAN_FIELDS = (
    "correct_pot",
    "cue_scratch",
    "wrong_pocket",
    "stopped",
    "timed_out",
    "numerical_failure",
)
FLOAT_FIELDS = (
    "reward",
    "stop_error",
    "cue_speed",
    "elapsed_time",
    "minimum_object_pocket_distance",
)


def _generated_actions(
    dataset: TwoBallTaskDataset,
    task_indices: np.ndarray,
) -> np.ndarray:
    actions = np.zeros((len(task_indices), 2), dtype=np.float32)
    actions[:, 1] = np.asarray(
        [
            encode_speed_action(float(dataset.generated_speeds[task_index]))
            for task_index in task_indices
        ],
        dtype=np.float32,
    )
    return actions


def _execute(
    environment: MJWarpMidLevelVecEnv,
    task_indices: np.ndarray,
    actions: np.ndarray,
) -> dict[str, np.ndarray]:
    if task_indices.shape != (environment.num_envs,):
        raise ValueError("Audit task batch does not fill every world.")
    if actions.shape != (environment.num_envs, 2):
        raise ValueError("Audit action batch has the wrong shape.")
    environment.set_options(
        [{"task_index": int(task_index)} for task_index in task_indices]
    )
    environment.reset()
    _, rewards, dones, infos = environment.step(actions)
    if not bool(np.all(dones)):
        raise RuntimeError("Every audited shot must terminate in one vector step.")
    observed_indices = np.asarray(
        [int(info["task_index"]) for info in infos],
        dtype=np.int64,
    )
    if not np.array_equal(observed_indices, task_indices):
        raise RuntimeError("MJWarp changed the requested audit task mapping.")
    result: dict[str, np.ndarray] = {
        "task_index": observed_indices,
        "world_slot": np.arange(environment.num_envs, dtype=np.int64),
        "cue_final": np.stack(
            [np.asarray(info["cue_ball_final_position"]) for info in infos]
        ).astype(np.float64),
        "object_final": np.stack(
            [np.asarray(info["object_ball_final_position"]) for info in infos]
        ).astype(np.float64),
    }
    for field in BOOLEAN_FIELDS:
        result[field] = np.asarray(
            [bool(info[field]) for info in infos],
            dtype=np.bool_,
        )
    for field in FLOAT_FIELDS:
        if field == "reward":
            result[field] = np.asarray(rewards, dtype=np.float64)
        else:
            result[field] = np.asarray(
                [float(info[field]) for info in infos],
                dtype=np.float64,
            )
    return result


def _align_by_task(
    result: dict[str, np.ndarray],
    canonical_task_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    lookup = {
        int(task_index): row
        for row, task_index in enumerate(result["task_index"])
    }
    if len(lookup) != len(canonical_task_indices):
        raise RuntimeError("Audit batch contains duplicate tasks.")
    order = np.asarray(
        [lookup[int(task_index)] for task_index in canonical_task_indices],
        dtype=np.int64,
    )
    return {name: values[order] for name, values in result.items()}


def _comparison_report(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, Any]:
    cue_delta = np.linalg.norm(
        candidate["cue_final"] - reference["cue_final"],
        axis=1,
    )
    object_delta = np.linalg.norm(
        candidate["object_final"] - reference["object_final"],
        axis=1,
    )
    reward_delta = np.abs(candidate["reward"] - reference["reward"])
    stop_error_delta = np.abs(
        candidate["stop_error"] - reference["stop_error"]
    )
    flag_changed = np.zeros(len(cue_delta), dtype=np.bool_)
    per_flag_mismatch: dict[str, float] = {}
    for field in BOOLEAN_FIELDS:
        mismatch = candidate[field] != reference[field]
        flag_changed |= mismatch
        per_flag_mismatch[f"{field}_mismatch_rate"] = float(np.mean(mismatch))
    return {
        "sample_count": int(len(cue_delta)),
        "world_slot_changed_rate": float(
            np.mean(candidate["world_slot"] != reference["world_slot"])
        ),
        "cue_final_delta_mean_m": float(np.mean(cue_delta)),
        "cue_final_delta_p95_m": float(np.percentile(cue_delta, 95)),
        "cue_final_delta_max_m": float(np.max(cue_delta)),
        "object_final_delta_mean_m": float(np.mean(object_delta)),
        "object_final_delta_p95_m": float(np.percentile(object_delta, 95)),
        "object_final_delta_max_m": float(np.max(object_delta)),
        "reward_delta_mean": float(np.mean(reward_delta)),
        "reward_delta_p95": float(np.percentile(reward_delta, 95)),
        "reward_delta_max": float(np.max(reward_delta)),
        "stop_error_delta_mean_m": float(np.mean(stop_error_delta)),
        "stop_error_delta_p95_m": float(np.percentile(stop_error_delta, 95)),
        "stop_error_delta_max_m": float(np.max(stop_error_delta)),
        "any_outcome_flag_mismatch_rate": float(np.mean(flag_changed)),
        **per_flag_mismatch,
    }


def _outcome_report(result: dict[str, np.ndarray]) -> dict[str, float | int]:
    valid_stop = (
        result["correct_pot"]
        & ~result["cue_scratch"]
        & result["stopped"]
    )
    return {
        "sample_count": int(len(result["task_index"])),
        "correct_pot_rate": float(np.mean(result["correct_pot"])),
        "scratch_rate": float(np.mean(result["cue_scratch"])),
        "wrong_pocket_rate": float(np.mean(result["wrong_pocket"])),
        "stopped_rate": float(np.mean(result["stopped"])),
        "valid_stop_rate": float(np.mean(valid_stop)),
        "joint_success_rate": float(
            np.mean(valid_stop & (result["stop_error"] <= 0.05))
        ),
        "mean_reward": float(np.mean(result["reward"])),
        "mean_stop_error_m": float(np.mean(result["stop_error"])),
        "p95_stop_error_m": float(np.percentile(result["stop_error"], 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "outputs/tasks/midlevel_two_ball_train.npz",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--physics-device", default="cuda:0")
    parser.add_argument("--num-worlds", type=int, default=4096)
    parser.add_argument("--batch-start", type=int)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument(
        "--cross-slot-shifts",
        type=int,
        nargs="+",
        default=(1, 1024),
    )
    parser.add_argument("--chunk-steps", type=int, default=64)
    parser.add_argument("--check-interval-steps", type=int, default=8192)
    parser.add_argument("--max-shot-time", type=float, default=8.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/diagnostics/midlevel_world_slot_audit.json",
    )
    args = parser.parse_args()

    if args.num_worlds <= 0:
        raise ValueError("--num-worlds must be positive.")
    if args.repeat_count < 2:
        raise ValueError("--repeat-count must be at least two.")
    dataset = TwoBallTaskDataset.load(args.tasks, validate_model=False)
    if dataset.physics_backend != MUJOCO_WARP_PHYSICS_BACKEND:
        raise ValueError("Slot audit requires an MJWarp task library.")
    if dataset.execution_max_time != args.max_shot_time:
        raise ValueError("--max-shot-time does not match the task library.")
    if args.batch_start is None:
        batch_start = int(
            slot_aligned_local_probe_batch_starts(
                dataset,
                args.num_worlds,
                num_worlds=args.num_worlds,
                seed=args.seed,
            )[0]
        )
    else:
        batch_start = int(args.batch_start)
    if batch_start < 0 or batch_start % args.num_worlds != 0:
        raise ValueError("--batch-start must be a non-negative canonical boundary.")
    if batch_start + args.num_worlds > len(dataset):
        raise ValueError("--batch-start does not select a complete canonical batch.")
    shifts = tuple(int(shift) % args.num_worlds for shift in args.cross_slot_shifts)
    if any(shift == 0 for shift in shifts) or len(set(shifts)) != len(shifts):
        raise ValueError("Cross-slot shifts must be unique and non-zero modulo width.")

    canonical_indices = np.arange(
        batch_start,
        batch_start + args.num_worlds,
        dtype=np.int64,
    )
    canonical_actions = _generated_actions(dataset, canonical_indices)
    environment = MJWarpMidLevelVecEnv(
        dataset,
        args.model,
        num_envs=args.num_worlds,
        device=args.physics_device,
        chunk_steps=args.chunk_steps,
        check_interval_steps=args.check_interval_steps,
        max_time=args.max_shot_time,
    )
    runs: dict[str, dict[str, np.ndarray]] = {}
    try:
        for repeat_index in range(args.repeat_count):
            name = f"canonical_repeat_{repeat_index}"
            print(f"audit_run={name}", flush=True)
            runs[name] = _execute(
                environment,
                canonical_indices,
                canonical_actions,
            )
        for shift in shifts:
            name = f"cross_slot_shift_{shift}"
            print(f"audit_run={name}", flush=True)
            runs[name] = _execute(
                environment,
                np.roll(canonical_indices, shift),
                np.roll(canonical_actions, shift, axis=0),
            )
    finally:
        environment.close()

    reference = runs["canonical_repeat_0"]
    repeat_reports: dict[str, dict[str, Any]] = {}
    for repeat_index in range(1, args.repeat_count):
        name = f"canonical_repeat_{repeat_index}"
        repeat_reports[name] = _comparison_report(reference, runs[name])
    cross_slot_reports: dict[str, dict[str, Any]] = {}
    for shift in shifts:
        name = f"cross_slot_shift_{shift}"
        aligned = _align_by_task(runs[name], canonical_indices)
        cross_slot_reports[name] = {
            **_comparison_report(reference, aligned),
            "outcome": _outcome_report(aligned),
        }

    same_slot_passed = all(
        report["cue_final_delta_max_m"] <= 1.0e-7
        and report["object_final_delta_max_m"] <= 1.0e-7
        and report["reward_delta_max"] <= 1.0e-7
        and report["any_outcome_flag_mismatch_rate"] == 0.0
        for report in repeat_reports.values()
    )
    canonical_outcome = _outcome_report(reference)
    canonical_certification_passed = (
        canonical_outcome["valid_stop_rate"] == 1.0
        and canonical_outcome["wrong_pocket_rate"] == 0.0
    )
    report = {
        "audit_version": "mujoco-warp-world-slot-v1",
        "task_library": str(args.tasks),
        "task_count": len(dataset),
        "canonical_num_worlds": args.num_worlds,
        "batch_start": batch_start,
        "action_source": "published_generated_action",
        "repeat_count": args.repeat_count,
        "cross_slot_shifts": list(shifts),
        "canonical_outcome": canonical_outcome,
        "same_slot_repeatability": repeat_reports,
        "cross_slot_bias": cross_slot_reports,
        "same_slot_passed": same_slot_passed,
        "canonical_certification_passed": canonical_certification_passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arrays: dict[str, np.ndarray] = {
        "canonical_task_indices": canonical_indices,
        "canonical_actions": canonical_actions,
    }
    for run_name, result in runs.items():
        for field, values in result.items():
            arrays[f"{run_name}__{field}"] = values
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    print(json.dumps(report, sort_keys=True), flush=True)
    if not same_slot_passed:
        raise RuntimeError("Same-slot MJWarp replay is not deterministic.")
    if not canonical_certification_passed:
        raise RuntimeError("Published generated actions failed canonical replay.")


if __name__ == "__main__":
    main()
