"""Collect real-physics reward curves around a frozen mid-level BC speed."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.midlevel_mujoco_warp_vec_env import (  # noqa: E402
    MJWarpMidLevelVecEnv,
)
from snooker_env.midlevel_ppo import (  # noqa: E402
    generated_behavior_cloning_data,
)
from snooker_env.midlevel_sac_her import (  # noqa: E402
    SingleStepTD3BC,
    slot_aligned_local_probe_batch_starts,
)
from snooker_env.midlevel_tasks import TwoBallTaskDataset  # noqa: E402
from snooker_env.midlevel_two_ball import (  # noqa: E402
    MAX_CUE_SPEED,
    MIN_CUE_SPEED,
)


BOOLEAN_FIELDS = (
    "correct_pot",
    "cue_scratch",
    "wrong_pocket",
    "stopped",
    "timed_out",
    "numerical_failure",
    "joint_success",
)
FLOAT_INFO_FIELDS = (
    "stop_error",
    "object_pocket_error",
    "reward_object_ball",
    "reward_cue_position",
    "reward_joint_success_bonus",
    "cue_speed",
)


def _run_curve_point(
    environment: MJWarpMidLevelVecEnv,
    task_indices: np.ndarray,
    actions: np.ndarray,
) -> dict[str, np.ndarray]:
    environment.set_options(
        [{"task_index": int(task_index)} for task_index in task_indices]
    )
    observations = np.asarray(environment.reset(), dtype=np.float32)
    _, rewards, dones, infos = environment.step(actions)
    if not bool(np.all(dones)):
        raise RuntimeError("Every speed-curve shot must terminate.")
    returned = np.asarray(
        [int(info["task_index"]) for info in infos],
        dtype=np.int64,
    )
    if not np.array_equal(returned, task_indices):
        raise RuntimeError("Speed-curve rollout changed task/world identity.")
    result: dict[str, np.ndarray] = {
        "observation": observations,
        "action": np.asarray(actions, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
        "cue_final": np.stack(
            [np.asarray(info["cue_ball_final_position"]) for info in infos]
        ).astype(np.float32),
        "object_final": np.stack(
            [np.asarray(info["object_ball_final_position"]) for info in infos]
        ).astype(np.float32),
    }
    for field in BOOLEAN_FIELDS:
        result[field] = np.asarray(
            [bool(info[field]) for info in infos],
            dtype=np.bool_,
        )
    for field in FLOAT_INFO_FIELDS:
        result[field] = np.asarray(
            [float(info[field]) for info in infos],
            dtype=np.float32,
        )
    return result


def _rate(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _curve_report(
    offsets: np.ndarray,
    results: dict[str, np.ndarray],
    generated_speed_correction_mps: np.ndarray,
) -> dict[str, Any]:
    rewards = results["reward"]
    safe = (
        results["correct_pot"]
        & ~results["cue_scratch"]
        & ~results["wrong_pocket"]
        & results["stopped"]
        & ~results["timed_out"]
        & ~results["numerical_failure"]
    )
    center_index = int(np.flatnonzero(offsets == 0.0)[0])
    per_offset: list[dict[str, float]] = []
    for index, offset in enumerate(offsets):
        per_offset.append(
            {
                "offset_mps": float(offset),
                "mean_reward": float(np.mean(rewards[index])),
                "correct_pot_rate": _rate(results["correct_pot"][index]),
                "joint_success_rate": _rate(results["joint_success"][index]),
                "safe_rate": _rate(safe[index]),
                "scratch_rate": _rate(results["cue_scratch"][index]),
                "timeout_rate": _rate(results["timed_out"][index]),
                "mean_stop_error_m": float(
                    np.mean(results["stop_error"][index])
                ),
            }
        )

    safe_rewards = np.where(safe, rewards, -np.inf)
    maximum_safe_reward = np.max(safe_rewards, axis=0)
    has_safe = np.isfinite(maximum_safe_reward)
    tied_best = (
        safe
        & np.isclose(
            rewards,
            maximum_safe_reward[None, :],
            atol=1.0e-7,
            rtol=0.0,
        )
    )
    tie_cost = np.where(tied_best, np.abs(offsets)[:, None], np.inf)
    best_indices = np.argmin(tie_cost, axis=0)
    best_indices[~has_safe] = center_index
    column_indices = np.arange(rewards.shape[1])
    best_offsets = offsets[best_indices]
    best_rewards = rewards[best_indices, column_indices]
    center_rewards = rewards[center_index]
    best_joint_success = results["joint_success"][best_indices, column_indices]
    best_correct_pot = results["correct_pot"][best_indices, column_indices]
    best_scratch = results["cue_scratch"][best_indices, column_indices]

    adjacent_reward_change = np.abs(np.diff(rewards, axis=0))
    outcome_vector = np.stack(
        [
            results["correct_pot"],
            results["cue_scratch"],
            results["stopped"],
            results["timed_out"],
            results["joint_success"],
        ],
        axis=2,
    )
    adjacent_outcome_flip = np.any(
        outcome_vector[1:] != outcome_vector[:-1],
        axis=2,
    )
    best_offset_counts = Counter(float(value) for value in best_offsets)

    symmetric_reports: list[dict[str, float | int]] = []
    for positive_offset in offsets[offsets > 0.0]:
        negative_matches = np.flatnonzero(offsets == -positive_offset)
        if len(negative_matches) != 1:
            continue
        plus_index = int(np.flatnonzero(offsets == positive_offset)[0])
        minus_index = int(negative_matches[0])
        physical_difference = rewards[plus_index] - rewards[minus_index]
        eligible = (
            np.abs(generated_speed_correction_mps) >= 0.005
        ) & (np.abs(physical_difference) >= 0.01)
        desired_sign = np.sign(generated_speed_correction_mps[eligible])
        margin = physical_difference[eligible] * desired_sign
        symmetric_reports.append(
            {
                "offset_mps": float(positive_offset),
                "eligible_count": int(np.count_nonzero(eligible)),
                "generated_correction_sign_agreement": (
                    float(np.mean(margin > 0.0))
                    if len(margin)
                    else float("nan")
                ),
                "mean_abs_reward_difference": float(
                    np.mean(np.abs(physical_difference))
                ),
                "p95_abs_reward_difference": float(
                    np.percentile(np.abs(physical_difference), 95)
                ),
            }
        )

    return {
        "per_offset": per_offset,
        "center": per_offset[center_index],
        "oracle_safe_candidate": {
            "has_safe_candidate_rate": _rate(has_safe),
            "correct_pot_rate": _rate(best_correct_pot),
            "joint_success_rate": _rate(best_joint_success),
            "scratch_rate": _rate(best_scratch),
            "mean_reward": float(np.mean(best_rewards)),
            "mean_reward_improvement_over_center": float(
                np.mean(best_rewards - center_rewards)
            ),
            "positive_reward_improvement_rate": float(
                np.mean(best_rewards > center_rewards + 1.0e-7)
            ),
            "best_offset_counts": {
                f"{offset:+.3f}": int(best_offset_counts.get(float(offset), 0))
                for offset in offsets
            },
        },
        "curve_roughness": {
            "adjacent_reward_change_mean": float(
                np.mean(adjacent_reward_change)
            ),
            "adjacent_reward_change_p95": float(
                np.percentile(adjacent_reward_change, 95)
            ),
            "adjacent_reward_change_max": float(
                np.max(adjacent_reward_change)
            ),
            "adjacent_reward_jump_ge_0_5_rate": float(
                np.mean(adjacent_reward_change >= 0.5)
            ),
            "adjacent_outcome_flip_rate": _rate(adjacent_outcome_flip),
            "tasks_with_any_outcome_flip_rate": float(
                np.mean(np.any(adjacent_outcome_flip, axis=0))
            ),
        },
        "symmetric_physical_rankings": symmetric_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "outputs/tasks/midlevel_two_ball_train.npz",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--physics-device", default="cuda:0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-worlds", type=int, default=4096)
    parser.add_argument("--batch-start", type=int)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument(
        "--offsets-mps",
        type=float,
        nargs="+",
        default=(-0.12, -0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.06, 0.12),
    )
    parser.add_argument("--chunk-steps", type=int, default=64)
    parser.add_argument("--check-interval-steps", type=int, default=8192)
    parser.add_argument("--max-shot-time", type=float, default=8.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/diagnostics/midlevel_bc_speed_curves.npz",
    )
    args = parser.parse_args()

    offsets = np.asarray(args.offsets_mps, dtype=np.float64)
    if (
        offsets.ndim != 1
        or len(offsets) < 3
        or not np.all(np.isfinite(offsets))
        or len(np.unique(offsets)) != len(offsets)
        or not np.all(np.diff(offsets) > 0.0)
        or not np.any(offsets == 0.0)
    ):
        raise ValueError("Offsets must be unique, sorted, finite, and include zero.")
    if not np.allclose(offsets, -offsets[::-1], atol=1.0e-12):
        raise ValueError("Speed-curve offsets must be symmetric around zero.")
    dataset = TwoBallTaskDataset.load(args.tasks, validate_model=False)
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
        raise ValueError("--batch-start must be an execution-batch boundary.")
    if batch_start + args.num_worlds > len(dataset):
        raise ValueError("Speed curve requires one complete task batch.")
    task_indices = np.arange(
        batch_start,
        batch_start + args.num_worlds,
        dtype=np.int64,
    )
    all_observations, generated_actions = generated_behavior_cloning_data(dataset)
    observations = all_observations[task_indices]
    generated_actions = generated_actions[task_indices].copy()
    generated_actions[:, 0] = 0.0
    policy = SingleStepTD3BC.load(args.checkpoint, device=args.device)
    bc_actions, _ = policy.predict(observations, deterministic=True)
    bc_actions = np.asarray(bc_actions, dtype=np.float32)
    bc_actions[:, 0] = 0.0
    speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    generated_speed_correction_mps = (
        generated_actions[:, 1].astype(np.float64)
        - bc_actions[:, 1].astype(np.float64)
    ) * speed_half_range
    speed_error = np.abs(generated_speed_correction_mps)

    environment = MJWarpMidLevelVecEnv(
        dataset,
        args.model,
        num_envs=args.num_worlds,
        device=args.physics_device,
        chunk_steps=args.chunk_steps,
        check_interval_steps=args.check_interval_steps,
        max_time=args.max_shot_time,
    )
    curve_points: list[dict[str, np.ndarray]] = []
    try:
        for offset in offsets:
            actions = bc_actions.copy()
            actions[:, 1] += np.float32(offset / speed_half_range)
            if np.any(actions[:, 1] < -1.0) or np.any(actions[:, 1] > 1.0):
                raise ValueError(
                    f"Offset {offset:+.3f} m/s exceeds the action range."
                )
            print(f"speed_curve_offset_mps={offset:+.3f}", flush=True)
            curve_points.append(
                _run_curve_point(environment, task_indices, actions)
            )
    finally:
        environment.close()

    stacked = {
        name: np.stack([point[name] for point in curve_points], axis=0)
        for name in curve_points[0]
    }
    report = {
        "curve_version": "frozen-bc-real-physics-speed-curve-v1",
        "checkpoint": str(args.checkpoint),
        "task_library": str(args.tasks),
        "batch_start": batch_start,
        "task_count": args.num_worlds,
        "num_worlds": args.num_worlds,
        "offsets_mps": [float(value) for value in offsets],
        "world_slot_aligned": True,
        "offset_execution": "serial_same_world_slot",
        "bc_speed_error_mps": {
            "mean": float(np.mean(speed_error)),
            "p50": float(np.percentile(speed_error, 50)),
            "p95": float(np.percentile(speed_error, 95)),
            "max": float(np.max(speed_error)),
        },
        **_curve_report(offsets, stacked, generated_speed_correction_mps),
    }
    if not all(
        math.isfinite(value)
        for value in (
            report["bc_speed_error_mps"]["mean"],
            report["curve_roughness"]["adjacent_reward_change_mean"],
        )
    ):
        raise FloatingPointError("Speed-curve report contains non-finite metrics.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        task_indices=task_indices,
        offsets_mps=offsets,
        bc_actions=bc_actions,
        generated_actions=generated_actions,
        generated_speed_correction_mps=generated_speed_correction_mps,
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
