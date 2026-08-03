#!/usr/bin/env python3
"""Check that MJWarp task execution is invariant to world slot and ordering."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL
from snooker_env.midlevel_mujoco_warp_tasks import _generated_actions
from snooker_env.midlevel_mujoco_warp_vec_env import MJWarpMidLevelVecEnv
from snooker_env.midlevel_tasks import TwoBallTaskDataset
from snooker_env.midlevel_two_ball import TwoBallShotSimulator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--max-tasks", type=int, default=64)
    parser.add_argument(
        "--num-worlds",
        type=int,
        default=None,
        help="Repeat the selected tasks to fill this many MJWarp worlds.",
    )
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--stop-tolerance",
        type=float,
        default=1.0e-5,
        help="Maximum per-task drift; defaults to a strict 10-micrometer guard.",
    )
    parser.add_argument("--max-time", type=float, default=8.0)
    parser.add_argument(
        "--skip-independent-environment",
        action="store_true",
        help="Skip rebuilding MJWarp and replaying the canonical layout once.",
    )
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help="Run one layout and check its stored generation stop points.",
    )
    return parser.parse_args()


def _rollout(
    environment: MJWarpMidLevelVecEnv,
    dataset: TwoBallTaskDataset,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    environment.set_options(
        [{"task_index": int(index)} for index in indices]
    )
    environment.reset()
    tasks = [dataset[int(index)] for index in indices]
    _, _, _, infos = environment.step(_generated_actions(tasks))
    returned_indices = np.asarray(
        [int(info["task_index"]) for info in infos],
        dtype=np.int64,
    )
    if not np.array_equal(returned_indices, indices):
        raise RuntimeError(
            "MJWarp returned task infos in a different slot order: "
            f"expected={indices.tolist()}, returned={returned_indices.tolist()}"
        )
    final_positions = np.stack(
        [
            np.asarray(info["cue_ball_final_position"], dtype=np.float64)[:2]
            for info in infos
        ]
    )
    feasible = np.asarray(
        [
            bool(
                info["correct_pot"]
                and info["legal_first_contact"]
                and not info["cue_scratch"]
                and not info["cushion_before_object"]
                and not info["object_cushion_before_pocket"]
                and info["stopped"]
                and not info["timed_out"]
                and not info["numerical_failure"]
            )
            for info in infos
        ],
        dtype=np.bool_,
    )
    return final_positions, feasible


def main() -> None:
    args = _parse_args()
    simulator = TwoBallShotSimulator(args.model)
    dataset = TwoBallTaskDataset.load(
        args.dataset,
        simulator=simulator,
        validate_model=True,
    )
    task_count = min(len(dataset), args.max_tasks)
    if task_count <= 1:
        raise ValueError("Slot/order smoke requires at least two tasks.")
    num_worlds = task_count if args.num_worlds is None else args.num_worlds
    if num_worlds < task_count:
        raise ValueError("--num-worlds must cover every selected task.")
    canonical_indices = np.arange(num_worlds, dtype=np.int64) % task_count
    permuted_indices = np.random.default_rng(args.seed).permutation(
        canonical_indices
    )
    environment = MJWarpMidLevelVecEnv(
        dataset,
        args.model,
        num_envs=num_worlds,
        max_time=args.max_time,
        validate_task_execution=args.max_time == dataset.execution_max_time,
    )
    try:
        canonical_positions, canonical_feasible = _rollout(
            environment,
            dataset,
            canonical_indices,
        )
        repeated_positions = None
        repeated_feasible = None
        permuted_positions = None
        permuted_feasible = None
        if not args.canonical_only:
            repeated_positions, repeated_feasible = _rollout(
                environment,
                dataset,
                canonical_indices,
            )
            permuted_positions, permuted_feasible = _rollout(
                environment,
                dataset,
                permuted_indices,
            )
        throughput = environment.last_world_steps_per_second
    finally:
        environment.close()

    independent_positions: np.ndarray | None = None
    independent_feasible: np.ndarray | None = None
    if not args.skip_independent_environment and not args.canonical_only:
        independent_environment = MJWarpMidLevelVecEnv(
            dataset,
            args.model,
            num_envs=num_worlds,
            max_time=args.max_time,
            validate_task_execution=(
                args.max_time == dataset.execution_max_time
            ),
        )
        try:
            independent_positions, independent_feasible = _rollout(
                independent_environment,
                dataset,
                canonical_indices,
            )
        finally:
            independent_environment.close()

    reference_positions = np.empty((task_count, 2), dtype=np.float64)
    reference_feasible = np.empty(task_count, dtype=np.bool_)
    for task_index in range(task_count):
        reference_slot = int(np.flatnonzero(canonical_indices == task_index)[0])
        reference_positions[task_index] = canonical_positions[reference_slot]
        reference_feasible[task_index] = canonical_feasible[reference_slot]
    canonical_error = np.linalg.norm(
        canonical_positions - reference_positions[canonical_indices],
        axis=1,
    )
    stored_stop_error = np.linalg.norm(
        canonical_positions
        - dataset.target_stop_positions[canonical_indices],
        axis=1,
    )
    repeated_error = (
        np.linalg.norm(
            repeated_positions - reference_positions[canonical_indices],
            axis=1,
        )
        if repeated_positions is not None
        else None
    )
    permuted_error = (
        np.linalg.norm(
            permuted_positions - reference_positions[permuted_indices],
            axis=1,
        )
        if permuted_positions is not None
        else None
    )
    independent_error = (
        np.linalg.norm(
            independent_positions - reference_positions[canonical_indices],
            axis=1,
        )
        if independent_positions is not None
        else None
    )
    print(
        "canonical_slot_max_error_m=",
        float(canonical_error.max()),
        "stored_stop_max_error_m=",
        float(stored_stop_error.max()),
        "same_order_max_error_m=",
        float(repeated_error.max()) if repeated_error is not None else "skipped",
        "permuted_max_error_m=",
        float(permuted_error.max()) if permuted_error is not None else "skipped",
        "independent_env_max_error_m=",
        (
            float(independent_error.max())
            if independent_error is not None
            else "skipped"
        ),
        "canonical_feasible=",
        f"{int(canonical_feasible.sum())}/{num_worlds}",
        "repeated_feasible=",
        (
            f"{int(repeated_feasible.sum())}/{num_worlds}"
            if repeated_feasible is not None
            else "skipped"
        ),
        "permuted_feasible=",
        (
            f"{int(permuted_feasible.sum())}/{num_worlds}"
            if permuted_feasible is not None
            else "skipped"
        ),
        "independent_env_feasible=",
        (
            f"{int(independent_feasible.sum())}/{num_worlds}"
            if independent_feasible is not None
            else "skipped"
        ),
        "world_steps_per_second=",
        f"{throughput:.0f}",
    )
    bad_canonical = (
        (canonical_error > args.stop_tolerance)
        | (canonical_feasible != reference_feasible[canonical_indices])
    )
    bad_stored = stored_stop_error > args.stop_tolerance
    bad_repeated = (
        (
            (repeated_error > args.stop_tolerance)
            | (repeated_feasible != reference_feasible[canonical_indices])
        )
        if repeated_error is not None and repeated_feasible is not None
        else np.zeros(num_worlds, dtype=np.bool_)
    )
    bad_permuted = (
        (
            (permuted_error > args.stop_tolerance)
            | (permuted_feasible != reference_feasible[permuted_indices])
        )
        if permuted_error is not None and permuted_feasible is not None
        else np.zeros(num_worlds, dtype=np.bool_)
    )
    bad_independent = (
        (
            (independent_error > args.stop_tolerance)
            | (
                independent_feasible
                != reference_feasible[canonical_indices]
            )
        )
        if independent_error is not None
        and independent_feasible is not None
        else np.zeros(num_worlds, dtype=np.bool_)
    )
    if (
        np.any(bad_canonical)
        or np.any(bad_stored)
        or np.any(bad_repeated)
        or np.any(bad_permuted)
        or np.any(bad_independent)
    ):
        failing_tasks = np.unique(
            np.concatenate(
                (
                    canonical_indices[bad_canonical],
                    canonical_indices[bad_stored],
                    canonical_indices[bad_repeated],
                    permuted_indices[bad_permuted],
                    canonical_indices[bad_independent],
                )
            )
        )
        details = ", ".join(
            f"task={int(task_index)}"
            for task_index in failing_tasks[:16]
        )
        raise RuntimeError(f"MJWarp slot/order invariance failed: {details}")


if __name__ == "__main__":
    main()
