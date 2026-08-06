"""Batched MJWarp generation and replay for feasible two-ball tasks."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL
from snooker_env.midlevel_mujoco_warp_vec_env import (
    MJWarpMidLevelVecEnv,
    active_mujoco_warp_backend_sha256,
)
from snooker_env.midlevel_ppo_env import (
    OBSERVATION_X_SCALE,
    OBSERVATION_Y_SCALE,
)
from snooker_env.midlevel_tasks import (
    EVENT_FLAG_NAMES,
    MUJOCO_WARP_PHYSICS_BACKEND,
    TaskValidationReport,
    TwoBallTask,
    TwoBallTaskDataset,
    _generation_schedule,
    _sample_candidate,
)
from snooker_env.midlevel_two_ball import (
    POCKET_NAMES,
    TwoBallShotSimulator,
    encode_speed_action,
)
from snooker_env.mujoco_warp_sdf import MUJOCO_WARP_NCONMAX, MUJOCO_WARP_NJMAX


GenerationProgress = Callable[[int, int, TwoBallTask], None]
GenerationStatus = Callable[[str], None]


def _candidate_task(
    pocket_name: str,
    candidate_seed: int,
    simulator: TwoBallShotSimulator,
) -> TwoBallTask | None:
    candidate = _sample_candidate(pocket_name, candidate_seed, simulator)
    if candidate is None:
        return None
    cue_position, object_position, direction, speed = candidate
    return TwoBallTask(
        cue_position=cue_position,
        object_position=object_position,
        pocket_name=pocket_name,
        pocket_position=simulator.pocket_positions[pocket_name].copy(),
        # Generation rollouts replace this placeholder with the measured
        # MJWarp stop point before the task is accepted.
        target_stop_position=cue_position.copy(),
        generated_direction=direction,
        generated_speed=speed,
        candidate_seed=candidate_seed,
        elapsed_time=0.0,
        min_object_pocket_distance=float(
            np.linalg.norm(object_position - simulator.pocket_positions[pocket_name])
        ),
        event_metrics={name: False for name in EVENT_FLAG_NAMES},
    )


def _generated_actions(tasks: list[TwoBallTask]) -> np.ndarray:
    return np.asarray(
        [
            (0.0, encode_speed_action(task.generated_speed))
            for task in tasks
        ],
        dtype=np.float32,
    )


def _require_slot_identity(
    tasks: list[TwoBallTask],
    infos: list[dict[str, object]],
    *,
    context: str,
) -> None:
    """Fail before accepting results that were returned for another task slot."""

    if len(infos) != len(tasks):
        raise RuntimeError(
            f"{context} returned {len(infos)} infos for {len(tasks)} task slots."
        )
    for slot, (task, info) in enumerate(zip(tasks, infos, strict=True)):
        returned_seed = int(info["candidate_seed"])
        returned_pocket = str(info["pocket_name"])
        if (
            returned_seed != task.candidate_seed
            or returned_pocket != task.pocket_name
        ):
            raise RuntimeError(
                f"{context} task identity mismatch at world slot {slot}: "
                f"expected seed={task.candidate_seed} pocket={task.pocket_name}, "
                f"returned seed={returned_seed} pocket={returned_pocket}."
            )


def _is_feasible_info(info: dict[str, object]) -> bool:
    return bool(
        info["correct_pot"]
        and info["legal_first_contact"]
        and not info["cue_scratch"]
        and not info["cushion_before_object"]
        and not info["object_cushion_before_pocket"]
        and info["stopped"]
        and not info["timed_out"]
        and not info["numerical_failure"]
    )


def _passes_generation_prefilter(info: dict[str, object]) -> bool:
    """Keep shots that already made a legal direct pot in a short rollout."""

    return bool(
        info["correct_pot"]
        and info["legal_first_contact"]
        and not info["cue_scratch"]
        and not info["cushion_before_object"]
        and not info["object_cushion_before_pocket"]
        and not info["numerical_failure"]
    )


def _accepted_task(
    candidate: TwoBallTask,
    info: dict[str, object],
) -> TwoBallTask | None:
    if not _is_feasible_info(info):
        return None
    cue_final = np.asarray(info["cue_ball_final_position"], dtype=np.float64)
    if (
        not np.all(np.isfinite(cue_final[:2]))
        or abs(float(cue_final[0])) > OBSERVATION_X_SCALE
        or abs(float(cue_final[1])) > OBSERVATION_Y_SCALE
    ):
        return None
    direction = np.asarray(info["shot_direction"], dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    if not np.all(np.isfinite(direction)) or direction_norm <= 1e-12:
        return None
    direction /= direction_norm
    expected_direction = np.asarray(
        candidate.generated_direction,
        dtype=np.float64,
    )
    expected_direction_norm = float(np.linalg.norm(expected_direction))
    if (
        not np.all(np.isfinite(expected_direction))
        or expected_direction_norm <= 1e-12
    ):
        return None
    expected_direction /= expected_direction_norm
    cue_speed = float(info["cue_speed"])
    if (
        float(np.linalg.norm(direction - expected_direction)) > 1e-6
        or not np.isfinite(cue_speed)
        or abs(cue_speed - candidate.generated_speed) > 1e-6
    ):
        return None
    return TwoBallTask(
        cue_position=candidate.cue_position.copy(),
        object_position=candidate.object_position.copy(),
        pocket_name=candidate.pocket_name,
        pocket_position=candidate.pocket_position.copy(),
        target_stop_position=cue_final[:2].copy(),
        generated_direction=direction,
        generated_speed=cue_speed,
        candidate_seed=candidate.candidate_seed,
        elapsed_time=float(info["elapsed_time"]),
        min_object_pocket_distance=float(
            info["minimum_object_pocket_distance"]
        ),
        event_metrics={
            "correct_pot": bool(info["correct_pot"]),
            "legal_first_contact": bool(info["legal_first_contact"]),
            "no_cushion_direct_pot": not bool(info["cushion_before_object"])
            and not bool(info["object_cushion_before_pocket"]),
            "cue_scratch": bool(info["cue_scratch"]),
            "stopped": bool(info["stopped"]),
            "timed_out": bool(info["timed_out"]),
            "numerical_failure": bool(info["numerical_failure"]),
        },
    )


def _canonical_replay_task(
    candidate: TwoBallTask,
    first_info: dict[str, object],
    replay_info: dict[str, object],
    *,
    stop_tolerance: float,
) -> TwoBallTask | None:
    """Accept two consistent feasible executions and keep the replay result."""

    first = _accepted_task(candidate, first_info)
    canonical = _accepted_task(candidate, replay_info)
    if first is None or canonical is None:
        return None
    if (
        float(
            np.linalg.norm(
                first.generated_direction - canonical.generated_direction
            )
        )
        > 1e-6
        or abs(first.generated_speed - canonical.generated_speed) > 1e-6
    ):
        return None
    stop_error = float(
        np.linalg.norm(
            first.target_stop_position - canonical.target_stop_position
        )
    )
    if stop_error > stop_tolerance:
        return None
    # The caller-selected independent replay is authoritative for this pair.
    # Final dataset generation repeats this operation after slots are fixed.
    return canonical


def _backend_dataset(
    tasks: list[TwoBallTask],
    simulator: TwoBallShotSimulator,
    generation_seed: int,
    backend_hash: str,
) -> TwoBallTaskDataset:
    return TwoBallTaskDataset.from_tasks(
        tasks,
        simulator,
        generation_seed,
        physics_backend=MUJOCO_WARP_PHYSICS_BACKEND,
        backend_hash=backend_hash,
    )


def _fixed_layout_indices(
    start: int,
    count: int,
    batch_size: int,
) -> list[int]:
    """Return the exact padded slot layout used by final validation."""

    valid_count = min(batch_size, count - start)
    return [
        start + min(offset, valid_count - 1)
        for offset in range(batch_size)
    ]


def _canonicalize_fixed_layout(
    tasks: Sequence[TwoBallTask],
    replacement_tasks: Mapping[str, Sequence[TwoBallTask]],
    *,
    simulator: TwoBallShotSimulator,
    generation_seed: int,
    backend_hash: str,
    num_worlds: int,
    device: str,
    chunk_steps: int,
    check_interval_steps: int,
    nconmax: int,
    njmax: int,
    max_time: float,
    stop_tolerance: float,
    max_rounds: int,
    status: GenerationStatus | None,
) -> TwoBallTaskDataset:
    """Replay twice in immutable final slots and replace failures in place.

    A task's target stop is only authoritative for the world slot and batch in
    which the published dataset will execute it.  Removing a failed task would
    shift every later slot, invalidating results that were already checked, so
    replacements always retain the failed task's global index and pocket.
    """

    if not tasks:
        raise ValueError("Fixed-layout canonicalization requires tasks.")
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive.")

    canonical_tasks = list(tasks)
    count = len(canonical_tasks)
    batch_size = min(num_worlds, count)
    replacements = {
        pocket_name: deque(replacement_tasks.get(pocket_name, ()))
        for pocket_name in POCKET_NAMES
    }
    occupied_seeds = {task.candidate_seed for task in canonical_tasks}
    if len(occupied_seeds) != count:
        raise RuntimeError(
            "Fixed-layout canonicalization requires unique candidate seeds."
        )

    first_environment: MJWarpMidLevelVecEnv | None = None
    second_environment: MJWarpMidLevelVecEnv | None = None
    try:
        for start in range(0, count, batch_size):
            valid_count = min(batch_size, count - start)
            indices = _fixed_layout_indices(start, count, batch_size)
            for round_index in range(1, max_rounds + 1):
                current_dataset = _backend_dataset(
                    canonical_tasks,
                    simulator,
                    generation_seed,
                    backend_hash,
                )
                slot_tasks = [current_dataset[index] for index in indices]

                if first_environment is None:
                    first_environment = MJWarpMidLevelVecEnv(
                        current_dataset,
                        simulator.model_path,
                        num_envs=batch_size,
                        seed=generation_seed,
                        device=device,
                        chunk_steps=chunk_steps,
                        check_interval_steps=check_interval_steps,
                        nconmax=nconmax,
                        njmax=njmax,
                        max_time=max_time,
                    )
                    second_environment = MJWarpMidLevelVecEnv(
                        current_dataset,
                        simulator.model_path,
                        num_envs=batch_size,
                        seed=generation_seed + 1,
                        device=device,
                        chunk_steps=chunk_steps,
                        check_interval_steps=check_interval_steps,
                        nconmax=nconmax,
                        njmax=njmax,
                        max_time=max_time,
                    )
                else:
                    first_environment.replace_task_dataset(current_dataset)
                    assert second_environment is not None
                    second_environment.replace_task_dataset(current_dataset)

                options = [{"task_index": index} for index in indices]
                first_environment.set_options(options)
                first_environment.reset()
                _, _, _, first_infos = first_environment.step(
                    _generated_actions(slot_tasks)
                )
                _require_slot_identity(
                    slot_tasks,
                    first_infos,
                    context="MJWarp fixed-layout first replay",
                )

                assert second_environment is not None
                second_environment.set_options(options)
                second_environment.reset()
                _, _, _, second_infos = second_environment.step(
                    _generated_actions(slot_tasks)
                )
                _require_slot_identity(
                    slot_tasks,
                    second_infos,
                    context="MJWarp fixed-layout independent replay",
                )

                failures: list[int] = []
                updates: dict[int, TwoBallTask] = {}
                for offset in range(valid_count):
                    task_index = start + offset
                    candidate = canonical_tasks[task_index]
                    canonical = _canonical_replay_task(
                        candidate,
                        first_infos[offset],
                        second_infos[offset],
                        stop_tolerance=stop_tolerance,
                    )
                    if canonical is None:
                        failures.append(task_index)
                    else:
                        updates[task_index] = canonical

                for task_index, task in updates.items():
                    canonical_tasks[task_index] = task

                if not failures:
                    if status is not None:
                        status(
                            "fixed-layout canonical batch "
                            f"{start // batch_size + 1}/"
                            f"{int(np.ceil(count / batch_size))} "
                            f"passed round={round_index} tasks={valid_count}"
                        )
                    break

                if status is not None:
                    preview = ",".join(str(index) for index in failures[:8])
                    status(
                        "fixed-layout canonical batch "
                        f"{start // batch_size + 1} round={round_index} "
                        f"replacing={len(failures)} slots={preview}"
                    )
                if round_index == max_rounds:
                    raise RuntimeError(
                        "MJWarp fixed-layout canonicalization did not converge "
                        f"after {max_rounds} rounds; failed slots={failures[:20]}."
                    )

                for task_index in failures:
                    old_task = canonical_tasks[task_index]
                    pocket_name = old_task.pocket_name
                    occupied_seeds.remove(old_task.candidate_seed)
                    replacement: TwoBallTask | None = None
                    while replacements[pocket_name]:
                        candidate = replacements[pocket_name].popleft()
                        if candidate.pocket_name != pocket_name:
                            raise RuntimeError(
                                "A fixed-layout replacement was queued under "
                                "the wrong pocket."
                            )
                        if candidate.candidate_seed not in occupied_seeds:
                            replacement = candidate
                            break
                    if replacement is None:
                        raise RuntimeError(
                            "MJWarp fixed-layout replacement pool exhausted for "
                            f"{pocket_name} at task slot {task_index}."
                        )
                    canonical_tasks[task_index] = replacement
                    occupied_seeds.add(replacement.candidate_seed)
            else:  # pragma: no cover - loop exits by pass or explicit failure
                raise AssertionError("Unreachable fixed-layout replay state.")
    finally:
        if first_environment is not None:
            first_environment.close()
        if second_environment is not None:
            second_environment.close()

    return _backend_dataset(
        canonical_tasks,
        simulator,
        generation_seed,
        backend_hash,
    )


def generate_mujoco_warp_task_dataset(
    count: int,
    *,
    seed: int,
    model_path: Path = DEFAULT_MIDLEVEL_MODEL,
    num_worlds: int = 1024,
    device: str = "cuda:0",
    chunk_steps: int = 16,
    check_interval_steps: int = 2048,
    nconmax: int = MUJOCO_WARP_NCONMAX,
    njmax: int = MUJOCO_WARP_NJMAX,
    max_time: float = 8.0,
    prefilter_time: float = 1.5,
    stability_stop_tolerance: float = 5e-3,
    canonical_reserve_per_pocket: int = 32,
    canonical_max_rounds: int = 8,
    max_attempts_per_task: int = 2_000,
    progress: GenerationProgress | None = None,
    status: GenerationStatus | None = None,
) -> TwoBallTaskDataset:
    """Generate a balanced library with immediate independent replay filtering."""

    if count <= 0:
        raise ValueError("count must be positive.")
    if num_worlds <= 0:
        raise ValueError("num_worlds must be positive.")
    if max_attempts_per_task <= 0:
        raise ValueError("max_attempts_per_task must be positive.")
    if canonical_reserve_per_pocket < 0:
        raise ValueError("canonical_reserve_per_pocket cannot be negative.")
    if canonical_max_rounds <= 0:
        raise ValueError("canonical_max_rounds must be positive.")
    if prefilter_time <= 0.0 or prefilter_time >= max_time:
        raise ValueError("prefilter_time must be positive and less than max_time.")
    if (
        not np.isfinite(stability_stop_tolerance)
        or stability_stop_tolerance <= 0.0
    ):
        raise ValueError("stability_stop_tolerance must be positive and finite.")

    simulator = TwoBallShotSimulator(model_path, max_time=max_time)
    xml_hash, model_hash, backend_hash = active_mujoco_warp_backend_sha256(
        model_path
    )
    if xml_hash != simulator.xml_hash or model_hash != simulator.model_hash:
        raise RuntimeError("MJWarp fingerprint model does not match task generation model.")

    schedule = _generation_schedule(count, seed)
    final_required = {
        pocket_name: sum(name == pocket_name for name, _ in schedule)
        for pocket_name in POCKET_NAMES
    }
    reserve_required = {
        pocket_name: (
            min(
                canonical_reserve_per_pocket,
                max(1, int(np.ceil(final_required[pocket_name] * 0.02))),
            )
            if final_required[pocket_name] > 0
            and canonical_reserve_per_pocket > 0
            else 0
        )
        for pocket_name in POCKET_NAMES
    }
    required = {
        pocket_name: (
            final_required[pocket_name] + reserve_required[pocket_name]
        )
        for pocket_name in POCKET_NAMES
    }
    pool_count = sum(required.values())
    accepted_per_pocket = {pocket_name: 0 for pocket_name in POCKET_NAMES}
    accepted: list[TwoBallTask] = []
    rng = np.random.default_rng(seed)
    attempts = 0
    maximum_attempts = pool_count * max_attempts_per_task
    prefilter_worlds = min(num_worlds, max(pool_count * 8, 1))
    full_worlds = min(num_worlds, count)
    prefilter_environment: MJWarpMidLevelVecEnv | None = None
    full_environment: MJWarpMidLevelVecEnv | None = None
    replay_environment: MJWarpMidLevelVecEnv | None = None
    survivors: list[TwoBallTask] = []
    pocket_cursor = 0
    try:
        while len(accepted) < pool_count:
            survivors = [
                candidate
                for candidate in survivors
                if accepted_per_pocket[candidate.pocket_name]
                < required[candidate.pocket_name]
            ]
            while len(survivors) < full_worlds and attempts < maximum_attempts:
                candidates: list[TwoBallTask] = []
                while (
                    len(candidates) < prefilter_worlds
                    and attempts < maximum_attempts
                ):
                    eligible = [
                        name
                        for name in POCKET_NAMES
                        if accepted_per_pocket[name] < required[name]
                    ]
                    pocket_name = eligible[pocket_cursor % len(eligible)]
                    pocket_cursor += 1
                    candidate_seed = int(
                        rng.integers(
                            0,
                            np.iinfo(np.uint64).max,
                            dtype=np.uint64,
                        )
                    )
                    attempts += 1
                    candidate = _candidate_task(
                        pocket_name,
                        candidate_seed,
                        simulator,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                if not candidates:
                    break
                candidate_count = len(candidates)
                if status is not None:
                    status(
                        f"prefilter candidates={candidate_count} "
                        f"attempts={attempts} survivors={len(survivors)}"
                    )
                while len(candidates) < prefilter_worlds:
                    candidates.append(candidates[-1])
                candidate_dataset = _backend_dataset(
                    candidates,
                    simulator,
                    seed,
                    backend_hash,
                )
                if prefilter_environment is None:
                    prefilter_environment = MJWarpMidLevelVecEnv(
                        candidate_dataset,
                        model_path,
                        num_envs=prefilter_worlds,
                        seed=seed,
                        device=device,
                        chunk_steps=chunk_steps,
                        check_interval_steps=check_interval_steps,
                        nconmax=nconmax,
                        njmax=njmax,
                        max_time=prefilter_time,
                        validate_task_execution=False,
                    )
                else:
                    prefilter_environment.replace_task_dataset(
                        candidate_dataset
                    )
                prefilter_environment.set_options(
                    [
                        {"task_index": index}
                        for index in range(prefilter_worlds)
                    ]
                )
                prefilter_environment.reset()
                _, _, _, prefilter_infos = prefilter_environment.step(
                    _generated_actions(candidates)
                )
                _require_slot_identity(
                    candidates,
                    prefilter_infos,
                    context="MJWarp generation prefilter",
                )
                passed_prefilter = [
                    candidate
                    for candidate, info in zip(
                        candidates[:candidate_count],
                        prefilter_infos[:candidate_count],
                        strict=True,
                    )
                    if _passes_generation_prefilter(info)
                ]
                survivors.extend(passed_prefilter)
                if status is not None:
                    status(
                        f"prefilter passed={len(passed_prefilter)} "
                        f"survivors={len(survivors)}"
                    )

            if not survivors:
                remaining = {
                    name: required[name] - accepted_per_pocket[name]
                    for name in POCKET_NAMES
                    if required[name] > accepted_per_pocket[name]
                }
                raise RuntimeError(
                    "MJWarp task generation exhausted its candidate budget; "
                    f"accepted={len(accepted)}/{pool_count}, "
                    f"remaining={remaining}."
                )

            valid_count = min(full_worlds, len(survivors))
            finalists = survivors[:valid_count]
            del survivors[:valid_count]
            if status is not None:
                status(
                    f"primary rollout candidates={valid_count} "
                    f"stable_pool={len(accepted)}/{pool_count}"
                )
            while len(finalists) < full_worlds:
                finalists.append(finalists[-1])
            finalist_dataset = _backend_dataset(
                finalists,
                simulator,
                seed,
                backend_hash,
            )
            if full_environment is None:
                full_environment = MJWarpMidLevelVecEnv(
                    finalist_dataset,
                    model_path,
                    num_envs=full_worlds,
                    seed=seed,
                    device=device,
                    chunk_steps=chunk_steps,
                    check_interval_steps=check_interval_steps,
                    nconmax=nconmax,
                    njmax=njmax,
                    max_time=max_time,
                )
            else:
                full_environment.replace_task_dataset(finalist_dataset)
            full_environment.set_options(
                [{"task_index": index} for index in range(full_worlds)]
            )
            full_environment.reset()
            _, _, _, infos = full_environment.step(
                _generated_actions(finalists)
            )
            _require_slot_identity(
                finalists,
                infos,
                context="MJWarp generation rollout",
            )
            # Compact only first-pass feasible candidates into a separately
            # constructed environment. This changes their world-slot layout
            # and removes any first-rollout state history before acceptance.
            provisional: list[tuple[TwoBallTask, dict[str, object]]] = []
            for candidate, info in zip(
                finalists[:valid_count],
                infos[:valid_count],
                strict=True,
            ):
                pocket_name = candidate.pocket_name
                if (
                    accepted_per_pocket[pocket_name] >= required[pocket_name]
                    or not _is_feasible_info(info)
                ):
                    continue
                provisional.append((candidate, info))

            if not provisional:
                if status is not None:
                    status("primary rollout produced no feasible finalists")
                continue
            replay_count = len(provisional)
            if status is not None:
                status(
                    f"independent canonical replay candidates={replay_count}"
                )
            replay_candidates = [candidate for candidate, _ in provisional]
            while len(replay_candidates) < full_worlds:
                replay_candidates.append(replay_candidates[-1])
            replay_dataset = _backend_dataset(
                replay_candidates,
                simulator,
                seed,
                backend_hash,
            )
            if replay_environment is None:
                replay_environment = MJWarpMidLevelVecEnv(
                    replay_dataset,
                    model_path,
                    num_envs=full_worlds,
                    seed=seed + 1,
                    device=device,
                    chunk_steps=chunk_steps,
                    check_interval_steps=check_interval_steps,
                    nconmax=nconmax,
                    njmax=njmax,
                    max_time=max_time,
                )
            else:
                replay_environment.replace_task_dataset(replay_dataset)
            replay_environment.set_options(
                [{"task_index": index} for index in range(full_worlds)]
            )
            replay_environment.reset()
            _, _, _, replay_infos = replay_environment.step(
                _generated_actions(replay_candidates)
            )
            _require_slot_identity(
                replay_candidates,
                replay_infos,
                context="MJWarp independent generation replay",
            )
            accepted_before_replay = len(accepted)
            for (candidate, first_info), replay_info in zip(
                provisional,
                replay_infos[:replay_count],
                strict=True,
            ):
                pocket_name = candidate.pocket_name
                if accepted_per_pocket[pocket_name] >= required[pocket_name]:
                    continue
                task = _canonical_replay_task(
                    candidate,
                    first_info,
                    replay_info,
                    stop_tolerance=stability_stop_tolerance,
                )
                if task is None:
                    continue
                accepted.append(task)
                accepted_per_pocket[pocket_name] += 1
                if progress is not None and len(accepted) <= count:
                    progress(len(accepted), attempts, task)
                if len(accepted) == pool_count:
                    break
            if status is not None:
                stable_count = len(accepted) - accepted_before_replay
                status(
                    f"canonical replay accepted={stable_count}/{replay_count} "
                    f"stable_pool={len(accepted)}/{pool_count}"
                )
    finally:
        if prefilter_environment is not None:
            prefilter_environment.close()
        if full_environment is not None:
            full_environment.close()
        if replay_environment is not None:
            replay_environment.close()
        # close() synchronizes but intentionally does not own a global Warp
        # allocator. Drop the Python references before allocating the two
        # full-size fixed-layout replay environments.
        prefilter_environment = None
        full_environment = None
        replay_environment = None

    selected_per_pocket = {pocket_name: 0 for pocket_name in POCKET_NAMES}
    final_tasks: list[TwoBallTask] = []
    replacement_tasks = {pocket_name: [] for pocket_name in POCKET_NAMES}
    for task in accepted:
        pocket_name = task.pocket_name
        if selected_per_pocket[pocket_name] < final_required[pocket_name]:
            final_tasks.append(task)
            selected_per_pocket[pocket_name] += 1
        else:
            replacement_tasks[pocket_name].append(task)
    if len(final_tasks) != count:
        raise RuntimeError(
            "MJWarp stable pool did not contain the requested final pocket "
            f"balance: selected={len(final_tasks)}/{count}."
        )
    for pocket_name in POCKET_NAMES:
        if len(replacement_tasks[pocket_name]) != reserve_required[pocket_name]:
            raise RuntimeError(
                "MJWarp stable pool did not contain its requested canonical "
                f"reserve for {pocket_name}."
            )

    if status is not None:
        status(
            "starting fixed-layout double replay "
            f"tasks={count} reserves={pool_count - count}"
        )
    return _canonicalize_fixed_layout(
        final_tasks,
        replacement_tasks,
        simulator=simulator,
        generation_seed=seed,
        backend_hash=backend_hash,
        num_worlds=num_worlds,
        device=device,
        chunk_steps=chunk_steps,
        check_interval_steps=check_interval_steps,
        nconmax=nconmax,
        njmax=njmax,
        max_time=max_time,
        stop_tolerance=stability_stop_tolerance,
        max_rounds=canonical_max_rounds,
        status=status,
    )


def _validate_mujoco_warp_dataset_compatibility(
    dataset: TwoBallTaskDataset,
    simulator: TwoBallShotSimulator,
    backend_hash: str,
    *,
    allow_backend_recanonicalization: bool = False,
) -> None:
    if (
        dataset.xml_hash != simulator.xml_hash
        or dataset.model_hash != simulator.model_hash
    ):
        raise ValueError("Task dataset does not match the active base model.")
    if dataset.physics_backend != MUJOCO_WARP_PHYSICS_BACKEND:
        raise ValueError(
            "Fixed-layout replay requires a MJWarp-generated task dataset."
        )
    if (
        dataset.backend_hash != backend_hash
        and not allow_backend_recanonicalization
    ):
        raise ValueError(
            "Task dataset MJWarp backend hash does not match active physics."
        )
    if (
        dataset.execution_max_time != simulator.max_time
        or dataset.stop_speed != simulator.stop_speed
        or dataset.stop_hold_time != simulator.stop_hold_time
    ):
        raise ValueError(
            "Task dataset shot timing/stopping settings do not match the "
            "fixed-layout replay environment."
        )


def canonicalize_mujoco_warp_task_dataset(
    dataset: TwoBallTaskDataset,
    replacement_tasks: Sequence[TwoBallTask] = (),
    *,
    model_path: Path = DEFAULT_MIDLEVEL_MODEL,
    num_worlds: int = 1024,
    device: str = "cuda:0",
    chunk_steps: int = 16,
    check_interval_steps: int = 2048,
    nconmax: int = MUJOCO_WARP_NCONMAX,
    njmax: int = MUJOCO_WARP_NJMAX,
    max_time: float = 8.0,
    stop_tolerance: float = 5e-3,
    max_rounds: int = 8,
    status: GenerationStatus | None = None,
) -> TwoBallTaskDataset:
    """Canonicalize an existing library without changing any final slot."""

    if num_worlds <= 0:
        raise ValueError("num_worlds must be positive.")
    if not np.isfinite(stop_tolerance) or stop_tolerance <= 0.0:
        raise ValueError("stop_tolerance must be positive and finite.")
    simulator = TwoBallShotSimulator(model_path, max_time=max_time)
    xml_hash, model_hash, backend_hash = active_mujoco_warp_backend_sha256(
        model_path
    )
    if xml_hash != simulator.xml_hash or model_hash != simulator.model_hash:
        raise RuntimeError(
            "MJWarp fingerprint model does not match canonical replay model."
        )
    _validate_mujoco_warp_dataset_compatibility(
        dataset,
        simulator,
        backend_hash,
    )

    grouped_replacements = {pocket_name: [] for pocket_name in POCKET_NAMES}
    for task in replacement_tasks:
        if task.pocket_name not in grouped_replacements:
            raise ValueError(
                f"Replacement task has an unknown pocket: {task.pocket_name}"
            )
        grouped_replacements[task.pocket_name].append(task)

    return _canonicalize_fixed_layout(
        [dataset[index] for index in range(len(dataset))],
        grouped_replacements,
        simulator=simulator,
        generation_seed=dataset.generation_seed,
        backend_hash=backend_hash,
        num_worlds=num_worlds,
        device=device,
        chunk_steps=chunk_steps,
        check_interval_steps=check_interval_steps,
        nconmax=nconmax,
        njmax=njmax,
        max_time=max_time,
        stop_tolerance=stop_tolerance,
        max_rounds=max_rounds,
        status=status,
    )


def repair_mujoco_warp_task_dataset(
    dataset: TwoBallTaskDataset,
    *,
    model_path: Path = DEFAULT_MIDLEVEL_MODEL,
    replacement_tasks_per_pocket: int = 16,
    replacement_seed: int | None = None,
    num_worlds: int = 1024,
    device: str = "cuda:0",
    chunk_steps: int = 16,
    check_interval_steps: int = 2048,
    nconmax: int = MUJOCO_WARP_NCONMAX,
    njmax: int = MUJOCO_WARP_NJMAX,
    max_time: float = 8.0,
    prefilter_time: float = 1.5,
    stop_tolerance: float = 5e-3,
    max_rounds: int = 8,
    max_attempts_per_task: int = 2_000,
    allow_backend_recanonicalization: bool = False,
    status: GenerationStatus | None = None,
) -> TwoBallTaskDataset:
    """Repair a staged library with independently generated pocket reserves.

    Backend migration is opt-in.  When enabled, only a backend-hash mismatch
    is tolerated on input; the model and shot-execution contract remain
    strict.  The returned dataset receives the active hash only after every
    fixed slot has passed two independent replays on the active backend.
    """

    if replacement_tasks_per_pocket <= 0:
        raise ValueError("replacement_tasks_per_pocket must be positive.")

    simulator = TwoBallShotSimulator(model_path, max_time=max_time)
    xml_hash, model_hash, backend_hash = active_mujoco_warp_backend_sha256(
        model_path
    )
    if xml_hash != simulator.xml_hash or model_hash != simulator.model_hash:
        raise RuntimeError("MJWarp fingerprint model does not match repair model.")
    _validate_mujoco_warp_dataset_compatibility(
        dataset,
        simulator,
        backend_hash,
        allow_backend_recanonicalization=allow_backend_recanonicalization,
    )
    if status is not None and dataset.backend_hash != backend_hash:
        status(
            "recanonicalizing staged backend "
            f"source_sha256={dataset.backend_hash} "
            f"active_sha256={backend_hash}"
        )

    if replacement_seed is None:
        replacement_seed = (
            dataset.generation_seed + 0x9E3779B97F4A7C15
        ) & int(np.iinfo(np.uint64).max)
    replacement_count = replacement_tasks_per_pocket * len(POCKET_NAMES)
    if status is not None:
        status(
            "generating independent fixed-layout replacements "
            f"tasks={replacement_count}"
        )

    def replacement_status(message: str) -> None:
        if status is not None:
            status(f"replacement pool: {message}")

    replacement_dataset = generate_mujoco_warp_task_dataset(
        replacement_count,
        seed=int(replacement_seed),
        model_path=model_path,
        num_worlds=num_worlds,
        device=device,
        chunk_steps=chunk_steps,
        check_interval_steps=check_interval_steps,
        nconmax=nconmax,
        njmax=njmax,
        max_time=max_time,
        prefilter_time=prefilter_time,
        stability_stop_tolerance=stop_tolerance,
        canonical_reserve_per_pocket=max(
            1,
            min(4, replacement_tasks_per_pocket // 4),
        ),
        canonical_max_rounds=max_rounds,
        max_attempts_per_task=max_attempts_per_task,
        status=replacement_status,
    )
    replacements = [
        replacement_dataset[index]
        for index in range(len(replacement_dataset))
    ]
    if status is not None:
        status("canonicalizing staged tasks in their final immutable slots")
    return _canonicalize_fixed_layout(
        [dataset[index] for index in range(len(dataset))],
        {
            pocket_name: [
                task
                for task in replacements
                if task.pocket_name == pocket_name
            ]
            for pocket_name in POCKET_NAMES
        },
        simulator=simulator,
        generation_seed=dataset.generation_seed,
        backend_hash=backend_hash,
        num_worlds=num_worlds,
        device=device,
        chunk_steps=chunk_steps,
        check_interval_steps=check_interval_steps,
        nconmax=nconmax,
        njmax=njmax,
        max_time=max_time,
        stop_tolerance=stop_tolerance,
        max_rounds=max_rounds,
        status=status,
    )


def validate_mujoco_warp_task_dataset(
    dataset: TwoBallTaskDataset,
    *,
    model_path: Path = DEFAULT_MIDLEVEL_MODEL,
    max_tasks: int | None = None,
    num_worlds: int = 1024,
    device: str = "cuda:0",
    chunk_steps: int = 16,
    check_interval_steps: int = 2048,
    nconmax: int = MUJOCO_WARP_NCONMAX,
    njmax: int = MUJOCO_WARP_NJMAX,
    max_time: float = 8.0,
    stop_tolerance: float = 5e-3,
) -> TaskValidationReport:
    """Replay stored generation actions on the same fingerprinted backend.

    The calibrated MJWarp backend uses deterministic per-world contact and
    Newton-reduction ordering.  The five-millimeter default remains a strict
    guard against physical/model changes while allowing future backend builds
    to differ below one tenth of the policy's 5 cm joint-success tolerance.
    """

    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("max_tasks must be positive when provided.")
    if num_worlds <= 0:
        raise ValueError("num_worlds must be positive.")
    if not np.isfinite(stop_tolerance) or stop_tolerance <= 0.0:
        raise ValueError("stop_tolerance must be positive and finite.")
    total_count = len(dataset)
    checked_count = (
        total_count if max_tasks is None else min(total_count, max_tasks)
    )
    if checked_count <= 0:
        raise ValueError("A replay check requires at least one task.")
    batch_size = min(num_worlds, total_count)
    failures: list[str] = []
    max_stop_error = 0.0
    passed = 0
    environment = MJWarpMidLevelVecEnv(
        dataset,
        model_path,
        num_envs=batch_size,
        device=device,
        chunk_steps=chunk_steps,
        check_interval_steps=check_interval_steps,
        nconmax=nconmax,
        njmax=njmax,
        max_time=max_time,
    )
    try:
        for start in range(0, checked_count, batch_size):
            valid_count = min(batch_size, total_count - start)
            checked_in_batch = min(valid_count, checked_count - start)
            indices = _fixed_layout_indices(start, total_count, batch_size)
            tasks = [dataset[index] for index in indices]
            environment.set_options(
                [{"task_index": index} for index in indices]
            )
            environment.reset()
            _, _, _, infos = environment.step(_generated_actions(tasks))
            _require_slot_identity(
                tasks,
                infos,
                context="MJWarp task replay",
            )
            for offset in range(checked_in_batch):
                task_index = start + offset
                info = infos[offset]
                cue_final = np.asarray(
                    info["cue_ball_final_position"],
                    dtype=np.float64,
                )
                stop_error = float(
                    np.linalg.norm(
                        cue_final[:2] - dataset.target_stop_positions[task_index]
                    )
                )
                max_stop_error = max(max_stop_error, stop_error)
                if not _is_feasible_info(info):
                    failures.append(
                        f"task {task_index}: generated action no longer "
                        "produces a feasible direct pot"
                    )
                elif stop_error > stop_tolerance:
                    failures.append(
                        f"task {task_index}: stop replay error "
                        f"{stop_error:.6g} m"
                    )
                else:
                    passed += 1
    finally:
        environment.close()
    return TaskValidationReport(
        checked_count=checked_count,
        passed_count=passed,
        max_stop_replay_error=max_stop_error,
        failures=tuple(failures),
    )
