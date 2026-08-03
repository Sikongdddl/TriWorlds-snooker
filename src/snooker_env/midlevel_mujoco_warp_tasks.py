"""Batched MJWarp generation and replay for feasible two-ball tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

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
        abs(float(cue_final[0])) > OBSERVATION_X_SCALE
        or abs(float(cue_final[1])) > OBSERVATION_Y_SCALE
    ):
        return None
    direction = np.asarray(info["shot_direction"], dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    return TwoBallTask(
        cue_position=candidate.cue_position.copy(),
        object_position=candidate.object_position.copy(),
        pocket_name=candidate.pocket_name,
        pocket_position=candidate.pocket_position.copy(),
        target_stop_position=cue_final[:2].copy(),
        generated_direction=direction,
        generated_speed=float(info["cue_speed"]),
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
    max_attempts_per_task: int = 2_000,
    progress: GenerationProgress | None = None,
) -> TwoBallTaskDataset:
    """Generate an evenly pocket-balanced library on the training backend."""

    if count <= 0:
        raise ValueError("count must be positive.")
    if num_worlds <= 0:
        raise ValueError("num_worlds must be positive.")
    if max_attempts_per_task <= 0:
        raise ValueError("max_attempts_per_task must be positive.")
    if prefilter_time <= 0.0 or prefilter_time >= max_time:
        raise ValueError("prefilter_time must be positive and less than max_time.")

    simulator = TwoBallShotSimulator(model_path, max_time=max_time)
    xml_hash, model_hash, backend_hash = active_mujoco_warp_backend_sha256(
        model_path
    )
    if xml_hash != simulator.xml_hash or model_hash != simulator.model_hash:
        raise RuntimeError("MJWarp fingerprint model does not match task generation model.")

    schedule = _generation_schedule(count, seed)
    required = {
        pocket_name: sum(name == pocket_name for name, _ in schedule)
        for pocket_name in POCKET_NAMES
    }
    accepted_per_pocket = {pocket_name: 0 for pocket_name in POCKET_NAMES}
    accepted: list[TwoBallTask] = []
    rng = np.random.default_rng(seed)
    attempts = 0
    maximum_attempts = count * max_attempts_per_task
    prefilter_worlds = min(num_worlds, max(count * 8, 1))
    full_worlds = min(num_worlds, count)
    prefilter_environment: MJWarpMidLevelVecEnv | None = None
    full_environment: MJWarpMidLevelVecEnv | None = None
    survivors: list[TwoBallTask] = []
    pocket_cursor = 0
    try:
        while len(accepted) < count:
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
                survivors.extend(
                    candidate
                    for candidate, info in zip(
                        candidates[:candidate_count],
                        prefilter_infos[:candidate_count],
                        strict=True,
                    )
                    if _passes_generation_prefilter(info)
                )

            if not survivors:
                remaining = {
                    name: required[name] - accepted_per_pocket[name]
                    for name in POCKET_NAMES
                    if required[name] > accepted_per_pocket[name]
                }
                raise RuntimeError(
                    "MJWarp task generation exhausted its candidate budget; "
                    f"accepted={len(accepted)}/{count}, remaining={remaining}."
                )

            valid_count = min(full_worlds, len(survivors))
            finalists = survivors[:valid_count]
            del survivors[:valid_count]
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
            for candidate, info in zip(
                finalists[:valid_count],
                infos[:valid_count],
                strict=True,
            ):
                pocket_name = candidate.pocket_name
                if accepted_per_pocket[pocket_name] >= required[pocket_name]:
                    continue
                task = _accepted_task(candidate, info)
                if task is None:
                    continue
                accepted.append(task)
                accepted_per_pocket[pocket_name] += 1
                if progress is not None:
                    progress(len(accepted), attempts, task)
                if len(accepted) == count:
                    break
    finally:
        if prefilter_environment is not None:
            prefilter_environment.close()
        if full_environment is not None:
            full_environment.close()

    return _backend_dataset(accepted, simulator, seed, backend_hash)


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
    count = len(dataset) if max_tasks is None else min(len(dataset), max_tasks)
    if count <= 0:
        raise ValueError("A replay check requires at least one task.")
    batch_size = min(num_worlds, count)
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
        for start in range(0, count, batch_size):
            valid_count = min(batch_size, count - start)
            indices = [
                start + min(offset, valid_count - 1)
                for offset in range(batch_size)
            ]
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
            for offset in range(valid_count):
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
        checked_count=count,
        passed_count=passed,
        max_stop_replay_error=max_stop_error,
        failures=tuple(failures),
    )
