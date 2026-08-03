"""Feasible two-ball task generation and versioned dataset storage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Callable, Mapping, Sequence

import numpy as np

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL
from snooker_env.midlevel_two_ball import (
    MAX_CUE_SPEED,
    MIN_CUE_SPEED,
    POCKET_NAMES,
    SHOT_EXECUTION_VERSION,
    TwoBallShotResult,
    TwoBallShotSimulator,
    ghost_ball_direction,
    quantize_cue_speed,
    rotate_direction,
)
from snooker_env.table_geometry import BALL_RADIUS


TASK_DATASET_VERSION = 4
CPU_PHYSICS_BACKEND = "mujoco_cpu"
MUJOCO_WARP_PHYSICS_BACKEND = "mujoco_warp"
DEFAULT_TRAIN_TASKS = 61_440
DEFAULT_VALIDATION_TASKS = 6_144
EVENT_FLAG_NAMES = (
    "correct_pot",
    "legal_first_contact",
    "no_cushion_direct_pot",
    "cue_scratch",
    "stopped",
    "timed_out",
    "numerical_failure",
)


@dataclass(frozen=True)
class TwoBallTask:
    """One guaranteed-feasible direct-pot task."""

    cue_position: np.ndarray
    object_position: np.ndarray
    pocket_name: str
    pocket_position: np.ndarray
    target_stop_position: np.ndarray
    generated_direction: np.ndarray
    generated_speed: float
    candidate_seed: int
    elapsed_time: float
    min_object_pocket_distance: float
    event_metrics: Mapping[str, bool]


@dataclass(frozen=True)
class TaskValidationReport:
    """Result of replaying all or part of a task library."""

    checked_count: int
    passed_count: int
    max_stop_replay_error: float
    failures: tuple[str, ...]


class TwoBallTaskDataset:
    """Compact NumPy-backed collection with strict model compatibility."""

    def __init__(
        self,
        *,
        cue_positions: np.ndarray,
        object_positions: np.ndarray,
        pocket_indices: np.ndarray,
        target_stop_positions: np.ndarray,
        generated_directions: np.ndarray,
        generated_speeds: np.ndarray,
        candidate_seeds: np.ndarray,
        elapsed_times: np.ndarray,
        min_object_pocket_distances: np.ndarray,
        event_flags: np.ndarray,
        xml_hash: str,
        model_hash: str,
        physics_backend: str,
        backend_hash: str,
        generation_seed: int,
        execution_max_time: float,
        stop_speed: float,
        stop_hold_time: float,
    ) -> None:
        self.cue_positions = np.asarray(cue_positions, dtype=np.float64)
        self.object_positions = np.asarray(object_positions, dtype=np.float64)
        self.pocket_indices = np.asarray(pocket_indices, dtype=np.int8)
        self.target_stop_positions = np.asarray(target_stop_positions, dtype=np.float64)
        self.generated_directions = np.asarray(generated_directions, dtype=np.float64)
        self.generated_speeds = np.asarray(generated_speeds, dtype=np.float64)
        self.candidate_seeds = np.asarray(candidate_seeds, dtype=np.uint64)
        self.elapsed_times = np.asarray(elapsed_times, dtype=np.float64)
        self.min_object_pocket_distances = np.asarray(min_object_pocket_distances, dtype=np.float64)
        self.event_flags = np.asarray(event_flags, dtype=np.bool_)
        self.xml_hash = str(xml_hash)
        self.model_hash = str(model_hash)
        self.physics_backend = str(physics_backend)
        self.backend_hash = str(backend_hash)
        self.generation_seed = int(generation_seed)
        self.execution_max_time = float(execution_max_time)
        self.stop_speed = float(stop_speed)
        self.stop_hold_time = float(stop_hold_time)
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        count = int(self.pocket_indices.shape[0])
        expected = {
            "cue_positions": (count, 2),
            "object_positions": (count, 2),
            "target_stop_positions": (count, 2),
            "generated_directions": (count, 2),
            "generated_speeds": (count,),
            "candidate_seeds": (count,),
            "elapsed_times": (count,),
            "min_object_pocket_distances": (count,),
            "event_flags": (count, len(EVENT_FLAG_NAMES)),
        }
        for name, shape in expected.items():
            values = getattr(self, name)
            if values.shape != shape:
                raise ValueError(f"{name} has shape {values.shape}, expected {shape}.")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains non-finite values.")
        if np.any(self.pocket_indices < 0) or np.any(self.pocket_indices >= len(POCKET_NAMES)):
            raise ValueError("pocket_indices contains an unknown pocket.")
        if self.physics_backend not in (
            CPU_PHYSICS_BACKEND,
            MUJOCO_WARP_PHYSICS_BACKEND,
        ):
            raise ValueError(f"Unsupported task physics backend: {self.physics_backend}")
        for name, value in (
            ("xml_hash", self.xml_hash),
            ("model_hash", self.model_hash),
            ("backend_hash", self.backend_hash),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} is not a lowercase SHA-256 digest.")
        for name in (
            "execution_max_time",
            "stop_speed",
            "stop_hold_time",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        direction_norms = np.linalg.norm(self.generated_directions, axis=1)
        if not np.allclose(direction_norms, 1.0, atol=1e-8):
            raise ValueError("Generated directions must be unit vectors.")
        if np.any(self.generated_speeds < MIN_CUE_SPEED) or np.any(
            self.generated_speeds > MAX_CUE_SPEED
        ):
            raise ValueError("Generated speeds are outside the action range.")

    def __len__(self) -> int:
        return int(self.pocket_indices.shape[0])

    def __getitem__(self, index: int) -> TwoBallTask:
        pocket_name = POCKET_NAMES[int(self.pocket_indices[index])]
        return TwoBallTask(
            cue_position=self.cue_positions[index].copy(),
            object_position=self.object_positions[index].copy(),
            pocket_name=pocket_name,
            pocket_position=np.asarray(
                # Stored pocket coordinates come from the same named sites;
                # retaining a fixed copy makes observations independent of a
                # live simulator after compatibility validation.
                _POCKET_ARRAY[int(self.pocket_indices[index])], dtype=np.float64
            ).copy(),
            target_stop_position=self.target_stop_positions[index].copy(),
            generated_direction=self.generated_directions[index].copy(),
            generated_speed=float(self.generated_speeds[index]),
            candidate_seed=int(self.candidate_seeds[index]),
            elapsed_time=float(self.elapsed_times[index]),
            min_object_pocket_distance=float(self.min_object_pocket_distances[index]),
            event_metrics={
                name: bool(self.event_flags[index, flag_index])
                for flag_index, name in enumerate(EVENT_FLAG_NAMES)
            },
        )

    def content_sha256(self) -> str:
        """Hash all task values and compatibility metadata."""

        digest = hashlib.sha256()
        for value in (
            str(TASK_DATASET_VERSION),
            SHOT_EXECUTION_VERSION,
            self.xml_hash,
            self.model_hash,
            self.physics_backend,
            self.backend_hash,
            str(self.generation_seed),
            repr(self.execution_max_time),
            repr(self.stop_speed),
            repr(self.stop_hold_time),
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        for name in (
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
        ):
            values = np.asarray(getattr(self, name))
            digest.update(name.encode("ascii"))
            digest.update(values.dtype.str.encode("ascii"))
            digest.update(repr(values.shape).encode("ascii"))
            digest.update(values.tobytes(order="C"))
        return digest.hexdigest()

    @classmethod
    def from_tasks(
        cls,
        tasks: Sequence[TwoBallTask],
        simulator: TwoBallShotSimulator,
        generation_seed: int,
        *,
        physics_backend: str = CPU_PHYSICS_BACKEND,
        backend_hash: str | None = None,
    ) -> "TwoBallTaskDataset":
        if not tasks:
            raise ValueError("A task dataset cannot be empty.")
        return cls(
            cue_positions=np.stack([task.cue_position for task in tasks]),
            object_positions=np.stack([task.object_position for task in tasks]),
            pocket_indices=np.asarray([POCKET_NAMES.index(task.pocket_name) for task in tasks]),
            target_stop_positions=np.stack([task.target_stop_position for task in tasks]),
            generated_directions=np.stack([task.generated_direction for task in tasks]),
            generated_speeds=np.asarray([task.generated_speed for task in tasks]),
            candidate_seeds=np.asarray([task.candidate_seed for task in tasks], dtype=np.uint64),
            elapsed_times=np.asarray([task.elapsed_time for task in tasks]),
            min_object_pocket_distances=np.asarray(
                [task.min_object_pocket_distance for task in tasks]
            ),
            event_flags=np.asarray(
                [
                    [bool(task.event_metrics[name]) for name in EVENT_FLAG_NAMES]
                    for task in tasks
                ],
                dtype=np.bool_,
            ),
            xml_hash=simulator.xml_hash,
            model_hash=simulator.model_hash,
            physics_backend=physics_backend,
            backend_hash=backend_hash or simulator.model_hash,
            generation_seed=generation_seed,
            execution_max_time=simulator.max_time,
            stop_speed=simulator.stop_speed,
            stop_hold_time=simulator.stop_hold_time,
        )

    def save(self, path: Path) -> None:
        """Write a compressed, pickle-free task library."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            {
                "dataset_version": TASK_DATASET_VERSION,
                "shot_execution_version": SHOT_EXECUTION_VERSION,
                "xml_sha256": self.xml_hash,
                "model_sha256": self.model_hash,
                "physics_backend": self.physics_backend,
                "backend_sha256": self.backend_hash,
                "generation_seed": self.generation_seed,
                "execution_max_time": self.execution_max_time,
                "stop_speed": self.stop_speed,
                "stop_hold_time": self.stop_hold_time,
                "task_count": len(self),
                "content_sha256": self.content_sha256(),
                "pocket_names": POCKET_NAMES,
                "event_flag_names": EVENT_FLAG_NAMES,
            },
            sort_keys=True,
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output.parent,
                prefix=f".{output.stem}.",
                suffix=".npz",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary_path,
                metadata=np.asarray(metadata),
                cue_positions=self.cue_positions,
                object_positions=self.object_positions,
                pocket_indices=self.pocket_indices,
                target_stop_positions=self.target_stop_positions,
                generated_directions=self.generated_directions,
                generated_speeds=self.generated_speeds,
                candidate_seeds=self.candidate_seeds,
                elapsed_times=self.elapsed_times,
                min_object_pocket_distances=self.min_object_pocket_distances,
                event_flags=self.event_flags,
            )
            temporary_path.replace(output)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        simulator: TwoBallShotSimulator | None = None,
        model_path: Path = DEFAULT_MIDLEVEL_MODEL,
        validate_model: bool = True,
        expected_backend: str | None = None,
        backend_hash: str | None = None,
    ) -> "TwoBallTaskDataset":
        """Load a library and reject stale physics/model fingerprints."""

        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if int(metadata.get("dataset_version", -1)) != TASK_DATASET_VERSION:
                raise ValueError("Unsupported mid-level task dataset version.")
            if metadata.get("shot_execution_version") != SHOT_EXECUTION_VERSION:
                raise ValueError("Task dataset was generated by a different shot executor.")
            if tuple(metadata.get("pocket_names", ())) != POCKET_NAMES:
                raise ValueError("Task dataset pocket ordering does not match this code version.")
            if tuple(metadata.get("event_flag_names", ())) != EVENT_FLAG_NAMES:
                raise ValueError("Task dataset event metric ordering does not match this code version.")
            dataset = cls(
                cue_positions=archive["cue_positions"],
                object_positions=archive["object_positions"],
                pocket_indices=archive["pocket_indices"],
                target_stop_positions=archive["target_stop_positions"],
                generated_directions=archive["generated_directions"],
                generated_speeds=archive["generated_speeds"],
                candidate_seeds=archive["candidate_seeds"],
                elapsed_times=archive["elapsed_times"],
                min_object_pocket_distances=archive["min_object_pocket_distances"],
                event_flags=archive["event_flags"],
                xml_hash=metadata["xml_sha256"],
                model_hash=metadata["model_sha256"],
                physics_backend=metadata["physics_backend"],
                backend_hash=metadata["backend_sha256"],
                generation_seed=int(metadata["generation_seed"]),
                execution_max_time=float(metadata["execution_max_time"]),
                stop_speed=float(metadata["stop_speed"]),
                stop_hold_time=float(metadata["stop_hold_time"]),
            )
        if int(metadata.get("task_count", -1)) != len(dataset):
            raise ValueError("Task dataset count does not match its metadata.")
        if metadata.get("content_sha256") != dataset.content_sha256():
            raise ValueError("Task dataset content hash does not match its metadata.")
        if validate_model:
            active_simulator = simulator or TwoBallShotSimulator(model_path)
            if dataset.xml_hash != active_simulator.xml_hash:
                raise ValueError("Task dataset XML hash does not match the active model.")
            if dataset.model_hash != active_simulator.model_hash:
                raise ValueError("Task dataset compiled-model hash does not match active physics.")
            if (
                dataset.execution_max_time != active_simulator.max_time
                or dataset.stop_speed != active_simulator.stop_speed
                or dataset.stop_hold_time != active_simulator.stop_hold_time
            ):
                raise ValueError(
                    "Task dataset shot timing/stopping settings do not match "
                    "the active simulator."
                )
        if expected_backend is not None and dataset.physics_backend != expected_backend:
            raise ValueError(
                "Task dataset physics backend does not match the requested backend: "
                f"{dataset.physics_backend!r} != {expected_backend!r}."
            )
        if backend_hash is not None and dataset.backend_hash != backend_hash:
            raise ValueError("Task dataset backend hash does not match active physics.")
        return dataset


_POCKET_ARRAY = np.asarray(
    [
        (-0.675, -1.31),
        (-0.675, 1.31),
        (0.675, -1.31),
        (0.675, 1.31),
        (-0.717426, 0.0),
        (0.717426, 0.0),
    ],
    dtype=np.float64,
)


def _sample_candidate(
    pocket_name: str,
    candidate_seed: int,
    simulator: TwoBallShotSimulator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    rng = np.random.default_rng(candidate_seed)
    pocket_position = simulator.pocket_positions[pocket_name]
    # Put the object in a short finishing lane, then sample the cue around the
    # ghost ball.  The independent cut angle is essential: placing the cue on
    # the pot line produces only trivial straight shots and leaves PPO unable
    # to learn speed transfer or rail/scratch avoidance for ordinary cuts.
    outward = pocket_position / max(float(np.linalg.norm(pocket_position)), 1e-12)
    lane_angle = rng.uniform(np.deg2rad(-2.0), np.deg2rad(2.0))
    pot_direction = rotate_direction(outward, lane_angle)
    distance_to_pocket = float(rng.uniform(0.18, 0.34))
    object_position = pocket_position - pot_direction * distance_to_pocket
    if abs(object_position[0]) > 0.56 or abs(object_position[1]) > 1.18:
        return None
    ghost_position = object_position - 2.0 * BALL_RADIUS * pot_direction
    cut_angle = rng.uniform(np.deg2rad(-48.0), np.deg2rad(48.0))
    shot_direction = rotate_direction(pot_direction, cut_angle)
    cue_distance = rng.uniform(0.28, 0.82)
    cue_position = ghost_position - shot_direction * cue_distance
    if abs(cue_position[0]) > 0.55 or abs(cue_position[1]) > 1.15:
        return None
    if np.linalg.norm(cue_position - object_position) < 3.0 * BALL_RADIUS:
        return None
    transfer = max(float(np.cos(cut_angle)), 0.55)
    nominal_speed = (0.38 + 1.05 * distance_to_pocket) / transfer
    speed = quantize_cue_speed(
        float(np.clip(nominal_speed + rng.uniform(-0.12, 0.18), MIN_CUE_SPEED, 1.8))
    )
    # Recompute through the public helper so the stored direction is exactly
    # the action-space baseline used during training.
    direction = ghost_ball_direction(cue_position, object_position, pocket_position)
    return cue_position, object_position, direction, speed


def _is_feasible(result: TwoBallShotResult) -> bool:
    return bool(
        result.correct_pot
        and result.legal_first_contact
        and not result.cue_scratch
        and not result.cushion_before_object
        and not result.object_cushion_before_pocket
        and result.stopped
        and not result.timed_out
        and not result.numerical_failure
    )


def _task_from_result(
    *,
    cue_position: np.ndarray,
    object_position: np.ndarray,
    pocket_name: str,
    direction: np.ndarray,
    speed: float,
    candidate_seed: int,
    result: TwoBallShotResult,
    simulator: TwoBallShotSimulator,
) -> TwoBallTask:
    return TwoBallTask(
        cue_position=cue_position.copy(),
        object_position=object_position.copy(),
        pocket_name=pocket_name,
        pocket_position=simulator.pocket_positions[pocket_name].copy(),
        target_stop_position=result.cue_ball_final_position[:2].copy(),
        generated_direction=direction.copy(),
        generated_speed=speed,
        candidate_seed=candidate_seed,
        elapsed_time=result.elapsed_time,
        min_object_pocket_distance=result.min_object_pocket_distance,
        event_metrics={
            "correct_pot": result.correct_pot,
            "legal_first_contact": result.legal_first_contact,
            "no_cushion_direct_pot": not result.cushion_before_object
            and not result.object_cushion_before_pocket,
            "cue_scratch": result.cue_scratch,
            "stopped": result.stopped,
            "timed_out": result.timed_out,
            "numerical_failure": result.numerical_failure,
        },
    )


def _generate_one_task(
    pocket_name: str,
    task_seed: int,
    simulator: TwoBallShotSimulator,
    max_attempts: int,
) -> tuple[int, TwoBallTask]:
    rng = np.random.default_rng(task_seed)
    for attempt in range(1, max_attempts + 1):
        candidate_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        candidate = _sample_candidate(pocket_name, candidate_seed, simulator)
        if candidate is None:
            continue
        cue_position, object_position, direction, speed = candidate
        result = simulator.execute(
            cue_position,
            object_position,
            pocket_name,
            direction,
            speed,
        )
        if _is_feasible(result):
            return attempt, _task_from_result(
                cue_position=cue_position,
                object_position=object_position,
                pocket_name=pocket_name,
                direction=direction,
                speed=speed,
                candidate_seed=candidate_seed,
                result=result,
                simulator=simulator,
            )
    raise RuntimeError(
        f"Could not generate a feasible task for {pocket_name} after {max_attempts} attempts."
    )


def _generation_schedule(count: int, seed: int) -> tuple[tuple[str, int], ...]:
    rng = np.random.default_rng(seed)
    repeats = int(np.ceil(count / len(POCKET_NAMES)))
    pocket_order = np.tile(np.asarray(POCKET_NAMES, dtype="U32"), repeats)
    rng.shuffle(pocket_order)
    pocket_order = pocket_order[:count]
    task_seeds = rng.integers(0, np.iinfo(np.uint64).max, size=count, dtype=np.uint64)
    return tuple(
        (str(pocket_name), int(task_seed))
        for pocket_name, task_seed in zip(pocket_order, task_seeds, strict=True)
    )


def generate_task_dataset(
    count: int,
    *,
    seed: int,
    simulator: TwoBallShotSimulator | None = None,
    model_path: Path = DEFAULT_MIDLEVEL_MODEL,
    max_attempts_per_task: int = 2_000,
    progress: Callable[[int, int, TwoBallTask], None] | None = None,
) -> TwoBallTaskDataset:
    """Generate an evenly pocket-balanced library by exact MuJoCo rejection."""

    if count <= 0:
        raise ValueError("count must be positive.")
    if max_attempts_per_task <= 0:
        raise ValueError("max_attempts_per_task must be positive.")
    active_simulator = simulator or TwoBallShotSimulator(model_path)
    tasks: list[TwoBallTask] = []
    for pocket_name, task_seed in _generation_schedule(count, seed):
        attempt, task = _generate_one_task(
            pocket_name, task_seed, active_simulator, max_attempts_per_task
        )
        tasks.append(task)
        if progress is not None:
            progress(len(tasks), attempt, task)
    return TwoBallTaskDataset.from_tasks(tasks, active_simulator, generation_seed=seed)


_WORKER_SIMULATOR: TwoBallShotSimulator | None = None


def _initialize_generation_worker(
    model_path: str,
    max_time: float,
    stop_speed: float,
    stop_hold_time: float,
) -> None:
    global _WORKER_SIMULATOR
    _WORKER_SIMULATOR = TwoBallShotSimulator(
        Path(model_path),
        max_time=max_time,
        stop_speed=stop_speed,
        stop_hold_time=stop_hold_time,
    )


def _generation_worker(arguments: tuple[str, int, int]) -> tuple[int, TwoBallTask]:
    if _WORKER_SIMULATOR is None:
        raise RuntimeError("Task generation worker was not initialized.")
    pocket_name, task_seed, max_attempts = arguments
    return _generate_one_task(pocket_name, task_seed, _WORKER_SIMULATOR, max_attempts)


def generate_task_dataset_parallel(
    count: int,
    *,
    seed: int,
    num_workers: int,
    simulator: TwoBallShotSimulator | None = None,
    model_path: Path = DEFAULT_MIDLEVEL_MODEL,
    max_attempts_per_task: int = 2_000,
    progress: Callable[[int, int, TwoBallTask], None] | None = None,
) -> TwoBallTaskDataset:
    """Generate deterministic tasks concurrently with spawn-safe workers."""

    if num_workers <= 1:
        return generate_task_dataset(
            count,
            seed=seed,
            simulator=simulator,
            model_path=model_path,
            max_attempts_per_task=max_attempts_per_task,
            progress=progress,
        )
    if count <= 0:
        raise ValueError("count must be positive.")
    if max_attempts_per_task <= 0:
        raise ValueError("max_attempts_per_task must be positive.")
    import multiprocessing

    active_simulator = simulator or TwoBallShotSimulator(model_path)
    work = tuple(
        (pocket_name, task_seed, max_attempts_per_task)
        for pocket_name, task_seed in _generation_schedule(count, seed)
    )
    tasks: list[TwoBallTask] = []
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=num_workers,
        initializer=_initialize_generation_worker,
        initargs=(
            str(active_simulator.model_path),
            active_simulator.max_time,
            active_simulator.stop_speed,
            active_simulator.stop_hold_time,
        ),
    ) as pool:
        for attempt, task in pool.imap(_generation_worker, work, chunksize=1):
            tasks.append(task)
            if progress is not None:
                progress(len(tasks), attempt, task)
    return TwoBallTaskDataset.from_tasks(tasks, active_simulator, generation_seed=seed)


def validate_task_dataset(
    dataset: TwoBallTaskDataset,
    *,
    simulator: TwoBallShotSimulator | None = None,
    model_path: Path = DEFAULT_MIDLEVEL_MODEL,
    max_tasks: int | None = None,
    stop_tolerance: float = 2e-4,
) -> TaskValidationReport:
    """Replay generated actions and verify all feasibility guarantees."""

    active_simulator = simulator or TwoBallShotSimulator(model_path)
    if dataset.xml_hash != active_simulator.xml_hash or dataset.model_hash != active_simulator.model_hash:
        raise ValueError("Cannot validate a task dataset against different physics.")
    if (
        dataset.physics_backend != CPU_PHYSICS_BACKEND
        or dataset.backend_hash != active_simulator.model_hash
    ):
        raise ValueError("CPU replay requires a task dataset generated by this CPU backend.")
    if (
        dataset.execution_max_time != active_simulator.max_time
        or dataset.stop_speed != active_simulator.stop_speed
        or dataset.stop_hold_time != active_simulator.stop_hold_time
    ):
        raise ValueError("CPU replay shot timing/stopping settings do not match the dataset.")
    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("max_tasks must be positive when provided.")
    count = len(dataset) if max_tasks is None else min(len(dataset), int(max_tasks))
    failures: list[str] = []
    max_stop_error = 0.0
    passed = 0
    for index in range(count):
        task = dataset[index]
        result = active_simulator.execute(
            task.cue_position,
            task.object_position,
            task.pocket_name,
            task.generated_direction,
            task.generated_speed,
        )
        stop_error = float(np.linalg.norm(result.cue_ball_final_position[:2] - task.target_stop_position))
        max_stop_error = max(max_stop_error, stop_error)
        if not _is_feasible(result):
            failures.append(f"task {index}: generated action no longer produces a feasible direct pot")
        elif stop_error > stop_tolerance:
            failures.append(f"task {index}: stop replay error {stop_error:.6g} m")
        else:
            passed += 1
    return TaskValidationReport(
        checked_count=count,
        passed_count=passed,
        max_stop_replay_error=max_stop_error,
        failures=tuple(failures),
    )
