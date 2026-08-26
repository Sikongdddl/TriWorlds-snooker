#!/usr/bin/env python3
"""Check the immediate independent-replay acceptance contract."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

import snooker_env.midlevel_mujoco_warp_tasks as generation  # noqa: E402
from snooker_env.midlevel_difficulty import difficulty_cell  # noqa: E402
from snooker_env.midlevel_mujoco_warp_tasks import (  # noqa: E402
    _canonical_replay_task,
    _fixed_layout_indices,
)
from snooker_env.midlevel_mujoco_warp_vec_env import _hash_python_tree  # noqa: E402
from snooker_env.midlevel_tasks import EVENT_FLAG_NAMES, TwoBallTask  # noqa: E402
from snooker_env.midlevel_tasks import TwoBallTaskDataset  # noqa: E402
from snooker_env.midlevel_two_ball import (  # noqa: E402
    POCKET_NAMES,
    POCKET_POSITIONS,
)


def _candidate() -> TwoBallTask:
    return TwoBallTask(
        cue_position=np.array([0.05, 0.25], dtype=np.float64),
        object_position=np.array([0.45, 0.70], dtype=np.float64),
        pocket_name="pocket_corner_posx_posy",
        pocket_position=np.array([0.675, 1.31], dtype=np.float64),
        target_stop_position=np.array([0.1, -0.4], dtype=np.float64),
        generated_direction=np.array([0.0, 1.0], dtype=np.float64),
        generated_speed=1.0,
        candidate_seed=17,
        elapsed_time=0.0,
        min_object_pocket_distance=1.0,
        event_metrics={name: False for name in EVENT_FLAG_NAMES},
    )


def _info(stop: tuple[float, float], *, feasible: bool = True) -> dict[str, object]:
    return {
        "correct_pot": feasible,
        "legal_first_contact": feasible,
        "cue_scratch": False,
        "cushion_before_object": False,
        "object_cushion_before_pocket": False,
        "stopped": feasible,
        "timed_out": False,
        "numerical_failure": False,
        "cue_ball_final_position": np.array([*stop, 1.0785]),
        "shot_direction": np.array([0.0, 1.0]),
        "cue_speed": 1.0,
        "elapsed_time": 2.0,
        "minimum_object_pocket_distance": 0.0,
    }


def _base_stop(seed: int) -> tuple[float, float]:
    return ((seed % 2_000) / 10_000.0 - 0.1, 0.05)


def _check_generator_orchestration() -> None:
    """Exercise compaction, replenishment, and canonical recording without CUDA."""

    class FakeSimulator:
        def __init__(self, model_path, max_time=8.0):
            self.model_path = model_path
            self.max_time = max_time
            self.stop_speed = 0.01
            self.stop_hold_time = 0.2
            self.xml_hash = "1" * 64
            self.model_hash = "2" * 64
            self.pocket_positions = {
                name: np.asarray(POCKET_POSITIONS[name][:2], dtype=np.float64)
                for name in POCKET_NAMES
            }

    class FakeEnvironment:
        created_roles: list[str] = []
        unstable_replays = 0
        fixed_bad_seed: int | None = None

        def __init__(self, dataset, _model_path, *, num_envs, max_time, **_kwargs):
            if max_time < 8.0:
                self.role = "prefilter"
            elif "primary" not in self.created_roles:
                self.role = "primary"
            elif "replay" not in self.created_roles:
                self.role = "replay"
            elif "fixed_primary" not in self.created_roles:
                self.role = "fixed_primary"
            else:
                self.role = "fixed_replay"
            self.created_roles.append(self.role)
            self.tasks = dataset
            self.num_envs = num_envs
            self.indices = list(range(num_envs))

        def replace_task_dataset(self, dataset) -> None:
            self.tasks = dataset

        def set_options(self, options) -> None:
            self.indices = [int(option["task_index"]) for option in options]

        def reset(self) -> None:
            return None

        def step(self, _actions):
            infos = []
            for slot, index in enumerate(self.indices):
                task = self.tasks[index]
                stop = _base_stop(task.candidate_seed)
                if self.role == "replay":
                    delta = 0.01 if task.candidate_seed % 5 == 0 else 0.001
                    if delta > 0.005:
                        type(self).unstable_replays += 1
                    stop = (stop[0] + delta, stop[1])
                elif self.role.startswith("fixed_"):
                    if type(self).fixed_bad_seed is None and slot == 0:
                        type(self).fixed_bad_seed = task.candidate_seed
                    stop = (stop[0] + 0.002 * slot, stop[1])
                feasible = not (
                    self.role.startswith("fixed_")
                    and task.candidate_seed == type(self).fixed_bad_seed
                )
                info = _info(stop, feasible=feasible)
                info["candidate_seed"] = task.candidate_seed
                info["pocket_name"] = task.pocket_name
                infos.append(info)
            return None, None, None, infos

        def close(self) -> None:
            return None

    def fake_candidate(
        pocket_name,
        difficulty_cell_index,
        candidate_seed,
        simulator,
    ):
        cell = difficulty_cell(difficulty_cell_index)
        cue_distance = float(np.mean(cell.cue_object_range_m))
        object_distance = float(np.mean(cell.object_pocket_range_m))
        pocket_position = simulator.pocket_positions[pocket_name]
        object_position = pocket_position - np.array(
            [object_distance, 0.0],
            dtype=np.float64,
        )
        cue_position = object_position - np.array(
            [cue_distance, 0.0],
            dtype=np.float64,
        )
        return TwoBallTask(
            cue_position=cue_position,
            object_position=object_position,
            pocket_name=pocket_name,
            pocket_position=pocket_position.copy(),
            target_stop_position=cue_position.copy(),
            generated_direction=np.array([0.0, 1.0], dtype=np.float64),
            generated_speed=1.0,
            candidate_seed=candidate_seed,
            elapsed_time=0.0,
            min_object_pocket_distance=1.0,
            event_metrics={name: False for name in EVENT_FLAG_NAMES},
        )

    originals = (
        generation.TwoBallShotSimulator,
        generation.MJWarpMidLevelVecEnv,
        generation.active_mujoco_warp_backend_sha256,
        generation._candidate_task,
    )
    try:
        generation.TwoBallShotSimulator = FakeSimulator
        generation.MJWarpMidLevelVecEnv = FakeEnvironment
        generation.active_mujoco_warp_backend_sha256 = lambda _path: (
            "1" * 64,
            "2" * 64,
            "3" * 64,
        )
        generation._candidate_task = fake_candidate
        dataset = generation.generate_mujoco_warp_task_dataset(
            12,
            seed=23,
            model_path="fake.xml",
            num_worlds=6,
            prefilter_time=1.0,
            stability_stop_tolerance=5e-3,
            canonical_reserve_per_pocket=1,
            canonical_max_rounds=3,
            max_attempts_per_task=100,
        )
    finally:
        (
            generation.TwoBallShotSimulator,
            generation.MJWarpMidLevelVecEnv,
            generation.active_mujoco_warp_backend_sha256,
            generation._candidate_task,
        ) = originals

    if FakeEnvironment.created_roles != [
        "prefilter",
        "primary",
        "replay",
        "fixed_primary",
        "fixed_replay",
    ]:
        raise RuntimeError(
            "Generation did not construct distinct prefilter, primary, and "
            f"replay environments: {FakeEnvironment.created_roles}"
        )
    if FakeEnvironment.unstable_replays == 0:
        raise RuntimeError("Synthetic unstable candidates were not exercised.")
    if len(dataset) != 12:
        raise RuntimeError("Rejected candidates were not replenished to target count.")
    if FakeEnvironment.fixed_bad_seed is None:
        raise RuntimeError("Synthetic fixed-layout failure was not exercised.")
    if FakeEnvironment.fixed_bad_seed in set(map(int, dataset.candidate_seeds)):
        raise RuntimeError("A failed fixed-layout task was not replaced in place.")
    for index in range(len(dataset)):
        task = dataset[index]
        expected = np.asarray(_base_stop(task.candidate_seed))
        expected[0] += 0.002 * (index % 6)
        np.testing.assert_allclose(
            task.target_stop_position,
            expected,
            atol=1e-12,
            rtol=0.0,
        )


def _check_backend_hash_scope() -> None:
    with tempfile.TemporaryDirectory(prefix="mjwarp-hash-scope-") as directory:
        root = Path(directory)
        runtime = root / "runtime.py"
        test = root / "runtime_test.py"
        fixture = root / "test_data.py"
        runtime.write_text("VALUE = 1\n")
        test.write_text("VALUE = 1\n")
        fixture.write_text("VALUE = 1\n")

        def digest() -> str:
            value = hashlib.sha256()
            _hash_python_tree(value, root, "scope")
            return value.hexdigest()

        baseline = digest()
        test.write_text("VALUE = 2\n")
        fixture.write_text("VALUE = 2\n")
        if digest() != baseline:
            raise RuntimeError("Test-only files changed the physics backend hash.")
        runtime.write_text("VALUE = 2\n")
        if digest() == baseline:
            raise RuntimeError("Runtime physics code did not change the backend hash.")


def _check_backend_recanonicalization_guard() -> None:
    class FakeSimulator:
        xml_hash = "1" * 64
        model_hash = "2" * 64
        max_time = 8.0
        stop_speed = 0.01
        stop_hold_time = 0.2

    simulator = FakeSimulator()
    old_backend_hash = "3" * 64
    active_backend_hash = "4" * 64
    dataset = TwoBallTaskDataset.from_tasks(
        [_candidate()],
        simulator,
        generation_seed=7,
        physics_backend=generation.MUJOCO_WARP_PHYSICS_BACKEND,
        backend_hash=old_backend_hash,
    )
    try:
        generation._validate_mujoco_warp_dataset_compatibility(
            dataset,
            simulator,
            active_backend_hash,
        )
    except ValueError as error:
        if "backend hash" not in str(error):
            raise
    else:
        raise RuntimeError("An old backend passed strict compatibility checking.")

    generation._validate_mujoco_warp_dataset_compatibility(
        dataset,
        simulator,
        active_backend_hash,
        allow_backend_recanonicalization=True,
    )

    cpu_dataset = TwoBallTaskDataset.from_tasks(
        [_candidate()],
        simulator,
        generation_seed=7,
        backend_hash=old_backend_hash,
    )
    try:
        generation._validate_mujoco_warp_dataset_compatibility(
            cpu_dataset,
            simulator,
            active_backend_hash,
            allow_backend_recanonicalization=True,
        )
    except ValueError as error:
        if "MJWarp-generated" not in str(error):
            raise
    else:
        raise RuntimeError("Backend migration accepted a CPU task library.")


def _check_batch_range_replay() -> None:
    """Ensure multi-GPU ranges reproduce the final padded batch layout."""

    class FakeSimulator:
        xml_hash = "1" * 64
        model_hash = "2" * 64
        max_time = 8.0
        stop_speed = 0.01
        stop_hold_time = 0.2

    class FakeEnvironment:
        last_indices: list[int] = []

        def __init__(self, dataset, _model_path, *, num_envs, **_kwargs):
            self.dataset = dataset
            self.num_envs = num_envs
            self.indices = list(range(num_envs))

        def set_options(self, options) -> None:
            self.indices = [int(option["task_index"]) for option in options]
            type(self).last_indices = self.indices

        def reset(self) -> None:
            return None

        def step(self, _actions):
            infos = []
            for index in self.indices:
                task = self.dataset[index]
                info = _info(tuple(task.target_stop_position))
                info["candidate_seed"] = task.candidate_seed
                info["pocket_name"] = task.pocket_name
                infos.append(info)
            return None, None, None, infos

        def close(self) -> None:
            return None

    tasks = [replace(_candidate(), candidate_seed=100 + index) for index in range(10)]
    dataset = TwoBallTaskDataset.from_tasks(
        tasks,
        FakeSimulator(),
        generation_seed=3,
        physics_backend=generation.MUJOCO_WARP_PHYSICS_BACKEND,
        backend_hash="3" * 64,
    )
    original_environment = generation.MJWarpMidLevelVecEnv
    try:
        generation.MJWarpMidLevelVecEnv = FakeEnvironment
        report = generation.validate_mujoco_warp_task_dataset(
            dataset,
            model_path="fake.xml",
            start_task=6,
            max_tasks=4,
            num_worlds=6,
        )
        if report.passed_count != 4 or report.checked_count != 4:
            raise RuntimeError("Batch-range replay did not validate its full range.")
        if FakeEnvironment.last_indices != [6, 7, 8, 9, 9, 9]:
            raise RuntimeError("Batch-range replay changed final padding semantics.")
        try:
            generation.validate_mujoco_warp_task_dataset(
                dataset,
                model_path="fake.xml",
                start_task=1,
                max_tasks=2,
                num_worlds=6,
            )
        except ValueError as error:
            if "align" not in str(error):
                raise
        else:
            raise RuntimeError("A non-aligned multi-GPU replay range was accepted.")
    finally:
        generation.MJWarpMidLevelVecEnv = original_environment


def main() -> None:
    if _fixed_layout_indices(6, 10, 6) != [6, 7, 8, 9, 9, 9]:
        raise RuntimeError("Final partial-batch padding changed unexpectedly.")
    candidate = _candidate()
    first = _info((0.2, -0.1))
    canonical = _info((0.203, -0.1))
    accepted = _canonical_replay_task(
        candidate,
        first,
        canonical,
        stop_tolerance=5e-3,
    )
    if accepted is None:
        raise RuntimeError("A stable double replay was rejected.")
    np.testing.assert_allclose(
        accepted.target_stop_position,
        canonical["cue_ball_final_position"][:2],
        atol=0.0,
        rtol=0.0,
    )

    unstable = _canonical_replay_task(
        candidate,
        first,
        _info((0.206, -0.1)),
        stop_tolerance=5e-3,
    )
    if unstable is not None:
        raise RuntimeError("A stop-point mismatch was accepted.")

    failed_replay = _canonical_replay_task(
        candidate,
        first,
        _info((0.2, -0.1), feasible=False),
        stop_tolerance=5e-3,
    )
    if failed_replay is not None:
        raise RuntimeError("An infeasible independent replay was accepted.")

    mismatched_action_info = _info((0.2, -0.1))
    mismatched_action_info["shot_direction"] = np.array([0.1, 0.995])
    mismatched_action = _canonical_replay_task(
        candidate,
        first,
        mismatched_action_info,
        stop_tolerance=5e-3,
    )
    if mismatched_action is not None:
        raise RuntimeError("A mismatched replay action was accepted.")

    _check_generator_orchestration()
    _check_backend_hash_scope()
    _check_backend_recanonicalization_guard()
    _check_batch_range_replay()

    print("mid-level MJWarp generation replay filter smoke passed")


if __name__ == "__main__":
    main()
