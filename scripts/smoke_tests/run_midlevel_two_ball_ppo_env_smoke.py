"""Exercise task storage, Gymnasium API, seeding, and one exact shot."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from gymnasium.utils.env_checker import check_reset_seed_determinism

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_ppo_env import MidLevelTwoBallPPOEnv  # noqa: E402
from snooker_env.midlevel_tasks import TwoBallTask, TwoBallTaskDataset  # noqa: E402
from snooker_env.midlevel_two_ball import (  # noqa: E402
    TwoBallShotResult,
    TwoBallShotSimulator,
    ghost_ball_direction,
    quantize_cue_speed,
)


def _fixture_task(simulator: TwoBallShotSimulator) -> TwoBallTask:
    cue_position = np.array([-0.2, 0.0], dtype=np.float64)
    object_position = np.array([0.45, 0.0], dtype=np.float64)
    pocket_name = "pocket_middle_posx"
    pocket_position = simulator.pocket_positions[pocket_name]
    return TwoBallTask(
        cue_position=cue_position,
        object_position=object_position,
        pocket_name=pocket_name,
        pocket_position=pocket_position.copy(),
        target_stop_position=np.array(
            [0.6320569632420524, 0.0025953078413919967], dtype=np.float64
        ),
        generated_direction=ghost_ball_direction(
            cue_position, object_position, pocket_position
        ),
        generated_speed=quantize_cue_speed(0.5),
        candidate_seed=1,
        elapsed_time=4.4716000000123595,
        min_object_pocket_distance=0.015298263138390215,
        event_metrics={
            "correct_pot": True,
            "legal_first_contact": True,
            "no_cushion_direct_pot": True,
            "cue_scratch": False,
            "stopped": True,
            "timed_out": False,
            "numerical_failure": False,
        },
    )


def main() -> None:
    simulator = TwoBallShotSimulator()
    dataset = TwoBallTaskDataset.from_tasks([_fixture_task(simulator)], simulator, generation_seed=7)
    with tempfile.TemporaryDirectory(prefix="midlevel-two-ball-smoke-") as directory:
        task_path = Path(directory) / "tasks.npz"
        dataset.save(task_path)
        loaded = TwoBallTaskDataset.load(task_path, simulator=simulator)
        if len(loaded) != 1 or loaded[0].candidate_seed != 1:
            raise RuntimeError("Task dataset did not round-trip losslessly.")
        stale_path = Path(directory) / "stale_tasks.npz"
        loaded.model_hash = "0" * 64
        loaded.save(stale_path)
        try:
            TwoBallTaskDataset.load(stale_path, simulator=simulator)
        except ValueError:
            pass
        else:
            raise RuntimeError("A stale task-library model hash was not rejected.")
        tampered_path = Path(directory) / "tampered_tasks.npz"
        with np.load(task_path, allow_pickle=False) as archive:
            tampered_values = {
                name: archive[name].copy()
                for name in archive.files
            }
        tampered_values["generated_speeds"][0] += 0.01
        np.savez_compressed(tampered_path, **tampered_values)
        try:
            TwoBallTaskDataset.load(tampered_path, simulator=simulator)
        except ValueError as error:
            if "content hash" not in str(error):
                raise
        else:
            raise RuntimeError("Tampered task values were not rejected.")
        try:
            MidLevelTwoBallPPOEnv(task_path, max_time=7.0)
        except ValueError as error:
            if "timing/stopping" not in str(error):
                raise
        else:
            raise RuntimeError("Mismatched shot termination settings were not rejected.")

        env = MidLevelTwoBallPPOEnv(task_path)
        check_reset_seed_determinism(env)
        observation, reset_info = env.reset(seed=123, options={"task_index": 0})
        if observation.shape != (8,) or observation.dtype != np.float32:
            raise RuntimeError("Observation contract is incorrect.")
        if not env.observation_space.contains(observation):
            raise RuntimeError("Normalized observation is outside its Box.")
        action = env.generated_action(env.tasks[0])
        if not env.action_space.contains(action):
            raise RuntimeError("Generated action is outside its Box.")
        _, reward, terminated, truncated, info = env.step(action)
        timeout_result = TwoBallShotResult(
            target_pocket=env.tasks[0].pocket_name,
            shot_direction=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            cue_speed=0.5,
            elapsed_time=8.0,
            cue_ball_final_position=np.array([0.0, 0.0, 1.0785]),
            object_ball_final_position=np.array([0.0, 0.5, 1.0785]),
            first_ball_contact_time=None,
            first_cushion_contact_time=None,
            object_pocket=None,
            cue_pocket=None,
            min_object_pocket_distance=0.2,
            initial_object_pocket_distance=0.5,
            stopped=False,
            timed_out=True,
            numerical_failure=False,
            cushion_before_object=False,
            object_cushion_before_pocket=False,
            any_cushion_contact=False,
            contact_events=(),
        )
        env.reset(options={"task_index": 0})
        env.simulator.execute = lambda *args, **kwargs: timeout_result
        _, _, timeout_terminated, timeout_truncated, _ = env.step(
            np.zeros(2, dtype=np.float32)
        )
        env.close()

    print(
        f"task={reset_info['task_index']} reward={reward:.4f} "
        f"pot={info['correct_pot']} joint={info['joint_success']} "
        f"stop_error={info['stop_error']:.6g}m elapsed={info['elapsed_time']:.4f}s"
    )
    if not terminated or truncated:
        raise RuntimeError("A settled shot must terminate without truncation.")
    if not timeout_terminated or timeout_truncated:
        raise RuntimeError(
            "A time-limited one-step shot must terminate without value bootstrapping."
        )
    if not np.isfinite(reward):
        raise RuntimeError("Reward is not finite.")
    if not info["correct_pot"] or info["cue_scratch"] or not info["stopped"]:
        raise RuntimeError("Generated fixture action did not reproduce its feasible shot.")
    if float(info["stop_error"]) > 2e-4 or not info["joint_success"]:
        raise RuntimeError("Generated stop point did not reproduce within tolerance.")


if __name__ == "__main__":
    main()
