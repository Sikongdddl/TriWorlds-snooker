"""Single-step contextual PPO environment for feasible two-ball shots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL
from snooker_env.midlevel_tasks import (
    CPU_PHYSICS_BACKEND,
    TwoBallTask,
    TwoBallTaskDataset,
)
from snooker_env.midlevel_two_ball import (
    MAX_CUE_SPEED,
    MAX_SHOT_TIME,
    MIN_CUE_SPEED,
    STOP_HOLD_TIME,
    STOP_SPEED_THRESHOLD,
    TwoBallShotResult,
    TwoBallShotSimulator,
    decode_action,
    encode_speed_action,
)


OBSERVATION_X_SCALE = 0.75
OBSERVATION_Y_SCALE = 1.40


@dataclass(frozen=True)
class RewardBreakdown:
    """Named terminal reward components for logging and tests."""

    total: float
    correct_pot: float
    legal_first_contact: float
    object_progress: float
    position_reward: float
    cue_scratch: float
    wrong_pocket: float
    no_ball_contact: float
    cushion_before_object: float
    timeout: float
    numerical_failure: float
    speed_penalty: float
    stop_error: float
    joint_success: bool

    def as_info(self) -> dict[str, float | bool]:
        return {
            "reward_total": self.total,
            "reward_correct_pot": self.correct_pot,
            "reward_legal_first_contact": self.legal_first_contact,
            "reward_object_progress": self.object_progress,
            "reward_position": self.position_reward,
            "penalty_cue_scratch": self.cue_scratch,
            "penalty_wrong_pocket": self.wrong_pocket,
            "penalty_no_ball_contact": self.no_ball_contact,
            "penalty_cushion_before_object": self.cushion_before_object,
            "penalty_timeout": self.timeout,
            "penalty_numerical_failure": self.numerical_failure,
            "penalty_speed": self.speed_penalty,
            "stop_error": self.stop_error,
            "joint_success": self.joint_success,
        }


def shot_observation(
    cue_position: np.ndarray,
    object_position: np.ndarray,
    pocket_position: np.ndarray,
    target_stop_position: np.ndarray,
) -> np.ndarray:
    """Build the normalized 8-D policy observation from world coordinates."""

    raw = np.concatenate(
        (
            np.asarray(cue_position, dtype=np.float64)[:2],
            np.asarray(object_position, dtype=np.float64)[:2],
            np.asarray(pocket_position, dtype=np.float64)[:2],
            np.asarray(target_stop_position, dtype=np.float64)[:2],
        )
    ).astype(np.float64)
    scales = np.array(
        [
            OBSERVATION_X_SCALE,
            OBSERVATION_Y_SCALE,
            OBSERVATION_X_SCALE,
            OBSERVATION_Y_SCALE,
            OBSERVATION_X_SCALE,
            OBSERVATION_Y_SCALE,
            OBSERVATION_X_SCALE,
            OBSERVATION_Y_SCALE,
        ],
        dtype=np.float64,
    )
    normalized = raw / scales
    if np.any(np.abs(normalized) > 1.0 + 1e-8):
        raise ValueError(f"Task coordinates are outside observation bounds: {raw}.")
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def task_observation(task: TwoBallTask) -> np.ndarray:
    """Build the normalized 8-D policy observation for a stored task."""

    return shot_observation(
        task.cue_position,
        task.object_position,
        task.pocket_position,
        task.target_stop_position,
    )


def compute_terminal_reward(
    result: TwoBallShotResult,
    target_stop_position: np.ndarray,
) -> RewardBreakdown:
    """Compute the terminal reward with position shaping gated by a legal pot."""

    initial_distance = max(float(result.initial_object_pocket_distance), 1e-12)
    progress_fraction = np.clip(
        (initial_distance - float(result.min_object_pocket_distance)) / initial_distance,
        0.0,
        1.0,
    )
    stop_error = float(
        np.linalg.norm(result.cue_ball_final_position[:2] - np.asarray(target_stop_position)[:2])
    )
    position_reward = 0.0
    if result.correct_pot and not result.cue_scratch and result.stopped:
        position_reward = 6.0 * float(np.exp(-((stop_error / 0.10) ** 2)))
        if stop_error <= 0.05:
            position_reward += 4.0

    correct_pot = 10.0 if result.correct_pot else 0.0
    legal_contact = 1.0 if result.legal_first_contact else 0.0
    object_progress = 2.0 * float(progress_fraction)
    cue_scratch = -15.0 if result.cue_scratch else 0.0
    wrong_pocket = -10.0 if result.wrong_pocket else 0.0
    no_contact = -2.0 if not result.legal_first_contact else 0.0
    cushion_before_object = -2.0 if result.cushion_before_object else 0.0
    timeout = -2.0 if result.timed_out else 0.0
    numerical_failure = -20.0 if result.numerical_failure else 0.0
    normalized_speed = np.clip(
        (float(result.cue_speed) - MIN_CUE_SPEED) / (MAX_CUE_SPEED - MIN_CUE_SPEED),
        0.0,
        1.0,
    )
    speed_penalty = -0.05 * float(normalized_speed**2)
    total = (
        correct_pot
        + legal_contact
        + object_progress
        + position_reward
        + cue_scratch
        + wrong_pocket
        + no_contact
        + cushion_before_object
        + timeout
        + numerical_failure
        + speed_penalty
    )
    return RewardBreakdown(
        total=float(total),
        correct_pot=correct_pot,
        legal_first_contact=legal_contact,
        object_progress=object_progress,
        position_reward=position_reward,
        cue_scratch=cue_scratch,
        wrong_pocket=wrong_pocket,
        no_ball_contact=no_contact,
        cushion_before_object=cushion_before_object,
        timeout=timeout,
        numerical_failure=numerical_failure,
        speed_penalty=speed_penalty,
        stop_error=stop_error,
        joint_success=bool(
            result.correct_pot
            and not result.cue_scratch
            and result.stopped
            and stop_error <= 0.05
        ),
    )


class MidLevelTwoBallPPOEnv(gym.Env[np.ndarray, np.ndarray]):
    """One shot per episode over a fixed feasible task library."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        task_dataset: TwoBallTaskDataset | Path,
        model_path: Path = DEFAULT_MIDLEVEL_MODEL,
        *,
        max_time: float = MAX_SHOT_TIME,
        stop_speed: float = STOP_SPEED_THRESHOLD,
        stop_hold_time: float = STOP_HOLD_TIME,
    ) -> None:
        super().__init__()
        self.simulator = TwoBallShotSimulator(
            model_path,
            max_time=max_time,
            stop_speed=stop_speed,
            stop_hold_time=stop_hold_time,
        )
        if isinstance(task_dataset, TwoBallTaskDataset):
            self.tasks = task_dataset
            if (
                self.tasks.xml_hash != self.simulator.xml_hash
                or self.tasks.model_hash != self.simulator.model_hash
                or self.tasks.physics_backend != CPU_PHYSICS_BACKEND
                or self.tasks.backend_hash != self.simulator.model_hash
                or self.tasks.execution_max_time != self.simulator.max_time
                or self.tasks.stop_speed != self.simulator.stop_speed
                or self.tasks.stop_hold_time != self.simulator.stop_hold_time
            ):
                raise ValueError("In-memory task dataset does not match active physics.")
        else:
            self.tasks = TwoBallTaskDataset.load(
                task_dataset,
                simulator=self.simulator,
                validate_model=True,
                expected_backend=CPU_PHYSICS_BACKEND,
                backend_hash=self.simulator.model_hash,
            )
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._task_index: int | None = None
        self._task: TwoBallTask | None = None
        self._awaiting_reset = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options is not None and "task_index" in options:
            task_index = int(options["task_index"])
            if not 0 <= task_index < len(self.tasks):
                raise IndexError(f"task_index {task_index} is outside the dataset.")
        else:
            task_index = int(self.np_random.integers(0, len(self.tasks)))
        self._task_index = task_index
        self._task = self.tasks[task_index]
        self._awaiting_reset = False
        observation = task_observation(self._task)
        return observation, {
            "task_index": task_index,
            "pocket_name": self._task.pocket_name,
            "candidate_seed": self._task.candidate_seed,
            "generated_speed": self._task.generated_speed,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._awaiting_reset or self._task is None or self._task_index is None:
            raise RuntimeError("reset() must be called before each single-step episode.")
        clipped_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        direction, speed = decode_action(
            clipped_action,
            self._task.cue_position,
            self._task.object_position,
            self._task.pocket_position,
        )
        result = self.simulator.execute(
            self._task.cue_position,
            self._task.object_position,
            self._task.pocket_name,
            direction,
            speed,
        )
        reward = compute_terminal_reward(result, self._task.target_stop_position)
        observation = task_observation(self._task)
        # A shot, including one that reaches the simulation deadline, is the
        # complete one-step contextual-bandit outcome.  Returning truncated
        # would make SB3 bootstrap gamma * V(terminal_observation) and change
        # the reward target away from the terminal shot reward.
        terminated = True
        truncated = False
        self._awaiting_reset = True
        info: dict[str, Any] = {
            "task_index": self._task_index,
            "pocket_name": self._task.pocket_name,
            "action": clipped_action.astype(np.float32),
            "shot_direction": direction.copy(),
            "cue_speed": speed,
            "correct_pot": result.correct_pot,
            "legal_first_contact": result.legal_first_contact,
            "cue_scratch": result.cue_scratch,
            "wrong_pocket": result.wrong_pocket,
            "cushion_before_object": result.cushion_before_object,
            "timed_out": result.timed_out,
            "numerical_failure": result.numerical_failure,
            "stopped": result.stopped,
            "object_pocket": result.object_pocket,
            "cue_pocket": result.cue_pocket,
            "elapsed_time": result.elapsed_time,
            **reward.as_info(),
        }
        return observation, reward.total, terminated, truncated, info

    @staticmethod
    def generated_action(task: TwoBallTask) -> np.ndarray:
        """Return the normalized action used to create a feasible task."""

        return np.array([0.0, encode_speed_action(task.generated_speed)], dtype=np.float32)

    def close(self) -> None:
        self._task = None
