"""Mid-level semantic shot policies.

Each mid-level policy receives one high-level skill call and returns a
low-level command trajectory. Internal impact-parameter search and trajectory
generation are hidden behind this interface so every mid-level policy has the
same external contract.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from snooker_env.pipeline_types import CueCommand, Pose3D, SceneState, SkillCommand, SkillId


class MidLevelSkill(Protocol):
    """Converts one semantic shot call into a cue-command trajectory."""

    skill_id: SkillId

    def rollout(self, command: SkillCommand, state: SceneState) -> tuple[CueCommand, ...]:
        """Return low-level cue pose/velocity commands."""


def _shot_direction(command: SkillCommand, state: SceneState) -> np.ndarray:
    intent = command.intent
    if intent.target_direction is not None:
        direction = intent.target_direction.astype(np.float64)
        norm = float(np.linalg.norm(direction[:2]))
        if norm < 1e-9:
            raise ValueError("target_direction must be non-zero.")
        return np.array([direction[0] / norm, direction[1] / norm, 0.0], dtype=np.float64)

    if intent.object_ball_name is None:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if intent.cue_ball_name not in state.balls:
        raise KeyError(f"Missing cue ball state: {intent.cue_ball_name}")
    if intent.object_ball_name not in state.balls:
        raise KeyError(f"Missing object ball state: {intent.object_ball_name}")

    cue_ball = state.balls[intent.cue_ball_name]
    object_ball = state.balls[intent.object_ball_name]
    delta = object_ball.position - cue_ball.position
    norm = float(np.linalg.norm(delta[:2]))
    if norm < 1e-9:
        raise ValueError("Cue ball and object ball are too close for shot planning.")
    return np.array([delta[0] / norm, delta[1] / norm, 0.0], dtype=np.float64)


def _impact_point(command: SkillCommand, state: SceneState, direction: np.ndarray) -> np.ndarray:
    cue_ball = state.balls[command.intent.cue_ball_name]
    # The cue approaches from behind the desired cue-ball travel direction, so
    # the tip contacts the rear side of the cue ball.
    return cue_ball.position - direction * float(command.params.get("ball_radius", 0.02625))


def _cue_pose_from_plan(command: SkillCommand, state: SceneState, backoff: float) -> Pose3D:
    direction = _shot_direction(command, state)
    cue_tip_offset = float(command.params.get("cue_tip_offset", 0.725))
    cue_elevation = float(command.params.get("cue_elevation", np.deg2rad(8.0)))
    axis = np.array(
        [
            direction[0] * np.cos(cue_elevation),
            direction[1] * np.cos(cue_elevation),
            -np.sin(cue_elevation),
        ],
        dtype=np.float64,
    )
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    position = _impact_point(command, state, direction) - axis * (cue_tip_offset + backoff)
    position = position.copy()
    position[2] = float(command.params.get("cue_height", position[2]))
    yaw = float(np.arctan2(direction[1], direction[0]))
    half_yaw = 0.5 * yaw
    half_elevation = 0.5 * cue_elevation
    q_yaw = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64)
    q_pitch = np.array([np.cos(half_elevation), 0.0, np.sin(half_elevation), 0.0], dtype=np.float64)
    quat_wxyz = np.array(
        [
            q_yaw[0] * q_pitch[0] - q_yaw[1] * q_pitch[1] - q_yaw[2] * q_pitch[2] - q_yaw[3] * q_pitch[3],
            q_yaw[0] * q_pitch[1] + q_yaw[1] * q_pitch[0] + q_yaw[2] * q_pitch[3] - q_yaw[3] * q_pitch[2],
            q_yaw[0] * q_pitch[2] - q_yaw[1] * q_pitch[3] + q_yaw[2] * q_pitch[0] + q_yaw[3] * q_pitch[1],
            q_yaw[0] * q_pitch[3] + q_yaw[1] * q_pitch[2] - q_yaw[2] * q_pitch[1] + q_yaw[3] * q_pitch[0],
        ],
        dtype=np.float64,
    )
    quat_wxyz /= max(float(np.linalg.norm(quat_wxyz)), 1e-9)
    return Pose3D(position=position, quat_wxyz=quat_wxyz)


class _BaseShotPolicy(MidLevelSkill):
    """Scripted trajectory scaffold for semantic shot policies."""

    skill_id: SkillId
    default_speed: float = 0.8
    setup_backoff: float = 0.35
    align_backoff: float = 0.16
    stroke_backoff: float = 0.08
    follow_through: float = 0.05

    def _command(
        self,
        command: SkillCommand,
        state: SceneState,
        backoff: float,
        speed: float,
    ) -> CueCommand:
        pose = _cue_pose_from_plan(command, state, backoff=backoff)
        direction = _shot_direction(command, state)
        return CueCommand(
            pose=pose,
            linear_velocity=direction * speed,
            angular_velocity=np.zeros(3, dtype=np.float64),
            debug_label=self.skill_id.value,
        )

    def rollout(self, command: SkillCommand, state: SceneState) -> tuple[CueCommand, ...]:
        default_speed = self.default_speed if command.intent.target_speed is None else command.intent.target_speed
        speed = float(command.params.get("target_speed", default_speed))
        return (
            self._command(command, state, backoff=self.setup_backoff, speed=0.0),
            self._command(command, state, backoff=self.align_backoff, speed=0.0),
            self._command(command, state, backoff=self.stroke_backoff, speed=0.0),
            self._command(command, state, backoff=self.stroke_backoff, speed=speed),
            self._command(command, state, backoff=-self.follow_through, speed=0.0),
        )


class PotShotPolicy(_BaseShotPolicy):
    """Offensive potting shot toward a selected ball/pocket/direction."""

    skill_id = SkillId.POT_SHOT


class SafetyShotPolicy(_BaseShotPolicy):
    """Defensive shot that favors low speed and controlled cue-ball placement."""

    skill_id = SkillId.SAFETY_SHOT
    default_speed = 0.45
    follow_through = 0.025


class PositionShotPolicy(_BaseShotPolicy):
    """Potting-style shot with an additional cue-ball position objective."""

    skill_id = SkillId.POSITION_SHOT
    default_speed = 0.7


class BreakShotPolicy(_BaseShotPolicy):
    """High-power opening/break shot scaffold."""

    skill_id = SkillId.BREAK_SHOT
    default_speed = 1.4
    setup_backoff = 0.45
    stroke_backoff = 0.12
    follow_through = 0.10


def default_skill_registry() -> dict[SkillId, MidLevelSkill]:
    """Return the built-in mid-level shot policies."""

    skills: tuple[MidLevelSkill, ...] = (
        PotShotPolicy(),
        SafetyShotPolicy(),
        PositionShotPolicy(),
        BreakShotPolicy(),
    )
    return {skill.skill_id: skill for skill in skills}
