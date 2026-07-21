"""Shared data contracts for the snooker robot policy pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class SkillId(str, Enum):
    """Mid-level skill identifiers selected by the high-level policy."""

    POT_SHOT = "pot_shot"
    SAFETY_SHOT = "safety_shot"
    POSITION_SHOT = "position_shot"
    BREAK_SHOT = "break_shot"


class RobotMode(str, Enum):
    """Robot mode used by non-RL system behaviors."""

    WALKING = "walking"
    PREPARE_SHOT = "prepare_shot"
    SHOOTING = "shooting"


@dataclass(frozen=True)
class Pose3D:
    """World-frame pose. Quaternion follows MuJoCo's ``wxyz`` convention."""

    position: FloatArray
    quat_wxyz: FloatArray


@dataclass(frozen=True)
class BallState:
    """Ball state in world coordinates."""

    name: str
    position: FloatArray
    linear_velocity: FloatArray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    angular_velocity: FloatArray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


@dataclass(frozen=True)
class TableState:
    """Table geometry useful for high-level planning."""

    length: float = 2.84
    width: float = 1.42
    cushion_height: float = 0.08
    pocket_positions: tuple[FloatArray, ...] = ()


@dataclass(frozen=True)
class VisionState:
    """Optional visual input placeholder for future VLM policies."""

    image_rgb: NDArray[np.uint8] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotState:
    """Policy-facing robot proprioception in model joint order."""

    joint_names: tuple[str, ...]
    joint_positions: FloatArray
    joint_velocities: FloatArray


@dataclass(frozen=True)
class SceneState:
    """Policy-facing state snapshot."""

    time: float
    table: TableState
    balls: Mapping[str, BallState]
    cue_pose: Pose3D | None = None
    cue_velocity: FloatArray | None = None
    robot: RobotState | None = None
    robot_mode: RobotMode = RobotMode.WALKING
    vision: VisionState | None = None


@dataclass(frozen=True)
class ShotIntent:
    """Semantic shot request emitted by high-level strategy."""

    cue_ball_name: str
    object_ball_name: str | None = None
    target_pocket: str | None = None
    target_direction: FloatArray | None = None
    desired_cue_ball_position: FloatArray | None = None
    target_speed: float | None = None
    risk_preference: float = 0.5


@dataclass(frozen=True)
class SkillCommand:
    """High-level command consumed by a mid-level skill."""

    skill_id: SkillId
    intent: ShotIntent
    params: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CueCommand:
    """Mid-level tool command consumed by low-level manipulation."""

    pose: Pose3D
    linear_velocity: FloatArray
    angular_velocity: FloatArray
    duration: float = 0.05
    debug_label: str | None = None


@dataclass(frozen=True)
class JointAction:
    """Low-level joint command."""

    joint_names: tuple[str, ...]
    position_targets: FloatArray
    velocity_targets: FloatArray
    torque_targets: FloatArray


@dataclass(frozen=True)
class MobileBaseCommand:
    """Base command for body positioning outside the RL stack."""

    linear_velocity_xy: FloatArray
    angular_velocity_z: float


@dataclass(frozen=True)
class BodyPositionCommand:
    """Command that moves the robot from walking into shot preparation."""

    base: MobileBaseCommand
    bend_preparation: float
    requested_mode: RobotMode


@dataclass(frozen=True)
class RecoveryCommand:
    """Command that returns the robot from shooting to walking."""

    requested_mode: RobotMode
    settle_time: float = 1.0


@dataclass(frozen=True)
class PipelineOutput:
    """One complete pipeline pass for debugging and smoke tests."""

    skills: tuple[SkillCommand, ...]
    cue_trajectories: tuple[tuple[CueCommand, ...], ...]
    cue_commands: tuple[CueCommand, ...]
    joint_actions: tuple[JointAction, ...]
    body_position: BodyPositionCommand | None = None
    recovery: RecoveryCommand | None = None
