"""Low-level tool manipulation interfaces and initial controllers."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from snooker_env.pipeline_types import CueCommand, JointAction, SceneState


class LowLevelPolicy(Protocol):
    """Maps cue-level commands to robot joint actions."""

    def act(self, cue_command: CueCommand, state: SceneState) -> JointAction:
        """Return a joint-level robot command."""


LIFT_ARM_JOINTS: tuple[str, ...] = (
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow",
    "left_wrist_1",
    "left_wrist_2",
    "left_wrist_3",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow",
    "right_wrist_1",
    "right_wrist_2",
    "right_wrist_3",
)


class PassiveToolDesignController(LowLevelPolicy):
    """Placeholder for a mechanical fixture mounted on the robot."""

    def act(self, cue_command: CueCommand, state: SceneState) -> JointAction:
        return JointAction(
            joint_names=(),
            position_targets=np.zeros(0, dtype=np.float64),
            velocity_targets=np.zeros(0, dtype=np.float64),
            torque_targets=np.zeros(0, dtype=np.float64),
        )


class DualArmImpedanceController(LowLevelPolicy):
    """Scaffold for front-hand stabilization and rear-hand stroke force."""

    def __init__(self, joint_names: tuple[str, ...] = LIFT_ARM_JOINTS) -> None:
        self.joint_names = joint_names

    def act(self, cue_command: CueCommand, state: SceneState) -> JointAction:
        n = len(self.joint_names)
        velocity_targets = np.zeros(n, dtype=np.float64)
        torque_targets = np.zeros(n, dtype=np.float64)
        if n:
            stroke_speed = float(np.linalg.norm(cue_command.linear_velocity))
            torque_targets[n // 2 :] = stroke_speed * 0.1
        return JointAction(
            joint_names=self.joint_names,
            position_targets=np.zeros(n, dtype=np.float64),
            velocity_targets=velocity_targets,
            torque_targets=torque_targets,
        )


class ResidualJointPolicy(LowLevelPolicy):
    """RL residual hook on top of a base low-level controller."""

    def __init__(self, base_controller: LowLevelPolicy | None = None) -> None:
        self.base_controller = base_controller or DualArmImpedanceController()

    def act(self, cue_command: CueCommand, state: SceneState) -> JointAction:
        base_action = self.base_controller.act(cue_command, state)
        # The learned residual is intentionally absent here. A future checkpoint
        # should add bounded deltas to these base targets.
        return base_action
