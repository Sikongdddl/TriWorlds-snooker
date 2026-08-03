"""Low-level tool manipulation interfaces and initial controllers."""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from snooker_env.pipeline_types import CueCommand, JointAction, SceneState


class LowLevelPolicy(Protocol):
    """Maps cue-level commands to robot joint actions."""

    def act(self, cue_command: CueCommand, state: SceneState) -> JointAction:
        """Return a joint-level robot command."""


LIFT_ARM_JOINTS: tuple[str, ...] = (
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
)


GENTO_ARM_JOINTS: tuple[str, ...] = tuple(
    f"gento_arm_{side}_joint_{index}"
    for side in ("right", "left")
    for index in range(1, 8)
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

    def __init__(
        self,
        base_controller: LowLevelPolicy | None = None,
        residual_provider: Callable[[CueCommand, SceneState, JointAction], np.ndarray] | None = None,
        residual_scale: float = 0.035,
    ) -> None:
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive.")
        self.base_controller = base_controller or DualArmImpedanceController()
        self.residual_provider = residual_provider
        self.residual_scale = float(residual_scale)

    def act(self, cue_command: CueCommand, state: SceneState) -> JointAction:
        base_action = self.base_controller.act(cue_command, state)
        if self.residual_provider is None or not base_action.joint_names:
            return base_action
        normalized_residual = np.asarray(
            self.residual_provider(cue_command, state, base_action),
            dtype=np.float64,
        )
        if normalized_residual.shape != base_action.position_targets.shape:
            raise ValueError(
                "Residual provider output must match the base position-target shape "
                f"{base_action.position_targets.shape}, got {normalized_residual.shape}."
            )
        position_targets = base_action.position_targets + self.residual_scale * np.clip(
            normalized_residual,
            -1.0,
            1.0,
        )
        return JointAction(
            joint_names=base_action.joint_names,
            position_targets=position_targets,
            velocity_targets=base_action.velocity_targets.copy(),
            torque_targets=base_action.torque_targets.copy(),
        )
