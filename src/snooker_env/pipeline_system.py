"""Non-RL system behaviors around the policy stack."""

from __future__ import annotations

import numpy as np

from snooker_env.pipeline_types import (
    BodyPositionCommand,
    MobileBaseCommand,
    Pose3D,
    RecoveryCommand,
    RobotMode,
    SceneState,
)


class GeometricBodyPositioningPlanner:
    """Move the mobile base behind the desired cue line."""

    def plan(
        self,
        cue_pose: Pose3D,
        target_ball_name: str,
        shot_direction: np.ndarray,
        state: SceneState,
    ) -> BodyPositionCommand:
        if target_ball_name not in state.balls:
            raise KeyError(f"Missing target ball state: {target_ball_name}")
        direction = shot_direction / max(float(np.linalg.norm(shot_direction)), 1e-9)
        desired_base_xy = cue_pose.position[:2] - direction[:2] * 0.7
        current_base_xy = np.zeros(2, dtype=np.float64)
        error_xy = desired_base_xy - current_base_xy
        speed_xy = np.clip(error_xy, -0.25, 0.25)
        return BodyPositionCommand(
            base=MobileBaseCommand(linear_velocity_xy=speed_xy, angular_velocity_z=0.0),
            bend_preparation=1.0,
            requested_mode=RobotMode.PREPARE_SHOT,
        )


class DefaultRecoveryPlanner:
    """Return from shooting mode to walking mode."""

    def plan(self, state: SceneState) -> RecoveryCommand:
        return RecoveryCommand(requested_mode=RobotMode.WALKING, settle_time=1.0)
