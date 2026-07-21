"""Verify that cue linear/angular velocity commands affect joint-level IK."""

from __future__ import annotations

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.lowlevel_residual_env import LowLevelResidualEnv  # noqa: E402
from snooker_env.pipeline_types import CueCommand, Pose3D  # noqa: E402


def _command(
    pose: Pose3D,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
) -> CueCommand:
    return CueCommand(
        pose=Pose3D(pose.position.copy(), pose.quat_wxyz.copy()),
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        duration=1.0,
        debug_label="velocity_ik_smoke",
    )


def main() -> None:
    env = LowLevelResidualEnv()
    env.reset(seed=0)
    pose = env._cue_pose()
    zeros = np.zeros(3, dtype=np.float64)

    stationary = env.controller.act(
        _command(pose, zeros, zeros),
        env.data,
        control_dt=env.control_dt,
    )
    linear = env.controller.act(
        _command(pose, np.array([0.50, 0.0, 0.0]), zeros),
        env.data,
        control_dt=env.control_dt,
    )
    angular = env.controller.act(
        _command(pose, zeros, np.array([0.0, 0.0, 0.50])),
        env.data,
        control_dt=env.control_dt,
    )

    linear_delta = linear.position_targets - stationary.position_targets
    angular_delta = angular.position_targets - stationary.position_targets
    print(f"linear_joint_target_delta={linear_delta}")
    print(f"angular_joint_target_delta={angular_delta}")
    print(f"linear_delta_norm={np.linalg.norm(linear_delta):.9f}")
    print(f"angular_delta_norm={np.linalg.norm(angular_delta):.9f}")
    print(f"linear_joint_velocity_targets={linear.velocity_targets}")
    print(f"angular_joint_velocity_targets={angular.velocity_targets}")

    if np.linalg.norm(linear_delta) < 1e-5:
        raise RuntimeError("Linear cue velocity did not change the joint IK target.")
    if np.linalg.norm(angular_delta) < 1e-5:
        raise RuntimeError("Angular cue velocity did not change the joint IK target.")
    if not np.all(np.isfinite(linear.velocity_targets)):
        raise RuntimeError("Linear velocity IK returned non-finite joint velocities.")
    if not np.all(np.isfinite(angular.velocity_targets)):
        raise RuntimeError("Angular velocity IK returned non-finite joint velocities.")
    env.close()


if __name__ == "__main__":
    main()
