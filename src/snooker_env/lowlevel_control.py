"""MuJoCo-aware controllers and actuator mapping for low-level cue tracking."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from snooker_env.pipeline_low_level import GENTO_ARM_JOINTS, LIFT_ARM_JOINTS
from snooker_env.pipeline_types import CueCommand, JointAction, Pose3D


GRIP_SITE_PAIRS: tuple[tuple[str, str], ...] = (
    ("lift_left_gripper_tcp", "cue_left_grip_site"),
    ("lift_right_gripper_tcp", "cue_right_grip_site"),
)

BRIDGE_STROKE_SITE_PAIRS: tuple[tuple[str, str], ...] = (
    ("lift_left_gripper_tcp", "cue_bridge_site"),
    ("lift_right_gripper_tcp", "cue_right_grip_site"),
)

GENTO_BRIDGE_STROKE_SITE_PAIRS: tuple[tuple[str, str], ...] = (
    # Robot-right fixes the forward cue line; robot-left drives the stroke.
    ("gento_right_gripper_tcp", "cue_left_grip_site"),
    ("gento_left_gripper_tcp", "cue_right_grip_site"),
)


def _named_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Model is missing {object_type.name}: {name}")
    return object_id


def quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a normalized MuJoCo wxyz quaternion to a rotation matrix."""

    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError("Quaternion norm must be non-zero.")
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quat / norm)
    return matrix.reshape(3, 3)


def rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return the world-frame rotation vector taking current to target."""

    relative = np.asarray(target) @ np.asarray(current).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-7:
        return 0.5 * skew
    sine = float(np.sin(angle))
    if abs(sine) < 1e-7:
        # The exact axis is ambiguous at pi. The eigenvector with eigenvalue one
        # is a stable enough choice for a bounded controller correction.
        values, vectors = np.linalg.eig(relative)
        axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return axis * angle
    return skew * (angle / (2.0 * sine))


@dataclass(frozen=True)
class TrackingDiagnostics:
    """Task-space errors from the most recent nominal-controller call."""

    position_error: float
    orientation_error: float


class JointPositionExecutor:
    """Map named arm position targets to MuJoCo position actuators."""

    def __init__(self, model: mujoco.MjModel, joint_names: tuple[str, ...] = LIFT_ARM_JOINTS) -> None:
        self.model = model
        self.joint_names = tuple(joint_names)
        self.actuator_ids = np.asarray(
            [
                _named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
                for name in self.joint_names
            ],
            dtype=int,
        )
        self.control_low = model.actuator_ctrlrange[self.actuator_ids, 0].copy()
        self.control_high = model.actuator_ctrlrange[self.actuator_ids, 1].copy()

    def clip(self, targets: np.ndarray) -> np.ndarray:
        values = np.asarray(targets, dtype=np.float64)
        if values.shape != (len(self.joint_names),):
            raise ValueError(f"Expected {(len(self.joint_names),)} position targets, got {values.shape}.")
        return np.clip(values, self.control_low, self.control_high)

    def apply(self, data: mujoco.MjData, action: JointAction | np.ndarray) -> np.ndarray:
        if isinstance(action, JointAction):
            if action.joint_names != self.joint_names:
                raise ValueError("JointAction names do not match the executor joint order.")
            targets = action.position_targets
        else:
            targets = action
        clipped = self.clip(targets)
        data.ctrl[self.actuator_ids] = clipped
        return clipped


class DualArmDifferentialIKController:
    """Damped least-squares nominal controller for the two cue grip sites."""

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_names: tuple[str, ...] = LIFT_ARM_JOINTS,
        *,
        site_pairs: tuple[tuple[str, str], ...] = GRIP_SITE_PAIRS,
        damping: float = 0.08,
        gain: float = 0.7,
        max_joint_step: float = 0.04,
        orientation_weight: float = 0.20,
    ) -> None:
        self.model = model
        self.joint_names = tuple(joint_names)
        self.damping = float(damping)
        self.gain = float(gain)
        self.max_joint_step = float(max_joint_step)
        self.orientation_weight = float(orientation_weight)
        self.executor = JointPositionExecutor(model, self.joint_names)

        joint_ids = [
            _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]
        self.qpos_ids = np.asarray([int(model.jnt_qposadr[idx]) for idx in joint_ids], dtype=int)
        self.dof_ids = np.asarray([int(model.jnt_dofadr[idx]) for idx in joint_ids], dtype=int)
        self.site_pairs = tuple(site_pairs)
        if len(self.site_pairs) != 2:
            raise ValueError("Dual-arm IK requires exactly two TCP/cue-site pairs.")
        self.tcp_site_ids = np.asarray(
            [_named_id(model, mujoco.mjtObj.mjOBJ_SITE, pair[0]) for pair in self.site_pairs],
            dtype=int,
        )
        self.cue_grip_site_ids = np.asarray(
            [_named_id(model, mujoco.mjtObj.mjOBJ_SITE, pair[1]) for pair in self.site_pairs],
            dtype=int,
        )
        self.cue_body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "cue_body")
        self._tcp_orientation_in_cue: tuple[np.ndarray, ...] | None = None
        self.last_diagnostics = TrackingDiagnostics(0.0, 0.0)

    @property
    def action_size(self) -> int:
        return len(self.joint_names)

    def reset_reference(self, data: mujoco.MjData) -> None:
        """Capture fixed TCP-to-cue orientation offsets in the ready pose."""

        mujoco.mj_forward(self.model, data)
        cue_rotation = data.xmat[self.cue_body_id].reshape(3, 3).copy()
        self._tcp_orientation_in_cue = tuple(
            cue_rotation.T @ data.site_xmat[site_id].reshape(3, 3)
            for site_id in self.tcp_site_ids
        )

    def joint_positions(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qpos_ids].copy()

    def joint_velocities(self, data: mujoco.MjData) -> np.ndarray:
        return data.qvel[self.dof_ids].copy()

    def _task(self, data: mujoco.MjData, pose: Pose3D) -> tuple[np.ndarray, np.ndarray]:
        if self._tcp_orientation_in_cue is None:
            self.reset_reference(data)
        assert self._tcp_orientation_in_cue is not None

        cue_rotation = quat_to_matrix(pose.quat_wxyz)
        errors: list[np.ndarray] = []
        jacobians: list[np.ndarray] = []
        position_norms: list[float] = []
        orientation_norms: list[float] = []
        for pair_index, tcp_site_id in enumerate(self.tcp_site_ids):
            grip_local = self.model.site_pos[self.cue_grip_site_ids[pair_index]]
            target_position = np.asarray(pose.position, dtype=np.float64) + cue_rotation @ grip_local
            current_position = data.site_xpos[tcp_site_id]
            position_delta = target_position - current_position

            target_rotation = cue_rotation @ self._tcp_orientation_in_cue[pair_index]
            current_rotation = data.site_xmat[tcp_site_id].reshape(3, 3)
            orientation_delta = rotation_error(target_rotation, current_rotation)

            jac_position = np.zeros((3, self.model.nv), dtype=np.float64)
            jac_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, data, jac_position, jac_rotation, int(tcp_site_id))
            errors.extend((position_delta, self.orientation_weight * orientation_delta))
            jacobians.extend(
                (
                    jac_position[:, self.dof_ids],
                    self.orientation_weight * jac_rotation[:, self.dof_ids],
                )
            )
            position_norms.append(float(np.linalg.norm(position_delta)))
            orientation_norms.append(float(np.linalg.norm(orientation_delta)))

        self.last_diagnostics = TrackingDiagnostics(
            position_error=max(position_norms),
            orientation_error=max(orientation_norms),
        )
        return np.concatenate(errors), np.vstack(jacobians)

    def _task_twist(self, command: CueCommand) -> np.ndarray:
        """Map the desired cue twist to both world-frame TCP twists."""

        linear = np.asarray(command.linear_velocity, dtype=np.float64)
        angular = np.asarray(command.angular_velocity, dtype=np.float64)
        if linear.shape != (3,) or angular.shape != (3,):
            raise ValueError("Cue linear and angular velocities must both have shape (3,).")
        if not np.all(np.isfinite(linear)) or not np.all(np.isfinite(angular)):
            raise ValueError("Cue linear and angular velocities must be finite.")

        cue_rotation = quat_to_matrix(command.pose.quat_wxyz)
        twists: list[np.ndarray] = []
        for cue_grip_site_id in self.cue_grip_site_ids:
            grip_offset_world = cue_rotation @ self.model.site_pos[cue_grip_site_id]
            grip_linear = linear + np.cross(angular, grip_offset_world)
            twists.extend((grip_linear, self.orientation_weight * angular))
        return np.concatenate(twists)

    def act(
        self,
        command: CueCommand,
        data: mujoco.MjData,
        *,
        control_dt: float = 0.0,
    ) -> JointAction:
        """Return a pose-feedback plus cue-twist-feedforward joint target."""

        if not np.isfinite(control_dt) or control_dt < 0.0:
            raise ValueError("control_dt must be finite and non-negative.")

        mujoco.mj_forward(self.model, data)
        error, jacobian = self._task(data, command.pose)
        desired_task_step = self.gain * error + control_dt * self._task_twist(command)
        regularized = jacobian @ jacobian.T + (self.damping**2) * np.eye(jacobian.shape[0])
        try:
            joint_delta = jacobian.T @ np.linalg.solve(regularized, desired_task_step)
        except np.linalg.LinAlgError:
            joint_delta = np.zeros(self.action_size, dtype=np.float64)
        joint_delta = np.clip(joint_delta, -self.max_joint_step, self.max_joint_step)
        targets = self.executor.clip(self.joint_positions(data) + joint_delta)
        zeros = np.zeros(self.action_size, dtype=np.float64)
        velocity_targets = joint_delta / control_dt if control_dt > 0.0 else zeros.copy()
        return JointAction(
            joint_names=self.joint_names,
            position_targets=targets,
            velocity_targets=velocity_targets,
            torque_targets=zeros.copy(),
        )


class BridgeStrokeDifferentialIKController(DualArmDifferentialIKController):
    """Keep the forward guide fixed while only the rear hand strokes."""

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_names: tuple[str, ...] = LIFT_ARM_JOINTS,
        *,
        site_pairs: tuple[tuple[str, str], ...] = BRIDGE_STROKE_SITE_PAIRS,
        **kwargs: float,
    ) -> None:
        if len(site_pairs) != 2:
            raise ValueError("Bridge-stroke IK requires support and push site pairs.")
        super().__init__(model, joint_names, site_pairs=site_pairs, **kwargs)
        self.bridge_tcp_site_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_SITE, site_pairs[0][0]
        )
        self.push_tcp_site_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_SITE, site_pairs[1][0]
        )
        self.push_cue_site_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_SITE, site_pairs[1][1]
        )
        self._bridge_anchor_position: np.ndarray | None = None
        self._bridge_anchor_rotation: np.ndarray | None = None
        self._push_orientation_in_cue: np.ndarray | None = None

    @property
    def bridge_anchor_position(self) -> np.ndarray:
        if self._bridge_anchor_position is None:
            raise RuntimeError("Call reset_reference() before reading the bridge anchor.")
        return self._bridge_anchor_position.copy()

    def reset_reference(self, data: mujoco.MjData) -> None:
        """Capture the stationary support pose and rear-hand cue orientation."""

        mujoco.mj_forward(self.model, data)
        cue_rotation = data.xmat[self.cue_body_id].reshape(3, 3).copy()
        self._bridge_anchor_position = data.site_xpos[self.bridge_tcp_site_id].copy()
        self._bridge_anchor_rotation = (
            data.site_xmat[self.bridge_tcp_site_id].reshape(3, 3).copy()
        )
        self._push_orientation_in_cue = (
            cue_rotation.T @ data.site_xmat[self.push_tcp_site_id].reshape(3, 3)
        )

    def _task(self, data: mujoco.MjData, pose: Pose3D) -> tuple[np.ndarray, np.ndarray]:
        if (
            self._bridge_anchor_position is None
            or self._bridge_anchor_rotation is None
            or self._push_orientation_in_cue is None
        ):
            self.reset_reference(data)
        assert self._bridge_anchor_position is not None
        assert self._bridge_anchor_rotation is not None
        assert self._push_orientation_in_cue is not None

        cue_rotation = quat_to_matrix(pose.quat_wxyz)
        push_offset = cue_rotation @ self.model.site_pos[self.push_cue_site_id]
        targets = (
            (
                self.bridge_tcp_site_id,
                self._bridge_anchor_position,
                self._bridge_anchor_rotation,
            ),
            (
                self.push_tcp_site_id,
                np.asarray(pose.position, dtype=np.float64) + push_offset,
                cue_rotation @ self._push_orientation_in_cue,
            ),
        )

        errors: list[np.ndarray] = []
        jacobians: list[np.ndarray] = []
        position_norms: list[float] = []
        orientation_norms: list[float] = []
        for tcp_site_id, target_position, target_rotation in targets:
            position_delta = target_position - data.site_xpos[tcp_site_id]
            current_rotation = data.site_xmat[tcp_site_id].reshape(3, 3)
            orientation_delta = rotation_error(target_rotation, current_rotation)
            jac_position = np.zeros((3, self.model.nv), dtype=np.float64)
            jac_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(
                self.model,
                data,
                jac_position,
                jac_rotation,
                int(tcp_site_id),
            )
            errors.extend((position_delta, self.orientation_weight * orientation_delta))
            jacobians.extend(
                (
                    jac_position[:, self.dof_ids],
                    self.orientation_weight * jac_rotation[:, self.dof_ids],
                )
            )
            position_norms.append(float(np.linalg.norm(position_delta)))
            orientation_norms.append(float(np.linalg.norm(orientation_delta)))

        self.last_diagnostics = TrackingDiagnostics(
            position_error=max(position_norms),
            orientation_error=max(orientation_norms),
        )
        return np.concatenate(errors), np.vstack(jacobians)

    def _task_twist(self, command: CueCommand) -> np.ndarray:
        """Apply cue feed-forward twist to the rear hand only."""

        linear = np.asarray(command.linear_velocity, dtype=np.float64)
        angular = np.asarray(command.angular_velocity, dtype=np.float64)
        if linear.shape != (3,) or angular.shape != (3,):
            raise ValueError("Cue linear and angular velocities must both have shape (3,).")
        if not np.all(np.isfinite(linear)) or not np.all(np.isfinite(angular)):
            raise ValueError("Cue linear and angular velocities must be finite.")
        cue_rotation = quat_to_matrix(command.pose.quat_wxyz)
        push_offset = cue_rotation @ self.model.site_pos[self.push_cue_site_id]
        push_linear = linear + np.cross(angular, push_offset)
        return np.concatenate(
            (
                np.zeros(6, dtype=np.float64),
                push_linear,
                self.orientation_weight * angular,
            )
        )

    def bridge_position_error(self, data: mujoco.MjData) -> float:
        return float(
            np.linalg.norm(
                self.bridge_anchor_position - data.site_xpos[self.bridge_tcp_site_id]
            )
        )

    def push_grip_error(self, data: mujoco.MjData) -> float:
        return float(
            np.linalg.norm(
                data.site_xpos[self.push_tcp_site_id]
                - data.site_xpos[self.push_cue_site_id]
            )
        )


class GentoBridgeStrokeDifferentialIKController(
    BridgeStrokeDifferentialIKController
):
    """Gento controller with robot-right support and robot-left speed roles."""

    def __init__(self, model: mujoco.MjModel, **kwargs: float) -> None:
        super().__init__(
            model,
            GENTO_ARM_JOINTS,
            site_pairs=GENTO_BRIDGE_STROKE_SITE_PAIRS,
            **kwargs,
        )
