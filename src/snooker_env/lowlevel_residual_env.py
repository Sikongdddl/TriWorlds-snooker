"""Gymnasium environment for residual dual-arm cue tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from snooker_env.contact_events import CollisionEventMonitor
from snooker_env.init_pose import set_lift_grip_ready_pose
from snooker_env.lowlevel_control import (
    GRIP_SITE_PAIRS,
    DualArmDifferentialIKController,
    JointPositionExecutor,
    quat_to_matrix,
    rotation_error,
)
from snooker_env.pipeline_types import CueCommand, Pose3D
from snooker_env.scene import default_model_path, load_model


def _normalize_quaternion(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quat.shape}.")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError("Cue target quaternion must have non-zero norm.")
    return quat / norm


def _slerp(start: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    q0 = _normalize_quaternion(start)
    q1 = _normalize_quaternion(target)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_quaternion(q0 + fraction * (q1 - q0))
    angle = float(np.arccos(dot))
    sine = float(np.sin(angle))
    return (np.sin((1.0 - fraction) * angle) * q0 + np.sin(fraction * angle) * q1) / sine


class LowLevelResidualEnv(gym.Env[np.ndarray, np.ndarray]):
    """Track timed cue commands with a nominal IK controller plus 12-D residuals.

    The policy action is normalized to ``[-1, 1]`` and converted to a bounded
    joint-position residual. The base, lift, head, and grippers remain fixed in
    this first training scaffold.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        command: CueCommand | None = None,
        control_decimation: int = 40,
        residual_scale: float = 0.035,
        randomize_command: bool = False,
        render_mode: str | None = None,
        render_width: int = 960,
        render_height: int = 540,
    ) -> None:
        super().__init__()
        if control_decimation <= 0:
            raise ValueError("control_decimation must be positive.")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive.")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.model_path = Path(model_path) if model_path is not None else default_model_path()
        self.model = load_model(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.controller = DualArmDifferentialIKController(self.model)
        self.executor = JointPositionExecutor(self.model, self.controller.joint_names)
        self.control_decimation = int(control_decimation)
        self.control_dt = float(self.model.opt.timestep * self.control_decimation)
        self.residual_scale = float(residual_scale)
        self.randomize_command = bool(randomize_command)
        self.render_mode = render_mode
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self._renderer: mujoco.Renderer | None = None

        self.cue_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "cue_body")
        cue_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "cue_free")
        self.cue_dof_id = int(self.model.jnt_dofadr[cue_joint_id])
        self.cue_tip_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "cue_tip")
        self.cue_ball_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "cue_ball_geom")
        cue_ball_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "cue_ball_free")
        self.cue_ball_dof_id = int(self.model.jnt_dofadr[cue_ball_joint_id])
        self.contact_monitor = CollisionEventMonitor(self.model)
        self.grip_site_ids = tuple(
            (
                self._id(mujoco.mjtObj.mjOBJ_SITE, robot_site),
                self._id(mujoco.mjtObj.mjOBJ_SITE, cue_site),
            )
            for robot_site, cue_site in GRIP_SITE_PAIRS
        )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.controller.action_size,), dtype=np.float32)
        # command(14) + q/qdot(24) + actual cue state(13) + nominal(12)
        # + previous residual(12) + cue pose error(6) + phase(1) = 82.
        self.observation_space = spaces.Box(-1.0e6, 1.0e6, shape=(82,), dtype=np.float32)

        self._configured_command = command
        self.command: CueCommand | None = None
        self._start_pose: Pose3D | None = None
        self._desired_command: CueCommand | None = None
        self._episode_start_time = 0.0
        self._previous_action = np.zeros(self.controller.action_size, dtype=np.float64)
        self._nominal_targets = np.zeros(self.controller.action_size, dtype=np.float64)
        self._first_cue_ball_contact_time: float | None = None
        self._peak_cue_ball_speed = 0.0
        self.contact_monitor.reset()
        self._has_reset = False

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Model is missing {object_type.name}: {name}")
        return object_id

    @property
    def elapsed_time(self) -> float:
        return max(0.0, float(self.data.time - self._episode_start_time))

    @property
    def phase(self) -> float:
        if self.command is None:
            return 0.0
        return float(np.clip(self.elapsed_time / self.command.duration, 0.0, 1.0))

    def _cue_pose(self) -> Pose3D:
        return Pose3D(
            position=self.data.xpos[self.cue_body_id].copy(),
            quat_wxyz=self.data.xquat[self.cue_body_id].copy(),
        )

    def _cue_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        velocity = self.data.qvel[self.cue_dof_id:self.cue_dof_id + 6].copy()
        return velocity[:3], velocity[3:]

    def _validated_command(self, command: CueCommand) -> CueCommand:
        position = np.asarray(command.pose.position, dtype=np.float64)
        linear_velocity = np.asarray(command.linear_velocity, dtype=np.float64)
        angular_velocity = np.asarray(command.angular_velocity, dtype=np.float64)
        if position.shape != (3,) or linear_velocity.shape != (3,) or angular_velocity.shape != (3,):
            raise ValueError("Cue position, linear velocity, and angular velocity must all have shape (3,).")
        if not np.isfinite(command.duration) or command.duration <= 0.0:
            raise ValueError("Cue command duration must be finite and positive.")
        return CueCommand(
            pose=Pose3D(position=position, quat_wxyz=_normalize_quaternion(command.pose.quat_wxyz)),
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            duration=float(command.duration),
            debug_label=command.debug_label,
        )

    def _default_command(self) -> CueCommand:
        start = self._cue_pose()
        # Conservative command found by the actuator-level contact sweep:
        # 60 mm forward travel, 5 mm high contact, and enough time for the
        # current position actuators to reach the cue ball before truncation.
        displacement = np.array([0.060, 0.0, 0.005], dtype=np.float64)
        linear_velocity = np.array([0.080, 0.0, 0.0], dtype=np.float64)
        duration = 1.50
        if self.randomize_command:
            displacement = np.array(
                [
                    self.np_random.uniform(0.045, 0.070),
                    self.np_random.uniform(-0.015, 0.015),
                    self.np_random.uniform(-0.010, 0.020),
                ],
                dtype=np.float64,
            )
            linear_velocity = np.array(
                [
                    self.np_random.uniform(0.060, 0.120),
                    self.np_random.uniform(-0.020, 0.020),
                    self.np_random.uniform(-0.020, 0.030),
                ],
                dtype=np.float64,
            )
            duration = float(self.np_random.uniform(1.50, 2.00))
        return CueCommand(
            pose=Pose3D(position=start.position + displacement, quat_wxyz=start.quat_wxyz.copy()),
            linear_velocity=linear_velocity,
            angular_velocity=np.zeros(3, dtype=np.float64),
            duration=duration,
            debug_label="residual_training",
        )

    def _trajectory_command(self, phase: float) -> CueCommand:
        assert self.command is not None and self._start_pose is not None
        smooth_fraction = phase * phase * (3.0 - 2.0 * phase)
        position = self._start_pose.position + smooth_fraction * (
            self.command.pose.position - self._start_pose.position
        )
        quat = _slerp(self._start_pose.quat_wxyz, self.command.pose.quat_wxyz, smooth_fraction)
        return CueCommand(
            pose=Pose3D(position=position, quat_wxyz=quat),
            linear_velocity=self.command.linear_velocity.copy(),
            angular_velocity=self.command.angular_velocity.copy(),
            duration=max(self.command.duration - self.elapsed_time, 0.0),
            debug_label=self.command.debug_label,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        set_lift_grip_ready_pose(self.model, self.data)
        self.controller.reset_reference(self.data)
        self._episode_start_time = float(self.data.time)
        self._start_pose = self._cue_pose()

        option_command = None if options is None else options.get("command")
        selected_command = option_command or self._configured_command or self._default_command()
        if not isinstance(selected_command, CueCommand):
            raise TypeError("reset option 'command' must be a CueCommand.")
        self.command = self._validated_command(selected_command)
        self._previous_action.fill(0.0)
        self._nominal_targets = self.controller.joint_positions(self.data)
        self._first_cue_ball_contact_time = None
        self._peak_cue_ball_speed = 0.0
        self.contact_monitor.reset()
        self._desired_command = self._trajectory_command(0.0)
        self._has_reset = True
        observation = self._observation()
        return observation, self._info(success=False)

    def _tracking_errors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self._desired_command is not None
        actual_pose = self._cue_pose()
        actual_linear, actual_angular = self._cue_velocity()
        position_delta = self._desired_command.pose.position - actual_pose.position
        orientation_delta = rotation_error(
            quat_to_matrix(self._desired_command.pose.quat_wxyz),
            quat_to_matrix(actual_pose.quat_wxyz),
        )
        linear_delta = self._desired_command.linear_velocity - actual_linear
        angular_delta = self._desired_command.angular_velocity - actual_angular
        return position_delta, orientation_delta, linear_delta, angular_delta

    def _grip_error(self) -> float:
        return max(
            float(np.linalg.norm(self.data.site_xpos[robot] - self.data.site_xpos[cue]))
            for robot, cue in self.grip_site_ids
        )

    def _reward(self, normalized_action: np.ndarray) -> tuple[float, dict[str, float]]:
        position, orientation, linear, angular = self._tracking_errors()
        position_error = float(np.linalg.norm(position))
        orientation_error = float(np.linalg.norm(orientation))
        linear_error = float(np.linalg.norm(linear))
        angular_error = float(np.linalg.norm(angular))
        grip_error = self._grip_error()
        action_cost = float(np.mean(np.square(normalized_action)))
        smoothness_cost = float(np.mean(np.square(normalized_action - self._previous_action)))

        reward = (
            2.0 * np.exp(-np.square(position_error / 0.025))
            + 1.0 * np.exp(-np.square(orientation_error / 0.15))
            + 0.75 * np.exp(-np.square(linear_error / 0.25))
            + 0.25 * np.exp(-np.square(angular_error / 0.50))
            - 0.02 * action_cost
            - 0.04 * smoothness_cost
            - 2.0 * grip_error
        )
        metrics = {
            "position_error": position_error,
            "orientation_error": orientation_error,
            "linear_velocity_error": linear_error,
            "angular_velocity_error": angular_error,
            "grip_error": grip_error,
            "action_cost": action_cost,
        }
        return float(reward), metrics

    def _observation(self) -> np.ndarray:
        assert self.command is not None and self._desired_command is not None
        actual_pose = self._cue_pose()
        actual_linear, actual_angular = self._cue_velocity()
        position_error, orientation_error, _, _ = self._tracking_errors()
        remaining_time = max(self.command.duration - self.elapsed_time, 0.0)
        observation = np.concatenate(
            (
                self._desired_command.pose.position,
                self._desired_command.pose.quat_wxyz,
                self._desired_command.linear_velocity,
                self._desired_command.angular_velocity,
                np.array([remaining_time]),
                self.controller.joint_positions(self.data),
                self.controller.joint_velocities(self.data),
                actual_pose.position,
                actual_pose.quat_wxyz,
                actual_linear,
                actual_angular,
                self._nominal_targets,
                self._previous_action * self.residual_scale,
                position_error,
                orientation_error,
                np.array([self.phase]),
            )
        ).astype(np.float32)
        if observation.shape != self.observation_space.shape:
            raise RuntimeError(f"Observation shape drifted to {observation.shape}.")
        return observation

    def _info(self, *, success: bool, **metrics: float) -> dict[str, Any]:
        return {
            "success": success,
            "phase": self.phase,
            "elapsed_time": self.elapsed_time,
            "control_dt": self.control_dt,
            "residual_scale": self.residual_scale,
            "first_cue_ball_contact_time": self._first_cue_ball_contact_time,
            "peak_cue_ball_speed": self._peak_cue_ball_speed,
            "contact_event_count": len(self.contact_monitor.events),
            "pocketed_balls": tuple(sorted(self.contact_monitor.pocketed_balls)),
            **metrics,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._has_reset:
            raise RuntimeError("Call reset() before step().")
        normalized_action = np.asarray(action, dtype=np.float64)
        if normalized_action.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {normalized_action.shape}.")
        normalized_action = np.clip(normalized_action, -1.0, 1.0)

        assert self.command is not None
        next_phase = float(np.clip((self.elapsed_time + self.control_dt) / self.command.duration, 0.0, 1.0))
        self._desired_command = self._trajectory_command(next_phase)
        nominal_action = self.controller.act(
            self._desired_command,
            self.data,
            control_dt=self.control_dt,
        )
        self._nominal_targets = nominal_action.position_targets.copy()
        final_targets = self._nominal_targets + self.residual_scale * normalized_action
        self.executor.apply(self.data, final_targets)

        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
            self.contact_monitor.scan(self.data)
            if self._first_cue_ball_contact_time is None:
                for contact_index in range(self.data.ncon):
                    contact = self.data.contact[contact_index]
                    if {contact.geom1, contact.geom2} == {self.cue_tip_geom_id, self.cue_ball_geom_id}:
                        self._first_cue_ball_contact_time = float(self.data.time)
                        break
            if self._first_cue_ball_contact_time is not None:
                self._peak_cue_ball_speed = max(
                    self._peak_cue_ball_speed,
                    float(
                        np.linalg.norm(
                            self.data.qvel[self.cue_ball_dof_id:self.cue_ball_dof_id + 3]
                        )
                    ),
                )

        reward, metrics = self._reward(normalized_action)
        finite = bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))
        failed = (not finite) or metrics["grip_error"] > 0.08 or metrics["position_error"] > 0.35
        truncated = self.elapsed_time + 1e-12 >= self.command.duration
        success = (
            truncated
            and metrics["position_error"] < 0.03
            and metrics["orientation_error"] < 0.20
            and metrics["grip_error"] < 0.04
        )
        self._previous_action = normalized_action.copy()
        observation = self._observation()
        return observation, reward, failed, truncated, self._info(success=success, **metrics)

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.render_width)
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.render_height)
            self._renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
        self._renderer.update_scene(self.data, camera="asset_overview")
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
