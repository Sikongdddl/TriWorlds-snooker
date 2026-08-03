"""Gento low-level IK environment with support/speed hand role separation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from snooker_env.init_pose import set_gento_side_grasp_ready_pose
from snooker_env.lowlevel_control import (
    GENTO_BRIDGE_STROKE_SITE_PAIRS,
    GentoBridgeStrokeDifferentialIKController,
)
from snooker_env.lowlevel_residual_env import LowLevelResidualEnv
from snooker_env.pipeline_types import CueCommand, Pose3D
from snooker_env.scene import project_root


def default_gento_ik_model_path() -> Path:
    return project_root() / "models" / "gento_side_grasp_scene.xml"


class GentoRoleIKEnv(LowLevelResidualEnv):
    """Train Gento IK with a fixed front guide and velocity-driven rear hand.

    Robot-right is the forward support hand. Its TCP position and orientation
    are anchored at reset, defining the cue line. Robot-left is the rear hand;
    only it receives the axial feed-forward velocity from the cue command.
    The PPO action is a 14-D residual on top of this role-aware differential
    IK controller.
    """

    SUPPORT_PAD_GEOMS = (
        "gento_right_upper_finger_pad",
        "gento_right_lower_finger_pad",
    )
    SPEED_PAD_GEOMS = (
        "gento_left_upper_finger_pad",
        "gento_left_lower_finger_pad",
    )
    PALM_GUARD_GEOMS = (
        "gento_right_palm_guard",
        "gento_left_palm_guard",
    )

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        command: CueCommand | None = None,
        control_decimation: int = 200,
        residual_scale: float = 0.006,
        randomize_command: bool = False,
        render_mode: str | None = None,
        render_width: int = 960,
        render_height: int = 640,
    ) -> None:
        self._support_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        self._lost_support_contact_steps = 0
        self._lost_speed_contact_steps = 0
        self._rear_speed_error_sum = 0.0
        self._rear_speed_sample_count = 0
        super().__init__(
            model_path=default_gento_ik_model_path() if model_path is None else model_path,
            command=command,
            control_decimation=control_decimation,
            residual_scale=residual_scale,
            randomize_command=randomize_command,
            render_mode=render_mode,
            render_width=render_width,
            render_height=render_height,
        )
        if not isinstance(
            self.controller,
            GentoBridgeStrokeDifferentialIKController,
        ):
            raise TypeError("GentoRoleIKEnv constructed the wrong nominal controller.")
        self._robot_geom_ids = self._geom_ids_with_prefix("gento_")
        self._table_geom_ids = self._geom_ids_with_prefix(
            "robot_table_collision_"
        )
        self._native_table_geom_ids = frozenset(
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_contype[geom_id]) & 1
        )
        self._support_pad_ids = {
            self._id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.SUPPORT_PAD_GEOMS
        }
        self._speed_pad_ids = {
            self._id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.SPEED_PAD_GEOMS
        }
        self._palm_guard_ids = {
            self._id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.PALM_GUARD_GEOMS
        }

    def _build_controller(self) -> GentoBridgeStrokeDifferentialIKController:
        return GentoBridgeStrokeDifferentialIKController(
            self.model,
            damping=0.10,
            gain=0.40,
            max_joint_step=0.012,
            orientation_weight=0.25,
        )

    def _grip_site_pairs(self) -> tuple[tuple[str, str], ...]:
        return GENTO_BRIDGE_STROKE_SITE_PAIRS

    def _geom_ids_with_prefix(self, prefix: str) -> frozenset[int]:
        return frozenset(
            geom_id
            for geom_id in range(self.model.ngeom)
            if (
                (name := mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    geom_id,
                ))
                is not None
                and name.startswith(prefix)
            )
        )

    def _set_ready_pose(self) -> None:
        set_gento_side_grasp_ready_pose(self.model, self.data)
        cue_rotation = self.data.xmat[self.cue_body_id].reshape(3, 3)
        self._support_axis = cue_rotation[:, 0].copy()
        self._support_axis /= max(np.linalg.norm(self._support_axis), 1.0e-12)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._lost_support_contact_steps = 0
        self._lost_speed_contact_steps = 0
        self._rear_speed_error_sum = 0.0
        self._rear_speed_sample_count = 0
        return super().reset(seed=seed, options=options)

    def _default_command(self) -> CueCommand:
        start = self._cue_pose()
        stroke = 0.060
        speed = 0.090
        duration = 1.50
        if self.randomize_command:
            stroke = float(self.np_random.uniform(0.045, 0.075))
            speed = float(self.np_random.uniform(0.060, 0.135))
            duration = float(self.np_random.uniform(1.45, 1.90))
        displacement = self._support_axis * stroke
        return CueCommand(
            pose=Pose3D(
                position=start.position + displacement,
                quat_wxyz=start.quat_wxyz.copy(),
            ),
            linear_velocity=self._support_axis * speed,
            angular_velocity=np.zeros(3, dtype=np.float64),
            duration=duration,
            debug_label="gento_front_support_rear_speed",
        )

    def _grip_error(self) -> float:
        return max(
            self.controller.bridge_position_error(self.data),
            self.controller.push_grip_error(self.data),
        )

    def _site_linear_velocity(self, site_id: int) -> np.ndarray:
        jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self.model,
            self.data,
            jacobian,
            None,
            int(site_id),
        )
        return jacobian @ self.data.qvel

    def _role_metrics(self) -> dict[str, float]:
        cue_axis = self.data.xmat[self.cue_body_id].reshape(3, 3)[:, 0]
        cosine = float(np.clip(np.dot(cue_axis, self._support_axis), -1.0, 1.0))
        direction_error = float(np.arccos(cosine))
        support_velocity = self._site_linear_velocity(
            self.controller.bridge_tcp_site_id
        )
        rear_velocity = self._site_linear_velocity(self.controller.push_tcp_site_id)
        rear_axial_speed = float(np.dot(rear_velocity, self._support_axis))
        rear_lateral_velocity = rear_velocity - rear_axial_speed * self._support_axis
        assert self._desired_command is not None
        desired_speed = float(
            np.dot(self._desired_command.linear_velocity, self._support_axis)
        )
        return {
            "support_direction_error": direction_error,
            "support_hand_speed": float(np.linalg.norm(support_velocity)),
            "rear_axial_speed": rear_axial_speed,
            "desired_rear_speed": desired_speed,
            "rear_speed_error": abs(desired_speed - rear_axial_speed),
            "rear_lateral_speed": float(np.linalg.norm(rear_lateral_velocity)),
        }

    def _contact_metrics(self) -> dict[str, float]:
        cue_shaft_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "cue_shaft")
        support_contacts = 0
        speed_contacts = 0
        table_contacts = 0
        palm_contacts = 0
        cue_table_contacts = 0
        max_table_penetration = 0.0
        max_palm_penetration = 0.0
        max_cue_table_penetration = 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if cue_shaft_id in pair:
                support_contacts += int(bool(pair & self._support_pad_ids))
                speed_contacts += int(bool(pair & self._speed_pad_ids))
                if pair & self._native_table_geom_ids:
                    cue_table_contacts += 1
                    max_cue_table_penetration = max(
                        max_cue_table_penetration,
                        max(0.0, -float(contact.dist)),
                    )
                if pair & self._palm_guard_ids:
                    palm_contacts += 1
                    max_palm_penetration = max(
                        max_palm_penetration,
                        max(0.0, -float(contact.dist)),
                    )
            if pair & self._robot_geom_ids and pair & self._table_geom_ids:
                table_contacts += 1
                max_table_penetration = max(
                    max_table_penetration,
                    max(0.0, -float(contact.dist)),
                )
        return {
            "support_cue_contact_count": float(support_contacts),
            "speed_cue_contact_count": float(speed_contacts),
            "robot_table_contact_count": float(table_contacts),
            "max_robot_table_penetration": max_table_penetration,
            "cue_palm_contact_count": float(palm_contacts),
            "max_cue_palm_penetration": max_palm_penetration,
            "cue_table_contact_count": float(cue_table_contacts),
            "max_cue_table_penetration": max_cue_table_penetration,
        }

    def _reward(self, normalized_action: np.ndarray) -> tuple[float, dict[str, float]]:
        reward, metrics = super()._reward(normalized_action)
        role = self._role_metrics()
        contacts = self._contact_metrics()
        reward += (
            1.25 * np.exp(-np.square(role["rear_speed_error"] / 0.045))
            + 0.75 * np.exp(-np.square(role["support_direction_error"] / 0.035))
            - 0.30 * np.square(role["rear_lateral_speed"] / 0.08)
            - 0.20 * np.square(role["support_hand_speed"] / 0.05)
            - 8.0 * np.square(contacts["max_robot_table_penetration"] / 0.01)
            - 12.0 * np.square(contacts["max_cue_palm_penetration"] / 0.005)
            - 16.0 * np.square(contacts["max_cue_table_penetration"] / 0.002)
        )
        if contacts["support_cue_contact_count"] < 1.0:
            reward -= 0.5
        if contacts["speed_cue_contact_count"] < 1.0:
            reward -= 1.0
        metrics.update(role)
        metrics.update(contacts)
        metrics["front_support_error"] = self.controller.bridge_position_error(
            self.data
        )
        metrics["rear_grip_error"] = self.controller.push_grip_error(self.data)
        return float(reward), metrics

    def _info(self, *, success: bool, **metrics: float) -> dict[str, Any]:
        mean_rear_speed_error = (
            self._rear_speed_error_sum / self._rear_speed_sample_count
            if self._rear_speed_sample_count
            else 0.0
        )
        return {
            **super()._info(success=success, **metrics),
            "support_hand": "right",
            "speed_hand": "left",
            "support_role": "fixed_position_and_direction",
            "speed_role": "axial_stroke_velocity",
            "mean_rear_speed_error": mean_rear_speed_error,
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        rear_speed_error = float(info["rear_speed_error"])
        self._rear_speed_error_sum += rear_speed_error
        self._rear_speed_sample_count += 1
        info["mean_rear_speed_error"] = (
            self._rear_speed_error_sum / self._rear_speed_sample_count
        )
        self._lost_support_contact_steps = (
            self._lost_support_contact_steps + 1
            if info["support_cue_contact_count"] < 1.0
            else 0
        )
        self._lost_speed_contact_steps = (
            self._lost_speed_contact_steps + 1
            if info["speed_cue_contact_count"] < 1.0
            else 0
        )
        info["lost_support_contact_steps"] = self._lost_support_contact_steps
        info["lost_speed_contact_steps"] = self._lost_speed_contact_steps

        physical_failure = (
            info["max_robot_table_penetration"] > 0.003
            or info["max_cue_palm_penetration"] > 0.002
            or info["max_cue_table_penetration"] > 0.001
            or self._lost_support_contact_steps > 12
            or self._lost_speed_contact_steps > 12
        )
        if physical_failure:
            terminated = True
            info["success"] = False
        elif truncated:
            info["success"] = bool(
                info["success"]
                and info["support_direction_error"] < 0.08
                and info["front_support_error"] < 0.025
                and info["mean_rear_speed_error"] < 0.12
            )
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self.model.vis.global_.offwidth = max(
                self.model.vis.global_.offwidth,
                self.render_width,
            )
            self.model.vis.global_.offheight = max(
                self.model.vis.global_.offheight,
                self.render_height,
            )
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.render_height,
                width=self.render_width,
            )
        self._renderer.update_scene(self.data, camera="gento_grip_closeup")
        return self._renderer.render().copy()
