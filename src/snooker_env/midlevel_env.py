"""MuJoCo environment scaffold for mid-level shot policy training.

This environment deliberately excludes the robot. It assumes an ideal low-level
cue controller and executes mid-level ``CueCommand`` trajectories directly on
the cue free joint while preserving MuJoCo cue/ball/table contacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from snooker_env.pipeline_types import BallState, CueCommand, SceneState, TableState


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIDLEVEL_MODEL = ROOT / "models" / "midlevel_train_scene.xml"


@dataclass(frozen=True)
class MidLevelRolloutResult:
    """Summary of one ideal-cue command trajectory rollout."""

    first_cue_ball_contact_time: float | None
    first_ball_ball_contact_time: float | None
    constraint_projection_count: int
    min_cue_table_clearance: float
    cue_ball_final_position: np.ndarray
    cue_ball_final_velocity: np.ndarray
    object_ball_final_position: np.ndarray
    object_ball_final_velocity: np.ndarray
    has_nan: bool
    exploded: bool


class MidLevelCueEnv:
    """Two-ball scene with direct cue pose/velocity control."""

    def __init__(self, model_path: Path = DEFAULT_MIDLEVEL_MODEL, action_repeat: int = 50) -> None:
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        if action_repeat <= 0:
            raise ValueError("action_repeat must be positive.")
        self.action_repeat = int(action_repeat)
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.cue_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "cue_free")
        self.cue_qpos_adr = int(self.model.jnt_qposadr[self.cue_joint_id])
        self.cue_dof_adr = int(self.model.jnt_dofadr[self.cue_joint_id])
        self.cue_tip_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "cue_tip")
        self.cue_ball_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "cue_ball_geom")
        self.object_ball_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "object_ball_0_geom")
        self.cue_ball_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, "cue_ball_free")
        self.object_ball_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, "object_ball_0_free")
        self._cue_local_samples = np.linspace(-0.725, 0.725, 25)
        self._cue_radius = 0.011
        self._table_obstacles = (
            (np.array([0.0, 0.0, 0.755], dtype=np.float64), np.array([1.07, 0.54, 0.006], dtype=np.float64)),
            (np.array([-0.55, 0.575, 0.795], dtype=np.float64), np.array([0.43, 0.035, 0.040], dtype=np.float64)),
            (np.array([0.55, 0.575, 0.795], dtype=np.float64), np.array([0.43, 0.035, 0.040], dtype=np.float64)),
            (np.array([-0.55, -0.575, 0.795], dtype=np.float64), np.array([0.43, 0.035, 0.040], dtype=np.float64)),
            (np.array([0.55, -0.575, 0.795], dtype=np.float64), np.array([0.43, 0.035, 0.040], dtype=np.float64)),
            (np.array([1.105, 0.0, 0.795], dtype=np.float64), np.array([0.035, 0.42, 0.040], dtype=np.float64)),
            (np.array([-1.105, 0.0, 0.795], dtype=np.float64), np.array([0.035, 0.42, 0.040], dtype=np.float64)),
        )

    @property
    def command_dt(self) -> float:
        """Fixed simulated duration of one cue command."""

        return float(self.action_repeat * self.model.opt.timestep)

    def reset(self) -> SceneState:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.scene_state()

    def scene_state(self) -> SceneState:
        mujoco.mj_forward(self.model, self.data)
        return SceneState(
            time=float(self.data.time),
            table=TableState(),
            balls={
                "cue_ball": self._ball_state("cue_ball", "cue_ball_free"),
                "object_ball_0": self._ball_state("object_ball_0", "object_ball_0_free"),
            },
        )

    def execute(self, commands: tuple[CueCommand, ...], settle_time: float = 0.8) -> MidLevelRolloutResult:
        first_cue_ball_contact: float | None = None
        first_ball_ball_contact: float | None = None
        constraint_projection_count = 0
        min_cue_table_clearance = float("inf")
        has_nan = False
        exploded = False

        if commands:
            first = commands[0]
            first_quat = first.pose.quat_wxyz.astype(np.float64).copy()
            first_quat /= max(float(np.linalg.norm(first_quat)), 1e-9)
            first_pos, clearance, projected = self.project_cue_position(first.pose.position.astype(np.float64), first_quat)
            constraint_projection_count += int(projected)
            min_cue_table_clearance = min(min_cue_table_clearance, clearance)
            self._set_cue_state(
                first_pos,
                first_quat,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
            mujoco.mj_forward(self.model, self.data)

        for command in commands:
            steps = self.action_repeat
            command_dt = self.command_dt
            current_pos = self.data.qpos[self.cue_qpos_adr:self.cue_qpos_adr + 3].copy()
            target_pos = command.pose.position.astype(np.float64).copy()
            quat = command.pose.quat_wxyz.astype(np.float64).copy()
            quat /= max(float(np.linalg.norm(quat)), 1e-9)
            command_speed = float(np.linalg.norm(command.linear_velocity))
            if command_speed < 1e-9:
                segment_velocity = (target_pos - current_pos) / command_dt
            else:
                segment_velocity = command.linear_velocity
            for step_idx in range(steps):
                elapsed = step_idx * self.model.opt.timestep
                if command_speed < 1e-9:
                    alpha = min(1.0, elapsed / command_dt)
                    pos = (1.0 - alpha) * current_pos + alpha * target_pos
                else:
                    pos = target_pos + command.linear_velocity * elapsed
                pos, clearance, projected = self.project_cue_position(pos, quat)
                if projected:
                    constraint_projection_count += 1
                self._set_cue_state(pos, quat, segment_velocity, command.angular_velocity)
                mujoco.mj_step(self.model, self.data)
                cue_contact, ball_contact = self._contact_flags()
                min_cue_table_clearance = min(min_cue_table_clearance, clearance)
                if cue_contact and first_cue_ball_contact is None:
                    first_cue_ball_contact = float(self.data.time)
                if ball_contact and first_ball_ball_contact is None:
                    first_ball_ball_contact = float(self.data.time)
                has_nan, exploded = self._stability_flags()
                if has_nan or exploded:
                    return self._result(
                        first_cue_ball_contact,
                        first_ball_ball_contact,
                        constraint_projection_count,
                        min_cue_table_clearance,
                        has_nan,
                        exploded,
                    )

        hold_pos = self.data.qpos[self.cue_qpos_adr:self.cue_qpos_adr + 3].copy()
        hold_quat = self.data.qpos[self.cue_qpos_adr + 3:self.cue_qpos_adr + 7].copy()
        hold_quat /= max(float(np.linalg.norm(hold_quat)), 1e-9)

        for _ in range(max(0, int(round(settle_time / self.model.opt.timestep)))):
            hold_pos, clearance, projected = self.project_cue_position(hold_pos, hold_quat)
            if projected:
                constraint_projection_count += 1
            self._set_cue_state(
                hold_pos,
                hold_quat,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
            mujoco.mj_step(self.model, self.data)
            cue_contact, ball_contact = self._contact_flags()
            min_cue_table_clearance = min(min_cue_table_clearance, clearance)
            if cue_contact and first_cue_ball_contact is None:
                first_cue_ball_contact = float(self.data.time)
            if ball_contact and first_ball_ball_contact is None:
                first_ball_ball_contact = float(self.data.time)
            has_nan, exploded = self._stability_flags()
            if has_nan or exploded:
                break

        return self._result(
            first_cue_ball_contact,
            first_ball_ball_contact,
            constraint_projection_count,
            min_cue_table_clearance,
            has_nan,
            exploded,
        )

    def _id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"Required {obj_type.name} name is missing: {name}")
        return obj_id

    def _ball_state(self, body_name: str, joint_name: str) -> BallState:
        body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, body_name)
        joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        dof_adr = int(self.model.jnt_dofadr[joint_id])
        return BallState(
            name=body_name,
            position=self.data.xpos[body_id].copy(),
            linear_velocity=self.data.qvel[dof_adr:dof_adr + 3].copy(),
            angular_velocity=self.data.qvel[dof_adr + 3:dof_adr + 6].copy(),
        )

    def _set_cue_state(
        self,
        position: np.ndarray,
        quat_wxyz: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> None:
        self.data.qpos[self.cue_qpos_adr:self.cue_qpos_adr + 3] = position
        self.data.qpos[self.cue_qpos_adr + 3:self.cue_qpos_adr + 7] = quat_wxyz
        self.data.qvel[self.cue_dof_adr:self.cue_dof_adr + 3] = linear_velocity
        self.data.qvel[self.cue_dof_adr + 3:self.cue_dof_adr + 6] = angular_velocity

    def project_cue_position(self, position: np.ndarray, quat_wxyz: np.ndarray, margin: float = 0.004) -> tuple[np.ndarray, float, bool]:
        """Project a requested cue pose into the table-feasible action space."""

        projected = position.astype(np.float64).copy()
        clearance = self._cue_table_clearance_for(projected, quat_wxyz)
        projected_any = False
        for _ in range(12):
            if clearance >= margin:
                return projected, clearance, projected_any
            projected[2] += margin - clearance
            projected_any = True
            clearance = self._cue_table_clearance_for(projected, quat_wxyz)
        return projected, clearance, projected_any

    def cue_table_clearance(self) -> float:
        """Return minimum signed clearance between cue samples and table proxies."""

        cue_pos = self.data.qpos[self.cue_qpos_adr:self.cue_qpos_adr + 3]
        cue_quat = self.data.qpos[self.cue_qpos_adr + 3:self.cue_qpos_adr + 7]
        return self._cue_table_clearance_for(cue_pos, cue_quat)

    def _cue_table_clearance_for(self, cue_pos: np.ndarray, cue_quat: np.ndarray) -> float:
        rot_flat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot_flat, cue_quat)
        rot = rot_flat.reshape(3, 3)
        min_clearance = float("inf")
        for sample_x in self._cue_local_samples:
            point = cue_pos + rot @ np.array([sample_x, 0.0, 0.0], dtype=np.float64)
            for center, half_size in self._table_obstacles:
                inflated = half_size + self._cue_radius
                delta = np.abs(point - center) - inflated
                outside = np.maximum(delta, 0.0)
                if np.any(delta > 0.0):
                    signed = float(np.linalg.norm(outside))
                else:
                    signed = float(np.max(delta))
                min_clearance = min(min_clearance, signed)
        return min_clearance

    def _contact_flags(self) -> tuple[bool, bool]:
        cue_ball_contact = False
        ball_ball_contact = False
        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            geom_pair = {contact.geom1, contact.geom2}
            cue_ball_contact = cue_ball_contact or geom_pair == {self.cue_tip_geom, self.cue_ball_geom}
            ball_ball_contact = ball_ball_contact or geom_pair == {self.cue_ball_geom, self.object_ball_geom}
        return cue_ball_contact, ball_ball_contact

    def _stability_flags(self) -> tuple[bool, bool]:
        has_nan = not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))
        exploded = bool(np.max(np.abs(self.data.qvel)) > 150.0 or np.max(np.abs(self.data.qpos)) > 25.0)
        return has_nan, exploded

    def _result(
        self,
        first_cue_ball_contact_time: float | None,
        first_ball_ball_contact_time: float | None,
        constraint_projection_count: int,
        min_cue_table_clearance: float,
        has_nan: bool,
        exploded: bool,
    ) -> MidLevelRolloutResult:
        cue_ball = self._ball_state("cue_ball", "cue_ball_free")
        object_ball = self._ball_state("object_ball_0", "object_ball_0_free")
        return MidLevelRolloutResult(
            first_cue_ball_contact_time=first_cue_ball_contact_time,
            first_ball_ball_contact_time=first_ball_ball_contact_time,
            constraint_projection_count=constraint_projection_count,
            min_cue_table_clearance=min_cue_table_clearance if np.isfinite(min_cue_table_clearance) else 0.0,
            cue_ball_final_position=cue_ball.position,
            cue_ball_final_velocity=cue_ball.linear_velocity,
            object_ball_final_position=object_ball.position,
            object_ball_final_velocity=object_ball.linear_velocity,
            has_nan=has_nan,
            exploded=exploded,
        )
