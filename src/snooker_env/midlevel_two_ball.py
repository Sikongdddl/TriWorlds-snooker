"""Deterministic two-ball shot execution for mid-level policy learning.

The simulator in this module is shared by task generation and the Gymnasium
environment.  A policy chooses only a horizontal shot direction and cue speed;
the cue itself is positioned and advanced by a deterministic kinematic driver.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from snooker_env.contact_events import CollisionEventMonitor, ContactEvent
from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL
from snooker_env.table_geometry import BALL_CENTER_Z, BALL_RADIUS


POCKET_NAMES = (
    "pocket_corner_negx_negy",
    "pocket_corner_negx_posy",
    "pocket_corner_posx_negy",
    "pocket_corner_posx_posy",
    "pocket_middle_negx",
    "pocket_middle_posx",
)
POCKET_POSITIONS = {
    "pocket_corner_negx_negy": np.array([-0.675, -1.31], dtype=np.float64),
    "pocket_corner_negx_posy": np.array([-0.675, 1.31], dtype=np.float64),
    "pocket_corner_posx_negy": np.array([0.675, -1.31], dtype=np.float64),
    "pocket_corner_posx_posy": np.array([0.675, 1.31], dtype=np.float64),
    "pocket_middle_negx": np.array([-0.717426, 0.0], dtype=np.float64),
    "pocket_middle_posx": np.array([0.717426, 0.0], dtype=np.float64),
}

MIN_CUE_SPEED = 0.3
MAX_CUE_SPEED = 2.5
MAX_ANGLE_RESIDUAL = np.deg2rad(15.0)
CUE_TIP_LOCAL_X = 0.725
CUE_START_BACKOFF = 0.10
CUE_FOLLOW_THROUGH = 0.05
STOP_SPEED_THRESHOLD = 0.01
STOP_HOLD_TIME = 0.20
MAX_SHOT_TIME = 8.0
SHOT_EXECUTION_VERSION = "two-ball-center-hit-v4-backend-fingerprint"


@dataclass(frozen=True)
class TwoBallShotResult:
    """Terminal events and measurements from one complete shot."""

    target_pocket: str
    shot_direction: np.ndarray
    cue_speed: float
    elapsed_time: float
    cue_ball_final_position: np.ndarray
    object_ball_final_position: np.ndarray
    first_ball_contact_time: float | None
    first_cushion_contact_time: float | None
    object_pocket: str | None
    cue_pocket: str | None
    min_object_pocket_distance: float
    initial_object_pocket_distance: float
    stopped: bool
    timed_out: bool
    numerical_failure: bool
    cushion_before_object: bool
    object_cushion_before_pocket: bool
    any_cushion_contact: bool
    contact_events: tuple[ContactEvent, ...]

    @property
    def legal_first_contact(self) -> bool:
        return self.first_ball_contact_time is not None

    @property
    def correct_pot(self) -> bool:
        return self.object_pocket == self.target_pocket

    @property
    def wrong_pocket(self) -> bool:
        return self.object_pocket is not None and not self.correct_pot

    @property
    def cue_scratch(self) -> bool:
        return self.cue_pocket is not None


def rotate_direction(direction: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a two-dimensional unit direction by ``angle`` radians."""

    direction = np.asarray(direction, dtype=np.float64)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotated = np.array(
        [cosine * direction[0] - sine * direction[1], sine * direction[0] + cosine * direction[1]],
        dtype=np.float64,
    )
    return rotated / max(float(np.linalg.norm(rotated)), 1e-12)


def ghost_ball_direction(
    cue_position: np.ndarray,
    object_position: np.ndarray,
    pocket_position: np.ndarray,
) -> np.ndarray:
    """Return the direct-pot direction from the cue ball to the ghost ball."""

    cue_xy = np.asarray(cue_position, dtype=np.float64)[:2]
    object_xy = np.asarray(object_position, dtype=np.float64)[:2]
    pocket_xy = np.asarray(pocket_position, dtype=np.float64)[:2]
    object_to_pocket = pocket_xy - object_xy
    object_distance = float(np.linalg.norm(object_to_pocket))
    if object_distance <= 2.0 * BALL_RADIUS:
        raise ValueError("Object ball is too close to the pocket for ghost-ball planning.")
    pot_direction = object_to_pocket / object_distance
    ghost_position = object_xy - 2.0 * BALL_RADIUS * pot_direction
    shot = ghost_position - cue_xy
    shot_norm = float(np.linalg.norm(shot))
    if shot_norm <= 2.0 * BALL_RADIUS:
        raise ValueError("Cue ball is too close to the ghost-ball position.")
    return shot / shot_norm


def decode_action(
    action: np.ndarray,
    cue_position: np.ndarray,
    object_position: np.ndarray,
    pocket_position: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Map a normalized PPO action to world-frame direction and cue speed."""

    clipped = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    if clipped.shape != (2,):
        raise ValueError(f"Expected a two-dimensional action, got {clipped.shape}.")
    baseline = ghost_ball_direction(cue_position, object_position, pocket_position)
    direction = rotate_direction(baseline, float(clipped[0]) * MAX_ANGLE_RESIDUAL)
    speed_fraction = 0.5 * (float(clipped[1]) + 1.0)
    speed = MIN_CUE_SPEED + speed_fraction * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    return direction, float(speed)


def encode_speed_action(speed: float) -> float:
    """Map a physical cue speed back into the normalized action range."""

    fraction = (float(speed) - MIN_CUE_SPEED) / (MAX_CUE_SPEED - MIN_CUE_SPEED)
    return float(np.clip(2.0 * fraction - 1.0, -1.0, 1.0))


def quantize_cue_speed(speed: float) -> float:
    """Round-trip a speed through the float32 Gym action representation."""

    normalized = np.float32(encode_speed_action(speed))
    fraction = 0.5 * (float(normalized) + 1.0)
    return float(MIN_CUE_SPEED + fraction * (MAX_CUE_SPEED - MIN_CUE_SPEED))


def _xml_dependencies(model_path: Path) -> tuple[Path, ...]:
    visited: set[Path] = set()

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in visited:
            return
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        visited.add(resolved)
        root = ET.fromstring(resolved.read_bytes())
        for include in root.iter("include"):
            include_file = include.attrib.get("file")
            if not include_file:
                continue
            nested = Path(include_file)
            if not nested.is_absolute():
                nested = resolved.parent / nested
            visit(nested)

    visit(Path(model_path))
    return tuple(sorted(visited, key=lambda path: str(path)))


def xml_sha256(model_path: Path = DEFAULT_MIDLEVEL_MODEL) -> str:
    """Hash the root MJCF and every recursively included XML file."""

    digest = hashlib.sha256()
    root = Path(model_path).resolve().parent
    for path in _xml_dependencies(Path(model_path)):
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = str(path)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def sdf_plugin_sha256() -> str:
    """Hash the installed SDF plugin binary that defines table collision physics."""

    plugin_path = Path(mujoco.__file__).resolve().parent / "plugin" / "libsdf_plugin.so"
    if not plugin_path.is_file():
        raise FileNotFoundError(
            f"Installed MuJoCo SDF plugin is missing: {plugin_path}"
        )
    return hashlib.sha256(plugin_path.read_bytes()).hexdigest()


def model_sha256(model: mujoco.MjModel, source_xml_sha256: str) -> str:
    """Hash compiled physics fields that affect two-ball trajectories."""

    digest = hashlib.sha256()
    digest.update(source_xml_sha256.encode("ascii"))
    digest.update(mujoco.__version__.encode("ascii"))
    digest.update(SHOT_EXECUTION_VERSION.encode("ascii"))
    digest.update(sdf_plugin_sha256().encode("ascii"))
    scalar_fields = (
        model.nq,
        model.nv,
        model.nbody,
        model.ngeom,
        model.npair,
        model.opt.timestep,
        model.opt.integrator,
        model.opt.solver,
        model.opt.iterations,
        model.opt.tolerance,
        model.opt.cone,
        model.opt.jacobian,
        model.opt.disableflags,
        model.opt.enableflags,
        model.opt.sdf_initpoints,
        model.opt.sdf_iterations,
    )
    digest.update(repr(scalar_fields).encode("ascii"))
    for name in (
        "body_mass",
        "body_inertia",
        "dof_damping",
        "geom_bodyid",
        "geom_type",
        "geom_contype",
        "geom_conaffinity",
        "geom_condim",
        "geom_priority",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_solmix",
        "geom_margin",
        "geom_gap",
        "geom_plugin",
        "pair_geom1",
        "pair_geom2",
        "pair_dim",
        "pair_friction",
        "pair_solref",
        "pair_solreffriction",
        "pair_solimp",
        "pair_margin",
        "pair_gap",
    ):
        values = np.asarray(getattr(model, name))
        digest.update(name.encode("ascii"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(repr(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
    digest.update(np.asarray(model.opt.gravity, dtype=np.float64).tobytes())
    return digest.hexdigest()


class TwoBallShotSimulator:
    """Execute horizontal center-ball strokes in the exact mid-level MJCF."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MIDLEVEL_MODEL,
        *,
        max_time: float = MAX_SHOT_TIME,
        stop_speed: float = STOP_SPEED_THRESHOLD,
        stop_hold_time: float = STOP_HOLD_TIME,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.max_time = float(max_time)
        self.stop_speed = float(stop_speed)
        self.stop_hold_time = float(stop_hold_time)
        if self.max_time <= 0.0 or self.stop_speed <= 0.0 or self.stop_hold_time <= 0.0:
            raise ValueError("Shot timing and stop thresholds must be positive.")

        self.cue_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "cue_free")
        self.cue_qpos_adr = int(self.model.jnt_qposadr[self.cue_joint_id])
        self.cue_dof_adr = int(self.model.jnt_dofadr[self.cue_joint_id])
        self.ball_joint_ids = {
            "cue_ball": self._id(mujoco.mjtObj.mjOBJ_JOINT, "cue_ball_free"),
            "object_ball_0": self._id(mujoco.mjtObj.mjOBJ_JOINT, "object_ball_0_free"),
        }
        self.ball_body_ids = {
            name: self._id(mujoco.mjtObj.mjOBJ_BODY, name) for name in self.ball_joint_ids
        }
        self.contact_monitor = CollisionEventMonitor(self.model)
        self.pocket_positions = self._read_pocket_positions()
        self.xml_hash = xml_sha256(self.model_path)
        self.model_hash = model_sha256(self.model, self.xml_hash)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Required MuJoCo name is missing: {name}")
        return object_id

    def _read_pocket_positions(self) -> dict[str, np.ndarray]:
        positions: dict[str, np.ndarray] = {}
        mujoco.mj_forward(self.model, self.data)
        for name in POCKET_NAMES:
            site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, name)
            positions[name] = self.data.site_xpos[site_id, :2].copy()
        return positions

    def _set_ball(self, name: str, position_xy: np.ndarray) -> None:
        joint_id = self.ball_joint_ids[name]
        qpos_adr = int(self.model.jnt_qposadr[joint_id])
        dof_adr = int(self.model.jnt_dofadr[joint_id])
        self.data.qpos[qpos_adr:qpos_adr + 3] = (
            float(position_xy[0]),
            float(position_xy[1]),
            BALL_CENTER_Z,
        )
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[dof_adr:dof_adr + 6] = 0.0

    def _park_ball(self, name: str) -> None:
        """Hold a pocketed ball away from all active collision geometry."""

        joint_id = self.ball_joint_ids[name]
        qpos_adr = int(self.model.jnt_qposadr[joint_id])
        dof_adr = int(self.model.jnt_dofadr[joint_id])
        self.data.qpos[qpos_adr:qpos_adr + 3] = (0.0, 0.0, 4.0)
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[dof_adr:dof_adr + 6] = 0.0

    def _set_cue(self, position: np.ndarray, quaternion: np.ndarray, velocity: np.ndarray) -> None:
        self.data.qpos[self.cue_qpos_adr:self.cue_qpos_adr + 3] = position
        self.data.qpos[self.cue_qpos_adr + 3:self.cue_qpos_adr + 7] = quaternion
        self.data.qvel[self.cue_dof_adr:self.cue_dof_adr + 3] = velocity
        self.data.qvel[self.cue_dof_adr + 3:self.cue_dof_adr + 6] = 0.0

    def reset(self, cue_position: np.ndarray, object_position: np.ndarray) -> None:
        """Reset both balls at rest and move the cue out of collision range."""

        cue_xy = np.asarray(cue_position, dtype=np.float64)[:2]
        object_xy = np.asarray(object_position, dtype=np.float64)[:2]
        if not np.all(np.isfinite(cue_xy)) or not np.all(np.isfinite(object_xy)):
            raise ValueError("Ball positions must be finite.")
        if np.linalg.norm(cue_xy - object_xy) <= 2.0 * BALL_RADIUS:
            raise ValueError("Initial balls overlap.")
        mujoco.mj_resetData(self.model, self.data)
        self._set_ball("cue_ball", cue_xy)
        self._set_ball("object_ball_0", object_xy)
        self._set_cue(
            np.array([0.0, 0.0, 6.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )
        mujoco.mj_forward(self.model, self.data)
        self.contact_monitor.reset()

    def _ball_position(self, name: str) -> np.ndarray:
        return self.data.xpos[self.ball_body_ids[name]].copy()

    def _ball_is_stopped(self, name: str) -> bool:
        joint_id = self.ball_joint_ids[name]
        dof_adr = int(self.model.jnt_dofadr[joint_id])
        linear_speed = float(np.linalg.norm(self.data.qvel[dof_adr:dof_adr + 3]))
        surface_angular_speed = BALL_RADIUS * float(np.linalg.norm(self.data.qvel[dof_adr + 3:dof_adr + 6]))
        return linear_speed < self.stop_speed and surface_angular_speed < self.stop_speed

    @staticmethod
    def _event_mentions(event: ContactEvent, geom_name: str) -> bool:
        return event.primary_name == geom_name or event.secondary_name == geom_name

    def execute(
        self,
        cue_position: np.ndarray,
        object_position: np.ndarray,
        target_pocket: str,
        shot_direction: np.ndarray,
        cue_speed: float,
    ) -> TwoBallShotResult:
        """Execute one shot and simulate until the required balls have stopped."""

        if target_pocket not in self.pocket_positions:
            raise KeyError(f"Unknown target pocket: {target_pocket}")
        direction_xy = np.asarray(shot_direction, dtype=np.float64)[:2]
        direction_norm = float(np.linalg.norm(direction_xy))
        if direction_norm <= 1e-12 or not np.isfinite(direction_norm):
            raise ValueError("Shot direction must be finite and non-zero.")
        direction_xy = direction_xy / direction_norm
        speed = float(cue_speed)
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError("Cue speed must be finite and positive.")

        cue_xy = np.asarray(cue_position, dtype=np.float64)[:2]
        object_xy = np.asarray(object_position, dtype=np.float64)[:2]
        self.reset(cue_xy, object_xy)
        pocket_xy = self.pocket_positions[target_pocket]
        initial_distance = float(np.linalg.norm(object_xy - pocket_xy))
        minimum_distance = initial_distance

        yaw = float(np.arctan2(direction_xy[1], direction_xy[0]))
        quaternion = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)
        direction = np.array([direction_xy[0], direction_xy[1], 0.0], dtype=np.float64)
        rear_contact = np.array(
            [cue_xy[0], cue_xy[1], BALL_CENTER_Z], dtype=np.float64
        ) - direction * BALL_RADIUS
        initial_tip = rear_contact - direction * CUE_START_BACKOFF
        initial_body = initial_tip - direction * CUE_TIP_LOCAL_X
        cue_travel = CUE_START_BACKOFF + CUE_FOLLOW_THROUGH
        cue_duration = cue_travel / speed

        first_ball_contact: float | None = None
        first_cushion_contact: float | None = None
        cushion_before_object = False
        object_cushion_before_pocket = False
        any_cushion = False
        object_pocket: str | None = None
        cue_pocket: str | None = None
        numerical_failure = False

        def scan_step() -> None:
            nonlocal first_ball_contact, first_cushion_contact
            nonlocal cushion_before_object, object_cushion_before_pocket
            nonlocal any_cushion, object_pocket, cue_pocket
            new_events = self.contact_monitor.scan(self.data)
            for event in new_events:
                if event.kind == "ball_ball" and first_ball_contact is None:
                    if self._event_mentions(event, "cue_ball_geom") and self._event_mentions(
                        event, "object_ball_0_geom"
                    ):
                        first_ball_contact = event.time
                elif event.kind == "ball_cushion":
                    # Once the target object has dropped, later cue-ball rail
                    # motion is position play, not a cushion-assisted pot.
                    # Direct-task filtering only concerns the path that made
                    # the pot itself.
                    if object_pocket is not None:
                        continue
                    ball_name = None
                    if self._event_mentions(event, "cue_ball_geom"):
                        ball_name = "cue_ball"
                    elif self._event_mentions(event, "object_ball_0_geom"):
                        ball_name = "object_ball_0"
                    near_pocket_jaw = False
                    if ball_name is not None:
                        ball_xy = self._ball_position(ball_name)[:2]
                        near_pocket_jaw = any(
                            np.linalg.norm(ball_xy - candidate) < 0.15
                            for candidate in self.pocket_positions.values()
                        )
                    if near_pocket_jaw:
                        continue
                    any_cushion = True
                    if first_cushion_contact is None:
                        first_cushion_contact = event.time
                    if first_ball_contact is None and self._event_mentions(event, "cue_ball_geom"):
                        cushion_before_object = True
                    if ball_name == "object_ball_0":
                        object_cushion_before_pocket = True
                elif event.kind == "ball_pocket":
                    if event.primary_name == "object_ball_0" and object_pocket is None:
                        object_pocket = event.secondary_name
                    elif event.primary_name == "cue_ball" and cue_pocket is None:
                        cue_pocket = event.secondary_name

        cue_steps = max(1, int(np.ceil(cue_duration / self.timestep)))
        for step_index in range(cue_steps):
            elapsed = min(step_index * self.timestep, cue_duration)
            body_position = initial_body + direction * (speed * elapsed)
            self._set_cue(body_position, quaternion, direction * speed)
            mujoco.mj_step(self.model, self.data)
            scan_step()
            object_now = self._ball_position("object_ball_0")
            minimum_distance = min(minimum_distance, float(np.linalg.norm(object_now[:2] - pocket_xy)))
            if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))):
                numerical_failure = True
                break
            if np.max(np.abs(self.data.qvel)) > 150.0 or np.max(np.abs(self.data.qpos)) > 25.0:
                numerical_failure = True
                break
            if cue_pocket is not None:
                break

        self._set_cue(
            np.array([0.0, 0.0, 6.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )
        mujoco.mj_forward(self.model, self.data)

        below_threshold_since: float | None = None
        stopped = False
        while not numerical_failure and cue_pocket is None and self.data.time < self.max_time:
            # The cue and any pocketed object are kinematically parked.  Using
            # a short native MuJoCo batch after the cue stroke preserves the
            # 10 us integration step while avoiding a Python round trip for
            # every free-rolling substep.
            self._set_cue(
                np.array([0.0, 0.0, 6.0], dtype=np.float64),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
            if object_pocket is not None:
                self._park_ball("object_ball_0")
            batch_steps = 5 if first_ball_contact is not None and object_pocket is None else 20
            remaining_steps = max(1, int((self.max_time - self.data.time) / self.timestep))
            batch_steps = min(batch_steps, remaining_steps)
            mujoco.mj_step(self.model, self.data, nstep=batch_steps)
            scan_step()
            object_now = self._ball_position("object_ball_0")
            minimum_distance = min(minimum_distance, float(np.linalg.norm(object_now[:2] - pocket_xy)))

            required_stopped = self._ball_is_stopped("cue_ball")
            if object_pocket is None:
                required_stopped = required_stopped and self._ball_is_stopped("object_ball_0")
            if required_stopped:
                if below_threshold_since is None:
                    below_threshold_since = float(self.data.time)
                elif self.data.time - below_threshold_since >= self.stop_hold_time:
                    stopped = True
                    break
            else:
                below_threshold_since = None

            finite = np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))
            exploded = np.max(np.abs(self.data.qvel)) > 150.0 or np.max(np.abs(self.data.qpos)) > 25.0
            if not finite or exploded:
                numerical_failure = True
                break

        timed_out = bool(
            not stopped and not numerical_failure and cue_pocket is None and self.data.time >= self.max_time
        )
        return TwoBallShotResult(
            target_pocket=target_pocket,
            shot_direction=direction.copy(),
            cue_speed=speed,
            elapsed_time=float(self.data.time),
            cue_ball_final_position=self._ball_position("cue_ball"),
            object_ball_final_position=self._ball_position("object_ball_0"),
            first_ball_contact_time=first_ball_contact,
            first_cushion_contact_time=first_cushion_contact,
            object_pocket=object_pocket,
            cue_pocket=cue_pocket,
            min_object_pocket_distance=minimum_distance,
            initial_object_pocket_distance=initial_distance,
            stopped=stopped,
            timed_out=timed_out,
            numerical_failure=numerical_failure,
            cushion_before_object=cushion_before_object,
            object_cushion_before_pocket=object_cushion_before_pocket,
            any_cushion_contact=any_cushion,
            contact_events=tuple(self.contact_monitor.events),
        )
