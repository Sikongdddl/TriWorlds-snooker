"""Batched MuJoCo Warp backend for contextual two-ball shot execution.

One Stable-Baselines3 vector step executes one complete shot in every Warp
world.  Physics state, contact-event reduction, stopping detection, and
terminal measurements remain on the GPU until the batched rollout completes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any
import warnings

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco_warp as mjw
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvStepReturn,
)
import warp as wp

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL
from snooker_env.midlevel_two_ball_env import compute_terminal_reward, task_observation
from snooker_env.midlevel_tasks import (
    MUJOCO_WARP_PHYSICS_BACKEND,
    TwoBallTaskDataset,
)
from snooker_env.midlevel_two_ball import (
    CUE_FOLLOW_THROUGH,
    CUE_START_BACKOFF,
    CUE_TIP_LOCAL_X,
    MAX_ANGLE_RESIDUAL,
    MAX_CUE_SPEED,
    MAX_SHOT_TIME,
    MIN_CUE_SPEED,
    POCKET_NAMES,
    STOP_HOLD_TIME,
    STOP_SPEED_THRESHOLD,
    TwoBallShotResult,
    TwoBallShotSimulator,
    model_sha256,
)
from snooker_env import mujoco_warp_sdf as mujoco_warp_sdf_module
from snooker_env.mujoco_warp_sdf import (
    MUJOCO_WARP_NCONMAX,
    MUJOCO_WARP_NJMAX,
    assert_mujoco_warp_capacity,
    calibrate_trapezoid_sdf_contact_damping,
    normalize_zero_friction_contact_dims,
    register_mujoco_billiards_sdf,
)
from snooker_env.table_geometry import (
    BALL_CENTER_Z,
    BALL_RADIUS,
    POCKET_ENTRY_Z,
    cushion_geom_ids,
)


_NO_EVENT_STEP = 2_000_000_000
MUJOCO_WARP_SHOT_EXECUTION_VERSION = (
    "two-ball-mujoco-warp-v8-deterministic-reductions-20us"
)


def prepare_mujoco_warp_model(model: mujoco.MjModel) -> None:
    """Apply every model-side transformation used by the MJWarp backend."""

    register_mujoco_billiards_sdf(model)
    normalize_zero_friction_contact_dims(model)
    calibrate_trapezoid_sdf_contact_damping(model)


def _hash_python_tree(digest: Any, root: Path, label: str) -> None:
    for path in sorted(root.rglob("*.py"), key=lambda candidate: str(candidate)):
        if (
            "__pycache__" in path.parts
            or path.name.endswith("_test.py")
            or path.name == "test_data.py"
        ):
            continue
        digest.update(f"{label}/{path.relative_to(root)}".encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")


def mujoco_warp_backend_sha256(
    model: mujoco.MjModel,
    source_xml_sha256: str,
) -> str:
    """Fingerprint calibrated model fields and all active MJWarp physics code."""

    digest = hashlib.sha256()
    digest.update(MUJOCO_WARP_SHOT_EXECUTION_VERSION.encode("ascii"))
    digest.update(model_sha256(model, source_xml_sha256).encode("ascii"))
    digest.update(str(wp.__version__).encode("ascii"))
    digest.update(str(Path(mjw.__file__).resolve()).encode("utf-8"))
    _hash_python_tree(
        digest,
        Path(mjw.__file__).resolve().parent,
        "mujoco_warp",
    )
    warp_root = Path(wp.__file__).resolve().parent
    for path in sorted((warp_root / "bin").glob("*.so")):
        digest.update(f"warp_native/{path.name}".encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    for label, path in (
        ("project_mujoco_warp_sdf", Path(mujoco_warp_sdf_module.__file__).resolve()),
        ("project_midlevel_warp", Path(__file__).resolve()),
    ):
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def active_mujoco_warp_backend_sha256(
    model_path: Path = DEFAULT_MIDLEVEL_MODEL,
) -> tuple[str, str, str]:
    """Return common XML/model hashes and the active calibrated MJWarp hash."""

    simulator = TwoBallShotSimulator(model_path)
    common_model_hash = simulator.model_hash
    prepare_mujoco_warp_model(simulator.model)
    backend_hash = mujoco_warp_backend_sha256(
        simulator.model,
        simulator.xml_hash,
    )
    return simulator.xml_hash, common_model_hash, backend_hash


@wp.kernel
def _initialize_shots(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    cue_positions: wp.array2d(dtype=float),
    object_positions: wp.array2d(dtype=float),
    target_pockets: wp.array(dtype=int),
    pocket_positions: wp.array(dtype=wp.vec2),
    cue_qpos: int,
    cue_dof: int,
    cue_ball_qpos: int,
    cue_ball_dof: int,
    object_ball_qpos: int,
    object_ball_dof: int,
    nv: int,
    first_ball_step: wp.array(dtype=int),
    first_cushion_step: wp.array(dtype=int),
    first_object_cushion_step: wp.array(dtype=int),
    shot_step: wp.array(dtype=int),
    object_pocket: wp.array(dtype=int),
    cue_pocket: wp.array(dtype=int),
    minimum_object_distance: wp.array(dtype=float),
    stop_hold_steps: wp.array(dtype=int),
    stopped: wp.array(dtype=int),
    timed_out: wp.array(dtype=int),
    numerical_failure: wp.array(dtype=int),
    overflow_seen: wp.array(dtype=int),
    done: wp.array(dtype=int),
    done_count: wp.array(dtype=int),
    final_cue_position: wp.array(dtype=wp.vec3),
    final_object_position: wp.array(dtype=wp.vec3),
    elapsed_time: wp.array(dtype=float),
) -> None:
    """Load one task per world and clear all terminal accumulators."""

    world_id = wp.tid()
    cue_x = cue_positions[world_id, 0]
    cue_y = cue_positions[world_id, 1]
    object_x = object_positions[world_id, 0]
    object_y = object_positions[world_id, 1]

    for dof_id in range(nv):
        qvel[world_id, dof_id] = 0.0

    qpos[world_id, cue_qpos + 0] = 0.0
    qpos[world_id, cue_qpos + 1] = 0.0
    qpos[world_id, cue_qpos + 2] = 6.0
    qpos[world_id, cue_qpos + 3] = 1.0
    qpos[world_id, cue_qpos + 4] = 0.0
    qpos[world_id, cue_qpos + 5] = 0.0
    qpos[world_id, cue_qpos + 6] = 0.0
    for offset in range(6):
        qvel[world_id, cue_dof + offset] = 0.0

    qpos[world_id, cue_ball_qpos + 0] = cue_x
    qpos[world_id, cue_ball_qpos + 1] = cue_y
    qpos[world_id, cue_ball_qpos + 2] = BALL_CENTER_Z
    qpos[world_id, cue_ball_qpos + 3] = 1.0
    qpos[world_id, cue_ball_qpos + 4] = 0.0
    qpos[world_id, cue_ball_qpos + 5] = 0.0
    qpos[world_id, cue_ball_qpos + 6] = 0.0
    for offset in range(6):
        qvel[world_id, cue_ball_dof + offset] = 0.0

    qpos[world_id, object_ball_qpos + 0] = object_x
    qpos[world_id, object_ball_qpos + 1] = object_y
    qpos[world_id, object_ball_qpos + 2] = BALL_CENTER_Z
    qpos[world_id, object_ball_qpos + 3] = 1.0
    qpos[world_id, object_ball_qpos + 4] = 0.0
    qpos[world_id, object_ball_qpos + 5] = 0.0
    qpos[world_id, object_ball_qpos + 6] = 0.0
    for offset in range(6):
        qvel[world_id, object_ball_dof + offset] = 0.0

    pocket_xy = pocket_positions[target_pockets[world_id]]
    delta_x = object_x - pocket_xy[0]
    delta_y = object_y - pocket_xy[1]
    minimum_object_distance[world_id] = wp.sqrt(
        delta_x * delta_x + delta_y * delta_y
    )
    first_ball_step[world_id] = _NO_EVENT_STEP
    first_cushion_step[world_id] = _NO_EVENT_STEP
    first_object_cushion_step[world_id] = _NO_EVENT_STEP
    shot_step[world_id] = 0
    object_pocket[world_id] = -1
    cue_pocket[world_id] = -1
    stop_hold_steps[world_id] = 0
    stopped[world_id] = 0
    timed_out[world_id] = 0
    numerical_failure[world_id] = 0
    overflow_seen[world_id] = 0
    done[world_id] = 0
    final_cue_position[world_id] = wp.vec3(cue_x, cue_y, BALL_CENTER_Z)
    final_object_position[world_id] = wp.vec3(
        object_x,
        object_y,
        BALL_CENTER_Z,
    )
    elapsed_time[world_id] = 0.0
    if world_id == 0:
        done_count[0] = 0


@wp.kernel
def _drive_and_park(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    shot_step: wp.array(dtype=int),
    shot_directions: wp.array2d(dtype=float),
    cue_speeds: wp.array(dtype=float),
    cue_positions: wp.array2d(dtype=float),
    object_pocket: wp.array(dtype=int),
    done: wp.array(dtype=int),
    cue_qpos: int,
    cue_dof: int,
    cue_ball_qpos: int,
    cue_ball_dof: int,
    object_ball_qpos: int,
    object_ball_dof: int,
    timestep: float,
) -> None:
    """Kinematically drive each cue and park inactive dynamic bodies."""

    world_id = wp.tid()
    if done[world_id] != 0:
        qpos[world_id, cue_qpos + 0] = 0.0
        qpos[world_id, cue_qpos + 1] = 0.0
        qpos[world_id, cue_qpos + 2] = 6.0
        qpos[world_id, cue_qpos + 3] = 1.0
        qpos[world_id, cue_qpos + 4] = 0.0
        qpos[world_id, cue_qpos + 5] = 0.0
        qpos[world_id, cue_qpos + 6] = 0.0
        qpos[world_id, cue_ball_qpos + 0] = 0.0
        qpos[world_id, cue_ball_qpos + 1] = 0.0
        qpos[world_id, cue_ball_qpos + 2] = 6.0
        qpos[world_id, cue_ball_qpos + 3] = 1.0
        qpos[world_id, cue_ball_qpos + 4] = 0.0
        qpos[world_id, cue_ball_qpos + 5] = 0.0
        qpos[world_id, cue_ball_qpos + 6] = 0.0
        qpos[world_id, object_ball_qpos + 0] = 0.0
        qpos[world_id, object_ball_qpos + 1] = 0.0
        qpos[world_id, object_ball_qpos + 2] = 6.0
        qpos[world_id, object_ball_qpos + 3] = 1.0
        qpos[world_id, object_ball_qpos + 4] = 0.0
        qpos[world_id, object_ball_qpos + 5] = 0.0
        qpos[world_id, object_ball_qpos + 6] = 0.0
        for offset in range(6):
            qvel[world_id, cue_dof + offset] = 0.0
            qvel[world_id, cue_ball_dof + offset] = 0.0
            qvel[world_id, object_ball_dof + offset] = 0.0
        return

    direction_x = shot_directions[world_id, 0]
    direction_y = shot_directions[world_id, 1]
    speed = cue_speeds[world_id]
    elapsed = float(shot_step[world_id]) * timestep
    stroke_duration = (CUE_START_BACKOFF + CUE_FOLLOW_THROUGH) / speed

    if elapsed < stroke_duration:
        cue_offset = (
            BALL_RADIUS + CUE_START_BACKOFF + CUE_TIP_LOCAL_X
            - speed * elapsed
        )
        qpos[world_id, cue_qpos + 0] = (
            cue_positions[world_id, 0] - direction_x * cue_offset
        )
        qpos[world_id, cue_qpos + 1] = (
            cue_positions[world_id, 1] - direction_y * cue_offset
        )
        qpos[world_id, cue_qpos + 2] = BALL_CENTER_Z
        half_yaw = 0.5 * wp.atan2(direction_y, direction_x)
        qpos[world_id, cue_qpos + 3] = wp.cos(half_yaw)
        qpos[world_id, cue_qpos + 4] = 0.0
        qpos[world_id, cue_qpos + 5] = 0.0
        qpos[world_id, cue_qpos + 6] = wp.sin(half_yaw)
        qvel[world_id, cue_dof + 0] = direction_x * speed
        qvel[world_id, cue_dof + 1] = direction_y * speed
        qvel[world_id, cue_dof + 2] = 0.0
        qvel[world_id, cue_dof + 3] = 0.0
        qvel[world_id, cue_dof + 4] = 0.0
        qvel[world_id, cue_dof + 5] = 0.0
    else:
        qpos[world_id, cue_qpos + 0] = 0.0
        qpos[world_id, cue_qpos + 1] = 0.0
        qpos[world_id, cue_qpos + 2] = 6.0
        qpos[world_id, cue_qpos + 3] = 1.0
        qpos[world_id, cue_qpos + 4] = 0.0
        qpos[world_id, cue_qpos + 5] = 0.0
        qpos[world_id, cue_qpos + 6] = 0.0
        for offset in range(6):
            qvel[world_id, cue_dof + offset] = 0.0

    if object_pocket[world_id] >= 0:
        qpos[world_id, object_ball_qpos + 0] = 0.0
        qpos[world_id, object_ball_qpos + 1] = 0.0
        qpos[world_id, object_ball_qpos + 2] = 4.0
        qpos[world_id, object_ball_qpos + 3] = 1.0
        qpos[world_id, object_ball_qpos + 4] = 0.0
        qpos[world_id, object_ball_qpos + 5] = 0.0
        qpos[world_id, object_ball_qpos + 6] = 0.0
        for offset in range(6):
            qvel[world_id, object_ball_dof + offset] = 0.0


@wp.kernel
def _scan_contacts(
    nacon: wp.array(dtype=int),
    contact_world: wp.array(dtype=int),
    contact_type: wp.array(dtype=int),
    contact_geom: wp.array(dtype=wp.vec2i),
    shot_step: wp.array(dtype=int),
    qpos: wp.array2d(dtype=float),
    pocket_positions: wp.array(dtype=wp.vec2),
    cushion_mask: wp.array(dtype=int),
    object_pocket: wp.array(dtype=int),
    done: wp.array(dtype=int),
    first_ball_step: wp.array(dtype=int),
    first_cushion_step: wp.array(dtype=int),
    first_object_cushion_step: wp.array(dtype=int),
    cue_ball_geom: int,
    object_ball_geom: int,
    cue_ball_qpos: int,
    object_ball_qpos: int,
    ngeom: int,
) -> None:
    """Reduce current contacts into first-contact event steps on the GPU."""

    contact_id = wp.tid()
    if contact_id >= nacon[0]:
        return
    if (contact_type[contact_id] & 1) == 0:
        return
    world_id = contact_world[contact_id]
    if done[world_id] != 0:
        return
    geom_pair = contact_geom[contact_id]
    geom_0 = geom_pair[0]
    geom_1 = geom_pair[1]
    step_index = shot_step[world_id] + 1

    is_ball_ball = (
        (geom_0 == cue_ball_geom and geom_1 == object_ball_geom)
        or (geom_1 == cue_ball_geom and geom_0 == object_ball_geom)
    )
    if is_ball_ball:
        wp.atomic_min(first_ball_step, world_id, step_index)
        return

    ball_geom = -1
    table_geom = -1
    if geom_0 == cue_ball_geom or geom_0 == object_ball_geom:
        ball_geom = geom_0
        table_geom = geom_1
    elif geom_1 == cue_ball_geom or geom_1 == object_ball_geom:
        ball_geom = geom_1
        table_geom = geom_0
    if table_geom < 0 or table_geom >= ngeom:
        return
    if ball_geom < 0 or cushion_mask[table_geom] == 0:
        return
    if object_pocket[world_id] >= 0:
        return

    ball_qpos = cue_ball_qpos
    if ball_geom == object_ball_geom:
        ball_qpos = object_ball_qpos
    ball_x = qpos[world_id, ball_qpos + 0]
    ball_y = qpos[world_id, ball_qpos + 1]
    near_pocket_jaw = False
    for pocket_id in range(6):
        pocket_xy = pocket_positions[pocket_id]
        delta_x = ball_x - pocket_xy[0]
        delta_y = ball_y - pocket_xy[1]
        if delta_x * delta_x + delta_y * delta_y < 0.15 * 0.15:
            near_pocket_jaw = True
    if near_pocket_jaw:
        return
    if ball_geom == cue_ball_geom:
        wp.atomic_min(first_cushion_step, world_id, step_index)
    elif ball_geom == object_ball_geom:
        wp.atomic_min(first_object_cushion_step, world_id, step_index)


@wp.kernel
def _update_terminal_state(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    shot_step: wp.array(dtype=int),
    overflow: wp.array(dtype=int),
    shot_speeds: wp.array(dtype=float),
    target_pockets: wp.array(dtype=int),
    pocket_positions: wp.array(dtype=wp.vec2),
    pocket_radii: wp.array(dtype=float),
    cue_ball_qpos: int,
    cue_ball_dof: int,
    object_ball_qpos: int,
    object_ball_dof: int,
    nq: int,
    nv: int,
    timestep: float,
    stop_speed: float,
    stop_hold_steps_required: int,
    max_steps: int,
    object_pocket: wp.array(dtype=int),
    cue_pocket: wp.array(dtype=int),
    minimum_object_distance: wp.array(dtype=float),
    stop_hold_steps: wp.array(dtype=int),
    stopped: wp.array(dtype=int),
    timed_out: wp.array(dtype=int),
    numerical_failure: wp.array(dtype=int),
    overflow_seen: wp.array(dtype=int),
    done: wp.array(dtype=int),
    done_count: wp.array(dtype=int),
    final_cue_position: wp.array(dtype=wp.vec3),
    final_object_position: wp.array(dtype=wp.vec3),
    elapsed_time: wp.array(dtype=float),
) -> None:
    """Detect pockets, stopping, timeout, numerical failure, and overflow."""

    world_id = wp.tid()
    if done[world_id] != 0:
        return

    failure = False
    if overflow[world_id] != 0:
        overflow_seen[world_id] = overflow_seen[world_id] | overflow[world_id]
        failure = True
    for qpos_id in range(nq):
        value = qpos[world_id, qpos_id]
        if value != value or wp.abs(value) > 25.0:
            failure = True
    for dof_id in range(nv):
        value = qvel[world_id, dof_id]
        if value != value or wp.abs(value) > 150.0:
            failure = True
    if failure:
        numerical_failure[world_id] = 1

    object_x = qpos[world_id, object_ball_qpos + 0]
    object_y = qpos[world_id, object_ball_qpos + 1]
    object_z = qpos[world_id, object_ball_qpos + 2]
    cue_x = qpos[world_id, cue_ball_qpos + 0]
    cue_y = qpos[world_id, cue_ball_qpos + 1]
    cue_z = qpos[world_id, cue_ball_qpos + 2]

    target_xy = pocket_positions[target_pockets[world_id]]
    object_delta_x = object_x - target_xy[0]
    object_delta_y = object_y - target_xy[1]
    object_distance = wp.sqrt(
        object_delta_x * object_delta_x + object_delta_y * object_delta_y
    )
    if object_pocket[world_id] < 0:
        minimum_object_distance[world_id] = wp.min(
            minimum_object_distance[world_id],
            object_distance,
        )
        if object_z < POCKET_ENTRY_Z:
            for pocket_id in range(6):
                pocket_xy = pocket_positions[pocket_id]
                delta_x = object_x - pocket_xy[0]
                delta_y = object_y - pocket_xy[1]
                radius = pocket_radii[pocket_id] + 0.03
                if delta_x * delta_x + delta_y * delta_y <= radius * radius:
                    object_pocket[world_id] = pocket_id

    if cue_pocket[world_id] < 0 and cue_z < POCKET_ENTRY_Z:
        for pocket_id in range(6):
            pocket_xy = pocket_positions[pocket_id]
            delta_x = cue_x - pocket_xy[0]
            delta_y = cue_y - pocket_xy[1]
            radius = pocket_radii[pocket_id] + 0.03
            if delta_x * delta_x + delta_y * delta_y <= radius * radius:
                cue_pocket[world_id] = pocket_id

    current_step = shot_step[world_id]
    elapsed = float(current_step + 1) * timestep
    stroke_duration = (
        CUE_START_BACKOFF + CUE_FOLLOW_THROUGH
    ) / shot_speeds[world_id]
    if (
        numerical_failure[world_id] == 0
        and cue_pocket[world_id] < 0
        and float(current_step) * timestep >= stroke_duration
    ):
        cue_linear_squared = 0.0
        cue_angular_squared = 0.0
        object_linear_squared = 0.0
        object_angular_squared = 0.0
        for axis in range(3):
            cue_linear = qvel[world_id, cue_ball_dof + axis]
            cue_angular = qvel[world_id, cue_ball_dof + 3 + axis]
            object_linear = qvel[world_id, object_ball_dof + axis]
            object_angular = qvel[world_id, object_ball_dof + 3 + axis]
            cue_linear_squared += cue_linear * cue_linear
            cue_angular_squared += cue_angular * cue_angular
            object_linear_squared += object_linear * object_linear
            object_angular_squared += object_angular * object_angular
        required_stopped = (
            wp.sqrt(cue_linear_squared) < stop_speed
            and BALL_RADIUS * wp.sqrt(cue_angular_squared) < stop_speed
        )
        if object_pocket[world_id] < 0:
            required_stopped = (
                required_stopped
                and wp.sqrt(object_linear_squared) < stop_speed
                and BALL_RADIUS * wp.sqrt(object_angular_squared) < stop_speed
            )
        if required_stopped:
            stop_hold_steps[world_id] = stop_hold_steps[world_id] + 1
            if stop_hold_steps[world_id] >= stop_hold_steps_required:
                stopped[world_id] = 1
        else:
            stop_hold_steps[world_id] = 0

    should_finish = (
        numerical_failure[world_id] != 0
        or cue_pocket[world_id] >= 0
        or stopped[world_id] != 0
    )
    if not should_finish and current_step + 1 >= max_steps:
        timed_out[world_id] = 1
        should_finish = True
    if should_finish:
        done[world_id] = 1
        wp.atomic_add(done_count, 0, 1)
        final_cue_position[world_id] = wp.vec3(cue_x, cue_y, cue_z)
        final_object_position[world_id] = wp.vec3(
            object_x,
            object_y,
            object_z,
        )
        elapsed_time[world_id] = elapsed
    shot_step[world_id] = current_step + 1


class MJWarpMidLevelVecEnv(VecEnv):
    """SB3 vector environment backed by batched MuJoCo Warp CUDA worlds."""

    render_mode = None

    def __init__(
        self,
        task_dataset: TwoBallTaskDataset | Path,
        model_path: Path = DEFAULT_MIDLEVEL_MODEL,
        *,
        num_envs: int = 1024,
        seed: int = 0,
        device: str = "cuda:0",
        chunk_steps: int = 16,
        check_interval_steps: int = 2048,
        nconmax: int = MUJOCO_WARP_NCONMAX,
        njmax: int = MUJOCO_WARP_NJMAX,
        max_time: float = MAX_SHOT_TIME,
        stop_speed: float = STOP_SPEED_THRESHOLD,
        stop_hold_time: float = STOP_HOLD_TIME,
        validate_task_execution: bool = True,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if chunk_steps <= 0 or check_interval_steps <= 0:
            raise ValueError("MJWarp chunk and check intervals must be positive.")
        if check_interval_steps < chunk_steps:
            raise ValueError("check_interval_steps must be at least chunk_steps.")
        if nconmax <= 0 or njmax <= 0:
            raise ValueError("MJWarp contact and constraint capacities must be positive.")
        if nconmax < MUJOCO_WARP_NCONMAX or njmax < MUJOCO_WARP_NJMAX:
            raise ValueError(
                "MJWarp capacity is below the validated safe floor: "
                f"nconmax must be >= {MUJOCO_WARP_NCONMAX} and "
                f"njmax must be >= {MUJOCO_WARP_NJMAX}. Undersized buffers "
                "can fault inside CUDA before overflow flags become readable."
            )
        if max_time <= 0.0 or stop_speed <= 0.0 or stop_hold_time <= 0.0:
            raise ValueError("Shot timing and stopping thresholds must be positive.")

        self.model_path = Path(model_path).resolve()
        # Needed while CUDA arrays and the step graph are built; VecEnv sets
        # the same value again after backend construction is complete.
        self.num_envs = int(num_envs)
        reference_simulator = TwoBallShotSimulator(
            self.model_path,
            max_time=max_time,
            stop_speed=stop_speed,
            stop_hold_time=stop_hold_time,
        )
        self.xml_hash = reference_simulator.xml_hash
        self.model_hash = reference_simulator.model_hash
        if isinstance(task_dataset, TwoBallTaskDataset):
            self.tasks = task_dataset
            if (
                self.tasks.xml_hash != reference_simulator.xml_hash
                or self.tasks.model_hash != reference_simulator.model_hash
            ):
                raise ValueError("In-memory task dataset does not match the active base model.")
        else:
            self.tasks = TwoBallTaskDataset.load(
                task_dataset,
                simulator=reference_simulator,
                validate_model=True,
            )

        self.model = reference_simulator.model
        prepare_mujoco_warp_model(self.model)
        self.backend_hash = mujoco_warp_backend_sha256(
            self.model,
            reference_simulator.xml_hash,
        )
        self.timestep = float(self.model.opt.timestep)
        self.max_time = float(max_time)
        self.max_steps = int(np.ceil(self.max_time / self.timestep))
        self.stop_speed = float(stop_speed)
        self.stop_hold_time = float(stop_hold_time)
        self.validate_task_execution = bool(validate_task_execution)
        self._validate_task_dataset(self.tasks)
        self.chunk_steps = int(chunk_steps)
        self.check_interval_steps = int(check_interval_steps)
        self.nconmax = int(nconmax)
        self.njmax = int(njmax)
        self.backend = "mujoco_warp"
        self.closed = False
        self.last_rollout_wall_seconds = 0.0
        self.last_rollout_steps = 0
        self.last_world_steps_per_second = 0.0
        self.last_terminal_rewards = np.zeros(num_envs, dtype=np.float32)
        self._pending_actions: np.ndarray | None = None
        self._awaiting_initial_reset = True
        self._task_indices = np.zeros(num_envs, dtype=np.int64)
        self._observations = np.zeros((num_envs, 8), dtype=np.float32)
        self._rngs = [
            np.random.default_rng(int(seed) + env_id) for env_id in range(num_envs)
        ]

        self._cue_qpos, self._cue_dof = self._joint_addresses("cue_free")
        self._cue_ball_qpos, self._cue_ball_dof = self._joint_addresses(
            "cue_ball_free"
        )
        self._object_ball_qpos, self._object_ball_dof = self._joint_addresses(
            "object_ball_0_free"
        )
        self._cue_ball_geom = self._required_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            "cue_ball_geom",
        )
        self._object_ball_geom = self._required_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            "object_ball_0_geom",
        )
        pocket_positions, pocket_radii = self._pocket_geometry(reference_simulator)
        self._pocket_positions_host = pocket_positions.copy()
        cushion_mask = np.zeros(self.model.ngeom, dtype=np.int32)
        cushion_mask[list(cushion_geom_ids(self.model))] = 1

        warnings.filterwarnings(
            "ignore",
            message=r"geom .* friction.*MJ_MINMU.*",
            category=UserWarning,
        )
        wp.config.log_level = wp.LOG_WARNING
        wp.init()
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise RuntimeError("MJWarp mid-level execution requires a CUDA device.")

        initial_data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, initial_data)
        with wp.ScopedDevice(self.device):
            self.warp_model = mjw.put_model(self.model)
            device_tolerance = float(
                self.warp_model.opt.tolerance.numpy()[0]
            )
            # MJWarp is float32 and intentionally floors CPU MuJoCo's much
            # tighter float64 tolerance. Zero-tolerance fixed iteration is not
            # safe in its incremental Newton path, so verify the calibrated
            # positive value instead of silently assuming CPU semantics.
            expected_device_tolerance = max(
                float(self.model.opt.tolerance),
                1.0e-6,
            )
            if not np.isclose(
                device_tolerance,
                expected_device_tolerance,
                rtol=1.0e-6,
                atol=0.0,
            ):
                raise RuntimeError(
                    "MJWarp solver tolerance calibration changed during model "
                    f"import: expected={expected_device_tolerance:.9g}, "
                    f"device={device_tolerance:.9g}."
                )
            self.warp_data = mjw.put_data(
                self.model,
                initial_data,
                nworld=num_envs,
                nconmax=self.nconmax,
                njmax=self.njmax,
            )
            self._cue_positions_device = wp.zeros(
                (num_envs, 2), dtype=float, device=self.device
            )
            self._object_positions_device = wp.zeros(
                (num_envs, 2), dtype=float, device=self.device
            )
            self._shot_directions_device = wp.zeros(
                (num_envs, 2), dtype=float, device=self.device
            )
            self._cue_speeds_device = wp.ones(
                num_envs, dtype=float, device=self.device
            )
            self._target_pockets_device = wp.zeros(
                num_envs, dtype=int, device=self.device
            )
            self._pocket_positions_device = wp.array(
                pocket_positions.astype(np.float32),
                dtype=wp.vec2,
                device=self.device,
            )
            self._pocket_radii_device = wp.array(
                pocket_radii.astype(np.float32),
                dtype=float,
                device=self.device,
            )
            self._cushion_mask_device = wp.array(
                cushion_mask,
                dtype=int,
                device=self.device,
            )
            self._first_ball_step = wp.empty(
                num_envs, dtype=int, device=self.device
            )
            self._first_cushion_step = wp.empty(
                num_envs, dtype=int, device=self.device
            )
            self._first_object_cushion_step = wp.empty(
                num_envs, dtype=int, device=self.device
            )
            self._shot_step = wp.empty(
                num_envs, dtype=int, device=self.device
            )
            self._object_pocket = wp.empty(
                num_envs, dtype=int, device=self.device
            )
            self._cue_pocket = wp.empty(num_envs, dtype=int, device=self.device)
            self._minimum_object_distance = wp.empty(
                num_envs, dtype=float, device=self.device
            )
            self._stop_hold_steps = wp.empty(
                num_envs, dtype=int, device=self.device
            )
            self._stopped = wp.empty(num_envs, dtype=int, device=self.device)
            self._timed_out = wp.empty(num_envs, dtype=int, device=self.device)
            self._numerical_failure = wp.empty(
                num_envs, dtype=int, device=self.device
            )
            self._overflow_seen = wp.empty(
                num_envs, dtype=int, device=self.device
            )
            self._done = wp.empty(num_envs, dtype=int, device=self.device)
            self._done_count = wp.zeros(1, dtype=int, device=self.device)
            self._final_cue_position = wp.empty(
                num_envs, dtype=wp.vec3, device=self.device
            )
            self._final_object_position = wp.empty(
                num_envs, dtype=wp.vec3, device=self.device
            )
            self._elapsed_time = wp.empty(
                num_envs, dtype=float, device=self.device
            )
            bootstrap_directions = np.tile(
                self.tasks.generated_directions[0].astype(np.float32),
                (num_envs, 1),
            )
            bootstrap_speeds = np.full(
                num_envs,
                self.tasks.generated_speeds[0],
                dtype=np.float32,
            )
            self._upload_and_initialize(
                bootstrap_directions,
                bootstrap_speeds,
            )
            wp.synchronize()
            self._capture_step_graph()
            mjw.reset_data(self.warp_model, self.warp_data)
            wp.synchronize()

        observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(8,),
            dtype=np.float32,
        )
        action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(2,),
            dtype=np.float32,
        )
        super().__init__(num_envs, observation_space, action_space)

    def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Required MuJoCo name is missing: {name}")
        return object_id

    def _validate_task_dataset(
        self,
        dataset: TwoBallTaskDataset,
    ) -> None:
        if (
            dataset.xml_hash != self.xml_hash
            or dataset.model_hash != self.model_hash
        ):
            raise ValueError("Task dataset does not match the active base model.")
        if dataset.physics_backend != MUJOCO_WARP_PHYSICS_BACKEND:
            raise ValueError(
                "MJWarp execution requires tasks generated and replayed by MJWarp; "
                f"received {dataset.physics_backend!r}."
            )
        if dataset.backend_hash != self.backend_hash:
            raise ValueError(
                "Task dataset MJWarp backend hash does not match the active "
                "calibrated model and physics implementation."
            )
        if self.validate_task_execution and (
            dataset.execution_max_time != self.max_time
            or dataset.stop_speed != self.stop_speed
            or dataset.stop_hold_time != self.stop_hold_time
        ):
            raise ValueError(
                "Task dataset shot timing/stopping settings do not match the "
                "MJWarp environment."
            )

    def replace_task_dataset(self, dataset: TwoBallTaskDataset) -> None:
        """Replace generation/replay tasks without rebuilding GPU state."""

        if self._pending_actions is not None:
            raise RuntimeError("Cannot replace tasks while a vector step is pending.")
        if len(dataset) < self.num_envs:
            raise ValueError(
                "Replacement dataset must contain at least one task per MJWarp world."
            )
        self._validate_task_dataset(dataset)
        self.tasks = dataset
        self._awaiting_initial_reset = True

    def _joint_addresses(self, name: str) -> tuple[int, int]:
        joint_id = self._required_id(mujoco.mjtObj.mjOBJ_JOINT, name)
        return (
            int(self.model.jnt_qposadr[joint_id]),
            int(self.model.jnt_dofadr[joint_id]),
        )

    def _pocket_geometry(
        self,
        simulator: TwoBallShotSimulator,
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = np.stack(
            [simulator.pocket_positions[name] for name in POCKET_NAMES]
        ).astype(np.float64)
        radii: list[float] = []
        for name in POCKET_NAMES:
            site_id = self._required_id(mujoco.mjtObj.mjOBJ_SITE, name)
            radii.append(float(self.model.site_size[site_id, 0]))
        return positions, np.asarray(radii, dtype=np.float64)

    def _capture_step_graph(self) -> None:
        stop_hold_steps_required = max(
            1,
            int(np.ceil(self.stop_hold_time / self.timestep)),
        )
        with wp.ScopedCapture() as capture:
            for _ in range(self.chunk_steps):
                wp.launch(
                    _drive_and_park,
                    dim=self.num_envs,
                    inputs=[
                        self.warp_data.qpos,
                        self.warp_data.qvel,
                        self._shot_step,
                        self._shot_directions_device,
                        self._cue_speeds_device,
                        self._cue_positions_device,
                        self._object_pocket,
                        self._done,
                        self._cue_qpos,
                        self._cue_dof,
                        self._cue_ball_qpos,
                        self._cue_ball_dof,
                        self._object_ball_qpos,
                        self._object_ball_dof,
                        self.timestep,
                    ],
                )
                mjw.step(self.warp_model, self.warp_data)
                wp.launch(
                    _scan_contacts,
                    dim=self.warp_data.naconmax,
                    inputs=[
                        self.warp_data.nacon,
                        self.warp_data.contact.worldid,
                        self.warp_data.contact.type,
                        self.warp_data.contact.geom,
                        self._shot_step,
                        self.warp_data.qpos,
                        self._pocket_positions_device,
                        self._cushion_mask_device,
                        self._object_pocket,
                        self._done,
                        self._first_ball_step,
                        self._first_cushion_step,
                        self._first_object_cushion_step,
                        self._cue_ball_geom,
                        self._object_ball_geom,
                        self._cue_ball_qpos,
                        self._object_ball_qpos,
                        self.model.ngeom,
                    ],
                )
                wp.launch(
                    _update_terminal_state,
                    dim=self.num_envs,
                    inputs=[
                        self.warp_data.qpos,
                        self.warp_data.qvel,
                        self._shot_step,
                        self.warp_data.overflow,
                        self._cue_speeds_device,
                        self._target_pockets_device,
                        self._pocket_positions_device,
                        self._pocket_radii_device,
                        self._cue_ball_qpos,
                        self._cue_ball_dof,
                        self._object_ball_qpos,
                        self._object_ball_dof,
                        self.model.nq,
                        self.model.nv,
                        self.timestep,
                        self.stop_speed,
                        stop_hold_steps_required,
                        self.max_steps,
                        self._object_pocket,
                        self._cue_pocket,
                        self._minimum_object_distance,
                        self._stop_hold_steps,
                        self._stopped,
                        self._timed_out,
                        self._numerical_failure,
                        self._overflow_seen,
                        self._done,
                        self._done_count,
                        self._final_cue_position,
                        self._final_object_position,
                        self._elapsed_time,
                    ],
                )
        self._step_graph = capture.graph

    def _select_tasks(self) -> None:
        for env_id, rng in enumerate(self._rngs):
            option = self._options[env_id]
            if "task_index" in option:
                task_index = int(option["task_index"])
                if not 0 <= task_index < len(self.tasks):
                    raise IndexError(
                        f"task_index {task_index} is outside the dataset."
                    )
            else:
                task_index = int(rng.integers(0, len(self.tasks)))
            self._task_indices[env_id] = task_index
            task = self.tasks[task_index]
            self._observations[env_id] = task_observation(task)
            self.reset_infos[env_id] = {
                "task_index": task_index,
                "pocket_name": task.pocket_name,
                "candidate_seed": task.candidate_seed,
                "generated_speed": task.generated_speed,
                "physics_backend": self.backend,
            }

    def reset(self) -> np.ndarray:
        if self.closed:
            raise RuntimeError("Cannot reset a closed MJWarp vector environment.")
        if any(seed is not None for seed in self._seeds):
            self._rngs = [
                np.random.default_rng(seed)
                if seed is not None
                else np.random.default_rng()
                for seed in self._seeds
            ]
        self._select_tasks()
        self._pending_actions = None
        self._awaiting_initial_reset = False
        self._reset_seeds()
        self._reset_options()
        return self._observations.copy()

    def step_async(self, actions: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("Cannot step a closed MJWarp vector environment.")
        if self._awaiting_initial_reset:
            raise RuntimeError("reset() must be called before the first vector step.")
        if self._pending_actions is not None:
            raise RuntimeError("step_async() called while another step is pending.")
        values = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.num_envs, 2)
        if values.shape != expected_shape:
            raise ValueError(
                f"Expected actions with shape {expected_shape}, got {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Mid-level actions must be finite.")
        self._pending_actions = np.clip(values, -1.0, 1.0)

    def _decode_actions(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        task_ids = self._task_indices
        cue_positions = self.tasks.cue_positions[task_ids]
        object_positions = self.tasks.object_positions[task_ids]
        pocket_indices = self.tasks.pocket_indices[task_ids].astype(np.int64)
        pocket_positions = self._pocket_positions_host[pocket_indices]
        object_to_pocket = pocket_positions - object_positions
        object_to_pocket /= np.linalg.norm(
            object_to_pocket,
            axis=1,
            keepdims=True,
        )
        ghost_positions = object_positions - 2.0 * BALL_RADIUS * object_to_pocket
        baseline = ghost_positions - cue_positions
        baseline /= np.linalg.norm(baseline, axis=1, keepdims=True)
        angles = actions[:, 0].astype(np.float64) * MAX_ANGLE_RESIDUAL
        cosine = np.cos(angles)
        sine = np.sin(angles)
        directions = np.column_stack(
            (
                cosine * baseline[:, 0] - sine * baseline[:, 1],
                sine * baseline[:, 0] + cosine * baseline[:, 1],
            )
        )
        speed_fraction = 0.5 * (actions[:, 1].astype(np.float64) + 1.0)
        speeds = MIN_CUE_SPEED + speed_fraction * (
            MAX_CUE_SPEED - MIN_CUE_SPEED
        )
        return (
            directions.astype(np.float32),
            speeds.astype(np.float32),
        )

    def _upload_and_initialize(
        self,
        directions: np.ndarray,
        speeds: np.ndarray,
    ) -> None:
        task_ids = self._task_indices
        cue_positions = self.tasks.cue_positions[task_ids].astype(np.float32)
        object_positions = self.tasks.object_positions[task_ids].astype(np.float32)
        target_pockets = self.tasks.pocket_indices[task_ids].astype(np.int32)
        self._cue_positions_device.assign(cue_positions)
        self._object_positions_device.assign(object_positions)
        self._shot_directions_device.assign(directions)
        self._cue_speeds_device.assign(speeds)
        self._target_pockets_device.assign(target_pockets)
        mjw.reset_data(self.warp_model, self.warp_data)
        wp.launch(
            _initialize_shots,
            dim=self.num_envs,
            inputs=[
                self.warp_data.qpos,
                self.warp_data.qvel,
                self._cue_positions_device,
                self._object_positions_device,
                self._target_pockets_device,
                self._pocket_positions_device,
                self._cue_qpos,
                self._cue_dof,
                self._cue_ball_qpos,
                self._cue_ball_dof,
                self._object_ball_qpos,
                self._object_ball_dof,
                self.model.nv,
                self._first_ball_step,
                self._first_cushion_step,
                self._first_object_cushion_step,
                self._shot_step,
                self._object_pocket,
                self._cue_pocket,
                self._minimum_object_distance,
                self._stop_hold_steps,
                self._stopped,
                self._timed_out,
                self._numerical_failure,
                self._overflow_seen,
                self._done,
                self._done_count,
                self._final_cue_position,
                self._final_object_position,
                self._elapsed_time,
            ],
        )

    def _execute_rollout(
        self,
        directions: np.ndarray,
        speeds: np.ndarray,
    ) -> None:
        with wp.ScopedDevice(self.device):
            self._upload_and_initialize(directions, speeds)
            wp.synchronize()
            start = time.perf_counter()
            maximum_chunks = int(
                np.ceil(self.max_steps / self.chunk_steps)
            ) + 1
            chunks_per_check = max(
                1,
                self.check_interval_steps // self.chunk_steps,
            )
            executed_chunks = 0
            done_count = 0
            while executed_chunks < maximum_chunks and done_count < self.num_envs:
                launch_count = min(
                    chunks_per_check,
                    maximum_chunks - executed_chunks,
                )
                for _ in range(launch_count):
                    wp.capture_launch(self._step_graph)
                executed_chunks += launch_count
                done_count = int(self._done_count.numpy()[0])
            wp.synchronize()
            self.last_rollout_wall_seconds = time.perf_counter() - start
            self.last_rollout_steps = executed_chunks * self.chunk_steps
            world_steps = self.last_rollout_steps * self.num_envs
            self.last_world_steps_per_second = (
                world_steps / self.last_rollout_wall_seconds
                if self.last_rollout_wall_seconds > 0.0
                else 0.0
            )
            if done_count != self.num_envs:
                raise RuntimeError(
                    "MJWarp rollout reached its host-side step bound before "
                    f"all worlds terminated ({done_count}/{self.num_envs})."
                )
            overflow_seen = np.asarray(
                self._overflow_seen.numpy(),
                dtype=np.int64,
            )
            if np.any(overflow_seen):
                world_ids = np.flatnonzero(overflow_seen)
                details = ", ".join(
                    f"{int(world_id)}:{int(overflow_seen[world_id])}"
                    for world_id in world_ids[:10]
                )
                raise RuntimeError(
                    "MJWarp capacity overflow occurred during the shot; "
                    f"world:flags={details}"
                )
            assert_mujoco_warp_capacity(
                self.warp_data,
                context="batched mid-level rollout",
            )

    def _terminal_outputs(
        self,
        actions: np.ndarray,
        directions: np.ndarray,
        speeds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        first_ball_steps = np.asarray(self._first_ball_step.numpy(), dtype=np.int64)
        first_cushion_steps = np.asarray(
            self._first_cushion_step.numpy(),
            dtype=np.int64,
        )
        first_object_cushion_steps = np.asarray(
            self._first_object_cushion_step.numpy(),
            dtype=np.int64,
        )
        object_pockets = np.asarray(self._object_pocket.numpy(), dtype=np.int64)
        cue_pockets = np.asarray(self._cue_pocket.numpy(), dtype=np.int64)
        minimum_distances = np.asarray(
            self._minimum_object_distance.numpy(),
            dtype=np.float64,
        )
        stopped = np.asarray(self._stopped.numpy(), dtype=np.bool_)
        timed_out = np.asarray(self._timed_out.numpy(), dtype=np.bool_)
        numerical_failure = np.asarray(
            self._numerical_failure.numpy(),
            dtype=np.bool_,
        )
        elapsed_times = np.asarray(self._elapsed_time.numpy(), dtype=np.float64)
        final_cue_positions = np.asarray(
            self._final_cue_position.numpy(),
            dtype=np.float64,
        )
        final_object_positions = np.asarray(
            self._final_object_position.numpy(),
            dtype=np.float64,
        )

        rewards = np.empty(self.num_envs, dtype=np.float32)
        dones = np.ones(self.num_envs, dtype=np.bool_)
        infos: list[dict[str, Any]] = []
        terminal_observations = self._observations.copy()
        for env_id in range(self.num_envs):
            task_index = int(self._task_indices[env_id])
            task = self.tasks[task_index]
            first_ball_step = int(first_ball_steps[env_id])
            first_cushion_step = int(first_cushion_steps[env_id])
            first_object_cushion_step = int(
                first_object_cushion_steps[env_id]
            )
            object_pocket_index = int(object_pockets[env_id])
            cue_pocket_index = int(cue_pockets[env_id])
            object_pocket = (
                POCKET_NAMES[object_pocket_index]
                if object_pocket_index >= 0
                else None
            )
            cue_pocket = (
                POCKET_NAMES[cue_pocket_index]
                if cue_pocket_index >= 0
                else None
            )
            cue_final = final_cue_positions[env_id]
            object_final = final_object_positions[env_id]
            min_distance = float(minimum_distances[env_id])
            if numerical_failure[env_id]:
                cue_final = np.nan_to_num(
                    cue_final,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                object_final = np.nan_to_num(
                    object_final,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                if not np.isfinite(min_distance):
                    min_distance = float(
                        np.linalg.norm(task.object_position - task.pocket_position)
                    )
            result = TwoBallShotResult(
                target_pocket=task.pocket_name,
                shot_direction=np.array(
                    [directions[env_id, 0], directions[env_id, 1], 0.0],
                    dtype=np.float64,
                ),
                cue_speed=float(speeds[env_id]),
                elapsed_time=float(elapsed_times[env_id]),
                cue_ball_final_position=cue_final,
                object_ball_final_position=object_final,
                first_ball_contact_time=(
                    first_ball_step * self.timestep
                    if first_ball_step < _NO_EVENT_STEP
                    else None
                ),
                first_cushion_contact_time=(
                    min(
                        first_cushion_step,
                        first_object_cushion_step,
                    )
                    * self.timestep
                    if min(
                        first_cushion_step,
                        first_object_cushion_step,
                    )
                    < _NO_EVENT_STEP
                    else None
                ),
                object_pocket=object_pocket,
                cue_pocket=cue_pocket,
                min_object_pocket_distance=min_distance,
                initial_object_pocket_distance=float(
                    np.linalg.norm(task.object_position - task.pocket_position)
                ),
                stopped=bool(stopped[env_id]),
                timed_out=bool(timed_out[env_id]),
                numerical_failure=bool(numerical_failure[env_id]),
                cushion_before_object=bool(
                    first_cushion_step < first_ball_step
                ),
                object_cushion_before_pocket=bool(
                    first_object_cushion_step < _NO_EVENT_STEP
                ),
                any_cushion_contact=bool(
                    first_cushion_step < _NO_EVENT_STEP
                    or first_object_cushion_step < _NO_EVENT_STEP
                ),
                contact_events=(),
            )
            reward = compute_terminal_reward(
                result,
                task.target_stop_position,
            )
            rewards[env_id] = reward.total
            infos.append(
                {
                    "task_index": task_index,
                    "candidate_seed": task.candidate_seed,
                    "pocket_name": task.pocket_name,
                    "action": actions[env_id].copy(),
                    "shot_direction": directions[env_id].copy(),
                    "cue_speed": float(speeds[env_id]),
                    "correct_pot": result.correct_pot,
                    "legal_first_contact": result.legal_first_contact,
                    "cue_scratch": result.cue_scratch,
                    "wrong_pocket": result.wrong_pocket,
                    "cushion_before_object": result.cushion_before_object,
                    "object_cushion_before_pocket": (
                        result.object_cushion_before_pocket
                    ),
                    "any_cushion_contact": result.any_cushion_contact,
                    "timed_out": result.timed_out,
                    "numerical_failure": result.numerical_failure,
                    "stopped": result.stopped,
                    "object_pocket": result.object_pocket,
                    "cue_pocket": result.cue_pocket,
                    "elapsed_time": result.elapsed_time,
                    "minimum_object_pocket_distance": (
                        result.min_object_pocket_distance
                    ),
                    "cue_ball_final_position": cue_final.copy(),
                    "object_ball_final_position": object_final.copy(),
                    "backend_hash": self.backend_hash,
                    "physics_backend": self.backend,
                    "terminal_observation": terminal_observations[env_id],
                    # Every shot is a terminal contextual-bandit outcome.
                    # Marking a timeout as TimeLimit truncation makes SB3 add
                    # gamma * V(s_terminal), corrupting this one-step target.
                    "TimeLimit.truncated": False,
                    "mujoco_warp_world_steps_per_second": (
                        self.last_world_steps_per_second
                    ),
                    **reward.as_info(),
                }
            )
        return rewards, dones, infos

    def step_wait(self) -> VecEnvStepReturn:
        if self._pending_actions is None:
            raise RuntimeError("step_wait() called without step_async().")
        actions = self._pending_actions
        self._pending_actions = None
        directions, speeds = self._decode_actions(actions)
        self._execute_rollout(directions, speeds)
        rewards, dones, infos = self._terminal_outputs(
            actions,
            directions,
            speeds,
        )
        self.last_terminal_rewards = rewards.copy()
        self._select_tasks()
        return self._observations.copy(), rewards, dones, infos

    def close(self) -> None:
        if self.closed:
            return
        with wp.ScopedDevice(self.device):
            wp.synchronize()
        self.closed = True
        self._pending_actions = None

    def get_attr(
        self,
        attr_name: str,
        indices: VecEnvIndices = None,
    ) -> list[Any]:
        if not hasattr(self, attr_name):
            raise AttributeError(attr_name)
        value = getattr(self, attr_name)
        return [value for _ in self._get_indices(indices)]

    def set_attr(
        self,
        attr_name: str,
        value: Any,
        indices: VecEnvIndices = None,
    ) -> None:
        selected = list(self._get_indices(indices))
        if len(selected) != self.num_envs:
            raise ValueError(
                "Per-world attributes are not supported by the batched MJWarp backend."
            )
        setattr(self, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        method = getattr(self, method_name)
        result = method(*method_args, **method_kwargs)
        return [result for _ in self._get_indices(indices)]

    def env_is_wrapped(
        self,
        wrapper_class: type[gym.Wrapper],
        indices: VecEnvIndices = None,
    ) -> list[bool]:
        return [False for _ in self._get_indices(indices)]

    def get_images(self) -> list[None]:
        return [None for _ in range(self.num_envs)]
