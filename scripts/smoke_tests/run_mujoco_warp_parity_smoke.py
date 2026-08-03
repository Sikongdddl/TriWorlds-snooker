"""Compare core snooker contacts between CPU MuJoCo and MuJoCo Warp."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
import warnings

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src.types import ContactType, MJ_MINMU
import numpy as np
import warp as wp

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.mujoco_warp_sdf import (  # noqa: E402
    MUJOCO_WARP_NCONMAX,
    MUJOCO_WARP_NJMAX,
    TRAPEZOID_MANIFOLD_CONTACTS,
    assert_mujoco_warp_capacity,
    calibrate_trapezoid_sdf_contact_damping,
    normalize_zero_friction_contact_dims,
    register_mujoco_billiards_sdf,
)
from snooker_env.table_geometry import (  # noqa: E402
    BALL_CENTER_Z,
    POCKET_ENTRY_Z,
    central_cloth_geom_id,
    nearest_cushion_geom_id,
)


@dataclass(frozen=True)
class Rollout:
    qpos: np.ndarray
    qvel: np.ndarray
    runtime: float
    pocket_crossing_time: float | None = None
    pocket_crossing_xy: np.ndarray | None = None


@dataclass(frozen=True)
class Scenario:
    name: str
    model: mujoco.MjModel
    data: mujoco.MjData
    duration: float
    cue_qpos: int
    cue_dof: int
    initial_speed: float
    object_qpos: int | None = None
    object_dof: int | None = None
    pocket_check: bool = False


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Missing {object_type.name}: {name}")
    return object_id


def _joint_addresses(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _only_collide_ids(model: mujoco.MjModel, geom_ids: tuple[int, ...]) -> None:
    active = frozenset(geom_ids)
    for geom_id in range(model.ngeom):
        if geom_id not in active:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0


def _park_cue(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    cue_qpos, cue_dof = _joint_addresses(model, "cue_free")
    data.qpos[cue_qpos:cue_qpos + 3] = (0.0, 0.0, 6.0)
    data.qpos[cue_qpos + 3:cue_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[cue_dof:cue_dof + 6] = 0.0


def _set_ball(
    data: mujoco.MjData,
    qpos_address: int,
    dof_address: int,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    data.qpos[qpos_address:qpos_address + 3] = position
    data.qpos[qpos_address + 3:qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[dof_address:dof_address + 3] = velocity
    data.qvel[dof_address + 3:dof_address + 6] = angular_velocity


def _load_model(model_path: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(model_path.resolve()))


def _ball_ball_scenario(model_path: Path) -> Scenario:
    model = _load_model(model_path)
    model.opt.gravity[:] = 0.0
    cue_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_ball_geom")
    object_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_ball_0_geom")
    _only_collide_ids(model, (cue_geom, object_geom))
    data = mujoco.MjData(model)
    _park_cue(model, data)
    cue_qpos, cue_dof = _joint_addresses(model, "cue_ball_free")
    object_qpos, object_dof = _joint_addresses(model, "object_ball_0_free")
    _set_ball(data, cue_qpos, cue_dof, (-0.10, 0.0, 1.0), (3.0, 0.0, 0.0))
    _set_ball(data, object_qpos, object_dof, (0.10, 0.0, 1.0), (0.0, 0.0, 0.0))
    mujoco.mj_forward(model, data)
    return Scenario(
        name="ball_ball",
        model=model,
        data=data,
        duration=0.20,
        cue_qpos=cue_qpos,
        cue_dof=cue_dof,
        object_qpos=object_qpos,
        object_dof=object_dof,
        initial_speed=3.0,
    )


def _cushion_scenario(model_path: Path) -> Scenario:
    model = _load_model(model_path)
    model.opt.gravity[:] = 0.0
    cue_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_ball_geom")
    cushion_geom = nearest_cushion_geom_id(
        model,
        np.array([0.6825, -0.663787, 1.085], dtype=np.float64),
    )
    _only_collide_ids(model, (cue_geom, cushion_geom))
    data = mujoco.MjData(model)
    _park_cue(model, data)
    cue_qpos, cue_dof = _joint_addresses(model, "cue_ball_free")
    object_qpos, object_dof = _joint_addresses(model, "object_ball_0_free")
    _set_ball(
        data,
        cue_qpos,
        cue_dof,
        (0.35, -0.40, BALL_CENTER_Z),
        (2.0, 0.0, 0.0),
    )
    _set_ball(data, object_qpos, object_dof, (0.0, 0.70, BALL_CENTER_Z), (0.0, 0.0, 0.0))
    mujoco.mj_forward(model, data)
    return Scenario(
        name="cushion",
        model=model,
        data=data,
        duration=0.30,
        cue_qpos=cue_qpos,
        cue_dof=cue_dof,
        initial_speed=2.0,
    )


def _trapezoid_cushion_scenario(model_path: Path) -> Scenario:
    model = _load_model(model_path)
    model.opt.gravity[:] = 0.0
    cue_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_ball_geom")
    target = np.array([0.70, -0.40], dtype=np.float64)
    candidates: list[int] = []
    for geom_id in range(model.ngeom):
        instance_id = int(model.geom_plugin[geom_id])
        if instance_id < 0:
            continue
        instance_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_PLUGIN,
            instance_id,
        )
        if instance_name is not None and instance_name.startswith("trapezoid"):
            candidates.append(geom_id)
    if not candidates:
        raise ValueError("No trapezoid SDF cushion geoms were found.")
    cushion_geom = min(
        candidates,
        key=lambda geom_id: float(np.linalg.norm(model.geom_pos[geom_id, :2] - target)),
    )
    _only_collide_ids(model, (cue_geom, cushion_geom))
    data = mujoco.MjData(model)
    _park_cue(model, data)
    cue_qpos, cue_dof = _joint_addresses(model, "cue_ball_free")
    object_qpos, object_dof = _joint_addresses(model, "object_ball_0_free")
    _set_ball(
        data,
        cue_qpos,
        cue_dof,
        (0.35, -0.40, BALL_CENTER_Z),
        (2.0, 0.0, 0.0),
    )
    _set_ball(data, object_qpos, object_dof, (0.0, 0.70, BALL_CENTER_Z), (0.0, 0.0, 0.0))
    mujoco.mj_forward(model, data)
    return Scenario(
        name="trapezoid_sdf_cushion",
        model=model,
        data=data,
        duration=0.30,
        cue_qpos=cue_qpos,
        cue_dof=cue_dof,
        initial_speed=2.0,
    )


def _rolling_scenario(model_path: Path) -> Scenario:
    model = _load_model(model_path)
    cue_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_ball_geom")
    _only_collide_ids(model, (cue_geom, central_cloth_geom_id(model)))
    data = mujoco.MjData(model)
    _park_cue(model, data)
    cue_qpos, cue_dof = _joint_addresses(model, "cue_ball_free")
    object_qpos, object_dof = _joint_addresses(model, "object_ball_0_free")
    radius = float(model.geom_size[cue_geom, 0])
    _set_ball(
        data,
        cue_qpos,
        cue_dof,
        (0.0, 0.0, BALL_CENTER_Z),
        (1.0, 0.0, 0.0),
        (0.0, 1.0 / radius, 0.0),
    )
    _set_ball(data, object_qpos, object_dof, (0.0, 0.70, BALL_CENTER_Z), (0.0, 0.0, 0.0))
    mujoco.mj_forward(model, data)
    return Scenario(
        name="rolling",
        model=model,
        data=data,
        duration=1.0,
        cue_qpos=cue_qpos,
        cue_dof=cue_dof,
        initial_speed=1.0,
    )


def _pocket_scenario(model_path: Path) -> Scenario:
    model = _load_model(model_path)
    data = mujoco.MjData(model)
    _park_cue(model, data)
    cue_qpos, cue_dof = _joint_addresses(model, "cue_ball_free")
    object_qpos, object_dof = _joint_addresses(model, "object_ball_0_free")
    _set_ball(
        data,
        cue_qpos,
        cue_dof,
        (0.40, 0.0, BALL_CENTER_Z),
        (0.80, 0.0, 0.0),
    )
    _set_ball(data, object_qpos, object_dof, (0.0, 0.70, BALL_CENTER_Z), (0.0, 0.0, 0.0))
    mujoco.mj_forward(model, data)
    return Scenario(
        name="middle_pocket",
        model=model,
        data=data,
        duration=1.2,
        cue_qpos=cue_qpos,
        cue_dof=cue_dof,
        initial_speed=0.8,
        pocket_check=True,
    )


def _run_cpu(scenario: Scenario, pocket_sample_steps: int) -> Rollout:
    model = scenario.model
    data = scenario.data
    steps = int(round(scenario.duration / model.opt.timestep))
    crossing_time: float | None = None
    crossing_xy: np.ndarray | None = None
    started = time.perf_counter()
    for step_index in range(steps):
        mujoco.mj_step(model, data)
        if (
            scenario.pocket_check
            and crossing_time is None
            and (step_index + 1) % pocket_sample_steps == 0
            and data.qpos[scenario.cue_qpos + 2] <= POCKET_ENTRY_Z
        ):
            crossing_time = float(data.time)
            crossing_xy = data.qpos[scenario.cue_qpos:scenario.cue_qpos + 2].copy()
    runtime = time.perf_counter() - started
    return Rollout(
        qpos=data.qpos.copy(),
        qvel=data.qvel.copy(),
        runtime=runtime,
        pocket_crossing_time=crossing_time,
        pocket_crossing_xy=crossing_xy,
    )


def _run_warp(
    scenario: Scenario,
    pocket_sample_steps: int,
) -> Rollout:
    normalize_zero_friction_contact_dims(scenario.model)
    calibrate_trapezoid_sdf_contact_damping(scenario.model)
    initial = mujoco.MjData(scenario.model)
    initial.qpos[:] = scenario.data.qpos
    initial.qvel[:] = scenario.data.qvel
    mujoco.mj_forward(scenario.model, initial)
    model = mjw.put_model(scenario.model)
    data = mjw.put_data(
        scenario.model,
        initial,
        nworld=1,
        nconmax=MUJOCO_WARP_NCONMAX,
        njmax=MUJOCO_WARP_NJMAX,
    )

    with wp.ScopedCapture() as capture:
        mjw.step(model, data)
    wp.synchronize()
    assert_mujoco_warp_capacity(data, context=f"{scenario.name} first step")

    steps = int(round(scenario.duration / scenario.model.opt.timestep))
    crossing_time: float | None = None
    crossing_xy: np.ndarray | None = None
    started = time.perf_counter()
    for completed_steps in range(2, steps + 1):
        wp.capture_launch(capture.graph)
        if (
            scenario.pocket_check
            and crossing_time is None
            and completed_steps % pocket_sample_steps == 0
        ):
            qpos = data.qpos.numpy()[0]
            if qpos[scenario.cue_qpos + 2] <= POCKET_ENTRY_Z:
                crossing_time = completed_steps * float(scenario.model.opt.timestep)
                crossing_xy = qpos[scenario.cue_qpos:scenario.cue_qpos + 2].copy()
    wp.synchronize()
    assert_mujoco_warp_capacity(data, context=f"{scenario.name} rollout")
    runtime = time.perf_counter() - started
    return Rollout(
        qpos=data.qpos.numpy()[0].astype(np.float64),
        qvel=data.qvel.numpy()[0].astype(np.float64),
        runtime=runtime,
        pocket_crossing_time=crossing_time,
        pocket_crossing_xy=crossing_xy,
    )


def _restitution(scenario: Scenario, rollout: Rollout) -> float:
    if scenario.object_dof is not None:
        return (
            rollout.qvel[scenario.object_dof] - rollout.qvel[scenario.cue_dof]
        ) / scenario.initial_speed
    return -rollout.qvel[scenario.cue_dof] / scenario.initial_speed


def _rolling_speed(scenario: Scenario, rollout: Rollout) -> float:
    return float(np.linalg.norm(rollout.qvel[scenario.cue_dof:scenario.cue_dof + 2]))


def _format_optional(value: float | None) -> str:
    return "none" if value is None else f"{value:.6f}"


def _run_batch_benchmark(model_path: Path, nworld: int, steps: int) -> float:
    if steps <= 1:
        raise ValueError("Batch benchmark requires at least two steps.")
    scenario = _rolling_scenario(model_path)
    normalize_zero_friction_contact_dims(scenario.model)
    calibrate_trapezoid_sdf_contact_damping(scenario.model)
    model = mjw.put_model(scenario.model)
    data = mjw.put_data(
        scenario.model,
        scenario.data,
        nworld=nworld,
        nconmax=16,
        njmax=64,
    )
    with wp.ScopedCapture() as capture:
        mjw.step(model, data)
    wp.synchronize()
    assert_mujoco_warp_capacity(data, context="batched rolling benchmark")
    started = time.perf_counter()
    for _ in range(steps - 1):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    runtime = time.perf_counter() - started
    if not np.isfinite(data.qpos.numpy()).all():
        raise RuntimeError("Batch benchmark produced non-finite state.")
    return nworld * (steps - 1) / runtime


def _count_batched_sdf_contacts(model_path: Path, nworld: int) -> int:
    scenario = _trapezoid_cushion_scenario(model_path)
    normalize_zero_friction_contact_dims(scenario.model)
    calibrate_trapezoid_sdf_contact_damping(scenario.model)
    scenario.data.qpos[scenario.cue_qpos] = 0.65421
    scenario.data.qvel[scenario.cue_dof:scenario.cue_dof + 6] = 0.0
    mujoco.mj_forward(scenario.model, scenario.data)
    model = mjw.put_model(scenario.model)
    data = mjw.put_data(
        scenario.model,
        scenario.data,
        nworld=nworld,
        nconmax=16,
        njmax=64,
    )
    data.nacon.zero_()
    mjw.collision(model, data)
    wp.synchronize()
    assert_mujoco_warp_capacity(data, context="batched trapezoid contact check")
    return int(data.nacon.numpy()[0])


def _check_two_ball_pocket_capacity(model_path: Path) -> tuple[int, int]:
    model = _load_model(model_path)
    normalize_zero_friction_contact_dims(model)
    calibrate_trapezoid_sdf_contact_damping(model)
    data = mujoco.MjData(model)
    _park_cue(model, data)
    cue_qpos, cue_dof = _joint_addresses(model, "cue_ball_free")
    object_qpos, object_dof = _joint_addresses(model, "object_ball_0_free")
    _set_ball(data, cue_qpos, cue_dof, (0.75, 0.0, 1.05), (0.0, 0.0, 0.0))
    _set_ball(data, object_qpos, object_dof, (-0.75, 0.0, 1.05), (0.0, 0.0, 0.0))
    mujoco.mj_forward(model, data)
    warp_model = mjw.put_model(model)
    warp_data = mjw.put_data(
        model,
        data,
        nworld=1,
        nconmax=MUJOCO_WARP_NCONMAX,
        njmax=MUJOCO_WARP_NJMAX,
    )
    mjw.step(warp_model, warp_data)
    assert_mujoco_warp_capacity(warp_data, context="two-ball pocket stress")
    return (
        int(warp_data.nacon.numpy()[0]),
        int(np.max(warp_data.nefc.numpy(), initial=0)),
    )


def _check_capacity_failure(model_path: Path) -> str:
    """Prove that the old undersized buffers are rejected, not truncated."""

    model = _load_model(model_path)
    normalize_zero_friction_contact_dims(model)
    calibrate_trapezoid_sdf_contact_damping(model)
    data = mujoco.MjData(model)
    _park_cue(model, data)
    cue_qpos, cue_dof = _joint_addresses(model, "cue_ball_free")
    object_qpos, object_dof = _joint_addresses(model, "object_ball_0_free")
    _set_ball(data, cue_qpos, cue_dof, (0.75, 0.0, 1.05), (0.0, 0.0, 0.0))
    _set_ball(data, object_qpos, object_dof, (-0.75, 0.0, 1.05), (0.0, 0.0, 0.0))
    mujoco.mj_forward(model, data)
    overflow_qpos = data.qpos.copy()
    _set_ball(
        data,
        cue_qpos,
        cue_dof,
        (-0.20, 0.0, BALL_CENTER_Z),
        (0.0, 0.0, 0.0),
    )
    _set_ball(
        data,
        object_qpos,
        object_dof,
        (0.20, 0.0, BALL_CENTER_Z),
        (0.0, 0.0, 0.0),
    )
    mujoco.mj_forward(model, data)
    warp_model = mjw.put_model(model)
    warp_data = mjw.put_data(
        model,
        data,
        nworld=1,
        nconmax=64,
        njmax=256,
    )
    qpos = warp_data.qpos.numpy()
    qpos[0] = overflow_qpos
    warp_data.qpos.assign(qpos)
    mjw.step(warp_model, warp_data)
    try:
        assert_mujoco_warp_capacity(
            warp_data,
            context="deliberately undersized two-ball pocket stress",
        )
    except RuntimeError as error:
        return str(error)
    raise RuntimeError("Undersized MJWarp buffers were silently accepted.")


def _check_explicit_pair_friction() -> tuple[int, np.ndarray]:
    """Exercise all missing-axis branches of explicit-pair normalization."""

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="a" type="sphere" size=".1"/>
            <geom name="b" type="sphere" size=".1" pos=".15 0 0"/>
          </worldbody>
          <contact>
            <pair geom1="a" geom2="b" condim="6"
                  friction=".8 0 0 .2 0"/>
          </contact>
        </mujoco>
        """
    )
    normalize_zero_friction_contact_dims(model)
    if int(model.pair_dim[0]) != 6:
        raise RuntimeError("A pair with rolling friction unexpectedly lost condim=6.")
    expected_positive = (0, 1, 2, 3, 4)
    if any(model.pair_friction[0, index] < MJ_MINMU for index in expected_positive):
        raise RuntimeError("Explicit pair retained a zero-width friction axis.")

    model.pair_dim[0] = 6
    model.pair_friction[0] = (0.8, 0.7, 0.1, 0.0, 0.0)
    normalize_zero_friction_contact_dims(model)
    if int(model.pair_dim[0]) != 4:
        raise RuntimeError("Zero rolling axes were not reduced from condim=6 to 4.")

    model.pair_dim[0] = 6
    model.pair_friction[0] = (0.8, 0.0, 0.0, 0.0, 0.0)
    normalize_zero_friction_contact_dims(model)
    if int(model.pair_dim[0]) != 3:
        raise RuntimeError("Zero spin/rolling axes were not reduced to condim=3.")
    if model.pair_friction[0, 1] < MJ_MINMU:
        raise RuntimeError("Second explicit-pair sliding axis was not repaired.")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        mjw.put_model(model)
    friction_warnings = [
        str(item.message) for item in captured if "friction" in str(item.message)
    ]
    if friction_warnings:
        raise RuntimeError(
            "Explicit-pair normalization left MJWarp warnings: "
            + "; ".join(friction_warnings)
        )
    return int(model.pair_dim[0]), model.pair_friction[0].copy()


def _check_trapezoid_collision_sensor(
    model_path: Path,
) -> tuple[int, float, np.ndarray]:
    """Ensure collision sensors retain the exact sphere/trapezoid path."""

    base = _trapezoid_cushion_scenario(model_path)
    target_candidates = [
        geom_id
        for geom_id in range(base.model.ngeom)
        if int(base.model.geom_plugin[geom_id]) >= 0
        and (
            mujoco.mj_id2name(
                base.model,
                mujoco.mjtObj.mjOBJ_PLUGIN,
                int(base.model.geom_plugin[geom_id]),
            )
            or ""
        ).startswith("trapezoid")
    ]
    target_geom = min(
        target_candidates,
        key=lambda geom_id: float(
            np.linalg.norm(base.model.geom_pos[geom_id, :2] - (0.70, -0.40))
        ),
    )

    spec = mujoco.MjSpec.from_file(str(model_path.resolve()))
    spec.geoms[target_geom].name = "warp_sensor_trapezoid"
    spec.add_sensor(
        name="trapezoid_distance",
        type=mujoco.mjtSensor.mjSENS_GEOMDIST,
        objtype=mujoco.mjtObj.mjOBJ_GEOM,
        objname="cue_ball_geom",
        reftype=mujoco.mjtObj.mjOBJ_GEOM,
        refname="warp_sensor_trapezoid",
        cutoff=10.0,
    )
    model = spec.compile()
    model.opt.gravity[:] = 0.0
    cue_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_ball_geom")
    target_geom = _id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "warp_sensor_trapezoid",
    )
    _only_collide_ids(model, (cue_geom, target_geom))
    normalize_zero_friction_contact_dims(model)
    calibrate_trapezoid_sdf_contact_damping(model)

    data = mujoco.MjData(model)
    _park_cue(model, data)
    cue_qpos, cue_dof = _joint_addresses(model, "cue_ball_free")
    object_qpos, object_dof = _joint_addresses(model, "object_ball_0_free")
    _set_ball(
        data,
        cue_qpos,
        cue_dof,
        (0.65421, -0.40, BALL_CENTER_Z),
        (0.0, 0.0, 0.0),
    )
    _set_ball(
        data,
        object_qpos,
        object_dof,
        (0.0, 0.70, BALL_CENTER_Z),
        (0.0, 0.0, 0.0),
    )
    mujoco.mj_forward(model, data)
    cpu_distance = float(data.sensordata[0])

    warp_model = mjw.put_model(model)
    warp_data = mjw.put_data(
        model,
        data,
        nworld=1,
        nconmax=16,
        njmax=64,
    )
    warp_data.nacon.zero_()
    mjw.collision(warp_model, warp_data)
    mjw.sensor_pos(warp_model, warp_data)
    assert_mujoco_warp_capacity(
        warp_data,
        context="trapezoid collision sensor",
    )
    contact_count = int(warp_data.nacon.numpy()[0])
    contact_types = warp_data.contact.type.numpy()[:contact_count].copy()
    required_type = int(ContactType.CONSTRAINT | ContactType.SENSOR)
    if contact_count != TRAPEZOID_MANIFOLD_CONTACTS:
        raise RuntimeError(
            "Trapezoid collision sensor fell back to the generic SDF contact cloud."
        )
    if np.any((contact_types & required_type) != required_type):
        raise RuntimeError("Trapezoid sensor contacts lost constraint or sensor flags.")
    warp_distance = float(warp_data.sensordata.numpy()[0, 0])
    return contact_count, abs(cpu_distance - warp_distance), contact_types


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pocket-sample-steps", type=int, default=100)
    parser.add_argument("--batch-worlds", type=int, default=256)
    parser.add_argument("--batch-steps", type=int, default=1_000)
    args = parser.parse_args()
    if args.pocket_sample_steps <= 0 or args.batch_worlds <= 0:
        raise ValueError("Sampling and batch sizes must be positive.")
    if args.batch_steps <= 1:
        raise ValueError("--batch-steps must be at least two.")

    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("The parity prototype requires a CUDA device.")
    wp.set_device(device)

    registration_model = _load_model(args.model)
    plugin_types = register_mujoco_billiards_sdf(registration_model)
    warnings.filterwarnings(
        "ignore",
        message=r"geom .* friction.*MJ_MINMU.*",
        category=UserWarning,
    )
    scenario_factories = (
        _ball_ball_scenario,
        _cushion_scenario,
        _trapezoid_cushion_scenario,
        _rolling_scenario,
        _pocket_scenario,
    )
    results: dict[str, tuple[Scenario, Rollout, Rollout]] = {}
    for factory in scenario_factories:
        cpu_scenario = factory(args.model)
        warp_scenario = factory(args.model)
        cpu_rollout = _run_cpu(cpu_scenario, args.pocket_sample_steps)
        warp_rollout = _run_warp(warp_scenario, args.pocket_sample_steps)
        if not (
            np.isfinite(cpu_rollout.qpos).all()
            and np.isfinite(cpu_rollout.qvel).all()
            and np.isfinite(warp_rollout.qpos).all()
            and np.isfinite(warp_rollout.qvel).all()
        ):
            raise RuntimeError(f"{cpu_scenario.name} produced non-finite state.")
        results[cpu_scenario.name] = (cpu_scenario, cpu_rollout, warp_rollout)

    ball_scenario, ball_cpu, ball_warp = results["ball_ball"]
    cushion_scenario, cushion_cpu, cushion_warp = results["cushion"]
    trapezoid_scenario, trapezoid_cpu, trapezoid_warp = results[
        "trapezoid_sdf_cushion"
    ]
    rolling_scenario, rolling_cpu, rolling_warp = results["rolling"]
    _, pocket_cpu, pocket_warp = results["middle_pocket"]
    ball_cpu_restitution = _restitution(ball_scenario, ball_cpu)
    ball_warp_restitution = _restitution(ball_scenario, ball_warp)
    cushion_cpu_restitution = _restitution(cushion_scenario, cushion_cpu)
    cushion_warp_restitution = _restitution(cushion_scenario, cushion_warp)
    trapezoid_cpu_restitution = _restitution(trapezoid_scenario, trapezoid_cpu)
    trapezoid_warp_restitution = _restitution(trapezoid_scenario, trapezoid_warp)
    rolling_cpu_speed = _rolling_speed(rolling_scenario, rolling_cpu)
    rolling_warp_speed = _rolling_speed(rolling_scenario, rolling_warp)
    pocket_time_error = (
        abs(pocket_cpu.pocket_crossing_time - pocket_warp.pocket_crossing_time)
        if pocket_cpu.pocket_crossing_time is not None
        and pocket_warp.pocket_crossing_time is not None
        else float("inf")
    )
    pocket_xy_error = (
        float(np.linalg.norm(pocket_cpu.pocket_crossing_xy - pocket_warp.pocket_crossing_xy))
        if pocket_cpu.pocket_crossing_xy is not None
        and pocket_warp.pocket_crossing_xy is not None
        else float("inf")
    )
    batch_worldsteps = _run_batch_benchmark(
        args.model,
        args.batch_worlds,
        args.batch_steps,
    )
    sdf_batch_contacts = _count_batched_sdf_contacts(
        args.model,
        args.batch_worlds,
    )
    pocket_stress_contacts, pocket_stress_constraints = (
        _check_two_ball_pocket_capacity(args.model)
    )
    capacity_failure = _check_capacity_failure(args.model)
    pair_dim, pair_friction = _check_explicit_pair_friction()
    sensor_contacts, sensor_distance_error, sensor_contact_types = (
        _check_trapezoid_collision_sensor(args.model)
    )
    expected_sdf_contacts = TRAPEZOID_MANIFOLD_CONTACTS * args.batch_worlds

    print(
        f"device={device} sdf_types={plugin_types} "
        f"batch_worldsteps_per_second={batch_worldsteps:.0f} "
        f"sdf_batch_contacts={sdf_batch_contacts}/{expected_sdf_contacts} "
        f"pocket_stress_contacts={pocket_stress_contacts} "
        f"pocket_stress_constraints={pocket_stress_constraints}"
    )
    print(f"capacity_failure={capacity_failure}")
    print(
        f"explicit_pair_dim={pair_dim} "
        f"explicit_pair_friction={pair_friction.tolist()} "
        f"sensor_contacts={sensor_contacts} "
        f"sensor_distance_error={sensor_distance_error:.9f} "
        f"sensor_contact_types={sensor_contact_types.tolist()}"
    )
    print(
        f"ball_ball_restitution cpu={ball_cpu_restitution:.6f} "
        f"warp={ball_warp_restitution:.6f} "
        f"error={abs(ball_cpu_restitution - ball_warp_restitution):.6f} "
        f"runtime_cpu={ball_cpu.runtime:.3f}s runtime_warp={ball_warp.runtime:.3f}s"
    )
    print(
        f"cushion_restitution cpu={cushion_cpu_restitution:.6f} "
        f"warp={cushion_warp_restitution:.6f} "
        f"error={abs(cushion_cpu_restitution - cushion_warp_restitution):.6f} "
        f"runtime_cpu={cushion_cpu.runtime:.3f}s runtime_warp={cushion_warp.runtime:.3f}s"
    )
    print(
        f"trapezoid_sdf_cushion_restitution cpu={trapezoid_cpu_restitution:.6f} "
        f"warp={trapezoid_warp_restitution:.6f} "
        f"error={abs(trapezoid_cpu_restitution - trapezoid_warp_restitution):.6f} "
        f"runtime_cpu={trapezoid_cpu.runtime:.3f}s runtime_warp={trapezoid_warp.runtime:.3f}s"
    )
    print(
        f"rolling_speed_after_1s cpu={rolling_cpu_speed:.6f} "
        f"warp={rolling_warp_speed:.6f} "
        f"error={abs(rolling_cpu_speed - rolling_warp_speed):.6f} "
        f"runtime_cpu={rolling_cpu.runtime:.3f}s runtime_warp={rolling_warp.runtime:.3f}s"
    )
    print(
        f"middle_pocket_crossing_time cpu={_format_optional(pocket_cpu.pocket_crossing_time)} "
        f"warp={_format_optional(pocket_warp.pocket_crossing_time)} "
        f"time_error={pocket_time_error:.6f} xy_error={pocket_xy_error:.6f} "
        f"runtime_cpu={pocket_cpu.runtime:.3f}s runtime_warp={pocket_warp.runtime:.3f}s"
    )

    failures: list[str] = []
    if abs(ball_cpu_restitution - ball_warp_restitution) > 0.02:
        failures.append("ball-ball restitution error > 0.02")
    if abs(cushion_cpu_restitution - cushion_warp_restitution) > 0.05:
        failures.append("cushion restitution error > 0.05")
    if abs(trapezoid_cpu_restitution - trapezoid_warp_restitution) > 0.05:
        failures.append("trapezoid SDF cushion restitution error > 0.05")
    if abs(rolling_cpu_speed - rolling_warp_speed) > 0.02:
        failures.append("rolling speed error > 0.02 m/s")
    if pocket_time_error > 0.02:
        failures.append("middle-pocket crossing-time error > 0.02 s")
    if pocket_xy_error > 0.01:
        failures.append("middle-pocket crossing-position error > 0.01 m")
    if sdf_batch_contacts != expected_sdf_contacts:
        failures.append(
            f"expected {TRAPEZOID_MANIFOLD_CONTACTS} sphere-SDF contacts per world"
        )
    if pocket_stress_contacts <= 64 or pocket_stress_constraints <= 256:
        failures.append("two-ball pocket stress did not exercise the old capacity limits")
    if sensor_distance_error > 0.0003:
        failures.append("trapezoid collision-sensor distance error > 0.3 mm")
    if failures:
        raise RuntimeError("MuJoCo Warp parity failed: " + "; ".join(failures))
    print("mujoco_warp_parity=PASS")


if __name__ == "__main__":
    main()
