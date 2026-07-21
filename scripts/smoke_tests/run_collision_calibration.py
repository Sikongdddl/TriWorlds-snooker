"""Calibrate ball transfer, cushion rebound, cloth rolling, and pocket entry."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.contact_events import CollisionEventMonitor  # noqa: E402
from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402


@dataclass(frozen=True)
class ImpactResult:
    restitution: float
    maximum_penetration: float
    incoming_speed: float
    outgoing_speed: float


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Missing {object_type.name}: {name}")
    return object_id


def _joint_addresses(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _only_collide(model: mujoco.MjModel, names: tuple[str, ...]) -> tuple[int, ...]:
    geom_ids = tuple(_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names)
    for geom_id in range(model.ngeom):
        if geom_id not in geom_ids:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
    return geom_ids


def _run_until_contact_ends(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_pair: frozenset[int],
    timeout: float,
) -> float:
    contact_seen = False
    maximum_penetration = 0.0
    for _ in range(int(round(timeout / model.opt.timestep))):
        mujoco.mj_step(model, data)
        contacts = [
            data.contact[index]
            for index in range(data.ncon)
            if frozenset((data.contact[index].geom1, data.contact[index].geom2)) == geom_pair
        ]
        if contacts:
            contact_seen = True
            maximum_penetration = max(
                maximum_penetration,
                max(max(0.0, -float(contact.dist)) for contact in contacts),
            )
        elif contact_seen:
            return maximum_penetration
    raise RuntimeError("Expected contact did not begin and end before timeout.")


def ball_ball_impact(speed: float = 3.0) -> ImpactResult:
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MIDLEVEL_MODEL))
    model.opt.gravity[:] = 0.0
    pair = _only_collide(model, ("cue_ball_geom", "object_ball_0_geom"))
    data = mujoco.MjData(model)
    cue_qpos, cue_dof = _joint_addresses(model, "cue_ball_free")
    object_qpos, object_dof = _joint_addresses(model, "object_ball_0_free")
    data.qpos[cue_qpos:cue_qpos + 3] = (-0.10, 0.0, 1.0)
    data.qpos[object_qpos:object_qpos + 3] = (0.10, 0.0, 1.0)
    data.qvel[cue_dof] = speed
    mujoco.mj_forward(model, data)
    penetration = _run_until_contact_ends(model, data, frozenset(pair), timeout=0.20)
    cue_out = float(data.qvel[cue_dof])
    object_out = float(data.qvel[object_dof])
    return ImpactResult(
        restitution=(object_out - cue_out) / speed,
        maximum_penetration=penetration,
        incoming_speed=speed,
        outgoing_speed=object_out,
    )


def cushion_impact(speed: float = 2.0) -> ImpactResult:
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MIDLEVEL_MODEL))
    model.opt.gravity[:] = 0.0
    pair = _only_collide(model, ("cue_ball_geom", "cushion_nose_y_pos_left"))
    data = mujoco.MjData(model)
    cue_qpos, cue_dof = _joint_addresses(model, "cue_ball_free")
    data.qpos[cue_qpos:cue_qpos + 3] = (-0.40, 0.35, 0.795)
    data.qvel[cue_dof + 1] = speed
    mujoco.mj_forward(model, data)
    penetration = _run_until_contact_ends(model, data, frozenset(pair), timeout=0.30)
    outgoing = float(data.qvel[cue_dof + 1])
    return ImpactResult(
        restitution=-outgoing / speed,
        maximum_penetration=penetration,
        incoming_speed=speed,
        outgoing_speed=outgoing,
    )


def rolling_speed_after(duration: float = 1.0, speed: float = 1.0) -> float:
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MIDLEVEL_MODEL))
    _only_collide(model, ("cue_ball_geom", "playfield_collision"))
    data = mujoco.MjData(model)
    cue_qpos, cue_dof = _joint_addresses(model, "cue_ball_free")
    radius = float(model.geom_size[_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_ball_geom"), 0])
    data.qpos[cue_qpos:cue_qpos + 3] = (0.0, 0.0, 0.789575)
    data.qvel[cue_dof] = speed
    data.qvel[cue_dof + 4] = -speed / radius
    mujoco.mj_forward(model, data)
    for _ in range(int(round(duration / model.opt.timestep))):
        mujoco.mj_step(model, data)
    return float(np.linalg.norm(data.qvel[cue_dof:cue_dof + 2]))


def middle_pocket_drop(speed: float = 0.80) -> tuple[bool, float]:
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MIDLEVEL_MODEL))
    data = mujoco.MjData(model)
    monitor = CollisionEventMonitor(model)
    cue_qpos, cue_dof = _joint_addresses(model, "cue_ball_free")
    object_qpos, object_dof = _joint_addresses(model, "object_ball_0_free")
    data.qpos[cue_qpos:cue_qpos + 3] = (0.0, 0.38, 0.789575)
    data.qvel[cue_dof + 1] = speed
    data.qpos[object_qpos:object_qpos + 3] = (0.70, 0.0, 0.789575)
    data.qvel[object_dof:object_dof + 6] = 0.0
    mujoco.mj_forward(model, data)
    minimum_z = float(data.qpos[cue_qpos + 2])
    for _ in range(int(round(1.2 / model.opt.timestep))):
        mujoco.mj_step(model, data)
        monitor.scan(data)
        minimum_z = min(minimum_z, float(data.qpos[cue_qpos + 2]))
    return "cue_ball" in monitor.pocketed_balls, minimum_z


def main() -> None:
    ball = ball_ball_impact()
    cushion = cushion_impact()
    rolling_speed = rolling_speed_after()
    pocketed, minimum_z = middle_pocket_drop()
    print(
        f"ball_ball_restitution={ball.restitution:.6f} "
        f"penetration={ball.maximum_penetration * 1000.0:.3f} mm "
        f"object_speed={ball.outgoing_speed:.6f} m/s"
    )
    print(
        f"cushion_restitution={cushion.restitution:.6f} "
        f"penetration={cushion.maximum_penetration * 1000.0:.3f} mm "
        f"outgoing_normal_speed={cushion.outgoing_speed:.6f} m/s"
    )
    print(f"rolling_speed_after_1s={rolling_speed:.6f} m/s")
    print(f"middle_pocketed={pocketed} minimum_ball_z={minimum_z:.6f} m")

    if not 0.88 <= ball.restitution <= 0.99 or ball.maximum_penetration > 0.002:
        raise RuntimeError("Ball-ball collision is outside the calibrated range.")
    if not 0.65 <= cushion.restitution <= 0.90 or cushion.maximum_penetration > 0.002:
        raise RuntimeError("Cushion collision is outside the calibrated range.")
    if not 0.0 < rolling_speed < 1.0:
        raise RuntimeError("Cloth rolling resistance did not reduce ball speed smoothly.")
    if not pocketed or minimum_z >= 0.745:
        raise RuntimeError("Ball did not enter the middle pocket region.")


if __name__ == "__main__":
    main()
