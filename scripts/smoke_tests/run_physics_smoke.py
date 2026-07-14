from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.scene import default_model_path, load_model  # noqa: E402
from snooker_env.init_pose import set_lift_grip_ready_pose  # noqa: E402


@dataclass
class SmokeResult:
    first_contact_time: float | None
    cue_tip_speed: float
    cue_ball_velocity: np.ndarray
    cue_ball_displacement: np.ndarray
    min_contact_distance: float
    has_nan: bool
    exploded: bool


def _id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Required {obj_type.name} name is missing: {name}")
    return obj_id


def _site_velocity(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp @ data.qvel


def _contact_between(data: mujoco.MjData, geom_a: int, geom_b: int) -> tuple[bool, float]:
    found = False
    min_dist = float("inf")
    for idx in range(data.ncon):
        contact = data.contact[idx]
        if {contact.geom1, contact.geom2} == {geom_a, geom_b}:
            found = True
            min_dist = min(min_dist, float(contact.dist))
    return found, min_dist


def run_smoke(model_path: Path, duration: float, strike_speed: float, keep_grip_constraints: bool) -> SmokeResult:
    model = load_model(model_path)
    data = mujoco.MjData(model)
    set_lift_grip_ready_pose(model, data)
    if not keep_grip_constraints and model.neq:
        data.eq_active[:] = 0
        mujoco.mj_forward(model, data)

    cue_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "cue_free")
    cue_ball_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "cue_ball_free")
    cue_tip_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_tip")
    cue_ball_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_ball_geom")
    cue_tip_site = _id(model, mujoco.mjtObj.mjOBJ_SITE, "cue_tip_site")

    cue_dof = int(model.jnt_dofadr[cue_joint])
    cue_ball_qpos = int(model.jnt_qposadr[cue_ball_joint])
    cue_ball_dof = int(model.jnt_dofadr[cue_ball_joint])
    start_pos = data.qpos[cue_ball_qpos:cue_ball_qpos + 3].copy()

    data.qvel[cue_dof:cue_dof + 3] = np.array([strike_speed, 0.0, 0.0])
    data.qvel[cue_dof + 3:cue_dof + 6] = 0.0
    mujoco.mj_forward(model, data)
    tip_speed = float(np.linalg.norm(_site_velocity(model, data, cue_tip_site)))

    first_contact: float | None = None
    min_dist = float("inf")
    has_nan = False
    exploded = False
    steps = int(duration / model.opt.timestep)

    for _ in range(steps):
        mujoco.mj_step(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            has_nan = True
            break
        if np.max(np.abs(data.qvel)) > 150.0 or np.max(np.abs(data.qpos)) > 25.0:
            exploded = True
            break
        touching, dist = _contact_between(data, cue_tip_geom, cue_ball_geom)
        if touching:
            min_dist = min(min_dist, dist)
            if first_contact is None:
                first_contact = float(data.time)

    ball_vel = data.qvel[cue_ball_dof:cue_ball_dof + 3].copy()
    final_pos = data.qpos[cue_ball_qpos:cue_ball_qpos + 3].copy()
    if min_dist == float("inf"):
        min_dist = 0.0

    return SmokeResult(
        first_contact_time=first_contact,
        cue_tip_speed=tip_speed,
        cue_ball_velocity=ball_vel,
        cue_ball_displacement=final_pos - start_pos,
        min_contact_distance=min_dist,
        has_nan=has_nan,
        exploded=exploded,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the asset-scene physics cue/ball chain.")
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--strike-speed", type=float, default=1.2)
    parser.add_argument(
        "--keep-grip-constraints",
        action="store_true",
        help="Keep LIFT/cue grip equality constraints active. Default disables them for cue-ball chain smoke testing.",
    )
    args = parser.parse_args()

    result = run_smoke(args.model, args.duration, args.strike_speed, args.keep_grip_constraints)
    print(f"Cue tip speed: {result.cue_tip_speed:.4f} m/s")
    print(f"First cue_tip/cue_ball contact time: {result.first_contact_time}")
    print(f"Cue ball velocity: {result.cue_ball_velocity} m/s")
    print(f"Cue ball displacement: {result.cue_ball_displacement} m")
    print(f"Minimum cue_tip/cue_ball contact distance: {result.min_contact_distance:.6f} m")
    print(f"NaN/Inf detected: {result.has_nan}")
    print(f"Numerical explosion detected: {result.exploded}")

    if result.first_contact_time is None:
        raise RuntimeError("Smoke test failed: cue tip never contacted cue ball.")
    if np.linalg.norm(result.cue_ball_velocity) < 1e-3:
        raise RuntimeError("Smoke test failed: cue ball did not acquire velocity.")
    if result.has_nan or result.exploded:
        raise RuntimeError("Smoke test failed: unstable simulation.")


if __name__ == "__main__":
    main()
