"""Verify that the initial object-ball rack has no overlap or lateral drift."""

from __future__ import annotations

import itertools

import mujoco
import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.init_pose import set_lift_grip_ready_pose  # noqa: E402
from snooker_env.scene import load_model  # noqa: E402


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Missing {object_type.name}: {name}")
    return object_id


def main() -> None:
    model = load_model()
    data = mujoco.MjData(model)
    set_lift_grip_ready_pose(model, data)

    names = [f"object_ball_{index}" for index in range(10)]
    body_ids = [_id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names]
    geom_ids = [_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom") for name in names]
    joint_ids = [_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free") for name in names]
    ball_radius = float(model.geom_size[geom_ids[0], 0])
    start_xy = data.xpos[body_ids, :2].copy()

    minimum_clearance = min(
        float(np.linalg.norm(data.xpos[first] - data.xpos[second]) - 2.0 * ball_radius)
        for first, second in itertools.combinations(body_ids, 2)
    )
    initial_ball_contacts = sum(
        data.contact[index].geom1 in geom_ids and data.contact[index].geom2 in geom_ids
        for index in range(data.ncon)
    )

    peak_horizontal_speed = 0.0
    for _ in range(int(round(1.0 / model.opt.timestep))):
        mujoco.mj_step(model, data)
        for joint_id in joint_ids:
            dof_id = int(model.jnt_dofadr[joint_id])
            peak_horizontal_speed = max(
                peak_horizontal_speed,
                float(np.linalg.norm(data.qvel[dof_id:dof_id + 2])),
            )

    horizontal_displacement = np.linalg.norm(data.xpos[body_ids, :2] - start_xy, axis=1)
    maximum_displacement = float(np.max(horizontal_displacement))
    print(f"minimum_initial_clearance={minimum_clearance:.9f} m")
    print(f"initial_ball_contacts={initial_ball_contacts}")
    print(f"peak_horizontal_speed={peak_horizontal_speed:.9f} m/s")
    print(f"maximum_horizontal_displacement={maximum_displacement:.9f} m")

    if minimum_clearance < 0.00049:
        raise RuntimeError("Initial rack clearance is smaller than the configured safety gap.")
    if initial_ball_contacts:
        raise RuntimeError("Initial rack contains ball-ball contacts.")
    if peak_horizontal_speed > 1e-5 or maximum_displacement > 1e-5:
        raise RuntimeError("Initial rack drifts laterally while settling.")


if __name__ == "__main__":
    main()

