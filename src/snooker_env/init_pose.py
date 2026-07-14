from __future__ import annotations

import mujoco


def set_lift_grip_ready_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Set the current scaffold pose used before dual-grip training.

    This is not a solved IK pose. It only raises the LIFT column so the
    placeholder TCP sites are roughly at cue-grip height.
    """

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_lift")
    if joint_id >= 0:
        qpos_id = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_id] = 0.26

    for actuator_name, value in {
        "joint_lift_pos": 0.26,
        "left_catch_joint1_pos": -0.025,
        "left_catch_joint2_pos": 0.025,
        "right_catch_joint1_pos": -0.025,
        "right_catch_joint2_pos": 0.025,
    }.items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id >= 0:
            data.ctrl[actuator_id] = value

    mujoco.mj_forward(model, data)
