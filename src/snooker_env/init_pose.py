from __future__ import annotations

import mujoco
import numpy as np


LIFT_READY_HEIGHT = 0.549


def set_lift_grip_ready_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Set the current scaffold pose used before dual-grip training.

    This is not a solved IK pose. It only raises the LIFT column so the
    placeholder TCP sites are roughly at cue-grip height.
    """

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_lift")
    if joint_id >= 0:
        qpos_id = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_id] = LIFT_READY_HEIGHT

    for actuator_name, value in {
        "joint_lift_pos": LIFT_READY_HEIGHT,
        "left_catch_joint1_pos": -0.025,
        "left_catch_joint2_pos": 0.025,
        "right_catch_joint1_pos": -0.025,
        "right_catch_joint2_pos": 0.025,
    }.items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id >= 0:
            data.ctrl[actuator_id] = value

    mujoco.mj_forward(model, data)


GENTO_SIDE_GRASP_ARM_QPOS = {
    # Robot-right is the forward direction/support hand; robot-left is the
    # rear speed hand. These angles come from the imported Gento URDF chain.
    "right": np.asarray(
        (
            -0.73536326,
            -1.18011590,
            0.20183900,
            -0.69284909,
            1.18262856,
            -0.57304080,
            -0.34681679,
        ),
        dtype=np.float64,
    ),
    "left": np.asarray(
        (
            0.73518951,
            -1.18231483,
            -0.20119228,
            -0.69343363,
            -1.18111677,
            -0.57254024,
            0.35012164,
        ),
        dtype=np.float64,
    ),
}


def set_gento_side_grasp_ready_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    """Place Gento in a physical side grasp above the dev-midlevel rail.

    The lift is raised so the horizontal cue center is 1.101 m high. With a
    10 mm shaft radius, its lower surface stays 1 mm above the source table's
    1.090 m rail top instead of passing through the wooden cushion body.
    """

    joint_targets: dict[str, float] = {
        "gento_waist_jiont_1_prismatic": 0.6612,
        "gento_waist_joint_2": 0.0,
        "gento_waist_joint_3": 0.0,
        "gento_head_joint_1": 0.0,
        "gento_head_joint_2": 0.0,
    }
    for side, arm_qpos in GENTO_SIDE_GRASP_ARM_QPOS.items():
        joint_targets.update(
            {
                f"gento_arm_{side}_joint_{index}": float(value)
                for index, value in enumerate(arm_qpos, start=1)
            }
        )
        joint_targets[f"gento_{side}_upper_finger_joint"] = -0.0265
        joint_targets[f"gento_{side}_lower_finger_joint"] = 0.0265

    for joint_name, value in joint_targets.items():
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise ValueError(f"Gento side-grasp model is missing joint {joint_name!r}.")
        qpos_id = int(model.jnt_qposadr[joint_id])
        dof_id = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_id] = value
        data.qvel[dof_id] = 0.0

        actuator_name = f"{joint_name}_pos"
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )
        if actuator_id < 0:
            raise ValueError(
                f"Gento side-grasp model is missing actuator {actuator_name!r}."
            )
        if "upper_finger" in joint_name:
            data.ctrl[actuator_id] = -0.03
        elif "lower_finger" in joint_name:
            data.ctrl[actuator_id] = 0.03
        else:
            data.ctrl[actuator_id] = value

    mujoco.mj_forward(model, data)
