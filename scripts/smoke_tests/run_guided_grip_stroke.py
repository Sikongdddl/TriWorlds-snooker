from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.init_pose import set_lift_grip_ready_pose  # noqa: E402
from snooker_env.scene import default_model_path, load_model  # noqa: E402


CONTROL_JOINTS = [
    "joint_lift",
    "joint_h_1",
    "joint_h_2",
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
]

SITE_PAIRS = [
    ("lift_left_gripper_tcp", "cue_left_grip_site"),
    ("lift_right_gripper_tcp", "cue_right_grip_site"),
]


def _id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing {obj_type.name}: {name}")
    return obj_id


def _site_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    return data.site_xpos[_id(model, mujoco.mjtObj.mjOBJ_SITE, name)].copy()


def _selected_dofs(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    dofs: list[int] = []
    qpos: list[int] = []
    for name in CONTROL_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        dofs.append(int(model.jnt_dofadr[joint_id]))
        qpos.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(dofs, dtype=int), np.asarray(qpos, dtype=int)


def _site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_name: str, dofs: np.ndarray) -> np.ndarray:
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    site_id = _id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp[:, dofs]


def _grip_error(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return max(
        float(np.linalg.norm(_site_pos(model, data, lift_site) - _site_pos(model, data, cue_site)))
        for lift_site, cue_site in SITE_PAIRS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Guided scaffold stroke using LIFT TCP Jacobian IK and cue grip constraints.")
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--duration", type=float, default=0.8)
    parser.add_argument("--stroke", type=float, default=0.06, help="Desired TCP displacement along world +X in meters.")
    parser.add_argument("--damping", type=float, default=0.08, help="Damped least-squares IK regularization.")
    parser.add_argument("--max-step", type=float, default=0.004, help="Max scalar joint update per sim step.")
    args = parser.parse_args()

    model = load_model(args.model)
    data = mujoco.MjData(model)
    set_lift_grip_ready_pose(model, data)

    dofs, qpos_ids = _selected_dofs(model)
    if len(dofs) == 0:
        raise RuntimeError("No controllable LIFT joints found.")

    initial_targets = {
        site: _site_pos(model, data, site)
        for site, _ in SITE_PAIRS
    }
    cue_tip_start = _site_pos(model, data, "cue_tip_site")
    cue_ball_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_ball_geom")
    cue_tip_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_tip")

    first_contact: float | None = None
    has_nan = False
    exploded = False
    max_grip_error = _grip_error(model, data)

    steps = int(args.duration / model.opt.timestep)
    for step in range(steps):
        phase = min(1.0, step / max(1, int(0.55 * steps)))
        offset = np.array([args.stroke * phase, 0.0, 0.0])

        mujoco.mj_forward(model, data)
        residuals: list[np.ndarray] = []
        jacobians: list[np.ndarray] = []
        for lift_site, _cue_site in SITE_PAIRS:
            target = initial_targets[lift_site] + offset
            residuals.append(target - _site_pos(model, data, lift_site))
            jacobians.append(_site_jacobian(model, data, lift_site, dofs))

        err = np.concatenate(residuals)
        jac = np.vstack(jacobians)
        lhs = jac @ jac.T + (args.damping ** 2) * np.eye(jac.shape[0])
        dq = jac.T @ np.linalg.solve(lhs, err)
        dq = np.clip(dq, -args.max_step, args.max_step)

        data.qpos[qpos_ids] += dq
        data.qvel[dofs] = dq / model.opt.timestep
        mujoco.mj_step(model, data)

        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            has_nan = True
            break
        if np.max(np.abs(data.qvel)) > 300.0 or np.max(np.abs(data.qpos)) > 30.0:
            exploded = True
            break
        max_grip_error = max(max_grip_error, _grip_error(model, data))

        for con_id in range(data.ncon):
            contact = data.contact[con_id]
            if {contact.geom1, contact.geom2} == {cue_tip_geom, cue_ball_geom} and first_contact is None:
                first_contact = float(data.time)

    cue_tip_end = _site_pos(model, data, "cue_tip_site")
    cue_tip_delta = cue_tip_end - cue_tip_start
    print(f"Controlled dofs: {len(dofs)}")
    print(f"Requested stroke: {args.stroke:.4f} m")
    print(f"Cue tip delta: {cue_tip_delta} m")
    print(f"Max grip error: {max_grip_error:.6f} m")
    print(f"First cue_tip/cue_ball contact time: {first_contact}")
    print(f"NaN/Inf detected: {has_nan}")
    print(f"Numerical explosion detected: {exploded}")

    if has_nan or exploded:
        raise RuntimeError("Guided grip stroke failed: unstable simulation.")
    if cue_tip_delta[0] < 0.01:
        raise RuntimeError("Guided grip stroke failed: cue tip did not move meaningfully along +X.")
    if max_grip_error > 0.05:
        raise RuntimeError(f"Guided grip stroke failed: grip error too large ({max_grip_error:.6f} m).")


if __name__ == "__main__":
    main()
