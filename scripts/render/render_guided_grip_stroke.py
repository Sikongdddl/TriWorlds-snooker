from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
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


def _hide_debug_groups(renderer: mujoco.Renderer) -> None:
    scene_option = getattr(renderer, "scene_option", None)
    if scene_option is None:
        scene_option = getattr(renderer, "_scene_option", None)
    if scene_option is not None:
        scene_option.geomgroup[3] = 0
        scene_option.sitegroup[3] = 0


def _camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.15
    cam.azimuth = -38.0
    cam.elevation = -14.0
    cam.lookat[:] = (-0.75, 0.18, 0.78)
    return cam


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
        if joint_id >= 0:
            dofs.append(int(model.jnt_dofadr[joint_id]))
            qpos.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(dofs, dtype=int), np.asarray(qpos, dtype=int)


def _site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_name: str, dofs: np.ndarray) -> np.ndarray:
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, _id(model, mujoco.mjtObj.mjOBJ_SITE, site_name))
    return jacp[:, dofs]


def _guided_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    dofs: np.ndarray,
    qpos_ids: np.ndarray,
    initial_targets: dict[str, np.ndarray],
    phase: float,
    stroke: float,
    damping: float,
    max_step: float,
) -> None:
    offset = np.array([stroke * phase, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    residuals: list[np.ndarray] = []
    jacobians: list[np.ndarray] = []
    for lift_site, _cue_site in SITE_PAIRS:
        residuals.append(initial_targets[lift_site] + offset - _site_pos(model, data, lift_site))
        jacobians.append(_site_jacobian(model, data, lift_site, dofs))

    err = np.concatenate(residuals)
    jac = np.vstack(jacobians)
    lhs = jac @ jac.T + (damping**2) * np.eye(jac.shape[0])
    dq = jac.T @ np.linalg.solve(lhs, err)
    dq = np.clip(dq, -max_step, max_step)

    data.qpos[qpos_ids] += dq
    data.qvel[dofs] = dq / model.opt.timestep
    mujoco.mj_step(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a video of the guided dual-grip stroke hitting the cue ball.")
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--output", type=Path, default=Path("outputs/videos_pool_asset/guided_grip_stroke_hit.mp4"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--stroke", type=float, default=0.075)
    parser.add_argument("--damping", type=float, default=0.08)
    parser.add_argument("--max-step", type=float, default=0.004)
    args = parser.parse_args()

    model = load_model(args.model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    set_lift_grip_ready_pose(model, data)

    dofs, qpos_ids = _selected_dofs(model)
    initial_targets = {site: _site_pos(model, data, site) for site, _ in SITE_PAIRS}

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    _hide_debug_groups(renderer)
    camera = _camera()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_frames = max(1, int(round(args.seconds * args.fps)))
    sim_steps_per_frame = max(1, int(round((1.0 / args.fps) / model.opt.timestep)))
    stroke_frames = max(1, int(0.55 * total_frames))

    cue_tip_start = _site_pos(model, data, "cue_tip_site")
    first_contact: float | None = None
    cue_tip_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_tip")
    cue_ball_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_ball_geom")

    with imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=9) as writer:
        for frame in range(total_frames):
            phase = min(1.0, frame / stroke_frames)
            for _ in range(sim_steps_per_frame):
                _guided_step(model, data, dofs, qpos_ids, initial_targets, phase, args.stroke, args.damping, args.max_step)
                for con_id in range(data.ncon):
                    contact = data.contact[con_id]
                    if {contact.geom1, contact.geom2} == {cue_tip_geom, cue_ball_geom} and first_contact is None:
                        first_contact = float(data.time)

            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())

    cue_tip_delta = _site_pos(model, data, "cue_tip_site") - cue_tip_start
    renderer.close()
    print(f"wrote={args.output}")
    print(f"cue_tip_delta={cue_tip_delta}")
    print(f"first_contact_time={first_contact}")


if __name__ == "__main__":
    main()
