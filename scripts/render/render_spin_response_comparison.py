"""Render side-by-side cue spin response comparisons."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL, MidLevelCueEnv  # noqa: E402
from snooker_env.pipeline_types import CueCommand, Pose3D  # noqa: E402


BALL_RADIUS = 0.028575
CUE_TIP_OFFSET = 0.725
CUE_TIP_RADIUS = 0.009


@dataclass(frozen=True)
class SpinCase:
    label: str
    offset_y: float
    offset_z: float


@dataclass
class PanelState:
    env: MidLevelCueEnv
    command: CueCommand
    renderer: mujoco.Renderer
    camera: mujoco.MjvCamera
    command_step: int = 0
    settle_step: int = 0
    current_pos: np.ndarray | None = None
    target_pos: np.ndarray | None = None
    quat: np.ndarray | None = None
    segment_velocity: np.ndarray | None = None
    trail: list[tuple[float, float]] | None = None


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.05
    camera.azimuth = -90.0
    camera.elevation = -32.0
    camera.lookat[:] = (0.10, 0.0, 0.79)
    return camera


def _quat_for_x_axis(elevation: float) -> np.ndarray:
    half_elevation = 0.5 * elevation
    quat = np.array([np.cos(half_elevation), 0.0, np.sin(half_elevation), 0.0], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-9)
    return quat


def _x_axis_from_quat(quat: np.ndarray) -> np.ndarray:
    rot_flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rot_flat, quat)
    axis = rot_flat.reshape(3, 3)[:, 0].copy()
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    return axis


def _set_ball(env: MidLevelCueEnv, joint_name: str, position: np.ndarray) -> None:
    joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Missing joint: {joint_name}")
    qadr = int(env.model.jnt_qposadr[joint_id])
    dadr = int(env.model.jnt_dofadr[joint_id])
    env.data.qpos[qadr:qadr + 3] = position
    env.data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    env.data.qvel[dadr:dadr + 6] = 0.0


def _make_stroke(case: SpinCase, cue_ball_pos: np.ndarray, speed: float, elevation: float, gap: float) -> CueCommand:
    quat = _quat_for_x_axis(elevation)
    axis = _x_axis_from_quat(quat)
    offset = np.array([0.0, case.offset_y, case.offset_z], dtype=np.float64)
    tip_start = cue_ball_pos - axis * (BALL_RADIUS + CUE_TIP_RADIUS + gap) + offset
    cue_body_pos = tip_start - axis * CUE_TIP_OFFSET
    return CueCommand(
        pose=Pose3D(position=cue_body_pos, quat_wxyz=quat),
        linear_velocity=axis * speed,
        angular_velocity=np.zeros(3, dtype=np.float64),
        debug_label=case.label,
    )


def _init_panel(
    case: SpinCase,
    args: argparse.Namespace,
    width: int,
    height: int,
) -> PanelState:
    env = MidLevelCueEnv(args.model, action_repeat=args.action_repeat)
    env.reset()
    _set_ball(env, "cue_ball_free", np.array([args.cue_ball_x, args.cue_ball_y, 0.7898], dtype=np.float64))
    _set_ball(env, "object_ball_0_free", np.array([args.object_ball_x, args.object_ball_y, 0.7898], dtype=np.float64))
    mujoco.mj_forward(env.model, env.data)

    cue_ball_pos = env.scene_state().balls["cue_ball"].position.copy()
    command = _make_stroke(case, cue_ball_pos, args.speed, args.elevation, args.gap)
    quat = command.pose.quat_wxyz.astype(np.float64).copy()
    first_pos, _, _ = env.project_cue_position(command.pose.position.astype(np.float64), quat)
    env._set_cue_state(first_pos, quat, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    mujoco.mj_forward(env.model, env.data)

    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, width)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    return PanelState(
        env=env,
        command=command,
        renderer=renderer,
        camera=_camera(),
        current_pos=env.data.qpos[env.cue_qpos_adr:env.cue_qpos_adr + 3].copy(),
        target_pos=command.pose.position.astype(np.float64).copy(),
        quat=quat,
        segment_velocity=command.linear_velocity,
        trail=[],
    )


def _world_xy_to_panel(point: np.ndarray, width: int, height: int) -> tuple[int, int]:
    x_min, x_max = -0.78, 1.08
    y_min, y_max = -0.34, 0.34
    px = int(np.clip((point[0] - x_min) / (x_max - x_min), 0.0, 1.0) * width)
    py = int((1.0 - np.clip((point[1] - y_min) / (y_max - y_min), 0.0, 1.0)) * height)
    return px, py


def _step_panel(panel: PanelState, settle_steps: int) -> None:
    env = panel.env
    command = panel.command
    if panel.command_step < env.action_repeat:
        elapsed = panel.command_step * env.model.opt.timestep
        assert panel.target_pos is not None
        assert panel.quat is not None
        pos = panel.target_pos + command.linear_velocity * elapsed
        pos, _, _ = env.project_cue_position(pos, panel.quat)
        env._set_cue_state(pos, panel.quat, command.linear_velocity, command.angular_velocity)
        panel.command_step += 1
    elif panel.settle_step < settle_steps:
        hold_pos = env.data.qpos[env.cue_qpos_adr:env.cue_qpos_adr + 3].copy()
        hold_quat = env.data.qpos[env.cue_qpos_adr + 3:env.cue_qpos_adr + 7].copy()
        hold_quat /= max(float(np.linalg.norm(hold_quat)), 1e-9)
        hold_pos, _, _ = env.project_cue_position(hold_pos, hold_quat)
        env._set_cue_state(hold_pos, hold_quat, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
        panel.settle_step += 1
    else:
        return

    mujoco.mj_step(env.model, env.data)
    cue_ball = env.scene_state().balls["cue_ball"]
    if panel.trail is not None:
        panel.trail.append((float(cue_ball.position[0]), float(cue_ball.position[1])))
        if len(panel.trail) > 220:
            del panel.trail[: len(panel.trail) - 220]


def _annotate(frame: np.ndarray, panel: PanelState, label: str) -> np.ndarray:
    env = panel.env
    cue_ball = env.scene_state().balls["cue_ball"]
    height, width = frame.shape[:2]
    annotated = frame.copy()
    if panel.trail:
        pts = [
            _world_xy_to_panel(np.array([x, y, 0.0], dtype=np.float64), width, height)
            for x, y in panel.trail
        ]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.line(annotated, p0, p1, (255, 220, 40), 2, lineType=cv2.LINE_AA)
    cv2.rectangle(annotated, (0, 0), (width, 88), (20, 20, 20), thickness=-1)
    speed = float(np.linalg.norm(cue_ball.linear_velocity[:2]))
    omega = cue_ball.angular_velocity
    cv2.putText(annotated, label, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        annotated,
        f"v_xy={speed:.3f} m/s",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 235, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"omega=[{omega[0]:+.1f}, {omega[1]:+.1f}, {omega[2]:+.1f}]",
        (18, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (230, 235, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated


def render_comparison(kind: str, args: argparse.Namespace) -> Path:
    if kind == "vertical":
        cases = (
            SpinCase("High / Top Spin", 0.0, args.offset),
            SpinCase("Center", 0.0, 0.0),
            SpinCase("Low / Draw", 0.0, -args.offset),
        )
        output = args.output_dir / "spin_vertical_top_center_bottom.mp4"
    elif kind == "side":
        cases = (
            SpinCase("Left English", args.offset, 0.0),
            SpinCase("Center", 0.0, 0.0),
            SpinCase("Right English", -args.offset, 0.0),
        )
        output = args.output_dir / "spin_side_left_center_right.mp4"
    else:
        raise ValueError(f"Unknown comparison kind: {kind}")

    panel_width = args.width // 3
    panel_height = args.height
    panels = [_init_panel(case, args, panel_width, panel_height) for case in cases]
    settle_steps = int(round(args.settle_time / panels[0].env.model.opt.timestep))
    sim_steps_per_frame = max(1, int(round((1.0 / args.fps) / panels[0].env.model.opt.timestep)))
    total_steps = panels[0].env.action_repeat + settle_steps
    total_frames = int(np.ceil(total_steps / sim_steps_per_frame))

    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=args.fps, codec="libx264", quality=9) as writer:
        for _ in range(total_frames):
            for _ in range(sim_steps_per_frame):
                for panel in panels:
                    _step_panel(panel, settle_steps)
            frames = []
            for case, panel in zip(cases, panels):
                panel.renderer.update_scene(panel.env.data, camera=panel.camera)
                frame = panel.renderer.render()
                frames.append(_annotate(frame, panel, case.label))
            writer.append_data(np.concatenate(frames, axis=1))

    for panel in panels:
        panel.renderer.close()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/videos_midlevel"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=float, default=0.85)
    parser.add_argument("--offset", type=float, default=0.010)
    parser.add_argument("--gap", type=float, default=0.004)
    parser.add_argument("--elevation", type=float, default=np.deg2rad(8.0))
    parser.add_argument("--action-repeat", type=int, default=360)
    parser.add_argument("--settle-time", type=float, default=2.8)
    parser.add_argument("--cue-ball-x", type=float, default=-0.616)
    parser.add_argument("--cue-ball-y", type=float, default=0.0)
    parser.add_argument("--object-ball-x", type=float, default=-0.18)
    parser.add_argument("--object-ball-y", type=float, default=0.0)
    parser.add_argument("--kind", choices=("vertical", "side", "both"), default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kinds = ("vertical", "side") if args.kind == "both" else (args.kind,)
    for kind in kinds:
        output = render_comparison(kind, args)
        print(f"wrote={output}")


if __name__ == "__main__":
    main()
