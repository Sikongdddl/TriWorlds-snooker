"""Render robot-actuated sweeps for every independent CueCommand component."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.init_pose import set_lift_grip_ready_pose  # noqa: E402
from snooker_env.lowlevel_control import (  # noqa: E402
    DualArmDifferentialIKController,
    JointPositionExecutor,
)
from snooker_env.pipeline_types import CueCommand, Pose3D  # noqa: E402
from snooker_env.scene import load_model  # noqa: E402


@dataclass(frozen=True)
class SweepCase:
    label: str
    kind: str
    value: np.ndarray
    duration: float


@dataclass
class Panel:
    case: SweepCase
    model: mujoco.MjModel
    data: mujoco.MjData
    controller: DualArmDifferentialIKController
    executor: JointPositionExecutor
    renderer: mujoco.Renderer
    camera: mujoco.MjvCamera
    start_pose: Pose3D


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.15
    camera.azimuth = -38.0
    camera.elevation = -14.0
    camera.lookat[:] = (-0.75, 0.18, 0.78)
    return camera


def _hide_debug_groups(renderer: mujoco.Renderer) -> None:
    scene_option = getattr(renderer, "scene_option", None)
    if scene_option is None:
        scene_option = getattr(renderer, "_scene_option", None)
    if scene_option is not None:
        scene_option.geomgroup[3] = 0
        scene_option.sitegroup[3] = 0


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    result = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    half = 0.5 * angle
    return np.concatenate(([np.cos(half)], axis * np.sin(half)))


def _smooth_phase(time: float, duration: float) -> tuple[float, float]:
    phase = float(np.clip(time / duration, 0.0, 1.0))
    smooth = phase * phase * (3.0 - 2.0 * phase)
    smooth_rate = 0.0 if phase >= 1.0 else 6.0 * phase * (1.0 - phase) / duration
    return smooth, smooth_rate


def _command_at(panel: Panel, time: float) -> CueCommand:
    case = panel.case
    start = panel.start_pose
    remaining = max(case.duration - time, 1e-6)
    zero = np.zeros(3, dtype=np.float64)

    if case.kind in ("position", "duration"):
        smooth, smooth_rate = _smooth_phase(time, case.duration)
        pose = Pose3D(start.position + smooth * case.value, start.quat_wxyz.copy())
        return CueCommand(pose, smooth_rate * case.value, zero, duration=remaining, debug_label=case.label)

    if case.kind == "orientation":
        axis = case.value[:3]
        angle = float(case.value[3])
        smooth, smooth_rate = _smooth_phase(time, case.duration)
        quat = _quat_multiply(_axis_angle_quat(axis, smooth * angle), start.quat_wxyz)
        return CueCommand(
            Pose3D(start.position.copy(), quat),
            zero,
            axis * angle * smooth_rate,
            duration=remaining,
            debug_label=case.label,
        )

    if case.kind == "linear_velocity":
        active_time = min(time, case.duration)
        velocity = case.value if time < case.duration else zero
        pose = Pose3D(start.position + case.value * active_time, start.quat_wxyz.copy())
        return CueCommand(pose, velocity, zero, duration=remaining, debug_label=case.label)

    if case.kind == "angular_velocity":
        active_time = min(time, case.duration)
        speed = float(np.linalg.norm(case.value))
        axis = case.value / max(speed, 1e-12)
        quat = _quat_multiply(_axis_angle_quat(axis, speed * active_time), start.quat_wxyz)
        angular_velocity = case.value if time < case.duration else zero
        return CueCommand(
            Pose3D(start.position.copy(), quat),
            zero,
            angular_velocity,
            duration=remaining,
            debug_label=case.label,
        )

    raise ValueError(f"Unknown sweep kind: {case.kind}")


def _make_panel(model_path: Path, case: SweepCase, width: int, height: int) -> Panel:
    model = load_model(model_path)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    # These videos isolate command tracking, so cue/table/ball contacts are disabled.
    for geom_name in ("cue_shaft", "cue_tip"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0

    data = mujoco.MjData(model)
    set_lift_grip_ready_pose(model, data)
    controller = DualArmDifferentialIKController(model, max_joint_step=0.025)
    controller.reset_reference(data)
    executor = JointPositionExecutor(model, controller.joint_names)
    renderer = mujoco.Renderer(model, width=width, height=height)
    _hide_debug_groups(renderer)
    cue_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cue_body")
    start_pose = Pose3D(data.xpos[cue_body].copy(), data.xquat[cue_body].copy())
    return Panel(case, model, data, controller, executor, renderer, _camera(), start_pose)


def _font(size: int) -> ImageFont.ImageFont:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(font_path, size=size) if font_path.exists() else ImageFont.load_default()


def _annotate(frame: np.ndarray, case: SweepCase, time: float) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 58), fill=(10, 10, 10))
    draw.text((16, 8), case.label, font=_font(20), fill=(255, 255, 255))
    draw.text((16, 34), f"t={time:.2f}s  command duration={case.duration:.2f}s", font=_font(14), fill=(190, 220, 255))
    return np.asarray(image)


def _cases(group: str) -> tuple[SweepCase, ...]:
    if group == "position":
        return (
            SweepCase("position px: +0.050 m", group, np.array([0.050, 0.0, 0.0]), 0.8),
            SweepCase("position py: +0.040 m", group, np.array([0.0, 0.040, 0.0]), 0.8),
            SweepCase("position pz: +0.040 m", group, np.array([0.0, 0.0, 0.040]), 0.8),
        )
    if group == "orientation":
        angle = np.deg2rad(12.0)
        return (
            SweepCase("quaternion qx: roll +12 deg", group, np.array([1.0, 0.0, 0.0, angle]), 0.9),
            SweepCase("quaternion qy: pitch +12 deg", group, np.array([0.0, 1.0, 0.0, angle]), 0.9),
            SweepCase("quaternion qz: yaw +12 deg", group, np.array([0.0, 0.0, 1.0, angle]), 0.9),
        )
    if group == "linear_velocity":
        return (
            SweepCase("linear velocity vx: +2.96 m/s", group, np.array([2.96, 0.0, 0.0]), 0.6),
            SweepCase("linear velocity vy: +0.83 m/s", group, np.array([0.0, 0.83, 0.0]), 0.6),
            SweepCase("linear velocity vz: +0.74 m/s", group, np.array([0.0, 0.0, 0.74]), 0.6),
        )
    if group == "angular_velocity":
        return (
            SweepCase("angular velocity wx: +0.40 rad/s", group, np.array([0.40, 0.0, 0.0]), 0.7),
            SweepCase("angular velocity wy: +0.30 rad/s", group, np.array([0.0, 0.30, 0.0]), 0.7),
            SweepCase("angular velocity wz: +0.30 rad/s", group, np.array([0.0, 0.0, 0.30]), 0.7),
        )
    if group == "duration":
        displacement = np.array([0.060, 0.0, 0.0])
        return (
            SweepCase("duration: 0.30 s", group, displacement, 0.30),
            SweepCase("duration: 0.70 s", group, displacement, 0.70),
            SweepCase("duration: 1.20 s", group, displacement, 1.20),
        )
    raise ValueError(group)


def render_group(model_path: Path, output: Path, group: str, width: int, height: int, fps: int, seconds: float) -> None:
    panels = [_make_panel(model_path, case, width, height) for case in _cases(group)]
    output.parent.mkdir(parents=True, exist_ok=True)
    control_dt = 0.01
    total_frames = int(round(seconds * fps))
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=9) as writer:
        for frame_index in range(total_frames):
            frame_time = frame_index / fps
            frames: list[np.ndarray] = []
            for panel in panels:
                while panel.data.time + 0.5 * panel.model.opt.timestep < frame_time:
                    command = _command_at(panel, float(panel.data.time))
                    nominal = panel.controller.act(command, panel.data, control_dt=control_dt)
                    panel.executor.apply(panel.data, nominal)
                    target_time = panel.data.time + control_dt
                    while panel.data.time + 0.5 * panel.model.opt.timestep < target_time:
                        mujoco.mj_step(panel.model, panel.data)
                panel.renderer.update_scene(panel.data, camera=panel.camera)
                frames.append(_annotate(panel.renderer.render(), panel.case, frame_time))
            writer.append_data(np.concatenate(frames, axis=1))

    for panel in panels:
        cue_body = mujoco.mj_name2id(panel.model, mujoco.mjtObj.mjOBJ_BODY, "cue_body")
        delta = panel.data.xpos[cue_body] - panel.start_pose.position
        print(f"{group}:{panel.case.label}: actual_position_delta={delta}")
        panel.renderer.close()
    print(f"wrote={output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "scene_pool_asset.xml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "cue_command_sweeps")
    parser.add_argument(
        "--group",
        choices=("position", "orientation", "linear_velocity", "angular_velocity", "duration", "all"),
        default="all",
    )
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    groups = ("position", "orientation", "linear_velocity", "angular_velocity", "duration") if args.group == "all" else (args.group,)
    for group in groups:
        render_group(
            args.model,
            args.output_dir / f"cue_command_{group}.mp4",
            group,
            args.panel_width,
            args.height,
            args.fps,
            args.seconds,
        )


if __name__ == "__main__":
    main()
