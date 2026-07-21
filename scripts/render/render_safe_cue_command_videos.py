"""Render actuator-level videos for conservative cue commands that hit the cue ball."""

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

from snooker_env.lowlevel_residual_env import LowLevelResidualEnv  # noqa: E402
from snooker_env.pipeline_types import CueCommand, Pose3D  # noqa: E402


@dataclass(frozen=True)
class SafeCase:
    label: str
    displacement: np.ndarray
    linear_velocity: np.ndarray
    duration: float


@dataclass
class Panel:
    case: SafeCase
    env: LowLevelResidualEnv
    renderer: mujoco.Renderer
    camera: mujoco.MjvCamera
    done: bool = False


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.15
    camera.azimuth = -38.0
    camera.elevation = -14.0
    camera.lookat[:] = (-0.75, 0.18, 0.78)
    return camera


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(path, size=size) if path.exists() else ImageFont.load_default()


def _hide_debug_groups(renderer: mujoco.Renderer) -> None:
    option = getattr(renderer, "scene_option", None)
    if option is None:
        option = getattr(renderer, "_scene_option", None)
    if option is not None:
        option.geomgroup[3] = 0
        option.sitegroup[3] = 0


def _cases(group: str) -> tuple[SafeCase, ...]:
    if group == "offsets":
        return (
            SafeCase("left offset: py=-0.010 m", np.array([0.060, -0.010, 0.005]), np.array([0.080, -0.010, 0.0]), 1.50),
            SafeCase("center: py=0.000 m", np.array([0.060, 0.000, 0.005]), np.array([0.080, 0.000, 0.0]), 1.50),
            SafeCase("right offset: py=+0.010 m", np.array([0.060, 0.010, 0.005]), np.array([0.080, 0.010, 0.0]), 1.50),
        )
    if group == "heights":
        return (
            SafeCase("low contact: pz=-0.005 m", np.array([0.060, 0.0, -0.005]), np.array([0.080, 0.0, -0.010]), 1.50),
            SafeCase("center-high: pz=+0.005 m", np.array([0.060, 0.0, 0.005]), np.array([0.080, 0.0, 0.0]), 1.50),
            SafeCase("high contact: pz=+0.015 m", np.array([0.060, 0.0, 0.015]), np.array([0.080, 0.0, 0.010]), 1.50),
        )
    if group == "durations":
        displacement = np.array([0.060, 0.0, 0.005])
        velocity = np.array([0.080, 0.0, 0.0])
        return (
            SafeCase("duration=1.50 s", displacement, velocity, 1.50),
            SafeCase("duration=1.75 s", displacement, velocity, 1.75),
            SafeCase("duration=2.00 s", displacement, velocity, 2.00),
        )
    raise ValueError(group)


def _make_panel(model_path: Path, case: SafeCase, width: int, height: int) -> Panel:
    env = LowLevelResidualEnv(model_path)
    env.reset(seed=0)
    start = env._cue_pose()
    command = CueCommand(
        pose=Pose3D(
            position=start.position + case.displacement,
            quat_wxyz=start.quat_wxyz.copy(),
        ),
        linear_velocity=case.linear_velocity.copy(),
        angular_velocity=np.zeros(3, dtype=np.float64),
        duration=case.duration,
        debug_label=case.label,
    )
    env.reset(seed=0, options={"command": command})
    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, width)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(env.model, width=width, height=height)
    _hide_debug_groups(renderer)
    return Panel(case, env, renderer, _camera())


def _advance(panel: Panel, target_time: float) -> None:
    zero_action = np.zeros(panel.env.action_space.shape, dtype=np.float32)
    while panel.env.data.time + 0.5 * panel.env.model.opt.timestep < target_time:
        if not panel.done:
            _, _, terminated, truncated, _ = panel.env.step(zero_action)
            panel.done = terminated or truncated
        else:
            step_target = panel.env.data.time + panel.env.control_dt
            while panel.env.data.time + 0.5 * panel.env.model.opt.timestep < step_target:
                mujoco.mj_step(panel.env.model, panel.env.data)


def _annotate(frame: np.ndarray, panel: Panel, time: float) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 82), fill=(10, 10, 10))
    draw.text((14, 5), panel.case.label, font=_font(19), fill=(255, 255, 255))
    draw.text(
        (14, 31),
        f"dp={np.round(panel.case.displacement, 3).tolist()}  v={np.round(panel.case.linear_velocity, 3).tolist()}",
        font=_font(13),
        fill=(190, 220, 255),
    )
    contact = panel.env._first_cue_ball_contact_time
    contact_text = "contact=no" if contact is None else f"contact=yes @{contact:.3f}s"
    draw.text(
        (14, 55),
        f"t={time:.2f}s  {contact_text}  peak={panel.env._peak_cue_ball_speed:.3f} m/s",
        font=_font(13),
        fill=(255, 210, 120) if contact is not None else (180, 180, 180),
    )
    return np.asarray(image)


def render_group(
    model_path: Path,
    output: Path,
    group: str,
    width: int,
    height: int,
    fps: int,
    seconds: float,
) -> None:
    panels = [_make_panel(model_path, case, width, height) for case in _cases(group)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=9) as writer:
        for frame_index in range(int(round(seconds * fps))):
            frame_time = frame_index / fps
            frames: list[np.ndarray] = []
            for panel in panels:
                _advance(panel, frame_time)
                panel.renderer.update_scene(panel.env.data, camera=panel.camera)
                frames.append(_annotate(panel.renderer.render(), panel, frame_time))
            writer.append_data(np.concatenate(frames, axis=1))

    for panel in panels:
        print(
            f"{group}:{panel.case.label}: contact={panel.env._first_cue_ball_contact_time} "
            f"peak_ball_speed={panel.env._peak_cue_ball_speed:.6f}"
        )
        panel.renderer.close()
        panel.env.close()
    print(f"wrote={output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "scene_pool_asset.xml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "safe_cue_commands")
    parser.add_argument("--group", choices=("offsets", "heights", "durations", "all"), default="all")
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=2.6)
    args = parser.parse_args()

    groups = ("offsets", "heights", "durations") if args.group == "all" else (args.group,)
    for group in groups:
        render_group(
            args.model,
            args.output_dir / f"safe_cue_{group}.mp4",
            group,
            args.panel_width,
            args.height,
            args.fps,
            args.seconds,
        )


if __name__ == "__main__":
    main()

