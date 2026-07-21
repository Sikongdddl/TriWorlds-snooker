"""Render equal-pose cue strokes whose only changed input is linear velocity."""

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


@dataclass
class Panel:
    target_vx: float
    env: LowLevelResidualEnv
    renderer: mujoco.Renderer
    camera: mujoco.MjvCamera
    cue_tip_site: int
    done: bool = False
    peak_cue_vx: float = 0.0
    peak_tip_vx: float = 0.0
    peak_joint_speed: float = 0.0


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(path, size=size) if path.exists() else ImageFont.load_default()


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.85
    camera.azimuth = -40.0
    camera.elevation = -22.0
    camera.lookat[:] = (-0.70, 0.10, 0.78)
    return camera


def _hide_debug_groups(renderer: mujoco.Renderer) -> None:
    option = getattr(renderer, "scene_option", None)
    if option is None:
        option = getattr(renderer, "_scene_option", None)
    if option is not None:
        option.geomgroup[3] = 0
        option.sitegroup[3] = 0


def _make_panel(
    model_path: Path,
    target_vx: float,
    displacement: float,
    duration: float,
    width: int,
    height: int,
) -> Panel:
    env = LowLevelResidualEnv(model_path)
    env.reset(seed=0)
    start = env._cue_pose()
    command = CueCommand(
        pose=Pose3D(
            position=start.position + np.array([displacement, 0.0, 0.005]),
            quat_wxyz=start.quat_wxyz.copy(),
        ),
        linear_velocity=np.array([target_vx, 0.0, 0.0]),
        angular_velocity=np.zeros(3, dtype=np.float64),
        duration=duration,
        debug_label=f"velocity_ik_vx_{target_vx:.2f}",
    )
    env.reset(seed=0, options={"command": command})
    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, width)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(env.model, width=width, height=height)
    _hide_debug_groups(renderer)
    cue_tip_site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "cue_tip_site")
    return Panel(target_vx, env, renderer, _camera(), cue_tip_site)


def _sample_metrics(panel: Panel) -> None:
    cue_linear, _ = panel.env._cue_velocity()
    panel.peak_cue_vx = max(panel.peak_cue_vx, float(cue_linear[0]))
    tip_velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        panel.env.model,
        panel.env.data,
        mujoco.mjtObj.mjOBJ_SITE,
        panel.cue_tip_site,
        tip_velocity,
        0,
    )
    panel.peak_tip_vx = max(panel.peak_tip_vx, float(tip_velocity[3]))
    joint_speeds = panel.env.controller.joint_velocities(panel.env.data)
    panel.peak_joint_speed = max(panel.peak_joint_speed, float(np.max(np.abs(joint_speeds))))


def _advance(panel: Panel, target_time: float) -> None:
    zero_action = np.zeros(panel.env.action_space.shape, dtype=np.float32)
    while panel.env.data.time + 0.5 * panel.env.model.opt.timestep < target_time:
        if not panel.done:
            _, _, terminated, truncated, _ = panel.env.step(zero_action)
            panel.done = terminated or truncated
            _sample_metrics(panel)
        else:
            step_target = panel.env.data.time + panel.env.control_dt
            while panel.env.data.time + 0.5 * panel.env.model.opt.timestep < step_target:
                mujoco.mj_step(panel.env.model, panel.env.data)
                panel.env.contact_monitor.scan(panel.env.data)
            _sample_metrics(panel)


def _annotate(
    frame: np.ndarray,
    panel: Panel,
    elapsed: float,
    displacement: float,
    duration: float,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 122), fill=(7, 9, 12))
    draw.text((14, 7), f"target linear vx = {panel.target_vx:.2f} m/s", font=_font(20), fill=(255, 255, 255))
    draw.text(
        (14, 34),
        f"same pose: dx={displacement:.3f}m  dz=0.005m  duration={duration:.2f}s",
        font=_font(12),
        fill=(180, 220, 255),
    )
    draw.text(
        (14, 55),
        "velocity-aware dual-arm IK -> 12 joint targets; RL residual=0",
        font=_font(12),
        fill=(190, 255, 195),
    )
    contact = panel.env._first_cue_ball_contact_time
    contact_text = "no" if contact is None else f"{contact:.3f}s"
    draw.text(
        (14, 77),
        f"t={elapsed:.2f}s  contact={contact_text}  cue vx peak={panel.peak_cue_vx:.3f}m/s",
        font=_font(12),
        fill=(255, 220, 145),
    )
    draw.text(
        (14, 99),
        f"tip vx peak={panel.peak_tip_vx:.3f}  ball peak={panel.env._peak_cue_ball_speed:.3f}m/s  joint peak={panel.peak_joint_speed:.2f}rad/s",
        font=_font(12),
        fill=(255, 220, 145),
    )
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "scene_pool_asset.xml")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "velocity_ik_validation" / "linear_velocity_ik_comparison.mp4",
    )
    parser.add_argument("--velocities", type=float, nargs=3, default=(0.0, 0.75, 1.50))
    parser.add_argument("--displacement", type=float, default=0.060)
    parser.add_argument("--duration", type=float, default=1.50)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=2.60)
    args = parser.parse_args()

    panels = [
        _make_panel(
            args.model,
            velocity,
            args.displacement,
            args.duration,
            args.panel_width,
            args.height,
        )
        for velocity in args.velocities
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=9) as writer:
        for frame_index in range(int(round(args.seconds * args.fps))):
            elapsed = frame_index / args.fps
            frames: list[np.ndarray] = []
            for panel in panels:
                _advance(panel, elapsed)
                panel.renderer.update_scene(panel.env.data, camera=panel.camera)
                frames.append(
                    _annotate(
                        panel.renderer.render(),
                        panel,
                        elapsed,
                        args.displacement,
                        args.duration,
                    )
                )
            writer.append_data(np.concatenate(frames, axis=1))

    for panel in panels:
        print(
            f"vx={panel.target_vx:.2f}: cue_vx_peak={panel.peak_cue_vx:.6f} "
            f"tip_vx_peak={panel.peak_tip_vx:.6f} "
            f"ball_peak={panel.env._peak_cue_ball_speed:.6f} "
            f"contact={panel.env._first_cue_ball_contact_time} "
            f"joint_speed_peak={panel.peak_joint_speed:.6f}"
        )
        panel.renderer.close()
        panel.env.close()
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
