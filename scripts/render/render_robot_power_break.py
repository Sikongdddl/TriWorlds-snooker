"""Render a calibrated actuator-level power break from a near top-down view.

The stroke is executed by the dual-arm differential IK controller and the
joint position actuators.  No RL residual action is applied.  A short
backswing lets the actuators build a repeatable forward velocity before the
cue tip reaches the cue ball.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.lowlevel_residual_env import LowLevelResidualEnv  # noqa: E402
from snooker_env.pipeline_types import CueCommand, Pose3D  # noqa: E402


ARM_SIDES = ("left", "right")
ARM_JOINT_COUNT = 6


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(path, size=size) if path.exists() else ImageFont.load_default()


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.35
    camera.azimuth = -90.0
    camera.elevation = -88.0
    camera.lookat[:] = (0.0, 0.0, 0.76)
    return camera


def _hide_debug_groups(renderer: mujoco.Renderer) -> None:
    option = getattr(renderer, "scene_option", None)
    if option is None:
        option = getattr(renderer, "_scene_option", None)
    if option is not None:
        option.geomgroup[3] = 0
        option.geomgroup[4] = 0
        option.sitegroup[3] = 0


def _move_overhead_light_to_hidden_group(model: mujoco.MjModel) -> None:
    """Keep the decorative ceiling lamp from occluding a top-down camera."""

    light_geom = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "ceiling_light_low_light_mat_0_visual",
    )
    if light_geom >= 0:
        model.geom_group[light_geom] = 4


def _smoothstep(value: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _configure_power_actuators(model: mujoco.MjModel, kp: float, force_limit: float) -> None:
    """Configure an explicit power-stroke preset without changing the XML."""

    for side in ARM_SIDES:
        for joint_index in range(1, ARM_JOINT_COUNT + 1):
            name = f"{side}_arm_joint{joint_index}_pos"
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise ValueError(f"Missing actuator: {name}")
            model.actuator_gainprm[actuator_id, 0] = kp
            model.actuator_biasprm[actuator_id, 1] = -kp
            model.actuator_forcerange[actuator_id] = (-force_limit, force_limit)


def _annotate(
    frame: np.ndarray,
    *,
    elapsed: float,
    first_cue_contact: float | None,
    first_rack_contact: float | None,
    peak_cue_ball_speed: float,
    peak_rack_speed: float,
    max_grip_error: float,
    event_count: int,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, 112), fill=(5, 8, 10, 225))
    draw.text((18, 8), "ROBOT POWER BREAK - TOP VIEW", font=_font(23), fill=(255, 255, 255, 255))
    draw.text(
        (18, 40),
        "nominal dual-arm IK -> joint position actuators (RL residual = 0)",
        font=_font(15),
        fill=(180, 220, 255, 255),
    )
    cue_text = "waiting" if first_cue_contact is None else f"{first_cue_contact:.3f}s"
    rack_text = "waiting" if first_rack_contact is None else f"{first_rack_contact:.3f}s"
    draw.text(
        (18, 69),
        f"t={elapsed:4.2f}s   cue contact={cue_text}   rack contact={rack_text}   events={event_count}",
        font=_font(14),
        fill=(255, 220, 140, 255),
    )
    draw.text(
        (18, 91),
        f"cue-ball peak={peak_cue_ball_speed:.2f} m/s   rack peak={peak_rack_speed:.2f} m/s   grip max={max_grip_error * 1000.0:.1f} mm",
        font=_font(14),
        fill=(185, 255, 190, 255),
    )
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "scene_pool_asset.xml")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "collision_validation" / "robot_power_break_topdown.mp4",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=5.2)
    parser.add_argument("--settle-seconds", type=float, default=0.25)
    parser.add_argument("--backswing", type=float, default=0.080)
    parser.add_argument("--backswing-seconds", type=float, default=0.70)
    parser.add_argument("--follow-through", type=float, default=0.150)
    parser.add_argument("--stroke-seconds", type=float, default=0.30)
    parser.add_argument("--aim-y", type=float, default=-0.0365)
    parser.add_argument("--aim-z", type=float, default=0.0080)
    parser.add_argument("--arm-kp", type=float, default=320.0)
    parser.add_argument("--arm-force-limit", type=float, default=320.0)
    args = parser.parse_args()

    positive = {
        "--seconds": args.seconds,
        "--settle-seconds": args.settle_seconds,
        "--backswing": args.backswing,
        "--backswing-seconds": args.backswing_seconds,
        "--follow-through": args.follow_through,
        "--stroke-seconds": args.stroke_seconds,
        "--arm-kp": args.arm_kp,
        "--arm-force-limit": args.arm_force_limit,
    }
    for name, value in positive.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive.")

    env = LowLevelResidualEnv(args.model)
    env.reset(seed=0)
    _configure_power_actuators(env.model, args.arm_kp, args.arm_force_limit)
    _move_overhead_light_to_hidden_group(env.model)

    # Let the rack and the held cue settle before measuring the stroke.
    env.executor.apply(env.data, env.controller.joint_positions(env.data))
    settle_steps = int(round(args.settle_seconds / env.model.opt.timestep))
    for _ in range(settle_steps):
        mujoco.mj_step(env.model, env.data)

    env.controller.reset_reference(env.data)
    start_pose = env._cue_pose()
    env.contact_monitor.reset()
    stroke_start_time = float(env.data.time)
    next_control_time = stroke_start_time

    rack_dofs: list[int] = []
    for ball_index in range(16):
        joint_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"object_ball_{ball_index}_free",
        )
        if joint_id >= 0:
            rack_dofs.append(int(env.model.jnt_dofadr[joint_id]))

    cue_tip_site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "cue_tip_site")
    first_cue_contact: float | None = None
    first_rack_contact: float | None = None
    peak_cue_ball_speed = 0.0
    peak_rack_speed = 0.0
    peak_cue_tip_speed = 0.0
    max_grip_error = env._grip_error()

    def advance(target_elapsed: float) -> None:
        nonlocal next_control_time
        nonlocal first_cue_contact, first_rack_contact
        nonlocal peak_cue_ball_speed, peak_rack_speed, peak_cue_tip_speed, max_grip_error

        target_time = stroke_start_time + target_elapsed
        while env.data.time + 0.5 * env.model.opt.timestep < target_time:
            elapsed = float(env.data.time - stroke_start_time)
            if env.data.time + 1e-12 >= next_control_time:
                if elapsed < args.backswing_seconds:
                    fraction = _smoothstep(elapsed / args.backswing_seconds)
                    x_offset = -args.backswing * fraction
                    y_offset = 0.0
                    z_offset = 0.0
                elif elapsed < args.backswing_seconds + args.stroke_seconds:
                    fraction = _smoothstep(
                        (elapsed - args.backswing_seconds) / args.stroke_seconds
                    )
                    x_offset = -args.backswing + (args.backswing + args.follow_through) * fraction
                    y_offset = args.aim_y * fraction
                    z_offset = args.aim_z * fraction
                else:
                    x_offset = args.follow_through
                    y_offset = args.aim_y
                    z_offset = args.aim_z

                desired_pose = Pose3D(
                    position=start_pose.position + np.array([x_offset, y_offset, z_offset]),
                    quat_wxyz=start_pose.quat_wxyz.copy(),
                )
                command = CueCommand(
                    pose=desired_pose,
                    linear_velocity=np.zeros(3, dtype=np.float64),
                    angular_velocity=np.zeros(3, dtype=np.float64),
                    duration=env.control_dt,
                    debug_label="robot_power_break",
                )
                env.executor.apply(env.data, env.controller.act(command, env.data))
                next_control_time += env.control_dt

            mujoco.mj_step(env.model, env.data)
            new_events = env.contact_monitor.scan(env.data)
            event_elapsed = float(env.data.time - stroke_start_time)
            for event in new_events:
                if event.kind == "cue_tip_ball" and first_cue_contact is None:
                    first_cue_contact = event_elapsed
                elif event.kind == "ball_ball" and first_rack_contact is None:
                    first_rack_contact = event_elapsed

            cue_ball_speed = float(
                np.linalg.norm(env.data.qvel[env.cue_ball_dof_id:env.cue_ball_dof_id + 2])
            )
            peak_cue_ball_speed = max(peak_cue_ball_speed, cue_ball_speed)
            if first_rack_contact is not None and rack_dofs:
                peak_rack_speed = max(
                    peak_rack_speed,
                    max(float(np.linalg.norm(env.data.qvel[dof:dof + 2])) for dof in rack_dofs),
                )
            tip_velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                env.model,
                env.data,
                mujoco.mjtObj.mjOBJ_SITE,
                cue_tip_site,
                tip_velocity,
                0,
            )
            peak_cue_tip_speed = max(peak_cue_tip_speed, float(np.linalg.norm(tip_velocity[3:])))
            max_grip_error = max(max_grip_error, env._grip_error())

    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, args.width)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(env.model, width=args.width, height=args.height)
    _hide_debug_groups(renderer)
    camera = _camera()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=9) as writer:
        frame_count = int(round(args.seconds * args.fps))
        for frame_index in range(frame_count):
            elapsed = frame_index / args.fps
            advance(elapsed)
            renderer.update_scene(env.data, camera=camera)
            frame = _annotate(
                renderer.render(),
                elapsed=elapsed,
                first_cue_contact=first_cue_contact,
                first_rack_contact=first_rack_contact,
                peak_cue_ball_speed=peak_cue_ball_speed,
                peak_rack_speed=peak_rack_speed,
                max_grip_error=max_grip_error,
                event_count=len(env.contact_monitor.events),
            )
            writer.append_data(frame)

    renderer.close()
    env.close()
    print(f"wrote={args.output}")
    print(f"first_cue_contact={first_cue_contact}")
    print(f"first_rack_contact={first_rack_contact}")
    print(f"peak_cue_tip_speed={peak_cue_tip_speed:.6f}")
    print(f"peak_cue_ball_speed={peak_cue_ball_speed:.6f}")
    print(f"peak_rack_speed={peak_rack_speed:.6f}")
    print(f"max_grip_error={max_grip_error:.6f}")
    print(f"contact_events={len(env.contact_monitor.events)}")


if __name__ == "__main__":
    main()
