"""Render a mid-level robot-free pot shot rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL, MidLevelCueEnv  # noqa: E402
from snooker_env.pipeline_mid_level import PotShotPolicy  # noqa: E402
from snooker_env.pipeline_types import ShotIntent, SkillCommand, SkillId  # noqa: E402


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.25
    camera.azimuth = -92.0
    camera.elevation = -18.0
    camera.lookat[:] = (-0.35, 0.05, 0.79)
    return camera


def _contact_flags(env: MidLevelCueEnv) -> tuple[bool, bool]:
    return env._contact_flags()


def _stability_flags(env: MidLevelCueEnv) -> tuple[bool, bool]:
    return env._stability_flags()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a robot-free mid-level pot shot.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--output", type=Path, default=Path("outputs/videos_midlevel/midlevel_pot_shot.mp4"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--settle-time", type=float, default=1.2)
    parser.add_argument("--target-speed", type=float, default=0.45)
    parser.add_argument("--cue-ball-x", type=float, default=-0.616)
    parser.add_argument("--cue-ball-y", type=float, default=0.0)
    parser.add_argument("--object-ball-x", type=float, default=-0.18)
    parser.add_argument("--object-ball-y", type=float, default=0.0)
    args = parser.parse_args()

    env = MidLevelCueEnv(args.model)
    env.reset()
    for joint_name, xy in (
        ("cue_ball_free", (args.cue_ball_x, args.cue_ball_y)),
        ("object_ball_0_free", (args.object_ball_x, args.object_ball_y)),
    ):
        joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Missing joint: {joint_name}")
        qpos_adr = int(env.model.jnt_qposadr[joint_id])
        dof_adr = int(env.model.jnt_dofadr[joint_id])
        env.data.qpos[qpos_adr:qpos_adr + 3] = np.array([xy[0], xy[1], 0.7898], dtype=np.float64)
        env.data.qvel[dof_adr:dof_adr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    state = env.scene_state()
    command = SkillCommand(
        skill_id=SkillId.POT_SHOT,
        intent=ShotIntent(
            cue_ball_name="cue_ball",
            object_ball_name="object_ball_0",
            target_pocket="demo_forward",
            target_speed=args.target_speed,
        ),
    )
    commands = PotShotPolicy().rollout(command, state)

    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, args.width)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    camera = _camera()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    first_cue_contact: float | None = None
    first_ball_contact: float | None = None
    constraint_projection_count = 0
    min_cue_table_clearance = float("inf")
    has_nan = False
    exploded = False

    sim_steps_per_frame = max(1, int(round((1.0 / args.fps) / env.model.opt.timestep)))

    if commands:
        first = commands[0]
        first_quat = first.pose.quat_wxyz.astype(np.float64).copy()
        first_quat /= max(float(np.linalg.norm(first_quat)), 1e-9)
        first_pos, clearance, projected = env.project_cue_position(first.pose.position.astype(np.float64), first_quat)
        constraint_projection_count += int(projected)
        min_cue_table_clearance = min(min_cue_table_clearance, clearance)
        env._set_cue_state(
            first_pos,
            first_quat,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )
        mujoco.mj_forward(env.model, env.data)

    with imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=9) as writer:
        for cue_command in commands:
            total_steps = env.action_repeat
            command_dt = env.command_dt
            rendered_steps = 0
            current_pos = env.data.qpos[env.cue_qpos_adr:env.cue_qpos_adr + 3].copy()
            target_pos = cue_command.pose.position.astype(np.float64).copy()
            quat = cue_command.pose.quat_wxyz.astype(np.float64).copy()
            quat /= max(float(np.linalg.norm(quat)), 1e-9)
            command_speed = float(np.linalg.norm(cue_command.linear_velocity))
            if command_speed < 1e-9:
                segment_velocity = (target_pos - current_pos) / command_dt
            else:
                segment_velocity = cue_command.linear_velocity

            while rendered_steps < total_steps:
                for _ in range(sim_steps_per_frame):
                    if rendered_steps >= total_steps:
                        break
                    elapsed = rendered_steps * env.model.opt.timestep
                    if command_speed < 1e-9:
                        alpha = min(1.0, elapsed / command_dt)
                        pos = (1.0 - alpha) * current_pos + alpha * target_pos
                    else:
                        pos = target_pos + cue_command.linear_velocity * elapsed
                    pos, clearance, projected = env.project_cue_position(pos, quat)
                    if projected:
                        constraint_projection_count += 1
                    env._set_cue_state(pos, quat, segment_velocity, cue_command.angular_velocity)
                    mujoco.mj_step(env.model, env.data)
                    min_cue_table_clearance = min(min_cue_table_clearance, clearance)
                    cue_contact, ball_contact = _contact_flags(env)
                    if cue_contact and first_cue_contact is None:
                        first_cue_contact = float(env.data.time)
                    if ball_contact and first_ball_contact is None:
                        first_ball_contact = float(env.data.time)
                    has_nan, exploded = _stability_flags(env)
                    rendered_steps += 1
                    if has_nan or exploded:
                        break
                renderer.update_scene(env.data, camera=camera)
                writer.append_data(renderer.render())
                if has_nan or exploded:
                    break
            if has_nan or exploded:
                break

        settle_steps = max(0, int(round(args.settle_time / env.model.opt.timestep)))
        rendered_steps = 0
        hold_pos = env.data.qpos[env.cue_qpos_adr:env.cue_qpos_adr + 3].copy()
        hold_quat = env.data.qpos[env.cue_qpos_adr + 3:env.cue_qpos_adr + 7].copy()
        hold_quat /= max(float(np.linalg.norm(hold_quat)), 1e-9)
        while rendered_steps < settle_steps and not (has_nan or exploded):
            for _ in range(sim_steps_per_frame):
                if rendered_steps >= settle_steps:
                    break
                hold_pos, clearance, projected = env.project_cue_position(hold_pos, hold_quat)
                if projected:
                    constraint_projection_count += 1
                env._set_cue_state(
                    hold_pos,
                    hold_quat,
                    np.zeros(3, dtype=np.float64),
                    np.zeros(3, dtype=np.float64),
                )
                mujoco.mj_step(env.model, env.data)
                min_cue_table_clearance = min(min_cue_table_clearance, clearance)
                cue_contact, ball_contact = _contact_flags(env)
                if cue_contact and first_cue_contact is None:
                    first_cue_contact = float(env.data.time)
                if ball_contact and first_ball_contact is None:
                    first_ball_contact = float(env.data.time)
                has_nan, exploded = _stability_flags(env)
                rendered_steps += 1
                if has_nan or exploded:
                    break
            renderer.update_scene(env.data, camera=camera)
            writer.append_data(renderer.render())

    renderer.close()
    cue_ball = env._ball_state("cue_ball", "cue_ball_free")
    object_ball = env._ball_state("object_ball_0", "object_ball_0_free")
    print(f"wrote={args.output}")
    print(f"commands={len(commands)}")
    print(f"first_cue_tip_cue_ball_contact={first_cue_contact}")
    print(f"first_cue_ball_object_ball_contact={first_ball_contact}")
    print(f"constraint_projection_count={constraint_projection_count}")
    print(f"min_cue_table_clearance={min_cue_table_clearance}")
    print(f"cue_ball_final_position={cue_ball.position}")
    print(f"object_ball_final_position={object_ball.position}")
    print(f"has_nan={has_nan}")
    print(f"exploded={exploded}")

    if first_cue_contact is None:
        raise RuntimeError("Render failed: cue did not contact cue ball.")
    if has_nan or exploded:
        raise RuntimeError("Render failed: unstable simulation.")
    if min_cue_table_clearance < -1e-6:
        raise RuntimeError("Render failed: cue trajectory penetrated table/cushion proxies after projection.")


if __name__ == "__main__":
    main()
