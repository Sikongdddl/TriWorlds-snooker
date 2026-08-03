#!/usr/bin/env python3
"""Search for a draw-shot configuration where the cue ball reverses after impact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL, MidLevelCueEnv  # noqa: E402
from snooker_env.pipeline_types import CueCommand, Pose3D  # noqa: E402
from snooker_env.table_geometry import BALL_CENTER_Z  # noqa: E402


BALL_RADIUS = 0.0285
CUE_TIP_OFFSET = 0.725
CUE_TIP_RADIUS = 0.009


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


def _make_command(cue_ball_pos: np.ndarray, speed: float, offset_z: float, elevation: float, gap: float) -> CueCommand:
    quat = _quat_for_x_axis(elevation)
    axis = _x_axis_from_quat(quat)
    tip_start = cue_ball_pos - axis * (BALL_RADIUS + CUE_TIP_RADIUS + gap) + np.array([0.0, 0.0, offset_z])
    cue_body_pos = tip_start - axis * CUE_TIP_OFFSET
    return CueCommand(
        pose=Pose3D(position=cue_body_pos, quat_wxyz=quat),
        linear_velocity=axis * speed,
        angular_velocity=np.zeros(3, dtype=np.float64),
        debug_label="draw_search",
    )


def _run_case(args: argparse.Namespace, speed: float, offset_z: float, object_x: float) -> tuple[float, float, float, float | None, float | None, bool]:
    env = MidLevelCueEnv(args.model, action_repeat=args.action_repeat)
    env.reset()
    cue_start = np.array([args.cue_ball_x, 0.0, BALL_CENTER_Z], dtype=np.float64)
    object_start = np.array([object_x, 0.0, BALL_CENTER_Z], dtype=np.float64)
    _set_ball(env, "cue_ball_free", cue_start)
    _set_ball(env, "object_ball_0_free", object_start)
    mujoco.mj_forward(env.model, env.data)

    command = _make_command(cue_start, speed=speed, offset_z=offset_z, elevation=args.elevation, gap=args.gap)
    first_cue_contact: float | None = None
    first_ball_contact: float | None = None
    post_contact_min_vx = float("inf")
    post_contact_min_x = float("inf")
    has_nan = False
    exploded = False

    quat = command.pose.quat_wxyz.astype(np.float64).copy()
    quat /= max(float(np.linalg.norm(quat)), 1e-9)
    first_pos, _, _ = env.project_cue_position(command.pose.position.astype(np.float64), quat)
    env._set_cue_state(first_pos, quat, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    mujoco.mj_forward(env.model, env.data)

    total_steps = env.action_repeat + int(round(args.settle_time / env.model.opt.timestep))
    for step_idx in range(total_steps):
        if step_idx < env.action_repeat:
            elapsed = step_idx * env.model.opt.timestep
            pos = command.pose.position + command.linear_velocity * elapsed
            pos, _, _ = env.project_cue_position(pos, quat)
            env._set_cue_state(pos, quat, command.linear_velocity, command.angular_velocity)
        else:
            hold_pos = env.data.qpos[env.cue_qpos_adr:env.cue_qpos_adr + 3].copy()
            hold_quat = env.data.qpos[env.cue_qpos_adr + 3:env.cue_qpos_adr + 7].copy()
            hold_quat /= max(float(np.linalg.norm(hold_quat)), 1e-9)
            hold_pos, _, _ = env.project_cue_position(hold_pos, hold_quat)
            env._set_cue_state(hold_pos, hold_quat, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64))

        mujoco.mj_step(env.model, env.data)
        cue_contact, ball_contact = env._contact_flags()
        if cue_contact and first_cue_contact is None:
            first_cue_contact = float(env.data.time)
        if ball_contact and first_ball_contact is None:
            first_ball_contact = float(env.data.time)
        if first_ball_contact is not None:
            cue_ball = env.scene_state().balls["cue_ball"]
            # Ignore rail rebound; only measure while the cue ball is still well
            # inside the table after contacting the object ball.
            if cue_ball.position[0] < args.rail_guard_x:
                post_contact_min_vx = min(post_contact_min_vx, float(cue_ball.linear_velocity[0]))
                post_contact_min_x = min(post_contact_min_x, float(cue_ball.position[0]))
        has_nan, exploded = env._stability_flags()
        if has_nan or exploded:
            break

    cue_ball = env.scene_state().balls["cue_ball"]
    dx = float(cue_ball.position[0] - cue_start[0])
    return (
        float(cue_ball.linear_velocity[0]),
        post_contact_min_vx,
        post_contact_min_x,
        dx,
        first_cue_contact,
        first_ball_contact,
        bool(has_nan or exploded),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--cue-ball-x", type=float, default=-0.35)
    parser.add_argument("--gap", type=float, default=0.004)
    parser.add_argument("--elevation", type=float, default=np.deg2rad(8.0))
    parser.add_argument("--action-repeat", type=int, default=12000)
    parser.add_argument("--settle-time", type=float, default=1.2)
    parser.add_argument("--rail-guard-x", type=float, default=0.85)
    parser.add_argument("--speeds", type=float, nargs="*", default=[0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6])
    parser.add_argument("--offsets", type=float, nargs="*", default=[-0.008, -0.010, -0.012, -0.014, -0.016])
    parser.add_argument("--object-xs", type=float, nargs="*", default=[-0.10, -0.04, 0.02, 0.08, 0.14])
    args = parser.parse_args()

    print("speed  offset_z object_x  final_vx  post_min_vx post_min_x final_dx  cue_t  ball_t  unstable")
    best: tuple[float, float, float, float, float, float] | None = None
    for object_x in args.object_xs:
        for offset_z in args.offsets:
            for speed in args.speeds:
                vx, min_vx, min_x, dx, cue_t, ball_t, unstable = _run_case(args, speed, offset_z, object_x)
                print(
                    f"{speed:5.2f} {offset_z:9.4f} {object_x:8.3f} {vx:9.4f} "
                    f"{min_vx:11.4f} {min_x:10.4f} {dx:9.4f} {cue_t!s:>6} {ball_t!s:>6} {unstable}"
                )
                if not unstable and ball_t is not None and min_vx < -0.02:
                    score = -min_vx
                    if best is None or score > best[0]:
                        best = (score, speed, offset_z, object_x, dx, min_vx)
    if best is None:
        raise RuntimeError("No draw-shot reversal found in the searched grid.")
    _, speed, offset_z, object_x, dx, min_vx = best
    print()
    print(f"best_draw speed={speed:.3f} offset_z={offset_z:.4f} object_x={object_x:.3f} final_dx={dx:.4f} post_min_vx={min_vx:.4f}")


if __name__ == "__main__":
    main()
