#!/usr/bin/env python3
"""Sweep cue-tip offsets and report cue-ball spin response."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL, MidLevelCueEnv  # noqa: E402
from snooker_env.pipeline_types import CueCommand, Pose3D  # noqa: E402


BALL_RADIUS = 0.028575
CUE_TIP_OFFSET = 0.725
CUE_TIP_RADIUS = 0.009


@dataclass(frozen=True)
class SpinCase:
    name: str
    offset_y: float
    offset_z: float


def _quat_for_x_axis(yaw: float, elevation: float) -> np.ndarray:
    """Return MuJoCo wxyz quaternion for cue local +X along shot axis."""

    half_yaw = 0.5 * yaw
    half_elevation = 0.5 * elevation
    q_yaw = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64)
    q_pitch = np.array([np.cos(half_elevation), 0.0, np.sin(half_elevation), 0.0], dtype=np.float64)
    quat = np.array(
        [
            q_yaw[0] * q_pitch[0] - q_yaw[1] * q_pitch[1] - q_yaw[2] * q_pitch[2] - q_yaw[3] * q_pitch[3],
            q_yaw[0] * q_pitch[1] + q_yaw[1] * q_pitch[0] + q_yaw[2] * q_pitch[3] - q_yaw[3] * q_pitch[2],
            q_yaw[0] * q_pitch[2] - q_yaw[1] * q_pitch[3] + q_yaw[2] * q_pitch[0] + q_yaw[3] * q_pitch[1],
            q_yaw[0] * q_pitch[3] + q_yaw[1] * q_pitch[2] - q_yaw[2] * q_pitch[1] + q_yaw[3] * q_pitch[0],
        ],
        dtype=np.float64,
    )
    quat /= max(float(np.linalg.norm(quat)), 1e-9)
    return quat


def _x_axis_from_quat(quat: np.ndarray) -> np.ndarray:
    rot_flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rot_flat, quat)
    return rot_flat.reshape(3, 3)[:, 0].copy()


def _move_object_ball_aside(env: MidLevelCueEnv) -> None:
    qadr = int(env.model.jnt_qposadr[env.object_ball_joint])
    dadr = int(env.model.jnt_dofadr[env.object_ball_joint])
    env.data.qpos[qadr:qadr + 3] = np.array([0.75, 0.35, 0.7898], dtype=np.float64)
    env.data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    env.data.qvel[dadr:dadr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def _make_stroke(case: SpinCase, cue_ball_pos: np.ndarray, speed: float, elevation: float, gap: float) -> CueCommand:
    quat = _quat_for_x_axis(yaw=0.0, elevation=elevation)
    axis = _x_axis_from_quat(quat)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    offset = np.array([0.0, case.offset_y, case.offset_z], dtype=np.float64)
    tip_start = cue_ball_pos - axis * (BALL_RADIUS + CUE_TIP_RADIUS + gap) + offset
    cue_body_pos = tip_start - axis * CUE_TIP_OFFSET
    return CueCommand(
        pose=Pose3D(position=cue_body_pos, quat_wxyz=quat),
        linear_velocity=axis * speed,
        angular_velocity=np.zeros(3, dtype=np.float64),
        debug_label=case.name,
    )


def run_sweep(args: argparse.Namespace) -> None:
    cases = (
        SpinCase("center", 0.0, 0.0),
        SpinCase("top", 0.0, args.offset),
        SpinCase("bottom", 0.0, -args.offset),
        SpinCase("left_side", args.offset, 0.0),
        SpinCase("right_side", -args.offset, 0.0),
    )
    env = MidLevelCueEnv(args.model, action_repeat=args.action_repeat)
    print(f"model={args.model}")
    print(f"action_repeat={env.action_repeat} command_dt={env.command_dt:.4f}s")
    print(f"speed={args.speed:.3f}m/s elevation={np.rad2deg(args.elevation):.2f}deg offset={args.offset:.4f}m")
    print()
    print("case          offset_y  offset_z  contact_t  cue_vxyz_after                 cue_wxyz_after")

    angular_results: list[np.ndarray] = []
    for case in cases:
        state = env.reset()
        _move_object_ball_aside(env)
        cue_ball_pos = state.balls["cue_ball"].position.copy()
        command = _make_stroke(case, cue_ball_pos, args.speed, args.elevation, args.gap)
        result = env.execute((command,), settle_time=args.settle_time)
        final_state = env.scene_state().balls["cue_ball"]
        angular_results.append(final_state.angular_velocity)
        print(
            f"{case.name:<13}"
            f"{case.offset_y:>8.4f} "
            f"{case.offset_z:>8.4f} "
            f"{str(result.first_cue_ball_contact_time):>9} "
            f"{np.round(final_state.linear_velocity, 5)!s:<31} "
            f"{np.round(final_state.angular_velocity, 5)}"
        )
        if result.has_nan or result.exploded:
            raise RuntimeError(f"Unstable simulation in spin case: {case.name}")

    spread = float(np.max(np.linalg.norm(np.array(angular_results) - angular_results[0], axis=1)))
    print()
    print(f"max angular response spread vs center: {spread:.6f} rad/s")
    if spread < args.min_angular_spread:
        raise RuntimeError(
            "Spin response is too small to distinguish cue offsets. "
            "Check tip/ball friction, contact geometry, and offset feasibility."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.010)
    parser.add_argument("--gap", type=float, default=0.004)
    parser.add_argument("--elevation", type=float, default=np.deg2rad(8.0))
    parser.add_argument("--action-repeat", type=int, default=80)
    parser.add_argument("--settle-time", type=float, default=0.25)
    parser.add_argument("--min-angular-spread", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())
