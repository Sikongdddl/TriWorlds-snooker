"""Show a repeating robot-free cue strike in the interactive MuJoCo viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402


def _required_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise ValueError(f"Model does not contain {kind.name}: {name}")
    return object_id


def view_strike(
    model_path: Path,
    *,
    speed: float,
    stroke: float,
    backoff: float,
    lead_in: float,
    cycle_seconds: float,
    fps: float,
    target_x: float,
    target_y: float,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    cue_joint = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cue_free")
    cue_tip = _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_tip")
    cue_ball = _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_ball_geom")
    object_ball = _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_ball_0_geom")
    object_joint = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_ball_0_free")
    cue_qpos = int(model.jnt_qposadr[cue_joint])
    cue_dof = int(model.jnt_dofadr[cue_joint])
    object_qpos = int(model.jnt_qposadr[object_joint])
    object_dof = int(model.jnt_dofadr[object_joint])
    initial_pose = model.qpos0[cue_qpos:cue_qpos + 7].copy()
    initial_pose[1] -= backoff
    stroke_seconds = stroke / speed
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    cue_contact_reported = False
    ball_contact_reported = False

    def reset_cycle() -> None:
        nonlocal cue_contact_reported, ball_contact_reported
        mujoco.mj_resetData(model, data)
        data.qpos[cue_qpos:cue_qpos + 7] = initial_pose
        data.qpos[object_qpos:object_qpos + 3] = (target_x, target_y, 1.0785)
        data.qvel[object_dof:object_dof + 6] = 0.0
        mujoco.mj_forward(model, data)
        cue_contact_reported = False
        ball_contact_reported = False

    reset_cycle()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (0.0, 0.0, 1.05)
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -40.0
        if hasattr(viewer, "opt"):
            # Hide project-only pocket markers while keeping the source cue
            # ball's red rotation markers (site group 0) visible.
            viewer.opt.sitegroup[3] = 0

        while viewer.is_running():
            frame_start = time.monotonic()
            for _ in range(steps_per_frame):
                if data.time >= cycle_seconds:
                    reset_cycle()

                elapsed = float(data.time)
                if elapsed < lead_in:
                    displacement = 0.0
                    velocity = 0.0
                elif elapsed < lead_in + stroke_seconds:
                    displacement = speed * (elapsed - lead_in)
                    velocity = speed
                else:
                    displacement = stroke
                    velocity = 0.0

                data.qpos[cue_qpos:cue_qpos + 7] = initial_pose
                data.qpos[cue_qpos + 1] += displacement
                data.qvel[cue_dof:cue_dof + 6] = 0.0
                data.qvel[cue_dof + 1] = velocity
                mujoco.mj_step(model, data)

                if not cue_contact_reported or not ball_contact_reported:
                    for contact in data.contact[:data.ncon]:
                        pair = {contact.geom1, contact.geom2}
                        if not cue_contact_reported and pair == {cue_tip, cue_ball}:
                            print(f"cue_tip/cue_ball contact at t={data.time:.5f} s", flush=True)
                            cue_contact_reported = True
                        if not ball_contact_reported and pair == {cue_ball, object_ball}:
                            print(f"cue_ball/object_ball contact at t={data.time:.5f} s", flush=True)
                            ball_contact_reported = True

            viewer.sync()
            remaining = 1.0 / fps - (time.monotonic() - frame_start)
            if remaining > 0.0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--speed", type=float, default=1.0, help="Forward cue speed in m/s.")
    parser.add_argument("--stroke", type=float, default=0.20, help="Forward cue travel in meters.")
    parser.add_argument("--backoff", type=float, default=0.10, help="Initial cue-tip backoff in meters.")
    parser.add_argument("--lead-in", type=float, default=0.40, help="Pause before each strike in seconds.")
    parser.add_argument("--cycle-seconds", type=float, default=2.5, help="Time between scene resets.")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=0.15)
    args = parser.parse_args()
    if args.speed <= 0.0 or args.stroke <= 0.0 or args.fps <= 0.0:
        raise ValueError("--speed, --stroke, and --fps must be positive.")
    if args.backoff < 0.0 or args.lead_in < 0.0:
        raise ValueError("--backoff and --lead-in must be non-negative.")
    if args.cycle_seconds <= args.lead_in + args.stroke / args.speed:
        raise ValueError("The cycle must extend beyond the lead-in and stroke.")
    view_strike(
        args.model,
        speed=args.speed,
        stroke=args.stroke,
        backoff=args.backoff,
        lead_in=args.lead_in,
        cycle_seconds=args.cycle_seconds,
        fps=args.fps,
        target_x=args.target_x,
        target_y=args.target_y,
    )


if __name__ == "__main__":
    main()
