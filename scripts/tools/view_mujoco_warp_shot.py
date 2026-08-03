"""Show a repeating two-ball cue strike simulated by MuJoCo Warp.

The native MuJoCo viewer is used only for display.  All dynamics steps,
including cue-tip, ball, cloth, cushion, and SDF contacts, run in MuJoCo Warp
on the selected CUDA device.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
import warnings

import mujoco
import mujoco.viewer
import mujoco_warp as mjw
import numpy as np
import warp as wp

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.mujoco_warp_sdf import (  # noqa: E402
    MUJOCO_WARP_NCONMAX,
    MUJOCO_WARP_NJMAX,
    assert_mujoco_warp_capacity,
    calibrate_trapezoid_sdf_contact_damping,
    normalize_zero_friction_contact_dims,
    register_mujoco_billiards_sdf,
)
from snooker_env.table_geometry import BALL_CENTER_Z, BALL_RADIUS  # noqa: E402


CUE_TIP_LOCAL_X = 0.725
CUE_FOLLOW_THROUGH = 0.05


@dataclass(frozen=True)
class ShotSetup:
    """Host model and indices needed by the online rollout."""

    model: mujoco.MjModel
    initial_data: mujoco.MjData
    cue_qpos: int
    cue_dof: int
    cue_ball_qpos: int
    cue_ball_dof: int
    object_ball_qpos: int
    object_ball_dof: int
    cue_body_position: np.ndarray
    cue_quaternion: np.ndarray
    direction: np.ndarray
    initial_cue_ball_position: np.ndarray
    initial_object_ball_position: np.ndarray


@wp.kernel
def _drive_cue(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    simulation_time: wp.array(dtype=float),
    cue_qpos: int,
    cue_dof: int,
    initial_body_position: wp.vec3,
    quaternion_wxyz: wp.vec4,
    direction: wp.vec3,
    lead_in: float,
    stroke_duration: float,
    speed: float,
) -> None:
    """Kinematically drive and then park the cue before a Warp physics step."""

    world_id = wp.tid()
    elapsed = simulation_time[world_id]
    displacement = 0.0
    velocity = 0.0
    parked = False
    if elapsed >= lead_in:
        if elapsed < lead_in + stroke_duration:
            displacement = speed * (elapsed - lead_in)
            velocity = speed
        else:
            parked = True

    position = initial_body_position + direction * displacement
    if parked:
        position = wp.vec3(0.0, 0.0, 6.0)

    qpos[world_id, cue_qpos + 0] = position[0]
    qpos[world_id, cue_qpos + 1] = position[1]
    qpos[world_id, cue_qpos + 2] = position[2]
    if parked:
        qpos[world_id, cue_qpos + 3] = 1.0
        qpos[world_id, cue_qpos + 4] = 0.0
        qpos[world_id, cue_qpos + 5] = 0.0
        qpos[world_id, cue_qpos + 6] = 0.0
    else:
        qpos[world_id, cue_qpos + 3] = quaternion_wxyz[0]
        qpos[world_id, cue_qpos + 4] = quaternion_wxyz[1]
        qpos[world_id, cue_qpos + 5] = quaternion_wxyz[2]
        qpos[world_id, cue_qpos + 6] = quaternion_wxyz[3]

    qvel[world_id, cue_dof + 0] = direction[0] * velocity
    qvel[world_id, cue_dof + 1] = direction[1] * velocity
    qvel[world_id, cue_dof + 2] = direction[2] * velocity
    qvel[world_id, cue_dof + 3] = 0.0
    qvel[world_id, cue_dof + 4] = 0.0
    qvel[world_id, cue_dof + 5] = 0.0


def _required_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Model does not contain {object_type.name}: {name}")
    return object_id


def _joint_addresses(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    joint_id = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _set_free_joint_default(
    model: mujoco.MjModel,
    qpos_address: int,
    position: np.ndarray,
    quaternion: np.ndarray,
) -> None:
    model.qpos0[qpos_address:qpos_address + 3] = position
    model.qpos0[qpos_address + 3:qpos_address + 7] = quaternion


def _build_shot_setup(args: argparse.Namespace) -> ShotSetup:
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    register_mujoco_billiards_sdf(model)
    normalize_zero_friction_contact_dims(model)
    calibrate_trapezoid_sdf_contact_damping(model)

    cue_qpos, cue_dof = _joint_addresses(model, "cue_free")
    cue_ball_qpos, cue_ball_dof = _joint_addresses(model, "cue_ball_free")
    object_ball_qpos, object_ball_dof = _joint_addresses(model, "object_ball_0_free")

    cue_ball_xy = np.array(
        [args.cue_ball_x, args.cue_ball_y],
        dtype=np.float64,
    )
    object_ball_xy = np.array(
        [args.object_ball_x, args.object_ball_y],
        dtype=np.float64,
    )
    direction_xy = object_ball_xy - cue_ball_xy
    separation = float(np.linalg.norm(direction_xy))
    if separation <= 2.0 * BALL_RADIUS:
        raise ValueError("The cue ball and object ball overlap.")
    direction_xy /= separation
    direction = np.array(
        [direction_xy[0], direction_xy[1], 0.0],
        dtype=np.float64,
    )

    yaw = float(np.arctan2(direction_xy[1], direction_xy[0]))
    quaternion = np.array(
        [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)],
        dtype=np.float64,
    )
    cue_ball_position = np.array(
        [cue_ball_xy[0], cue_ball_xy[1], BALL_CENTER_Z],
        dtype=np.float64,
    )
    object_ball_position = np.array(
        [object_ball_xy[0], object_ball_xy[1], BALL_CENTER_Z],
        dtype=np.float64,
    )
    rear_contact = cue_ball_position - direction * BALL_RADIUS
    initial_tip = rear_contact - direction * args.backoff
    cue_body_position = initial_tip - direction * CUE_TIP_LOCAL_X

    _set_free_joint_default(
        model,
        cue_qpos,
        cue_body_position,
        quaternion,
    )
    _set_free_joint_default(
        model,
        cue_ball_qpos,
        cue_ball_position,
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    _set_free_joint_default(
        model,
        object_ball_qpos,
        object_ball_position,
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    initial_data = mujoco.MjData(model)
    mujoco.mj_forward(model, initial_data)
    return ShotSetup(
        model=model,
        initial_data=initial_data,
        cue_qpos=cue_qpos,
        cue_dof=cue_dof,
        cue_ball_qpos=cue_ball_qpos,
        cue_ball_dof=cue_ball_dof,
        object_ball_qpos=object_ball_qpos,
        object_ball_dof=object_ball_dof,
        cue_body_position=cue_body_position,
        cue_quaternion=quaternion,
        direction=direction,
        initial_cue_ball_position=cue_ball_position,
        initial_object_ball_position=object_ball_position,
    )


def _capture_step_chunk(
    setup: ShotSetup,
    warp_model: object,
    warp_data: object,
    *,
    lead_in: float,
    stroke_duration: float,
    speed: float,
    chunk_steps: int,
) -> object:
    initial_body_position = wp.vec3(*setup.cue_body_position)
    quaternion = wp.vec4(*setup.cue_quaternion)
    direction = wp.vec3(*setup.direction)
    with wp.ScopedCapture() as capture:
        for _ in range(chunk_steps):
            wp.launch(
                _drive_cue,
                dim=warp_data.nworld,
                inputs=[
                    warp_data.qpos,
                    warp_data.qvel,
                    warp_data.time,
                    setup.cue_qpos,
                    setup.cue_dof,
                    initial_body_position,
                    quaternion,
                    direction,
                    lead_in,
                    stroke_duration,
                    speed,
                ],
            )
            mjw.step(warp_model, warp_data)
    wp.synchronize()
    mjw.reset_data(warp_model, warp_data)
    wp.synchronize()
    return capture.graph


def _copy_state_for_display(
    setup: ShotSetup,
    warp_data: object,
    display_data: mujoco.MjData,
) -> None:
    qpos = warp_data.qpos.numpy()[0]
    qvel = warp_data.qvel.numpy()[0]
    simulation_time = float(warp_data.time.numpy()[0])
    display_data.qpos[:] = qpos
    display_data.qvel[:] = qvel
    display_data.time = simulation_time
    mujoco.mj_forward(setup.model, display_data)


def _print_state(
    setup: ShotSetup,
    warp_data: object,
    *,
    simulated_seconds: float,
    wall_seconds: float,
) -> None:
    qpos = warp_data.qpos.numpy()[0].astype(np.float64)
    qvel = warp_data.qvel.numpy()[0].astype(np.float64)
    cue_position = qpos[setup.cue_ball_qpos:setup.cue_ball_qpos + 3]
    object_position = qpos[
        setup.object_ball_qpos:setup.object_ball_qpos + 3
    ]
    cue_speed = float(
        np.linalg.norm(qvel[setup.cue_ball_dof:setup.cue_ball_dof + 3])
    )
    object_speed = float(
        np.linalg.norm(qvel[setup.object_ball_dof:setup.object_ball_dof + 3])
    )
    print(f"simulated_seconds={simulated_seconds:.6f}")
    print(f"wall_seconds={wall_seconds:.3f}")
    if wall_seconds > 0.0:
        print(f"simulation_realtime_factor={simulated_seconds / wall_seconds:.3f}")
    print(f"cue_ball_position={cue_position.tolist()}")
    print(f"cue_ball_speed={cue_speed:.6f}")
    print(f"object_ball_position={object_position.tolist()}")
    print(f"object_ball_speed={object_speed:.6f}")
    print(
        "object_ball_displacement="
        f"{np.linalg.norm(object_position - setup.initial_object_ball_position):.6f}"
    )
    print(f"finite_state={bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())}")


def _run_headless(
    setup: ShotSetup,
    warp_data: object,
    graph: object,
    *,
    seconds: float,
    chunk_steps: int,
) -> None:
    timestep = float(setup.model.opt.timestep)
    blocks = max(1, int(np.ceil(seconds / (timestep * chunk_steps))))
    start = time.perf_counter()
    for _ in range(blocks):
        wp.capture_launch(graph)
    wp.synchronize()
    wall_seconds = time.perf_counter() - start
    assert_mujoco_warp_capacity(warp_data, context="headless viewer validation")
    simulated_seconds = float(warp_data.time.numpy()[0])
    _print_state(
        setup,
        warp_data,
        simulated_seconds=simulated_seconds,
        wall_seconds=wall_seconds,
    )
    if not (
        np.isfinite(warp_data.qpos.numpy()).all()
        and np.isfinite(warp_data.qvel.numpy()).all()
    ):
        raise RuntimeError("MJWarp shot produced a non-finite state.")


def _run_viewer(
    setup: ShotSetup,
    warp_model: object,
    warp_data: object,
    graph: object,
    *,
    fps: float,
    time_scale: float,
    cycle_seconds: float,
    chunk_steps: int,
) -> None:
    display_data = mujoco.MjData(setup.model)
    _copy_state_for_display(setup, warp_data, display_data)
    viewer_state = {"paused": False, "reset": False}

    def key_callback(key: int) -> None:
        if key == 32:
            viewer_state["paused"] = not viewer_state["paused"]
            print(f"paused={viewer_state['paused']}", flush=True)
        elif key in (82, 114):
            viewer_state["reset"] = True

    timestep = float(setup.model.opt.timestep)
    requested_steps_per_frame = time_scale / (fps * timestep)
    step_credit = 0.0
    frame_period = 1.0 / fps
    report_wall_time = time.monotonic()
    report_sim_time = 0.0

    print("controls: Space=pause/resume, R=restart shot, Esc/close window=quit")
    print(
        f"device={wp.get_device()} timestep={timestep:g} "
        f"chunk_steps={chunk_steps} target_fps={fps:g} time_scale={time_scale:g}"
    )
    with mujoco.viewer.launch_passive(
        setup.model,
        display_data,
        key_callback=key_callback,
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (0.0, -0.05, 1.05)
        viewer.cam.distance = 3.1
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -42.0
        if hasattr(viewer, "opt"):
            viewer.opt.sitegroup[3] = 0

        while viewer.is_running():
            frame_start = time.monotonic()
            simulation_time = float(warp_data.time.numpy()[0])
            if viewer_state["reset"] or simulation_time >= cycle_seconds:
                mjw.reset_data(warp_model, warp_data)
                wp.synchronize()
                viewer_state["reset"] = False
                step_credit = 0.0
                simulation_time = 0.0

            if not viewer_state["paused"]:
                step_credit += requested_steps_per_frame
                block_count = int(step_credit // chunk_steps)
                step_credit -= block_count * chunk_steps
                for _ in range(block_count):
                    wp.capture_launch(graph)
                wp.synchronize()

            with viewer.lock():
                _copy_state_for_display(
                    setup,
                    warp_data,
                    display_data,
                )
            viewer.sync()

            now = time.monotonic()
            if now - report_wall_time >= 2.0:
                current_sim_time = float(warp_data.time.numpy()[0])
                simulated_delta = (
                    current_sim_time - report_sim_time
                    if current_sim_time >= report_sim_time
                    else current_sim_time
                )
                print(
                    f"viewer_sim_time={current_sim_time:.3f}s "
                    f"achieved_realtime_factor={simulated_delta / (now - report_wall_time):.2f}",
                    flush=True,
                )
                report_wall_time = now
                report_sim_time = current_sim_time

            remaining = frame_period - (time.monotonic() - frame_start)
            if remaining > 0.0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speed", type=float, default=1.5)
    parser.add_argument("--backoff", type=float, default=0.10)
    parser.add_argument("--follow-through", type=float, default=CUE_FOLLOW_THROUGH)
    parser.add_argument("--lead-in", type=float, default=0.05)
    parser.add_argument("--cycle-seconds", type=float, default=1.5)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--time-scale",
        type=float,
        default=0.05,
        help="Simulation seconds per wall second; 0.05 gives smooth 20x slow motion.",
    )
    parser.add_argument("--chunk-steps", type=int, default=32)
    parser.add_argument("--cue-ball-x", type=float, default=0.0)
    parser.add_argument("--cue-ball-y", type=float, default=-0.616)
    parser.add_argument("--object-ball-x", type=float, default=0.0)
    parser.add_argument("--object-ball-y", type=float, default=-0.25)
    parser.add_argument(
        "--headless-seconds",
        type=float,
        default=0.0,
        help="Run without opening a window for this many simulated seconds.",
    )
    args = parser.parse_args()

    positive = {
        "--speed": args.speed,
        "--follow-through": args.follow_through,
        "--cycle-seconds": args.cycle_seconds,
        "--fps": args.fps,
        "--time-scale": args.time_scale,
    }
    invalid = [name for name, value in positive.items() if value <= 0.0]
    if invalid:
        raise ValueError(f"Arguments must be positive: {', '.join(invalid)}")
    if args.backoff < 0.0 or args.lead_in < 0.0:
        raise ValueError("--backoff and --lead-in must be non-negative.")
    if args.chunk_steps <= 0:
        raise ValueError("--chunk-steps must be positive.")
    if args.headless_seconds < 0.0:
        raise ValueError("--headless-seconds must be non-negative.")
    stroke_duration = (args.backoff + args.follow_through) / args.speed
    if args.cycle_seconds <= args.lead_in + stroke_duration:
        raise ValueError("The cycle must extend beyond the cue stroke.")

    warnings.filterwarnings(
        "ignore",
        message=r"geom .* friction.*MJ_MINMU.*",
        category=UserWarning,
    )
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("The online MJWarp shot viewer requires a CUDA device.")
    wp.set_device(device)

    setup = _build_shot_setup(args)
    print("uploading model to MJWarp...", flush=True)
    warp_model = mjw.put_model(setup.model)
    warp_data = mjw.put_data(
        setup.model,
        setup.initial_data,
        nworld=1,
        nconmax=MUJOCO_WARP_NCONMAX,
        njmax=MUJOCO_WARP_NJMAX,
    )
    print(
        f"compiling {args.chunk_steps}-step CUDA graph...",
        flush=True,
    )
    graph = _capture_step_chunk(
        setup,
        warp_model,
        warp_data,
        lead_in=args.lead_in,
        stroke_duration=stroke_duration,
        speed=args.speed,
        chunk_steps=args.chunk_steps,
    )
    print("CUDA graph ready", flush=True)

    if args.headless_seconds > 0.0:
        _run_headless(
            setup,
            warp_data,
            graph,
            seconds=args.headless_seconds,
            chunk_steps=args.chunk_steps,
        )
    else:
        _run_viewer(
            setup,
            warp_model,
            warp_data,
            graph,
            fps=args.fps,
            time_scale=args.time_scale,
            cycle_seconds=args.cycle_seconds,
            chunk_steps=args.chunk_steps,
        )


if __name__ == "__main__":
    main()
