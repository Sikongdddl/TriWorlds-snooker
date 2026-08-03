#!/usr/bin/env python3
"""Show four distinct shots advanced by one batched MJWarp CUDA graph."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
import warnings

import cv2
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.midlevel_mujoco_warp_vec_env import (  # noqa: E402
    prepare_mujoco_warp_model,
)
from snooker_env.mujoco_warp_sdf import (  # noqa: E402
    MUJOCO_WARP_NCONMAX,
    MUJOCO_WARP_NJMAX,
    assert_mujoco_warp_capacity,
)
from snooker_env.table_geometry import BALL_CENTER_Z, BALL_RADIUS  # noqa: E402


CUE_TIP_LOCAL_X = 0.725
WORLD_COUNT = 4
WINDOW_NAME = "MJWarp: four parallel worlds (one CUDA batch)"


@dataclass(frozen=True)
class ShotSpec:
    cue_xy: tuple[float, float]
    object_xy: tuple[float, float]
    aim_offset: tuple[float, float]
    speed: float
    label: str


SHOT_SPECS = (
    ShotSpec((0.00, -0.72), (0.00, -0.24), (0.00, 0.00), 0.75, "world 0 | straight | 0.75 m/s"),
    ShotSpec((-0.30, -0.70), (-0.08, -0.25), (0.010, 0.00), 1.00, "world 1 | right cut | 1.00 m/s"),
    ShotSpec((0.30, -0.70), (0.08, -0.25), (-0.012, 0.00), 1.25, "world 2 | left cut | 1.25 m/s"),
    ShotSpec((-0.34, -0.38), (0.16, -0.02), (0.018, -0.01), 1.50, "world 3 | oblique | 1.50 m/s"),
)


@wp.kernel
def _drive_cues(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    simulation_time: wp.array(dtype=float),
    initial_positions: wp.array(dtype=wp.vec3),
    quaternions_wxyz: wp.array(dtype=wp.vec4),
    directions: wp.array(dtype=wp.vec3),
    speeds: wp.array(dtype=float),
    cue_qpos: int,
    cue_dof: int,
    lead_in: float,
    stroke_distances: wp.array(dtype=float),
) -> None:
    """Drive one cue per world while all worlds share the same kernel launch."""

    world_id = wp.tid()
    elapsed = simulation_time[world_id]
    speed = speeds[world_id]
    stroke_distance = stroke_distances[world_id]
    stroke_duration = stroke_distance / speed
    displacement = 0.0
    velocity = 0.0
    parked = False
    if elapsed >= lead_in:
        if elapsed < lead_in + stroke_duration:
            displacement = speed * (elapsed - lead_in)
            velocity = speed
        else:
            parked = True

    direction = directions[world_id]
    position = initial_positions[world_id] + direction * displacement
    quaternion = quaternions_wxyz[world_id]
    if parked:
        position = wp.vec3(0.0, 0.0, 6.0)
        quaternion = wp.vec4(1.0, 0.0, 0.0, 0.0)

    qpos[world_id, cue_qpos + 0] = position[0]
    qpos[world_id, cue_qpos + 1] = position[1]
    qpos[world_id, cue_qpos + 2] = position[2]
    qpos[world_id, cue_qpos + 3] = quaternion[0]
    qpos[world_id, cue_qpos + 4] = quaternion[1]
    qpos[world_id, cue_qpos + 5] = quaternion[2]
    qpos[world_id, cue_qpos + 6] = quaternion[3]
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
        raise ValueError(f"Required MuJoCo object is missing: {name}")
    return object_id


def _joint_addresses(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    joint_id = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _world_initial_states(
    model: mujoco.MjModel,
    *,
    backoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cue_qpos, _ = _joint_addresses(model, "cue_free")
    cue_ball_qpos, _ = _joint_addresses(model, "cue_ball_free")
    object_ball_qpos, _ = _joint_addresses(model, "object_ball_0_free")
    qpos = np.tile(model.qpos0.astype(np.float32), (WORLD_COUNT, 1))
    cue_body_positions = np.empty((WORLD_COUNT, 3), dtype=np.float32)
    cue_quaternions = np.empty((WORLD_COUNT, 4), dtype=np.float32)
    directions = np.empty((WORLD_COUNT, 3), dtype=np.float32)
    speeds = np.empty(WORLD_COUNT, dtype=np.float32)

    for world_id, spec in enumerate(SHOT_SPECS):
        cue_xy = np.asarray(spec.cue_xy, dtype=np.float64)
        object_xy = np.asarray(spec.object_xy, dtype=np.float64)
        aim_xy = object_xy + np.asarray(spec.aim_offset, dtype=np.float64)
        direction_xy = aim_xy - cue_xy
        direction_xy /= np.linalg.norm(direction_xy)
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
            [cue_xy[0], cue_xy[1], BALL_CENTER_Z],
            dtype=np.float64,
        )
        object_ball_position = np.array(
            [object_xy[0], object_xy[1], BALL_CENTER_Z],
            dtype=np.float64,
        )
        rear_contact = cue_ball_position - direction * BALL_RADIUS
        initial_tip = rear_contact - direction * backoff
        cue_body_position = initial_tip - direction * CUE_TIP_LOCAL_X

        qpos[world_id, cue_qpos:cue_qpos + 3] = cue_body_position
        qpos[world_id, cue_qpos + 3:cue_qpos + 7] = quaternion
        qpos[world_id, cue_ball_qpos:cue_ball_qpos + 3] = cue_ball_position
        qpos[world_id, cue_ball_qpos + 3:cue_ball_qpos + 7] = (
            1.0,
            0.0,
            0.0,
            0.0,
        )
        qpos[world_id, object_ball_qpos:object_ball_qpos + 3] = (
            object_ball_position
        )
        qpos[world_id, object_ball_qpos + 3:object_ball_qpos + 7] = (
            1.0,
            0.0,
            0.0,
            0.0,
        )
        cue_body_positions[world_id] = cue_body_position
        cue_quaternions[world_id] = quaternion
        directions[world_id] = direction
        speeds[world_id] = spec.speed
    return qpos, cue_body_positions, cue_quaternions, directions, speeds


def _reset_worlds(
    warp_model: object,
    warp_data: object,
    initial_qpos: np.ndarray,
) -> None:
    mjw.reset_data(warp_model, warp_data)
    warp_data.qpos.assign(initial_qpos)
    warp_data.qvel.zero_()
    wp.synchronize()


def _display_frames(
    model: mujoco.MjModel,
    display_data: list[mujoco.MjData],
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
    qpos: np.ndarray,
    qvel: np.ndarray,
    simulation_times: np.ndarray,
) -> np.ndarray:
    frames: list[np.ndarray] = []
    for world_id, data in enumerate(display_data):
        data.qpos[:] = qpos[world_id]
        data.qvel[:] = qvel[world_id]
        data.time = float(simulation_times[world_id])
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(
            frame,
            SHOT_SPECS[world_id].label,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        frames.append(frame)
    return np.concatenate(
        (
            np.concatenate((frames[0], frames[1]), axis=1),
            np.concatenate((frames[2], frames[3]), axis=1),
        ),
        axis=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backoff", type=float, default=0.10)
    parser.add_argument("--follow-through", type=float, default=0.05)
    parser.add_argument("--lead-in", type=float, default=0.05)
    parser.add_argument("--cycle-seconds", type=float, default=2.5)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--time-scale", type=float, default=0.05)
    parser.add_argument("--chunk-steps", type=int, default=32)
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=360)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Close automatically after this many display frames; 0 runs until closed.",
    )
    args = parser.parse_args()
    positive = (
        args.follow_through,
        args.cycle_seconds,
        args.fps,
        args.time_scale,
        args.chunk_steps,
        args.tile_width,
        args.tile_height,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Timing, rendering, and chunk arguments must be positive.")
    if args.backoff < 0.0 or args.lead_in < 0.0 or args.max_frames < 0:
        raise ValueError("Backoff, lead-in, and max-frames must be non-negative.")

    warnings.filterwarnings(
        "ignore",
        message=r"geom .* friction.*MJ_MINMU.*",
        category=UserWarning,
    )
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("The four-world viewer requires a CUDA device.")
    wp.set_device(device)

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    prepare_mujoco_warp_model(model)
    cue_qpos, cue_dof = _joint_addresses(model, "cue_free")
    initial_qpos, cue_positions, cue_quaternions, directions, speeds = (
        _world_initial_states(model, backoff=args.backoff)
    )
    initial_data = mujoco.MjData(model)
    mujoco.mj_forward(model, initial_data)
    warp_model = mjw.put_model(model)
    warp_data = mjw.put_data(
        model,
        initial_data,
        nworld=WORLD_COUNT,
        nconmax=MUJOCO_WARP_NCONMAX,
        njmax=MUJOCO_WARP_NJMAX,
    )
    cue_positions_device = wp.array(cue_positions, dtype=wp.vec3)
    cue_quaternions_device = wp.array(cue_quaternions, dtype=wp.vec4)
    directions_device = wp.array(directions, dtype=wp.vec3)
    speeds_device = wp.array(speeds, dtype=float)
    stroke_distances_device = wp.array(
        np.full(
            WORLD_COUNT,
            args.backoff + args.follow_through,
            dtype=np.float32,
        ),
        dtype=float,
    )
    _reset_worlds(warp_model, warp_data, initial_qpos)
    with wp.ScopedCapture() as capture:
        for _ in range(args.chunk_steps):
            wp.launch(
                _drive_cues,
                dim=WORLD_COUNT,
                inputs=[
                    warp_data.qpos,
                    warp_data.qvel,
                    warp_data.time,
                    cue_positions_device,
                    cue_quaternions_device,
                    directions_device,
                    speeds_device,
                    cue_qpos,
                    cue_dof,
                    args.lead_in,
                    stroke_distances_device,
                ],
            )
            mjw.step(warp_model, warp_data)
    graph = capture.graph
    _reset_worlds(warp_model, warp_data, initial_qpos)

    display_data = [mujoco.MjData(model) for _ in range(WORLD_COUNT)]
    renderer = mujoco.Renderer(
        model,
        height=args.tile_height,
        width=args.tile_width,
    )
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, -0.05, 1.05)
    camera.distance = 3.1
    camera.azimuth = 90.0
    camera.elevation = -42.0

    timestep = float(model.opt.timestep)
    requested_steps_per_frame = args.time_scale / (args.fps * timestep)
    step_credit = 0.0
    frame_period = 1.0 / args.fps
    paused = False
    reset_requested = False
    frame_count = 0
    report_wall_time = time.monotonic()
    report_sim_time = 0.0
    print(
        f"worlds={WORLD_COUNT} device={device} timestep={timestep:g} "
        f"layout=2x2 chunk_steps={args.chunk_steps}",
        flush=True,
    )
    print("controls: Space=pause, R=restart, Esc/Q=close", flush=True)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        WINDOW_NAME,
        args.tile_width * 2,
        args.tile_height * 2,
    )
    try:
        while True:
            frame_start = time.monotonic()
            simulation_time = float(warp_data.time.numpy()[0])
            if reset_requested or simulation_time >= args.cycle_seconds:
                assert_mujoco_warp_capacity(
                    warp_data,
                    context="four-world online viewer cycle",
                )
                _reset_worlds(warp_model, warp_data, initial_qpos)
                reset_requested = False
                step_credit = 0.0

            if not paused:
                step_credit += requested_steps_per_frame
                block_count = int(step_credit // args.chunk_steps)
                step_credit -= block_count * args.chunk_steps
                for _ in range(block_count):
                    wp.capture_launch(graph)
                wp.synchronize()

            qpos = warp_data.qpos.numpy()
            qvel = warp_data.qvel.numpy()
            simulation_times = warp_data.time.numpy()
            composite = _display_frames(
                model,
                display_data,
                renderer,
                camera,
                qpos,
                qvel,
                simulation_times,
            )
            cv2.imshow(WINDOW_NAME, composite)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                paused = not paused
                print(f"paused={paused}", flush=True)
            elif key in (ord("r"), ord("R")):
                reset_requested = True
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
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
                    f"achieved_realtime_factor="
                    f"{simulated_delta / (now - report_wall_time):.3f}",
                    flush=True,
                )
                report_wall_time = now
                report_sim_time = current_sim_time
            remaining = frame_period - (time.monotonic() - frame_start)
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        assert_mujoco_warp_capacity(
            warp_data,
            context="four-world online viewer shutdown",
        )
        renderer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
