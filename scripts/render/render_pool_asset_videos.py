from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.scene import project_root  # noqa: E402
from snooker_env.init_pose import set_lift_grip_ready_pose  # noqa: E402


def _hide_group_3(renderer: mujoco.Renderer) -> None:
    scene_option = getattr(renderer, "scene_option", None)
    if scene_option is None:
        scene_option = getattr(renderer, "_scene_option", None)
    if scene_option is not None:
        scene_option.geomgroup[3] = 0
        scene_option.sitegroup[3] = 0


def _camera(distance: float, azimuth: float, elevation: float, lookat: tuple[float, float, float]) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.lookat[:] = lookat
    return cam


def _write_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output: Path,
    width: int,
    height: int,
    fps: int,
    seconds: float,
    camera_fn,
) -> None:
    renderer = mujoco.Renderer(model, height=height, width=width)
    _hide_group_3(renderer)
    frames = max(1, int(round(fps * seconds)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=9) as writer:
        for frame in range(frames):
            t = frame / max(1, frames - 1)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera_fn(t))
            writer.append_data(renderer.render())
    renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render several videos for the mujoco-billiards table scene.")
    parser.add_argument("--model", type=Path, default=project_root() / "models" / "scene_pool_asset.xml")
    parser.add_argument("--output-dir", type=Path, default=project_root() / "outputs" / "videos_pool_asset")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    set_lift_grip_ready_pose(model, data)

    def orbit(t: float) -> mujoco.MjvCamera:
        return _camera(3.6, -135.0 + 360.0 * t, -24.0, (0.0, 0.0, 1.05))

    def pocket_sweep(t: float) -> mujoco.MjvCamera:
        az = -82.0 + 34.0 * math.sin(2.0 * math.pi * t)
        dist = 1.95 - 0.25 * math.sin(math.pi * t)
        return _camera(dist, az, -12.0, (-0.58 + 1.16 * t, 0.0, 1.05))

    def cue_lane(t: float) -> mujoco.MjvCamera:
        return _camera(2.15, 0.0, -10.0, (0.0, -0.90 + 1.80 * t, 1.05))

    def lift_table_side(t: float) -> mujoco.MjvCamera:
        az = 35.0 + 18.0 * math.sin(2.0 * math.pi * t)
        return _camera(2.55, az, -16.0, (0.0, -0.90, 1.05))

    jobs = {
        "pool_asset_orbit_detail.mp4": orbit,
        "pool_asset_pocket_sweep.mp4": pocket_sweep,
        "pool_asset_cue_lane.mp4": cue_lane,
        "pool_lift_table_side.mp4": lift_table_side,
    }
    for filename, cam_fn in jobs.items():
        out = args.output_dir / filename
        _write_video(model, data, out, args.width, args.height, args.fps, args.seconds, cam_fn)
        print(f"wrote={out}")


if __name__ == "__main__":
    main()
