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


def _make_camera(distance: float, azimuth: float, elevation: float, lookat: tuple[float, float, float]) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.lookat[:] = lookat
    return camera


def _hide_collision_group(renderer: mujoco.Renderer) -> None:
    scene_option = getattr(renderer, "scene_option", None)
    if scene_option is None:
        scene_option = getattr(renderer, "_scene_option", None)
    if scene_option is not None:
        scene_option.geomgroup[3] = 0
        scene_option.sitegroup[3] = 0


def _render_png(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output: Path,
    width: int,
    height: int,
    distance: float,
    azimuth: float,
    elevation: float,
    lookat: tuple[float, float, float],
) -> None:
    renderer = mujoco.Renderer(model, height=height, width=width)
    _hide_collision_group(renderer)
    camera = _make_camera(distance, azimuth, elevation, lookat)
    renderer.update_scene(data, camera=camera)
    imageio.imwrite(output, renderer.render())
    renderer.close()


def render_images(model: mujoco.MjModel, data: mujoco.MjData, output_dir: Path, width: int, height: int) -> None:
    views = {
        "overview": (4.6, -130.0, -28.0, (0.0, 0.0, 1.05)),
        "robot_table_side": (3.2, -35.0, -18.0, (0.0, -0.75, 1.08)),
        "strike_lane": (2.4, 0.0, -14.0, (0.0, -0.55, 1.08)),
    }
    for name, (distance, azimuth, elevation, lookat) in views.items():
        _render_png(
            model,
            data,
            output_dir / f"{name}.png",
            width,
            height,
            distance,
            azimuth,
            elevation,
            lookat,
        )


def render_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output: Path,
    width: int,
    height: int,
    fps: int,
    seconds: float,
) -> None:
    renderer = mujoco.Renderer(model, height=height, width=width)
    _hide_collision_group(renderer)
    frame_count = max(1, int(round(fps * seconds)))
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=8) as writer:
        for frame in range(frame_count):
            phase = 2.0 * math.pi * frame / frame_count
            azimuth = -135.0 + math.degrees(phase)
            camera = _make_camera(4.2, azimuth, -24.0, (0.0, 0.0, 1.05))
            mujoco.mj_step(model, data)
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render images and video for the snooker scene.")
    parser.add_argument("--model", type=Path, default=project_root() / "models" / "scene_pool_asset.xml")
    parser.add_argument("--output-dir", type=Path, default=project_root() / "outputs" / "renders_pool_asset")
    parser.add_argument("--video", type=Path, default=project_root() / "outputs" / "videos" / "pool_asset_orbit.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.video.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(args.model))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    set_lift_grip_ready_pose(model, data)

    if not np.all(np.isfinite(data.qpos)):
        raise RuntimeError("Initial scene state contains non-finite qpos values.")

    render_images(model, data, args.output_dir, args.width, args.height)
    render_video(model, data, args.video, args.width, args.height, args.fps, args.seconds)
    print(f"images={args.output_dir}")
    print(f"video={args.video}")
    print(f"nbody={model.nbody} ngeom={model.ngeom} nmesh={model.nmesh}")


if __name__ == "__main__":
    main()
