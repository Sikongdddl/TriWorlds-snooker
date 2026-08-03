from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.scene import default_model_path, load_model  # noqa: E402
from snooker_env.init_pose import set_lift_grip_ready_pose  # noqa: E402


def view_scene(model_path: Path, fixed_camera: str | None = None) -> None:
    model = load_model(model_path)
    data = mujoco.MjData(model)
    set_lift_grip_ready_pose(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        if fixed_camera is not None:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, fixed_camera)
            if camera_id < 0:
                raise ValueError(f"Model does not contain camera: {fixed_camera}")
            viewer.cam.fixedcamid = camera_id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        else:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = (0.0, 0.0, 1.05)
            viewer.cam.distance = 3.2
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -40.0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the minimal snooker MuJoCo scene.")
    parser.add_argument("--model", type=Path, default=default_model_path(), help="Path to scene XML.")
    parser.add_argument(
        "--fixed-camera",
        help="Use a named fixed camera such as cam1 or cam2; default is mouse-controlled.",
    )
    args = parser.parse_args()
    view_scene(args.model, fixed_camera=args.fixed_camera)


if __name__ == "__main__":
    main()
