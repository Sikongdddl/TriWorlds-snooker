from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.scene import default_model_path, load_model  # noqa: E402


def view_scene(model_path: Path) -> None:
    model = load_model(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        if "overview" in [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]:
            viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the minimal snooker MuJoCo scene.")
    parser.add_argument("--model", type=Path, default=default_model_path(), help="Path to scene XML.")
    args = parser.parse_args()
    view_scene(args.model)


if __name__ == "__main__":
    main()
