from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.init_pose import set_lift_grip_ready_pose  # noqa: E402
from snooker_env.scene import default_model_path, load_model  # noqa: E402


PAIRINGS = [
    ("lift_left_gripper_tcp", "cue_left_grip_site"),
    ("lift_right_gripper_tcp", "cue_right_grip_site"),
]


def _site_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise ValueError(f"Missing site: {name}")
    return data.site_xpos[site_id].copy()


def _max_grip_error(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return max(
        float(np.linalg.norm(_site_pos(model, data, lift_site) - _site_pos(model, data, cue_site)))
        for lift_site, cue_site in PAIRINGS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test LIFT TCP to cue grip soft equality constraints.")
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--max-error", type=float, default=0.02)
    args = parser.parse_args()

    model = load_model(args.model)
    data = mujoco.MjData(model)
    set_lift_grip_ready_pose(model, data)

    initial_error = _max_grip_error(model, data)
    has_nan = False
    exploded = False
    max_error = initial_error

    steps = int(args.duration / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            has_nan = True
            break
        if np.max(np.abs(data.qvel)) > 150.0 or np.max(np.abs(data.qpos)) > 25.0:
            exploded = True
            break
        max_error = max(max_error, _max_grip_error(model, data))

    final_error = _max_grip_error(model, data)
    print(f"Equality constraints: {model.neq}")
    print(f"Initial grip error: {initial_error:.6f} m")
    print(f"Final grip error: {final_error:.6f} m")
    print(f"Max grip error: {max_error:.6f} m")
    print(f"NaN/Inf detected: {has_nan}")
    print(f"Numerical explosion detected: {exploded}")

    if model.neq < 2:
        raise RuntimeError("Expected at least two grip equality constraints.")
    if has_nan or exploded:
        raise RuntimeError("Grip constraint smoke failed: unstable simulation.")
    if max_error > args.max_error:
        raise RuntimeError(f"Grip constraint smoke failed: max grip error {max_error:.6f} m exceeds {args.max_error:.6f} m.")


if __name__ == "__main__":
    main()
