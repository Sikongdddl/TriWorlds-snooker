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


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the current LIFT/cue dual-grip scaffold alignment.")
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--max-distance", type=float, default=0.08)
    args = parser.parse_args()

    model = load_model(args.model)
    data = mujoco.MjData(model)
    set_lift_grip_ready_pose(model, data)

    failures: list[str] = []
    for lift_site, cue_site in PAIRINGS:
        lift_pos = _site_pos(model, data, lift_site)
        cue_pos = _site_pos(model, data, cue_site)
        delta = lift_pos - cue_pos
        dist = float(np.linalg.norm(delta))
        print(f"{lift_site} -> {cue_site}")
        print(f"  lift: {lift_pos}")
        print(f"  cue:  {cue_pos}")
        print(f"  delta: {delta}, distance={dist:.4f} m")
        if dist > args.max_distance:
            failures.append(f"{lift_site}/{cue_site}: {dist:.4f} m")

    print(f"nu={model.nu}, nq={model.nq}, nv={model.nv}")
    if failures:
        raise RuntimeError("Dual-grip scaffold alignment exceeds threshold: " + ", ".join(failures))


if __name__ == "__main__":
    main()
