from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.scene import default_model_path, load_model  # noqa: E402


def _names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int) -> list[str]:
    result: list[str] = []
    for idx in range(count):
        name = mujoco.mj_id2name(model, obj_type, idx)
        result.append(name if name is not None else f"<unnamed:{idx}>")
    return result


def _require_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Required {obj_type.name} name is missing: {name}")
    return obj_id


def _check_numeric_model(model: mujoco.MjModel) -> None:
    arrays = {
        "body_mass": model.body_mass,
        "body_inertia": model.body_inertia,
        "geom_size": model.geom_size,
    }
    for label, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Model array {label} contains NaN or Inf")

    for body_id in range(1, model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        mass = float(model.body_mass[body_id])
        inertia = model.body_inertia[body_id]
        if mass <= 0.0:
            raise ValueError(f"Body {body_name} has invalid mass {mass}")
        if np.any(inertia <= 0.0) or not np.all(np.isfinite(inertia)):
            raise ValueError(f"Body {body_name} has invalid inertia {inertia}")


def inspect_model(model_path: Path, check_strike_required: bool = False) -> None:
    model = load_model(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        raise ValueError("Initial qpos/qvel contains NaN or Inf")
    _check_numeric_model(model)

    print(f"Loaded model: {model_path}")
    print(f"nbody={model.nbody}, njnt={model.njnt}, ngeom={model.ngeom}, nsite={model.nsite}")
    print()

    print("Bodies:")
    for name in _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody):
        print(f"  {name}")
    print("Joints:")
    for name in _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt):
        print(f"  {name}")
    print("Geoms:")
    for name in _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom):
        print(f"  {name}")
    print("Sites:")
    for name in _names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite):
        print(f"  {name}")
    print()

    if check_strike_required:
        required = {
            "cue_body": mujoco.mjtObj.mjOBJ_BODY,
            "cue_tip": mujoco.mjtObj.mjOBJ_GEOM,
            "cue_tip_site": mujoco.mjtObj.mjOBJ_SITE,
            "cue_ball": mujoco.mjtObj.mjOBJ_BODY,
            "cue_ball_geom": mujoco.mjtObj.mjOBJ_GEOM,
            "object_ball_0": mujoco.mjtObj.mjOBJ_BODY,
            "object_ball_0_geom": mujoco.mjtObj.mjOBJ_GEOM,
        }
        print("Required IDs:")
        for name, obj_type in required.items():
            print(f"  {name}: {_require_name(model, obj_type, name)}")
        print()

    if not math.isfinite(float(model.opt.timestep)) or model.opt.timestep <= 0:
        raise ValueError(f"Invalid timestep: {model.opt.timestep}")
    print("Simulation options:")
    print(f"  timestep: {model.opt.timestep}")
    print(f"  solver: {model.opt.solver}")
    print(f"  iterations: {model.opt.iterations}")
    print(f"  tolerance: {model.opt.tolerance}")
    print(f"  gravity: {model.opt.gravity}")
    print("Model inspection passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the minimal snooker MuJoCo model.")
    parser.add_argument("--model", type=Path, default=default_model_path(), help="Path to scene XML.")
    parser.add_argument(
        "--check-strike-required",
        action="store_true",
        help="Check for the old primitive strike-test object names.",
    )
    parser.add_argument(
        "--skip-required",
        action="store_true",
        help="Deprecated no-op kept for compatibility.",
    )
    args = parser.parse_args()
    inspect_model(args.model, check_strike_required=args.check_strike_required)


if __name__ == "__main__":
    main()
