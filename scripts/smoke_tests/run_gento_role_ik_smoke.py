"""Validate the imported Gento side grasp, role-aware IK, and PPO residual."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mujoco
import numpy as np
from stable_baselines3 import PPO

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.gento_ik_env import GentoRoleIKEnv  # noqa: E402


class ActionPolicy(Protocol):
    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray, object]: ...


@dataclass(frozen=True)
class RolloutResult:
    label: str
    success: bool
    steps: int
    peak_ball_speed: float
    ball_displacement: np.ndarray
    max_robot_table_penetration: float
    max_cue_palm_penetration: float
    max_cue_table_penetration: float
    max_direction_error: float
    max_support_error: float
    max_rear_grip_error: float
    mean_rear_speed_error: float


def _body_position(env: GentoRoleIKEnv, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"Gento scene is missing body {name!r}.")
    return env.data.xpos[body_id].copy()


def _rollout(label: str, policy: ActionPolicy | None) -> RolloutResult:
    env = GentoRoleIKEnv()
    observation, _ = env.reset(seed=7)
    if observation.shape != (90,) or env.action_space.shape != (14,):
        raise RuntimeError(
            "Gento IK contract drifted: expected observation (90,) and action (14,)."
        )

    initial_contacts = env._contact_metrics()
    if initial_contacts["support_cue_contact_count"] < 2.0:
        raise RuntimeError("The forward support fingers do not physically contact the cue.")
    if initial_contacts["speed_cue_contact_count"] < 2.0:
        raise RuntimeError("The rear speed fingers do not physically contact the cue.")

    cue_shaft_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "cue_shaft",
    )
    cue_bottom = float(env.data.xpos[env.cue_body_id, 2] - env.model.geom_size[cue_shaft_id, 0])
    rail_top = 1.090
    if cue_bottom < rail_top:
        raise RuntimeError(
            f"Cue starts inside the rail: bottom={cue_bottom:.6f}, rail={rail_top:.6f}."
        )

    start_ball = _body_position(env, "cue_ball")
    maxima = {
        "max_robot_table_penetration": 0.0,
        "max_cue_palm_penetration": 0.0,
        "max_cue_table_penetration": 0.0,
        "support_direction_error": 0.0,
        "front_support_error": 0.0,
        "rear_grip_error": 0.0,
    }
    steps = 0
    terminated = truncated = False
    info: dict[str, object] = {}
    while not (terminated or truncated):
        if policy is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        steps += 1
        for key in maxima:
            maxima[key] = max(maxima[key], float(info[key]))

    end_ball = _body_position(env, "cue_ball")
    result = RolloutResult(
        label=label,
        success=bool(info["success"]),
        steps=steps,
        peak_ball_speed=float(info["peak_cue_ball_speed"]),
        ball_displacement=end_ball - start_ball,
        max_robot_table_penetration=maxima["max_robot_table_penetration"],
        max_cue_palm_penetration=maxima["max_cue_palm_penetration"],
        max_cue_table_penetration=maxima["max_cue_table_penetration"],
        max_direction_error=maxima["support_direction_error"],
        max_support_error=maxima["front_support_error"],
        max_rear_grip_error=maxima["rear_grip_error"],
        mean_rear_speed_error=float(info["mean_rear_speed_error"]),
    )
    env.close()

    if terminated or not truncated or not result.success:
        raise RuntimeError(f"{label} rollout did not complete successfully.")
    if not np.all(np.isfinite(observation)):
        raise RuntimeError(f"{label} rollout returned a non-finite observation.")
    if result.peak_ball_speed < 0.03 or result.ball_displacement[1] < 0.02:
        raise RuntimeError(f"{label} rollout did not strike the ball along world +Y.")
    if result.max_robot_table_penetration > 0.003:
        raise RuntimeError(f"{label} robot penetrated the table collision body.")
    if result.max_cue_palm_penetration > 0.002:
        raise RuntimeError(f"{label} cue penetrated a solid palm guard.")
    if result.max_cue_table_penetration > 0.001:
        raise RuntimeError(f"{label} cue penetrated the source table geometry.")
    return result


def main() -> None:
    checkpoint = ROOT / "assets" / "policies" / "gento_role_ik_residual_final.zip"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing imported PPO checkpoint: {checkpoint}")
    policy = PPO.load(checkpoint, device="cpu")
    if policy.observation_space.shape != (90,) or policy.action_space.shape != (14,):
        raise RuntimeError("Imported PPO checkpoint is incompatible with the Gento IK environment.")

    for result in (
        _rollout("nominal_ik", None),
        _rollout("imported_ppo", policy),
    ):
        print(
            f"{result.label}: success={result.success} steps={result.steps} "
            f"ball_peak={result.peak_ball_speed:.6f}m/s "
            f"ball_dxyz={result.ball_displacement}"
        )
        print(
            "  max_penetration="
            f"robot/table:{result.max_robot_table_penetration:.9f}m "
            f"cue/palm:{result.max_cue_palm_penetration:.9f}m "
            f"cue/table:{result.max_cue_table_penetration:.9f}m"
        )
        print(
            "  role_errors="
            f"direction:{result.max_direction_error:.6f}rad "
            f"support:{result.max_support_error:.6f}m "
            f"rear_grip:{result.max_rear_grip_error:.6f}m "
            f"rear_speed_mean:{result.mean_rear_speed_error:.6f}m/s"
        )


if __name__ == "__main__":
    main()
