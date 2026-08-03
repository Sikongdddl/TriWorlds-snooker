"""Smoke test for the 12-D residual joint-position training environment."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.lowlevel_residual_env import LowLevelResidualEnv  # noqa: E402


def main() -> None:
    env = LowLevelResidualEnv()
    check_env(env, skip_render_check=True)

    observation, info = env.reset(seed=7)
    start_position = env._cue_pose().position.copy()
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        observation, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        total_reward += reward
        steps += 1

    final_position = env._cue_pose().position.copy()
    print(f"observation_shape={observation.shape}")
    print(f"action_shape={env.action_space.shape}")
    print(f"control_dt={env.control_dt:.4f}s steps={steps}")
    print(f"cue_displacement={final_position - start_position}")
    print(f"total_reward={total_reward:.4f}")
    print(f"terminated={terminated} truncated={truncated} success={info['success']}")
    print(
        "final_errors="
        f"position:{info['position_error']:.6f} "
        f"orientation:{info['orientation_error']:.6f} "
        f"grip:{info['grip_error']:.6f}"
    )

    if observation.shape != (82,) or not np.all(np.isfinite(observation)):
        raise RuntimeError("Residual environment returned an invalid observation.")
    if env.action_space.shape != (12,):
        raise RuntimeError("Residual action must contain 12 arm joint deltas.")
    if terminated:
        raise RuntimeError("Nominal zero-residual rollout hit a failure condition.")
    if not truncated or not info["success"]:
        raise RuntimeError("Nominal zero-residual rollout did not complete successfully.")
    if final_position[1] - start_position[1] < 0.01:
        raise RuntimeError("Nominal controller did not move the cue along the commanded stroke.")

    env.close()


if __name__ == "__main__":
    main()
