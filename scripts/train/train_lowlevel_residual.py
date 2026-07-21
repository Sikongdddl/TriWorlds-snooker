"""Train a PPO residual joint-position policy for timed cue commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.lowlevel_residual_env import LowLevelResidualEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "scene_pool_asset.xml")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "checkpoints" / "lowlevel_residual_ppo")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = Monitor(LowLevelResidualEnv(args.model, randomize_command=True))
    policy = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        device=args.device,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
    )
    policy.learn(total_timesteps=args.total_timesteps)
    policy.save(args.output)
    env.close()
    print(f"checkpoint={args.output}.zip")


if __name__ == "__main__":
    main()
