"""Train PPO residual control for Gento's role-aware physical IK."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from snooker_env.gento_ik_env import GentoRoleIKEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-frequency", type=int, default=10_000)
    args = parser.parse_args()
    if args.total_timesteps <= 0 or args.checkpoint_frequency <= 0:
        raise ValueError("Timesteps and checkpoint frequency must be positive.")

    artifacts = EXPERIMENT_DIR / "artifacts"
    checkpoint_dir = artifacts / "checkpoints"
    log_dir = artifacts / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    monitor_path = log_dir / "training.monitor.csv"
    env = Monitor(GentoRoleIKEnv(randomize_command=True), filename=str(monitor_path))
    callback = CheckpointCallback(
        save_freq=args.checkpoint_frequency,
        save_path=str(checkpoint_dir),
        name_prefix="gento_role_ik_residual",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    policy = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        device=args.device,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        learning_rate=3.0e-4,
        gamma=0.99,
        policy_kwargs={"log_std_init": -1.0},
    )

    started = time.time()
    policy.learn(total_timesteps=args.total_timesteps, callback=callback)
    elapsed_seconds = time.time() - started
    final_path = checkpoint_dir / "gento_role_ik_residual_final"
    policy.save(final_path)
    env.close()

    metadata = {
        "total_timesteps_requested": args.total_timesteps,
        "total_timesteps_completed": int(policy.num_timesteps),
        "seed": args.seed,
        "device": args.device,
        "elapsed_seconds": elapsed_seconds,
        "checkpoint": str(final_path.with_suffix(".zip")),
        "monitor": str(monitor_path),
    }
    (artifacts / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
