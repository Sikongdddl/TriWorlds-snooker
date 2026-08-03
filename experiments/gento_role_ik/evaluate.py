"""Compare learned Gento residual IK with its zero-residual baseline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from snooker_env.gento_ik_env import GentoRoleIKEnv  # noqa: E402


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    successes: int
    success_rate: float
    mean_return: float
    mean_final_position_error: float
    mean_direction_error: float
    mean_front_support_error: float
    mean_rear_grip_error: float
    mean_rear_speed_error: float
    mean_peak_cue_ball_speed: float
    maximum_robot_table_penetration: float
    maximum_cue_palm_penetration: float
    maximum_cue_table_penetration: float
    maximum_lost_support_steps: int
    maximum_lost_speed_steps: int
    mean_absolute_action: float


def evaluate(
    action_provider: Callable[[np.ndarray], np.ndarray],
    *,
    episodes: int,
) -> EvaluationResult:
    env = GentoRoleIKEnv(randomize_command=True)
    returns: list[float] = []
    position_errors: list[float] = []
    direction_errors: list[float] = []
    support_errors: list[float] = []
    grip_errors: list[float] = []
    speed_errors: list[float] = []
    ball_speeds: list[float] = []
    action_magnitudes: list[float] = []
    successes = 0
    max_robot_table = max_cue_palm = max_cue_table = 0.0
    max_lost_support = max_lost_speed = 0

    for seed in range(episodes):
        observation, _ = env.reset(seed=seed)
        episode_return = 0.0
        terminated = truncated = False
        info: dict[str, object] = {}
        while not (terminated or truncated):
            action = np.asarray(action_provider(observation), dtype=np.float32)
            action_magnitudes.append(float(np.mean(np.abs(action))))
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            max_robot_table = max(
                max_robot_table,
                float(info["max_robot_table_penetration"]),
            )
            max_cue_palm = max(
                max_cue_palm,
                float(info["max_cue_palm_penetration"]),
            )
            max_cue_table = max(
                max_cue_table,
                float(info["max_cue_table_penetration"]),
            )
            max_lost_support = max(
                max_lost_support,
                int(info["lost_support_contact_steps"]),
            )
            max_lost_speed = max(
                max_lost_speed,
                int(info["lost_speed_contact_steps"]),
            )
        successes += int(bool(info["success"]))
        returns.append(episode_return)
        position_errors.append(float(info["position_error"]))
        direction_errors.append(float(info["support_direction_error"]))
        support_errors.append(float(info["front_support_error"]))
        grip_errors.append(float(info["rear_grip_error"]))
        speed_errors.append(float(info["mean_rear_speed_error"]))
        ball_speeds.append(float(info["peak_cue_ball_speed"]))

    env.close()
    return EvaluationResult(
        episodes=episodes,
        successes=successes,
        success_rate=successes / episodes,
        mean_return=float(np.mean(returns)),
        mean_final_position_error=float(np.mean(position_errors)),
        mean_direction_error=float(np.mean(direction_errors)),
        mean_front_support_error=float(np.mean(support_errors)),
        mean_rear_grip_error=float(np.mean(grip_errors)),
        mean_rear_speed_error=float(np.mean(speed_errors)),
        mean_peak_cue_ball_speed=float(np.mean(ball_speeds)),
        maximum_robot_table_penetration=max_robot_table,
        maximum_cue_palm_penetration=max_cue_palm,
        maximum_cue_table_penetration=max_cue_table,
        maximum_lost_support_steps=max_lost_support,
        maximum_lost_speed_steps=max_lost_speed,
        mean_absolute_action=float(np.mean(action_magnitudes)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "assets" / "policies" / "gento_role_ik_residual_final.zip",
    )
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    policy = PPO.load(args.checkpoint, device="cpu")
    zeros = np.zeros(14, dtype=np.float32)
    baseline = evaluate(lambda _observation: zeros, episodes=args.episodes)
    learned = evaluate(
        lambda observation: policy.predict(observation, deterministic=True)[0],
        episodes=args.episodes,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "baseline_zero_residual": asdict(baseline),
        "learned_residual": asdict(learned),
    }
    output_dir = EXPERIMENT_DIR / "artifacts" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"evaluation={output_path}")


if __name__ == "__main__":
    main()
