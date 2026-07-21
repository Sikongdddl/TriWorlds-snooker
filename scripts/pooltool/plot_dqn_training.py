#!/usr/bin/env python3
"""Plot PoolTool DQN training curves from the text log."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


EPISODE_RE = re.compile(
    r"^episode:\s+"
    r"(?P<episode>\d+)/(?P<total>\d+)\s+"
    r"return=(?P<return>-?\d+(?:\.\d+)?)\s+"
    r"cleared=(?P<cleared>True|False)\s+"
    r"shots=(?P<shots>\d+)\s+"
    r"epsilon=(?P<epsilon>\d+(?:\.\d+)?)\s+"
    r"replay=(?P<replay>\d+)\s+"
    r"updates=(?P<updates>\d+)\s+"
    r"loss=(?P<loss>None|-?\d+(?:\.\d+)?)"
)

EVAL_RE = re.compile(
    r"^eval:\s+"
    r"episode=(?P<episode>\d+)\s+"
    r"avg_return=(?P<avg_return>-?\d+(?:\.\d+)?)\s+"
    r"clear_rate=(?P<clear_rate>\d+(?:\.\d+)?)\s+"
    r"epsilon=(?P<epsilon>\d+(?:\.\d+)?)"
)


@dataclass(frozen=True)
class EpisodePoint:
    episode: int
    return_value: float
    cleared: bool
    shots: int
    epsilon: float
    replay: int
    updates: int
    loss: float | None


@dataclass(frozen=True)
class EvalPoint:
    episode: int
    avg_return: float
    clear_rate: float
    epsilon: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("outputs/pooltool/dqn_high_level_v2_train.log"))
    parser.add_argument("--output", type=Path, default=Path("outputs/pooltool/dqn_high_level_v2_training_curves.png"))
    parser.add_argument("--window", type=int, default=25, help="Moving average window over logged episode rows.")
    parser.add_argument("--title", default="PoolTool DQN High-Level Training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes, evals = _read_log(args.log)
    if not episodes:
        raise SystemExit(f"No episode rows found in log: {args.log}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot(episodes, evals, args.output, args.window, args.title)
    print(f"wrote={args.output}")
    print(f"episodes={episodes[-1].episode}")
    if evals:
        best_eval = max(evals, key=lambda point: point.avg_return)
        best_clear = max(evals, key=lambda point: point.clear_rate)
        print(
            "best_eval_return="
            f"{best_eval.avg_return:.3f}@episode{best_eval.episode} "
            f"best_eval_clear_rate={best_clear.clear_rate:.3f}@episode{best_clear.episode}"
        )


def _read_log(path: Path) -> tuple[list[EpisodePoint], list[EvalPoint]]:
    if not path.exists():
        raise SystemExit(f"Training log does not exist: {path}")

    episodes: list[EpisodePoint] = []
    evals: list[EvalPoint] = []
    for line in path.read_text().splitlines():
        episode_match = EPISODE_RE.match(line)
        if episode_match:
            groups = episode_match.groupdict()
            loss_text = groups["loss"]
            episodes.append(
                EpisodePoint(
                    episode=int(groups["episode"]),
                    return_value=float(groups["return"]),
                    cleared=groups["cleared"] == "True",
                    shots=int(groups["shots"]),
                    epsilon=float(groups["epsilon"]),
                    replay=int(groups["replay"]),
                    updates=int(groups["updates"]),
                    loss=None if loss_text == "None" else float(loss_text),
                )
            )
            continue

        eval_match = EVAL_RE.match(line)
        if eval_match:
            groups = eval_match.groupdict()
            evals.append(
                EvalPoint(
                    episode=int(groups["episode"]),
                    avg_return=float(groups["avg_return"]),
                    clear_rate=float(groups["clear_rate"]),
                    epsilon=float(groups["epsilon"]),
                )
            )

    return episodes, evals


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed: list[float] = []
    running_sum = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running_sum += value
        if len(queue) > window:
            running_sum -= queue.pop(0)
        smoothed.append(running_sum / len(queue))
    return smoothed


def _plot(
    episodes: list[EpisodePoint],
    evals: list[EvalPoint],
    output: Path,
    window: int,
    title: str,
) -> None:
    x = [point.episode for point in episodes]
    returns = [point.return_value for point in episodes]
    clear_flags = [1.0 if point.cleared else 0.0 for point in episodes]
    shots = [float(point.shots) for point in episodes]
    epsilons = [point.epsilon for point in episodes]
    replays = [float(point.replay) for point in episodes]
    updates = [float(point.updates) for point in episodes]
    loss_points = [point for point in episodes if point.loss is not None]
    loss_x = [point.episode for point in loss_points]
    losses = [float(point.loss) for point in loss_points if point.loss is not None]

    eval_x = [point.episode for point in evals]
    eval_returns = [point.avg_return for point in evals]
    eval_clear_rates = [point.clear_rate for point in evals]

    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.28,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.titlesize": 16,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(title)

    ax = axes[0][0]
    ax.plot(x, returns, color="#B7C6D9", linewidth=1.0, label="train return")
    ax.plot(x, _moving_average(returns, window), color="#1F77B4", linewidth=2.0, label=f"train return MA{window}")
    if evals:
        ax.plot(eval_x, eval_returns, color="#D62728", marker="o", linewidth=2.0, label="eval avg return")
    ax.set_title("Return")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.legend(loc="best")

    ax = axes[0][1]
    ax.plot(x, _moving_average(clear_flags, window), color="#2CA02C", linewidth=2.0, label=f"train clear rate MA{window}")
    if evals:
        ax.plot(eval_x, eval_clear_rates, color="#D62728", marker="o", linewidth=2.0, label="eval clear rate")
    ax.set_title("Clear Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")

    ax = axes[1][0]
    ax.plot(loss_x, losses, color="#9467BD", linewidth=1.2, alpha=0.7, label="TD loss")
    ax.plot(loss_x, _moving_average(losses, window), color="#4B2A75", linewidth=2.0, label=f"TD loss MA{window}")
    ax.set_title("Training Loss")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")

    ax = axes[1][1]
    ax.plot(x, epsilons, color="#FF7F0E", linewidth=2.0, label="train epsilon")
    if evals:
        ax.plot(eval_x, [point.epsilon for point in evals], color="#8C564B", marker="o", linewidth=1.8, label="eval epsilon")
    ax.set_title("Exploration")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best")

    ax = axes[2][0]
    ax.plot(x, shots, color="#17BECF", linewidth=1.2, alpha=0.55, label="shots")
    ax.plot(x, _moving_average(shots, window), color="#087C8F", linewidth=2.0, label=f"shots MA{window}")
    ax.set_title("Shots Per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Shots")
    ax.legend(loc="best")

    ax = axes[2][1]
    ax.plot(x, replays, color="#7F7F7F", linewidth=2.0, label="replay size")
    ax.plot(x, updates, color="#BCBD22", linewidth=2.0, label="updates")
    ax.set_title("Replay And Updates")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Count")
    ax.legend(loc="best")

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
