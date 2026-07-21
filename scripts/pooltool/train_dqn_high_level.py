#!/usr/bin/env python3
"""Train a DQN high-level PoolTool policy over ball/pocket/landing actions."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snooker_env.pooltool_dqn import DQNTransition, PoolToolDQNEnv, QNetwork, ReplayBuffer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-shots", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=5000)
    parser.add_argument("--learning-starts", type=int, default=64)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--target-update-every", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-break-rack", action="store_true")
    parser.add_argument("--break-speed", type=float, default=10.0)
    parser.add_argument("--no-prune-blocked-actions", action="store_true")
    parser.add_argument("--no-cue-landing", action="store_true")
    parser.add_argument("--landing-x-bins", type=int, default=8)
    parser.add_argument("--landing-y-bins", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-epsilon", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("outputs/pooltool/dqn_high_level_training.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/pooltool/dqn_high_level.pt"))
    parser.add_argument("--latest-checkpoint", type=Path, default=Path("outputs/pooltool/dqn_high_level_latest.pt"))
    parser.add_argument("--best-eval-checkpoint", type=Path, default=Path("outputs/pooltool/dqn_high_level_best_eval.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = PoolToolDQNEnv(
        random_seed=args.seed,
        break_rack=not args.no_break_rack,
        break_speed=args.break_speed,
        prune_blocked_actions=not args.no_prune_blocked_actions,
        include_cue_landing=not args.no_cue_landing,
        landing_x_bins=args.landing_x_bins,
        landing_y_bins=args.landing_y_bins,
    )
    device = torch.device(args.device)
    q_net = QNetwork(env.state_dim, env.action_dim, hidden_dim=args.hidden_dim).to(device)
    target_net = QNetwork(env.state_dim, env.action_dim, hidden_dim=args.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer_size, seed=args.seed)

    global_step = 0
    updates = 0
    episode_records: list[dict[str, Any]] = []
    best_return = -math.inf
    best_eval_return = -math.inf
    eval_records: list[dict[str, Any]] = []

    for episode in range(args.episodes):
        state, mask = env.reset()
        episode_return = 0.0
        cleared = False
        shots: list[dict[str, Any]] = []
        last_loss: float | None = None

        for shot_idx in range(args.max_shots):
            epsilon = _epsilon(args, global_step)
            action = _select_action(q_net, state, mask, epsilon, device)
            next_state, reward, done, next_mask, info = env.step(action)
            replay.append(DQNTransition(state, action, reward, next_state, done, next_mask))
            episode_return += reward
            global_step += 1

            if len(replay) >= args.learning_starts and global_step % args.train_every == 0:
                last_loss = _train_step(q_net, target_net, optimizer, replay, args.batch_size, args.gamma, device)
                updates += 1
                if updates % args.target_update_every == 0:
                    target_net.load_state_dict(q_net.state_dict())

            action_obj = info["action"]
            shots.append(
                {
                    "shot_index": shot_idx,
                    "action_index": action,
                    "target_ball_id": action_obj.target_ball_id,
                    "target_pocket_id": action_obj.target_pocket_id,
                    "cue_landing_cell": action_obj.cue_landing_cell,
                    "reward": reward,
                    "success": bool(info["success"]),
                    "foul": bool(info["foul"]),
                    "reason": str(info["reason"]),
                    "pot_success": bool(info.get("pot_success")),
                    "landing_success": info.get("landing_success"),
                    "actual_cue_landing_cell": info.get("cue_landing_cell"),
                    "cue_landing_distance": info.get("cue_landing_distance"),
                    "remaining_balls": tuple(info.get("remaining_balls", ())),
                }
            )
            state, mask = next_state, next_mask
            cleared = not info.get("remaining_balls", ()) and bool(info["success"]) and not bool(info["foul"])
            if done:
                break

        record = {
            "episode": episode,
            "return": episode_return,
            "cleared": cleared,
            "shots": len(shots),
            "epsilon": _epsilon(args, global_step),
            "replay_size": len(replay),
            "updates": updates,
            "last_loss": last_loss,
            "trajectory": shots,
        }
        episode_records.append(record)
        if episode_return > best_return:
            best_return = episode_return
            _save_checkpoint(args.checkpoint, q_net, env, args, episode, episode_return, "best_training_return")

        if args.eval_every > 0 and (episode + 1) % args.eval_every == 0:
            eval_record = _evaluate_policy(q_net, args, device, episode)
            eval_records.append(eval_record)
            if eval_record["avg_return"] > best_eval_return:
                best_eval_return = float(eval_record["avg_return"])
                _save_checkpoint(
                    args.best_eval_checkpoint,
                    q_net,
                    env,
                    args,
                    episode,
                    best_eval_return,
                    "best_eval_return",
                )
            print(
                "eval: "
                f"episode={episode + 1} "
                f"avg_return={eval_record['avg_return']:.3f} "
                f"clear_rate={eval_record['clear_rate']:.3f} "
                f"epsilon={args.eval_epsilon:.3f}",
                flush=True,
            )

        if args.log_interval > 0 and (episode + 1) % args.log_interval == 0:
            print(
                "episode: "
                f"{episode + 1}/{args.episodes} "
                f"return={episode_return:.3f} "
                f"cleared={cleared} "
                f"shots={len(shots)} "
                f"epsilon={record['epsilon']:.3f} "
                f"replay={len(replay)} "
                f"updates={updates} "
                f"loss={last_loss}",
                flush=True,
            )

    summary = {
        "episodes": args.episodes,
        "best_return": best_return,
        "best_eval_return": best_eval_return,
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "actions": [(a.target_ball_id, a.target_pocket_id, a.cue_landing_cell) for a in env.actions],
        "records": episode_records,
        "eval_records": eval_records,
        "checkpoint": str(args.checkpoint),
        "latest_checkpoint": str(args.latest_checkpoint),
        "best_eval_checkpoint": str(args.best_eval_checkpoint),
    }
    _save_checkpoint(args.latest_checkpoint, q_net, env, args, args.episodes - 1, episode_records[-1]["return"], "latest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote={args.output}", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"latest_checkpoint={args.latest_checkpoint}", flush=True)
    print(f"best_eval_checkpoint={args.best_eval_checkpoint}", flush=True)


def _epsilon(args: argparse.Namespace, step: int) -> float:
    frac = min(1.0, step / max(1, args.epsilon_decay_steps))
    return args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)


def _select_action(
    q_net: QNetwork,
    state: tuple[float, ...],
    mask: tuple[bool, ...],
    epsilon: float,
    device: torch.device,
) -> int:
    valid = [idx for idx, ok in enumerate(mask) if ok]
    if not valid:
        raise RuntimeError("No valid DQN actions are available.")
    if random.random() < epsilon:
        return random.choice(valid)
    with torch.no_grad():
        q_values = q_net(torch.tensor([list(state)], dtype=torch.float32, device=device))[0]
        invalid = torch.tensor([not ok for ok in mask], dtype=torch.bool, device=device)
        q_values = q_values.masked_fill(invalid, -1.0e9)
        return int(torch.argmax(q_values).item())


def _evaluate_policy(
    q_net: QNetwork,
    args: argparse.Namespace,
    device: torch.device,
    train_episode: int,
) -> dict[str, Any]:
    eval_env = PoolToolDQNEnv(
        random_seed=args.seed,
        break_rack=not args.no_break_rack,
        break_speed=args.break_speed,
        prune_blocked_actions=not args.no_prune_blocked_actions,
        include_cue_landing=not args.no_cue_landing,
        landing_x_bins=args.landing_x_bins,
        landing_y_bins=args.landing_y_bins,
    )
    returns: list[float] = []
    cleared = 0
    shot_counts: list[int] = []
    was_training = q_net.training
    q_net.eval()
    for _ in range(args.eval_episodes):
        state, mask = eval_env.reset()
        episode_return = 0.0
        episode_cleared = False
        shots = 0
        for _shot_idx in range(args.max_shots):
            action = _select_action(q_net, state, mask, args.eval_epsilon, device)
            next_state, reward, done, next_mask, info = eval_env.step(action)
            episode_return += reward
            shots += 1
            episode_cleared = not info.get("remaining_balls", ()) and bool(info["success"]) and not bool(info["foul"])
            state, mask = next_state, next_mask
            if done:
                break
        returns.append(episode_return)
        cleared += int(episode_cleared)
        shot_counts.append(shots)
    if was_training:
        q_net.train()
    return {
        "train_episode": train_episode,
        "avg_return": sum(returns) / len(returns),
        "clear_rate": cleared / len(returns),
        "avg_shots": sum(shot_counts) / len(shot_counts),
        "episodes": args.eval_episodes,
        "epsilon": args.eval_epsilon,
    }


def _save_checkpoint(
    path: Path,
    q_net: QNetwork,
    env: PoolToolDQNEnv,
    args: argparse.Namespace,
    episode: int,
    score: float,
    kind: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": q_net.state_dict(),
            "state_dim": env.state_dim,
            "action_dim": env.action_dim,
            "actions": [(a.target_ball_id, a.target_pocket_id, a.cue_landing_cell) for a in env.actions],
            "args": vars(args),
            "episode": episode,
            "return": score,
            "kind": kind,
        },
        path,
    )


def _train_step(
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float:
    batch = replay.sample(batch_size)
    states = torch.tensor([list(item.state) for item in batch], dtype=torch.float32, device=device)
    actions = torch.tensor([item.action for item in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=device)
    next_states = torch.tensor([list(item.next_state) for item in batch], dtype=torch.float32, device=device)
    dones = torch.tensor([item.done for item in batch], dtype=torch.bool, device=device)
    next_invalid = torch.tensor(
        [[not ok for ok in item.next_action_mask] for item in batch],
        dtype=torch.bool,
        device=device,
    )

    q_sa = q_net(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states).masked_fill(next_invalid, -1.0e9).max(dim=1).values
        target = rewards + gamma * torch.where(dones, torch.zeros_like(next_q), next_q)
    loss = F.smooth_l1_loss(q_sa, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.detach().cpu().item())


if __name__ == "__main__":
    main()
