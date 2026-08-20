#!/usr/bin/env python3
"""Train two turn-taking DQN agents in the PoolTool high-level environment."""

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

from snooker_env.pooltool_dqn import DQNTransition, QNetwork, ReplayBuffer, TwoPlayerPoolToolDQNEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Stop after the active player reaches this many gradient updates. Episodes remain an upper bound.",
    )
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--learning-starts", type=int, default=64)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--target-update-every", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-reward-scale", type=float, default=1.0)
    parser.add_argument("--train-reward-clip", type=float, default=100.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-break-rack", action="store_true")
    parser.add_argument("--break-speed", type=float, default=10.0)
    parser.add_argument("--randomize-break", action="store_true")
    parser.add_argument("--break-speed-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--break-phi-jitter-degrees", type=float, default=0.0)
    parser.add_argument(
        "--prune-blocked-actions",
        action="store_true",
        help=(
            "Mask geometrically blocked pocket actions. Disabled by default because this leaks "
            "shot feasibility into the policy."
        ),
    )
    parser.add_argument("--self-play-clone", action="store_true")
    parser.add_argument("--active-player", type=int, default=0)
    parser.add_argument("--clone-window", type=int, default=50)
    parser.add_argument("--clone-min-episodes", type=int, default=50)
    parser.add_argument("--clone-win-rate", type=float, default=0.75)
    parser.add_argument("--clone-reward-advantage", type=float, default=5.0)
    parser.add_argument("--initial-opponent-epsilon", type=float, default=1.0)
    parser.add_argument("--opponent-epsilon", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("outputs/pooltool/two_player_dqn_training.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/pooltool/two_player_dqn.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    env = _make_env(args)
    q_nets = [QNetwork(env.state_dim, env.action_dim, hidden_dim=args.hidden_dim).to(device) for _ in range(2)]
    target_nets = [QNetwork(env.state_dim, env.action_dim, hidden_dim=args.hidden_dim).to(device) for _ in range(2)]
    for idx in range(2):
        target_nets[idx].load_state_dict(q_nets[idx].state_dict())
    optimizers = [torch.optim.Adam(q_nets[idx].parameters(), lr=args.lr) for idx in range(2)]
    replays = [ReplayBuffer(args.buffer_size, seed=args.seed + idx) for idx in range(2)]
    updates = [0, 0]
    global_steps = [0, 0]
    records: list[dict[str, Any]] = []
    clone_events: list[dict[str, Any]] = []
    best_clear_turns = math.inf

    for episode in range(args.episodes):
        state, mask = env.reset()
        episode_returns = [0.0, 0.0]
        turns: list[dict[str, Any]] = []
        winner: int | None = None
        last_loss: list[float | None] = [None, None]

        for turn_idx in range(args.max_turns):
            player = env.current_player
            epsilon = _player_epsilon(args, player, sum(global_steps), bool(clone_events))
            valid_action_count = sum(1 for ok in mask if ok)
            action = _select_action(q_nets[player], state, mask, epsilon, device)
            next_state, reward, done, next_mask, info = env.step(action)
            episode_returns[player] += reward
            should_train_player = (not args.self_play_clone) or player == args.active_player
            if should_train_player:
                train_reward = _training_reward(reward, args.train_reward_scale, args.train_reward_clip)
                replays[player].append(DQNTransition(state, action, train_reward, next_state, done, next_mask))
                global_steps[player] += 1
            if (
                should_train_player
                and len(replays[player]) >= args.learning_starts
                and global_steps[player] % args.train_every == 0
            ):
                last_loss[player] = _train_step(
                    q_nets[player],
                    target_nets[player],
                    optimizers[player],
                    replays[player],
                    args.batch_size,
                    args.gamma,
                    device,
                )
                updates[player] += 1
                if updates[player] % args.target_update_every == 0:
                    target_nets[player].load_state_dict(q_nets[player].state_dict())
            stop_for_updates = (
                args.max_updates is not None
                and updates[args.active_player if args.self_play_clone else player] >= args.max_updates
            )

            action_obj = info["action"]
            turns.append(
                {
                    "turn": turn_idx,
                    "player": player,
                    "action_index": action,
                    "valid_action_count": valid_action_count,
                    "epsilon": epsilon,
                    "trained_player": should_train_player,
                    "target_ball_id": action_obj.target_ball_id,
                    "target_pocket_id": action_obj.target_pocket_id,
                    "reward": reward,
                    "success": bool(info.get("success", False)),
                    "foul": bool(info.get("foul", False)),
                    "reason": str(info.get("reason")),
                    "shot_path_type": info.get("shot_path_type"),
                    "cue_ball_displacement": info.get("cue_ball_displacement"),
                    "cue_ball_restored": bool(info.get("cue_ball_restored", False)),
                    "switch_turn": bool(info.get("switch_turn", False)),
                    "remaining_balls": tuple(info.get("remaining_balls", ())),
                    "player_scores": tuple(info.get("player_scores", ())),
                }
            )
            if env.env.is_cleared():
                winner = player
            state, mask = next_state, next_mask
            if done or stop_for_updates:
                break

        record = {
            "episode": episode,
            "turns": len(turns),
            "cleared": env.env.is_cleared(),
            "winner": winner,
            "returns": episode_returns,
            "player_scores": tuple(env.player_scores),
            "player_turns": tuple(env.player_turns),
            "epsilon": _epsilon(args, sum(global_steps)),
            "replay_sizes": [len(replay) for replay in replays],
            "updates": tuple(updates),
            "last_loss": last_loss,
            "clone_count": len(clone_events),
            "trajectory": turns,
        }
        records.append(record)
        clone_event = _maybe_clone_opponent(args, q_nets, target_nets, records, clone_events)
        if clone_event is not None:
            clone_events.append(clone_event)
            print(
                "clone: "
                f"episode={episode + 1} "
                f"active={args.active_player} "
                f"opponent={clone_event['opponent_player']} "
                f"win_rate={clone_event['window_win_rate']:.3f} "
                f"reward_advantage={clone_event['window_reward_advantage']:.3f}",
                flush=True,
            )
        if record["cleared"] and len(turns) < best_clear_turns:
            best_clear_turns = len(turns)
            _save_checkpoint(args.checkpoint, q_nets, env, args, episode, record, "best_clear_turns")

        if args.log_interval > 0 and (episode + 1) % args.log_interval == 0:
            print(
                "episode: "
                f"{episode + 1}/{args.episodes} "
                f"cleared={record['cleared']} "
                f"winner={winner} "
                f"turns={len(turns)} "
                f"returns=({episode_returns[0]:.2f},{episode_returns[1]:.2f}) "
                f"scores={tuple(env.player_scores)} "
                f"epsilon={record['epsilon']:.3f} "
                f"clones={len(clone_events)} "
                f"updates={tuple(updates)} "
                f"loss={last_loss}",
                flush=True,
            )

        if args.save_interval > 0 and (episode + 1) % args.save_interval == 0:
            _write_summary(args.output, env, args, records, clone_events)
            _save_checkpoint(args.checkpoint, q_nets, env, args, episode, record, "periodic")
        if args.max_updates is not None and updates[args.active_player if args.self_play_clone else 0] >= args.max_updates:
            break

    _save_checkpoint(args.checkpoint, q_nets, env, args, args.episodes - 1, records[-1], "latest")
    _write_summary(args.output, env, args, records, clone_events)
    print(f"wrote={args.output}")
    print(f"checkpoint={args.checkpoint}")


def _make_env(args: argparse.Namespace) -> TwoPlayerPoolToolDQNEnv:
    return TwoPlayerPoolToolDQNEnv(
        random_seed=args.seed,
        break_rack=not args.no_break_rack,
        break_speed=args.break_speed,
        randomize_break=args.randomize_break,
        break_speed_range=None if args.break_speed_range is None else tuple(args.break_speed_range),
        break_phi_jitter_degrees=args.break_phi_jitter_degrees,
        prune_blocked_actions=args.prune_blocked_actions,
        landing_mask_cache_path=None,
    )


def _write_summary(
    path: Path,
    env: TwoPlayerPoolToolDQNEnv,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    clone_events: list[dict[str, Any]],
) -> None:
    summary = {
        "episodes": args.episodes,
        "completed_episodes": len(records),
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "actions": [(a.target_ball_id, a.target_pocket_id, a.cue_landing_cell) for a in env.actions],
        "records": records,
        "clone_events": clone_events,
        "checkpoint": str(args.checkpoint),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _epsilon(args: argparse.Namespace, step: int) -> float:
    frac = min(1.0, step / max(1, args.epsilon_decay_steps))
    return args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)


def _training_reward(reward: float, scale: float, clip: float) -> float:
    value = float(reward) * scale
    if clip > 0.0:
        value = max(-clip, min(clip, value))
    return value


def _player_epsilon(args: argparse.Namespace, player: int, total_trained_steps: int, has_cloned: bool) -> float:
    if not args.self_play_clone or player == args.active_player:
        return _epsilon(args, total_trained_steps)
    return args.opponent_epsilon if has_cloned else args.initial_opponent_epsilon


def _maybe_clone_opponent(
    args: argparse.Namespace,
    q_nets: list[QNetwork],
    target_nets: list[QNetwork],
    records: list[dict[str, Any]],
    clone_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not args.self_play_clone:
        return None
    if args.active_player not in {0, 1}:
        raise ValueError("--active-player must be 0 or 1.")
    if len(records) < args.clone_min_episodes or len(records) < args.clone_window:
        return None
    if clone_events and int(clone_events[-1]["episode"]) + args.clone_window > len(records):
        return None
    window = records[-args.clone_window :]
    active = args.active_player
    opponent = 1 - active
    active_wins = sum(1 for record in window if record.get("winner") == active)
    win_rate = active_wins / len(window)
    reward_advantage = sum(float(record["returns"][active]) - float(record["returns"][opponent]) for record in window)
    reward_advantage /= len(window)
    if win_rate < args.clone_win_rate or reward_advantage < args.clone_reward_advantage:
        return None
    q_nets[opponent].load_state_dict(q_nets[active].state_dict())
    target_nets[opponent].load_state_dict(q_nets[active].state_dict())
    return {
        "episode": len(records),
        "active_player": active,
        "opponent_player": opponent,
        "window": args.clone_window,
        "window_win_rate": win_rate,
        "window_reward_advantage": reward_advantage,
    }


def _select_action(
    q_net: QNetwork,
    state: tuple[float, ...],
    mask: tuple[bool, ...],
    epsilon: float,
    device: torch.device,
) -> int:
    valid = [idx for idx, ok in enumerate(mask) if ok]
    if not valid:
        raise RuntimeError("No valid two-player DQN actions are available.")
    if random.random() < epsilon:
        return random.choice(valid)
    with torch.no_grad():
        q_values = q_net(torch.tensor([list(state)], dtype=torch.float32, device=device))[0]
        q_values = torch.nan_to_num(q_values, nan=-1.0e9, posinf=1.0e9, neginf=-1.0e9)
        invalid = torch.tensor([not ok for ok in mask], dtype=torch.bool, device=device)
        q_values = q_values.masked_fill(invalid, -1.0e9)
        return int(torch.argmax(q_values).item())


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
    next_invalid = torch.tensor([[not ok for ok in item.next_action_mask] for item in batch], dtype=torch.bool, device=device)
    states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
    next_states = torch.nan_to_num(next_states, nan=0.0, posinf=0.0, neginf=0.0)
    rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
    q_sa = q_net(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_q_all = target_net(next_states).masked_fill(next_invalid, -1.0e9)
        next_q = next_q_all.max(dim=1).values
        next_q = torch.where(next_invalid.all(dim=1), torch.zeros_like(next_q), next_q)
        next_q = torch.nan_to_num(next_q, nan=0.0, posinf=0.0, neginf=0.0)
        target = rewards + gamma * torch.where(dones, torch.zeros_like(next_q), next_q)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    loss = F.smooth_l1_loss(q_sa, target)
    if not torch.isfinite(loss):
        return float("nan")
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.detach().cpu().item())


def _save_checkpoint(
    path: Path,
    q_nets: list[QNetwork],
    env: TwoPlayerPoolToolDQNEnv,
    args: argparse.Namespace,
    episode: int,
    record: dict[str, Any],
    kind: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "player_model_state_dicts": [net.state_dict() for net in q_nets],
            "state_dim": env.state_dim,
            "action_dim": env.action_dim,
            "actions": [(a.target_ball_id, a.target_pocket_id, a.cue_landing_cell) for a in env.actions],
            "args": vars(args),
            "episode": episode,
            "record": record,
            "kind": kind,
        },
        path,
    )


if __name__ == "__main__":
    main()
