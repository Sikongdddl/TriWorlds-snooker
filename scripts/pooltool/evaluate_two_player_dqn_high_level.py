#!/usr/bin/env python3
"""Evaluate a two-player PoolTool DQN checkpoint and save a renderable rollout."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snooker_env.pooltool_dqn import QNetwork, TwoPlayerPoolToolDQNEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multisystem-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")

    device = torch.device(args.device)
    action_random = random.Random(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})
    state_dicts = checkpoint.get("player_model_state_dicts")
    if not isinstance(state_dicts, list) or len(state_dicts) != 2:
        raise SystemExit("Checkpoint does not contain two player_model_state_dicts.")

    env = TwoPlayerPoolToolDQNEnv(
        random_seed=args.seed,
        break_rack=not bool(train_args.get("no_break_rack", False)),
        break_speed=float(train_args.get("break_speed", 10.0)),
        randomize_break=bool(train_args.get("randomize_break", False)),
        break_speed_range=None
        if train_args.get("break_speed_range") is None
        else tuple(float(value) for value in train_args["break_speed_range"]),
        break_phi_jitter_degrees=float(train_args.get("break_phi_jitter_degrees", 0.0)),
        prune_blocked_actions=bool(train_args.get("prune_blocked_actions", False)),
        landing_mask_cache_path=None,
    )
    if env.action_dim != int(checkpoint["action_dim"]):
        raise SystemExit(
            f"Checkpoint action_dim={checkpoint['action_dim']} does not match environment action_dim={env.action_dim}."
        )

    hidden_dim = int(train_args.get("hidden_dim", 256))
    networks = [
        QNetwork(int(checkpoint["state_dim"]), int(checkpoint["action_dim"]), hidden_dim=hidden_dim).to(device)
        for _ in range(2)
    ]
    for network, state_dict in zip(networks, state_dicts):
        network.load_state_dict(state_dict)
        network.eval()

    state, mask = env.reset()
    systems = [env.env.system.copy()]
    shots: list[dict[str, Any]] = []
    returns = [0.0, 0.0]

    for turn_index in range(args.max_turns):
        player = env.current_player
        action_index, q_values = select_action(
            networks[player], state, mask, args.epsilon, action_random, device
        )
        next_state, reward, done, next_mask, info = env.step(action_index)
        returns[player] += float(reward)
        action = info["action"]
        solution = info.get("solution")
        record = {
            "label": f"Player {player} turn {turn_index + 1}",
            "rollout_title": "Two-player DQN rollout",
            "turn_index": turn_index,
            "player": player,
            "action_index": action_index,
            "q_values": q_values,
            "target_ball_id": action.target_ball_id,
            "target_pocket_id": action.target_pocket_id,
            "reward": float(reward),
            "score": float(reward),
            "success": bool(info.get("success", False)),
            "foul": bool(info.get("foul", False)),
            "reason": str(info.get("reason")),
            "switch_turn": bool(info.get("switch_turn", False)),
            "player_scores": list(info.get("player_scores", ())),
            "player_turns": list(info.get("player_turns", ())),
            "remaining_balls": list(info.get("remaining_balls", ())),
            "solution": None
            if solution is None
            else {
                "speed": float(solution.speed),
                "phi": float(solution.phi),
                "side_spin": float(solution.side_spin),
                "top_spin": float(solution.top_spin),
                "elevation": float(solution.elevation),
                "path_type": str(solution.path_type),
            },
        }
        shots.append(record)
        rendered_system = env.env.system.copy()
        if solution is None:
            # An unsolved action does not run PoolTool physics. Clear the
            # previous shot history so the renderer shows a stationary state
            # instead of replaying the preceding shot after the turn switch.
            rendered_system.reset_history()
        systems.append(rendered_system)
        print(
            f"turn={turn_index + 1} player={player} ball={action.target_ball_id} "
            f"pocket={action.target_pocket_id} success={record['success']} "
            f"switch={record['switch_turn']} scores={record['player_scores']}",
            flush=True,
        )
        state, mask = next_state, next_mask
        if done:
            break

    cleared = env.env.is_cleared()
    winner = shots[-1]["player"] if cleared and shots else None
    plan = {
        "title": "Two-player DQN rollout",
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "epsilon": args.epsilon,
        "cleared": cleared,
        "winner": winner,
        "returns": returns,
        "player_scores": list(env.player_scores),
        "player_turns": list(env.player_turns),
        "break": {
            "label": "scripted break",
            "rollout_title": "Two-player DQN rollout",
            "target_ball_id": "1",
            "target_pocket_id": "-",
            "success": True,
            "foul": False,
            "reason": "break",
            "score": 0.0,
            "player": "-",
            "player_scores": [0, 0],
            "switch_turn": False,
            "speed": float(train_args.get("break_speed", 10.0)),
        },
        "shots": shots,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    multisystem = env.env.pt.MultiSystem()
    for system in systems:
        multisystem.append(system)
    multisystem.save(args.multisystem_output)
    print(
        f"cleared={cleared} winner={winner} turns={len(shots)} scores={env.player_scores} "
        f"returns={returns}"
    )
    print(f"plan={args.output}")
    print(f"multisystem={args.multisystem_output}")


def select_action(
    network: QNetwork,
    state: tuple[float, ...],
    mask: tuple[bool, ...],
    epsilon: float,
    action_random: random.Random,
    device: torch.device,
) -> tuple[int, list[float]]:
    if not any(mask):
        raise RuntimeError("No valid actions are available during rollout.")
    with torch.no_grad():
        values = network(torch.tensor([list(state)], dtype=torch.float32, device=device))[0]
        values = torch.nan_to_num(values, nan=-1.0e9, posinf=1.0e9, neginf=-1.0e9)
        invalid = torch.tensor([not valid for valid in mask], dtype=torch.bool, device=device)
        masked = values.masked_fill(invalid, -1.0e9)
        valid_actions = [index for index, valid in enumerate(mask) if valid]
        action = (
            action_random.choice(valid_actions)
            if action_random.random() < epsilon
            else int(torch.argmax(masked).item())
        )
    return action, [float(value) for value in values.detach().cpu().tolist()]


if __name__ == "__main__":
    main()
