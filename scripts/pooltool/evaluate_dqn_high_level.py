#!/usr/bin/env python3
"""Evaluate a saved PoolTool DQN high-level policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snooker_env.pooltool_dqn import PoolToolDQNEnv, QNetwork  # noqa: E402
from snooker_env.pooltool_visualization import write_static_rollout_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/pooltool/dqn_high_level.pt"))
    parser.add_argument("--max-shots", type=int, default=9)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("outputs/pooltool/dqn_eval.json"))
    parser.add_argument("--multisystem-output", type=Path, default=Path("outputs/pooltool/dqn_eval.msgpack"))
    parser.add_argument("--html-output", type=Path, default=Path("outputs/pooltool/dqn_eval.html"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    train_args = checkpoint.get("args", {})
    checkpoint_action_dim = int(checkpoint["action_dim"])
    ordered_landing_actions = bool(train_args.get("ordered_landing_actions", checkpoint_action_dim == 192))
    ordered_pocket_actions = bool(train_args.get("ordered_pocket_actions", checkpoint_action_dim == 6)) or ordered_landing_actions
    include_cue_landing = not bool(train_args.get("no_cue_landing", checkpoint_action_dim == 54))
    if ordered_landing_actions:
        include_cue_landing = True
    elif ordered_pocket_actions:
        include_cue_landing = False
    env = PoolToolDQNEnv(
        random_seed=args.seed,
        break_speed=float(train_args.get("break_speed", 10.0)),
        randomize_break=bool(train_args.get("randomize_break", False)),
        break_speed_range=None
        if train_args.get("break_speed_range") is None
        else tuple(float(value) for value in train_args["break_speed_range"]),
        break_phi_jitter_degrees=float(train_args.get("break_phi_jitter_degrees", 0.0)),
        ordered_pocket_actions=ordered_pocket_actions,
        ordered_landing_actions=ordered_landing_actions,
        reset_max_attempts=int(train_args.get("reset_max_attempts", 100)),
        include_cue_landing=include_cue_landing,
        landing_x_bins=int(train_args.get("landing_x_bins", 8)),
        landing_y_bins=int(train_args.get("landing_y_bins", 4)),
        fast_landing_solver=not bool(train_args.get("no_fast_landing_solver", False)),
        fast_landing_max_trials=int(train_args.get("fast_landing_max_trials", 160)),
        prune_unreachable_landing_actions=not bool(train_args.get("no_prune_unreachable_landing_actions", False)),
        mask_unreachable_landing_actions=bool(train_args.get("mask_unreachable_landing_actions", False)),
        landing_mask_cache_path=None
        if bool(train_args.get("no_landing_mask_cache", False))
        else Path(str(train_args.get("landing_mask_cache", "outputs/pooltool/landing_mask_cache.sqlite"))),
        next_pocket_reward=float(train_args.get("next_pocket_reward", 1.5)),
        no_next_shot_penalty=float(train_args.get("no_next_shot_penalty", -3.0)),
        max_position_reward=float(train_args.get("max_position_reward", 6.0)),
    )
    if env.action_dim != checkpoint_action_dim:
        raise SystemExit(
            f"Checkpoint action_dim={checkpoint_action_dim} is incompatible with env action_dim={env.action_dim}."
        )
    hidden_dim = int(train_args.get("hidden_dim", 256))
    q_net = QNetwork(int(checkpoint["state_dim"]), checkpoint_action_dim, hidden_dim=hidden_dim)
    q_net.load_state_dict(checkpoint["model_state_dict"])
    q_net.to(torch.device(args.device))
    q_net.eval()

    state, mask = env.reset()
    rollout_systems = [env.env.system.copy()]
    rollout_records: list[dict[str, object]] = [
        {
            "label": "scripted break",
            "target_ball_id": "1",
            "target_pocket_id": "-",
            "success": True,
            "foul": False,
            "score": 0.0,
            "reason": "break",
            "remaining_balls": env.env.legal_ball_ids(env.env.system),
            "solution": None,
            "candidates": (),
        }
    ]
    shots: list[dict[str, Any]] = []
    total_return = 0.0

    for shot_idx in range(args.max_shots):
        action_index = _greedy_action(q_net, state, mask, torch.device(args.device))
        next_state, reward, done, next_mask, info = env.step(action_index)
        total_return += reward
        action = info["action"]
        solution = info.get("solution")
        record = {
            "shot_index": shot_idx,
            "action_index": action_index,
            "target_ball_id": action.target_ball_id,
            "target_pocket_id": action.target_pocket_id,
            "cue_landing_cell": action.cue_landing_cell,
            "reward": reward,
            "success": bool(info["success"]),
            "foul": bool(info["foul"]),
            "reason": str(info["reason"]),
            "position_reward": float(info.get("position_reward", 0.0)),
            "next_valid_pockets": info.get("next_valid_pockets"),
            "pot_success": bool(info.get("pot_success")),
            "landing_success": info.get("landing_success"),
            "actual_cue_landing_cell": info.get("cue_landing_cell"),
            "cue_landing_distance": info.get("cue_landing_distance"),
            "remaining_balls": tuple(info.get("remaining_balls", ())),
            "solution": None
            if solution is None
            else {
                "speed": solution.speed,
                "phi": solution.phi,
                "side_spin": solution.side_spin,
                "top_spin": solution.top_spin,
                "elevation": solution.elevation,
                "path_type": solution.path_type,
            },
            "candidates": (),
        }
        shots.append(record)
        rollout_systems.append(env.env.system.copy())
        rollout_records.append({"label": f"dqn shot {shot_idx}", **record})
        print(
            f"{shot_idx}: action={action_index} ball={action.target_ball_id} pocket={action.target_pocket_id} "
            f"landing={action.cue_landing_cell} "
            f"reward={reward:.3f} success={info['success']} foul={info['foul']} remaining={record['remaining_balls']}",
            flush=True,
        )
        state, mask = next_state, next_mask
        if done:
            break

    cleared = env.env.is_cleared()
    summary = {
        "checkpoint": str(args.checkpoint),
        "return": total_return,
        "cleared": cleared,
        "shots": shots,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    multisystem = env.env.pt.MultiSystem()
    for system in rollout_systems:
        multisystem.append(system)
    multisystem.save(args.multisystem_output)
    write_static_rollout_report(
        pt=env.env.pt,
        path=args.html_output,
        systems=rollout_systems,
        records=rollout_records,
        title="PoolTool DQN high-level rollout",
    )
    print(f"cleared={cleared} return={total_return:.3f} wrote={args.output}")
    print(f"multisystem={args.multisystem_output}")
    print(f"html={args.html_output}")


def _greedy_action(
    q_net: QNetwork,
    state: tuple[float, ...],
    mask: tuple[bool, ...],
    device: torch.device,
) -> int:
    with torch.no_grad():
        q_values = q_net(torch.tensor([list(state)], dtype=torch.float32, device=device))[0]
        invalid = torch.tensor([not ok for ok in mask], dtype=torch.bool, device=device)
        q_values = q_values.masked_fill(invalid, -1.0e9)
        return int(torch.argmax(q_values).item())


if __name__ == "__main__":
    main()
