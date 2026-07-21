#!/usr/bin/env python3
"""Train and run a discrete value-iteration high-level PoolTool policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snooker_env.pooltool_discrete_rl import DiscreteValueIterationPolicy, PoolTableDiscretizer  # noqa: E402
from snooker_env.pooltool_high_level import PoolToolSinglePlayerEnv  # noqa: E402
from snooker_env.pooltool_visualization import write_static_rollout_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-type", choices=("example", "nineball"), default="nineball")
    parser.add_argument("--legal-mode", choices=("any", "lowest"), default="any")
    parser.add_argument("--x-bins", type=int, default=8)
    parser.add_argument("--y-bins", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--max-depth", type=int, default=0, help="Reachable-graph expansion depth; 0 expands until the discrete reachable frontier closes.")
    parser.add_argument("--max-states", type=int, default=0, help="Safety cap for sampled discrete states; 0 means no cap.")
    parser.add_argument("--action-prune", type=int, default=0, help="Debug-only heuristic action cap; 0 evaluates every legal (ball, pocket) action.")
    parser.add_argument(
        "--prune-blocked-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop actions without clear direct-pot ghost-ball geometry. This is geometric feasibility pruning, not score/beam pruning.",
    )
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--log-interval", type=int, default=25, help="Print graph/value-iteration progress every N expanded states or VI iterations; 0 disables progress logs.")
    parser.add_argument("--max-shots", type=int, default=9)
    parser.add_argument(
        "--online-refit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refit a local value table if rollout reaches an uncovered discrete state.",
    )
    parser.add_argument("--break-rack", action="store_true")
    parser.add_argument("--break-speed", type=float, default=10.0)
    parser.add_argument("--break-target-ball", default="1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/pooltool/discrete_value_iteration_plan.json"))
    parser.add_argument("--multisystem-output", type=Path, default=Path("outputs/pooltool/discrete_value_iteration_rollout.msgpack"))
    parser.add_argument("--html-output", type=Path, default=Path("outputs/pooltool/discrete_value_iteration_rollout.html"))
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = PoolToolSinglePlayerEnv(game_type=args.game_type, legal_mode=args.legal_mode, random_seed=args.seed)
    env.reset()

    break_result = None
    rollout_systems: list[object] = []
    rollout_records: list[dict[str, object]] = []
    if args.break_rack:
        break_result = env.break_rack(speed=args.break_speed, target_ball_id=args.break_target_ball)
        rollout_systems.append(env.system.copy())
        rollout_records.append(
            {
                "label": "scripted break",
                "target_ball_id": break_result.target_ball_id,
                "target_pocket_id": "-",
                "success": not break_result.cue_scratch,
                "foul": break_result.cue_scratch,
                "score": break_result.ball_spread,
                "reason": "break",
                "remaining_balls": env.legal_ball_ids(env.system),
                "solution": {
                    "speed": break_result.speed,
                    "phi": break_result.phi,
                    "side_spin": 0.0,
                    "top_spin": 0.0,
                    "elevation": 0.0,
                },
                "candidates": (),
            }
        )

    discretizer = PoolTableDiscretizer(x_bins=args.x_bins, y_bins=args.y_bins, cue_ball_id=env.cue_ball_id)
    policy = DiscreteValueIterationPolicy(env, discretizer, gamma=args.gamma)
    result = policy.fit(
        env.system,
        max_depth=None if args.max_depth <= 0 else args.max_depth,
        max_states=None if args.max_states <= 0 else args.max_states,
        action_prune=None if args.action_prune <= 0 else args.action_prune,
        prune_blocked_actions=args.prune_blocked_actions,
        iterations=args.iterations,
        log_interval=args.log_interval,
    )
    print(
        "fit: "
        f"states={len(result.states)} transitions={len(result.transitions)} "
        f"iterations={result.iterations} max_delta={result.max_delta:.6f}",
        flush=True,
    )

    shots: list[dict[str, object]] = []
    for shot_idx in range(args.max_shots):
        if env.is_cleared():
            break
        state = discretizer.encode(env, env.system)
        try:
            action = policy.choose_action(env.system)
        except RuntimeError:
            if not args.online_refit:
                raise
            print(f"{shot_idx}: uncovered_state={state}; refitting local value table", flush=True)
            result = policy.fit(
                env.system,
                max_depth=None if args.max_depth <= 0 else max(1, args.max_depth - shot_idx),
                max_states=None if args.max_states <= 0 else args.max_states,
                action_prune=None if args.action_prune <= 0 else args.action_prune,
                prune_blocked_actions=args.prune_blocked_actions,
                iterations=args.iterations,
                log_interval=args.log_interval,
            )
            action = policy.choose_action(env.system)
        evaluation = env.step(action)
        transition = result.transitions.get((state, (action.target_ball_id, action.target_pocket_id)))
        solution = evaluation.solution
        record = {
            "shot_index": shot_idx,
            "state": state,
            "state_value": result.values.get(state, 0.0),
            "target_ball_id": action.target_ball_id,
            "target_pocket_id": action.target_pocket_id,
            "reward": None if transition is None else transition.reward,
            "success": evaluation.success,
            "foul": evaluation.foul,
            "score": evaluation.score,
            "reason": evaluation.reason,
            "remaining_balls": env.legal_ball_ids(env.system),
            "solution": None
            if solution is None
            else {
                "speed": solution.speed,
                "phi": solution.phi,
                "side_spin": solution.side_spin,
                "top_spin": solution.top_spin,
                "elevation": solution.elevation,
            },
            "candidates": _state_candidates(result, state, args.gamma),
        }
        shots.append(record)
        if evaluation.next_system is not None:
            rollout_systems.append(evaluation.next_system.copy())
            rollout_records.append({"label": f"value-iteration shot {shot_idx}", **record})
        print(
            f"{shot_idx}: state_value={record['state_value']:.3f} "
            f"ball={action.target_ball_id} pocket={action.target_pocket_id} "
            f"reward={record['reward']} success={evaluation.success} "
            f"foul={evaluation.foul} remaining={record['remaining_balls']}",
            flush=True,
        )
        if evaluation.foul or not evaluation.success:
            break

    summary = {
        "game_type": args.game_type,
        "legal_mode": args.legal_mode,
        "seed": args.seed,
        "grid": {"x_bins": args.x_bins, "y_bins": args.y_bins},
        "value_iteration": {
            "gamma": args.gamma,
            "max_depth": None if args.max_depth <= 0 else args.max_depth,
            "max_states": None if args.max_states <= 0 else args.max_states,
            "action_prune": args.action_prune,
            "prune_blocked_actions": args.prune_blocked_actions,
            "full_reachable_expansion": args.max_depth <= 0 and args.max_states <= 0 and args.action_prune <= 0,
            "states": len(result.states),
            "transitions": len(result.transitions),
            "iterations": result.iterations,
            "max_delta": result.max_delta,
            "log_interval": args.log_interval,
        },
        "break": None
        if break_result is None
        else {
            "speed": break_result.speed,
            "phi": break_result.phi,
            "target_ball_id": break_result.target_ball_id,
            "cue_scratch": break_result.cue_scratch,
            "pocketed_ball_ids": break_result.pocketed_ball_ids,
            "ball_spread": break_result.ball_spread,
        },
        "cleared": env.is_cleared(),
        "shots": shots,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if rollout_systems:
        multisystem = env.pt.MultiSystem()
        for system in rollout_systems:
            multisystem.append(system)
        args.multisystem_output.parent.mkdir(parents=True, exist_ok=True)
        multisystem.save(args.multisystem_output)
        write_static_rollout_report(
            pt=env.pt,
            path=args.html_output,
            systems=rollout_systems,
            records=rollout_records,
            title=f"PoolTool discrete value-iteration rollout",
        )
        summary["multisystem_output"] = str(args.multisystem_output)
        summary["html_output"] = str(args.html_output)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if args.show:
            env.pt.show(multisystem, title="Discrete value-iteration rollout")

    print(f"cleared={summary['cleared']} wrote={args.output}", flush=True)
    if rollout_systems:
        print(f"visualization={args.html_output}", flush=True)
        print(f"multisystem={args.multisystem_output}", flush=True)


def _state_candidates(result: object, state: tuple[int, ...], gamma: float) -> tuple[dict[str, object], ...]:
    rows = []
    transitions = getattr(result, "transitions")
    values = getattr(result, "values")
    for (candidate_state, action), transition in transitions.items():
        if candidate_state != state:
            continue
        q_value = transition.reward if transition.terminal else transition.reward + gamma * values.get(transition.next_state, 0.0)
        rows.append(
            {
                "target_ball_id": action[0],
                "target_pocket_id": action[1],
                "score": q_value,
                "reward": transition.reward,
                "success": transition.evaluation.success,
                "foul": transition.evaluation.foul,
                "reason": transition.evaluation.reason,
            }
        )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return tuple(rows[:8])


if __name__ == "__main__":
    main()
