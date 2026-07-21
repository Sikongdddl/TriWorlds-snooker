#!/usr/bin/env python3
"""Run the PoolTool-backed heuristic clearance planner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snooker_env.pooltool_high_level import HeuristicClearancePlanner, PoolToolSinglePlayerEnv  # noqa: E402
from snooker_env.pooltool_visualization import write_static_rollout_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-type", choices=("example", "nineball"), default="example")
    parser.add_argument("--legal-mode", choices=("any", "lowest"), default="any")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-shots", type=int, default=9)
    parser.add_argument("--break-rack", action="store_true", help="Apply a strong scripted break before planning.")
    parser.add_argument("--break-speed", type=float, default=10.0)
    parser.add_argument("--break-target-ball", default="1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/pooltool/clearance_plan.json"))
    parser.add_argument("--multisystem-output", type=Path, default=Path("outputs/pooltool/clearance_rollout.msgpack"))
    parser.add_argument("--html-output", type=Path, default=Path("outputs/pooltool/clearance_rollout.html"))
    parser.add_argument("--trajectory-dt", type=float, default=0.02, help="Trajectory sampling interval for the HTML report.")
    parser.add_argument("--show", action="store_true", help="Open PoolTool GUI with the full rollout after planning.")
    return parser.parse_args()


def _candidate_record(candidate: object) -> dict[str, object]:
    solution = getattr(candidate, "solution", None)
    action = getattr(candidate, "action")
    return {
        "target_ball_id": action.target_ball_id,
        "target_pocket_id": action.target_pocket_id,
        "score": float(getattr(candidate, "score")),
        "success": bool(getattr(candidate, "success")),
        "foul": bool(getattr(candidate, "foul")),
        "reason": str(getattr(candidate, "reason")),
        "solution": None
        if solution is None
        else {
            "speed": float(solution.speed),
            "phi": float(solution.phi),
            "side_spin": float(solution.side_spin),
            "top_spin": float(solution.top_spin),
            "elevation": float(solution.elevation),
        },
    }


def main() -> None:
    args = parse_args()
    env = PoolToolSinglePlayerEnv(game_type=args.game_type, legal_mode=args.legal_mode, random_seed=args.seed)
    planner = HeuristicClearancePlanner(env, depth=args.depth, beam_width=args.beam_width)
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
                "pocketed_ball_ids": break_result.pocketed_ball_ids,
                "candidates": (),
            }
        )
        print(
            "break: "
            f"target={break_result.target_ball_id} V0={break_result.speed:.2f} "
            f"phi={break_result.phi:.2f} cue_scratch={break_result.cue_scratch} "
            f"pocketed={break_result.pocketed_ball_ids} spread={break_result.ball_spread:.3f}"
        )

    shots: list[dict[str, object]] = []
    for shot_idx in range(args.max_shots):
        if env.is_cleared():
            break
        plan = planner.choose_action()
        result = env.step(plan.action)
        solution = result.solution
        record = {
            "shot_index": shot_idx,
            "target_ball_id": result.action.target_ball_id,
            "target_pocket_id": result.action.target_pocket_id,
            "success": result.success,
            "foul": result.foul,
            "score": result.score,
            "reason": result.reason,
            "remaining_balls": env.legal_ball_ids(env.system),
            "candidates": tuple(_candidate_record(candidate) for candidate in plan.candidates),
            "solution": None
            if solution is None
            else {
                "speed": solution.speed,
                "phi": solution.phi,
                "side_spin": solution.side_spin,
                "top_spin": solution.top_spin,
                "elevation": solution.elevation,
            },
        }
        shots.append(record)
        if result.next_system is not None:
            rollout_systems.append(result.next_system.copy())
            rollout_records.append({"label": f"planner shot {shot_idx}", **record})
        print(
            f"{shot_idx}: ball={result.action.target_ball_id} "
            f"pocket={result.action.target_pocket_id} "
            f"success={result.success} foul={result.foul} "
            f"score={result.score:.3f} remaining={record['remaining_balls']}"
        )
        if result.foul or not result.success:
            break

    summary = {
        "game_type": args.game_type,
        "legal_mode": args.legal_mode,
        "seed": args.seed,
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
            title=f"PoolTool {args.game_type} clearance rollout",
            trajectory_dt=args.trajectory_dt,
        )
        summary["multisystem_output"] = str(args.multisystem_output)
        summary["html_output"] = str(args.html_output)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"visualization={args.html_output}")
        print(f"multisystem={args.multisystem_output}")
        if args.show:
            env.pt.show(multisystem, title="PoolTool planner rollout: n/p switches shots, Enter toggles parallel view")

    print(f"cleared={summary['cleared']} wrote={args.output}")


if __name__ == "__main__":
    main()
