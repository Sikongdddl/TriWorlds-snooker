#!/usr/bin/env python3
"""Smoke-test PoolTool-backed high-level shot selection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snooker_env.pooltool_high_level import HeuristicClearancePlanner, PoolToolSinglePlayerEnv  # noqa: E402


def main() -> None:
    env = PoolToolSinglePlayerEnv(
        game_type="example",
        legal_mode="any",
        speed_grid=(0.8, 1.2, 1.6, 2.0, 2.6, 3.2),
        cut_offsets=(0.0, -1.0, 1.0),
        max_events=80,
    )
    planner = HeuristicClearancePlanner(env, depth=2, beam_width=4)
    system = env.reset()

    print(f"legal balls: {env.legal_ball_ids(system)}")
    print(f"pockets: {env.pocket_ids(system)}")
    print(f"candidate actions: {len(env.enumerate_actions(system))}")

    plan = planner.choose_action(system)
    print(
        "selected: "
        f"ball={plan.action.target_ball_id} pocket={plan.action.target_pocket_id} "
        f"score={plan.evaluation.score:.3f} reason={plan.evaluation.reason}"
    )
    if plan.evaluation.solution is not None:
        solution = plan.evaluation.solution
        print(
            "solution: "
            f"V0={solution.speed:.3f} phi={solution.phi:.3f} "
            f"a={solution.side_spin:.3f} b={solution.top_spin:.3f} theta={solution.elevation:.3f}"
        )
    print("top candidates:")
    for idx, candidate in enumerate(plan.candidates[:5]):
        solution_text = "-"
        if candidate.solution is not None:
            solution_text = f"V0={candidate.solution.speed:.2f}, phi={candidate.solution.phi:.2f}"
        print(
            f"  {idx}: ball={candidate.action.target_ball_id} "
            f"pocket={candidate.action.target_pocket_id} "
            f"success={candidate.success} foul={candidate.foul} "
            f"score={candidate.score:.3f} reason={candidate.reason} {solution_text}"
        )

    result = env.step(plan.action)
    print(
        "executed: "
        f"success={result.success} foul={result.foul} reason={result.reason} "
        f"remaining={env.legal_ball_ids(env.system)}"
    )
    if result.foul:
        raise RuntimeError("Selected PoolTool high-level action resulted in a foul.")
    if not result.success:
        raise RuntimeError("Selected PoolTool high-level action did not pot the target ball.")


if __name__ == "__main__":
    main()
