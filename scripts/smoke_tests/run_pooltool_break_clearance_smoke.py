#!/usr/bin/env python3
"""Smoke-test scripted break followed by heuristic PoolTool clearance."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snooker_env.pooltool_high_level import HeuristicClearancePlanner, PoolToolSinglePlayerEnv  # noqa: E402


def main() -> None:
    env = PoolToolSinglePlayerEnv(game_type="nineball", legal_mode="any", random_seed=42)
    env.reset()
    break_result = env.break_rack(speed=10.0, target_ball_id="1")
    print(
        "break: "
        f"cue_scratch={break_result.cue_scratch} "
        f"pocketed={break_result.pocketed_ball_ids} "
        f"spread={break_result.ball_spread:.3f}"
    )
    if break_result.cue_scratch:
        raise RuntimeError("Scripted break scratched the cue ball.")
    if break_result.ball_spread < 0.2:
        raise RuntimeError("Scripted break did not scatter the rack enough.")

    planner = HeuristicClearancePlanner(env, depth=2, beam_width=4)
    for shot_idx in range(9):
        if env.is_cleared():
            break
        result = env.step(planner.choose_action().action)
        print(
            f"{shot_idx}: ball={result.action.target_ball_id} "
            f"pocket={result.action.target_pocket_id} "
            f"success={result.success} foul={result.foul} "
            f"remaining={env.legal_ball_ids(env.system)}"
        )
        if result.foul or not result.success:
            raise RuntimeError("Break clearance planner failed before clearing the table.")

    if not env.is_cleared():
        raise RuntimeError(f"Break clearance planner did not clear the table: remaining={env.legal_ball_ids(env.system)}")


if __name__ == "__main__":
    main()
