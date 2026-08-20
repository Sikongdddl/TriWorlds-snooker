#!/usr/bin/env python3
"""Measure cue-ball landing coverage for one fixed cue/object-ball layout."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snooker_env.pooltool_high_level import CueLandingGrid, PoolToolSinglePlayerEnv, ShotAction, ShotSolution  # noqa: E402


@dataclass(frozen=True)
class SolverFamily:
    name: str
    shot_path_modes: tuple[str, ...]
    cut_offsets: tuple[float, ...]
    speeds: tuple[float, ...]
    side_spins: tuple[float, ...]
    top_spins: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/pooltool/single_ball_landing_reachability.json"))
    parser.add_argument("--cue-world-xy", type=float, nargs=2, default=(-0.28, -0.62))
    parser.add_argument("--object-world-xy", type=float, nargs=2, default=(0.08, 0.10))
    parser.add_argument("--landing-x-bins", type=int, default=8)
    parser.add_argument("--landing-y-bins", type=int, default=4)
    parser.add_argument("--max-events", type=int, default=80)
    parser.add_argument("--log", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = CueLandingGrid(x_bins=args.landing_x_bins, y_bins=args.landing_y_bins)
    families = _solver_families()
    results: list[dict[str, object]] = []
    start = time.perf_counter()

    for family in families:
        env = PoolToolSinglePlayerEnv(
            game_type="example",
            legal_mode="any",
            landing_grid=grid,
            shot_path_modes=family.shot_path_modes,
            max_events=args.max_events,
            random_seed=42,
        )
        system = _fixed_system(env, args.cue_world_xy, args.object_world_xy)
        result = _measure_family(env, system, family, log=args.log)
        results.append(result)
        elapsed = time.perf_counter() - start
        print(
            f"{family.name}: combos={result['reachable_combo_count']}/"
            f"{result['possible_combo_count']} successful_shots={result['successful_shot_count']} "
            f"trials={result['trial_count']} elapsed={elapsed:.1f}s",
            flush=True,
        )

    summary = {
        "cue_world_xy": [float(v) for v in args.cue_world_xy],
        "object_world_xy": [float(v) for v in args.object_world_xy],
        "landing_grid": {"x_bins": grid.x_bins, "y_bins": grid.y_bins, "cell_count": grid.cell_count},
        "families": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote={args.output}")


def _solver_families() -> tuple[SolverFamily, ...]:
    narrow_offsets = (0.0, -1.0, 1.0, -2.0, 2.0)
    wide_offsets = (0.0, -1.5, 1.5, -3.0, 3.0, -5.0, 5.0, -7.5, 7.5, -10.0, 10.0)
    speeds = (0.8, 1.2, 1.6, 2.0, 2.6, 3.2, 4.0)
    top_back = (0.0, -0.4, 0.4, -0.8, 0.8)
    return (
        SolverFamily(
            name="direct_center_fixed_speed",
            shot_path_modes=("direct",),
            cut_offsets=(0.0,),
            speeds=(1.6,),
            side_spins=(0.0,),
            top_spins=(0.0,),
        ),
        SolverFamily(
            name="direct_narrow_aim_offsets",
            shot_path_modes=("direct",),
            cut_offsets=narrow_offsets,
            speeds=(1.6,),
            side_spins=(0.0,),
            top_spins=(0.0,),
        ),
        SolverFamily(
            name="direct_wide_aim_offsets",
            shot_path_modes=("direct",),
            cut_offsets=wide_offsets,
            speeds=(1.6,),
            side_spins=(0.0,),
            top_spins=(0.0,),
        ),
        SolverFamily(
            name="direct_wide_aim_plus_speed",
            shot_path_modes=("direct",),
            cut_offsets=wide_offsets,
            speeds=speeds,
            side_spins=(0.0,),
            top_spins=(0.0,),
        ),
        SolverFamily(
            name="direct_wide_aim_speed_top_back",
            shot_path_modes=("direct",),
            cut_offsets=wide_offsets,
            speeds=speeds,
            side_spins=(0.0,),
            top_spins=top_back,
        ),
        SolverFamily(
            name="direct_and_bank_wide_aim_speed_top_back",
            shot_path_modes=("direct", "cue_bank", "object_bank"),
            cut_offsets=wide_offsets,
            speeds=speeds,
            side_spins=(0.0,),
            top_spins=top_back,
        ),
    )


def _fixed_system(env: PoolToolSinglePlayerEnv, cue_world_xy: Iterable[float], object_world_xy: Iterable[float]) -> object:
    system = env.reset()
    cue_xy = env.world_to_pool_xy(np.asarray(tuple(cue_world_xy), dtype=np.float64))
    object_xy = env.world_to_pool_xy(np.asarray(tuple(object_world_xy), dtype=np.float64))
    radius = env.table_spec.ball_radius
    for ball_id, xy in ((env.cue_ball_id, cue_xy), ("1", object_xy)):
        ball = system.balls[ball_id]
        ball.state.rvw[:] = 0.0
        ball.state.rvw[0] = [float(xy[0]), float(xy[1]), radius]
        ball.state.s = 0
    env.system = system
    return system


def _measure_family(
    env: PoolToolSinglePlayerEnv,
    system: object,
    family: SolverFamily,
    *,
    log: bool = False,
) -> dict[str, object]:
    reachable: dict[str, set[int]] = {pocket_id: set() for pocket_id in env.pocket_ids(system)}
    examples: dict[str, dict[str, object]] = {}
    combo_keys_by_aim_offset: dict[float, set[str]] = {}
    combo_keys_by_impact_bin: dict[str, set[str]] = {}
    successes_by_aim_offset: dict[float, int] = {}
    successes_by_impact_bin: dict[str, int] = {}
    successful_shots = 0
    trial_count = 0
    aim_candidate_count = 0

    for pocket_id in env.pocket_ids(system):
        action = ShotAction("1", pocket_id)
        aim_candidates = env._aim_candidates(system, action)
        aim_candidate_count += len(aim_candidates)
        for base_phi, path_type in aim_candidates:
            for cut_offset in family.cut_offsets:
                phi = (base_phi + cut_offset) % 360.0
                for speed in family.speeds:
                    for side_spin in family.side_spins:
                        for top_spin in family.top_spins:
                            if side_spin * side_spin + top_spin * top_spin >= 0.98:
                                continue
                            trial_count += 1
                            solution = ShotSolution(
                                speed=speed,
                                phi=phi,
                                side_spin=side_spin,
                                top_spin=top_spin,
                                path_type=path_type,
                            )
                            impact_fraction = _impact_fraction(env, system, phi, path_type)
                            impact_bin = _impact_bin(impact_fraction)
                            evaluation = env.evaluate_solution(system, action, solution)
                            if not evaluation.success or evaluation.foul or evaluation.cue_ball_xy is None:
                                continue
                            successful_shots += 1
                            cell = env.landing_grid.encode_xy(evaluation.next_system, evaluation.cue_ball_xy)
                            combo_key = f"{pocket_id}:{cell}"
                            reachable[pocket_id].add(cell)
                            combo_keys_by_aim_offset.setdefault(cut_offset, set()).add(combo_key)
                            combo_keys_by_impact_bin.setdefault(impact_bin, set()).add(combo_key)
                            successes_by_aim_offset[cut_offset] = successes_by_aim_offset.get(cut_offset, 0) + 1
                            successes_by_impact_bin[impact_bin] = successes_by_impact_bin.get(impact_bin, 0) + 1
                            key = f"{pocket_id}:{cell}"
                            examples.setdefault(
                                key,
                                {
                                    "pocket_id": pocket_id,
                                    "landing_cell": cell,
                                    "speed": speed,
                                    "phi": phi,
                                    "cut_offset": cut_offset,
                                    "side_spin": side_spin,
                                    "top_spin": top_spin,
                                    "path_type": path_type,
                                    "impact_fraction": impact_fraction,
                                    "impact_bin": impact_bin,
                                    "cue_world_xy": [
                                        float(value)
                                        for value in env.pool_to_world_xy(
                                            np.asarray(evaluation.cue_ball_xy, dtype=np.float64)
                                        )
                                    ],
                                },
                            )
        if log:
            print(f"  {family.name}:{pocket_id} cells={sorted(reachable[pocket_id])}", flush=True)

    combo_count = sum(len(cells) for cells in reachable.values())
    return {
        "name": family.name,
        "shot_path_modes": list(family.shot_path_modes),
        "cut_offsets": list(family.cut_offsets),
        "speeds": list(family.speeds),
        "side_spins": list(family.side_spins),
        "top_spins": list(family.top_spins),
        "trial_count": trial_count,
        "aim_candidate_count": aim_candidate_count,
        "successful_shot_count": successful_shots,
        "possible_combo_count": len(reachable) * env.landing_grid.cell_count,
        "reachable_combo_count": combo_count,
        "reachable_by_pocket": {pocket_id: sorted(cells) for pocket_id, cells in reachable.items()},
        "aim_offset_stats": [
            {
                "cut_offset": offset,
                "successful_shots": successes_by_aim_offset.get(offset, 0),
                "reachable_combo_count": len(combo_keys_by_aim_offset.get(offset, set())),
                "reachable_combos": sorted(combo_keys_by_aim_offset.get(offset, set())),
            }
            for offset in family.cut_offsets
        ],
        "impact_fraction_stats": [
            {
                "impact_bin": impact_bin,
                "successful_shots": successes_by_impact_bin.get(impact_bin, 0),
                "reachable_combo_count": len(combo_keys_by_impact_bin.get(impact_bin, set())),
                "reachable_combos": sorted(combo_keys_by_impact_bin.get(impact_bin, set())),
            }
            for impact_bin in sorted(combo_keys_by_impact_bin, key=_impact_bin_sort_key)
        ],
        "examples": sorted(examples.values(), key=lambda item: (str(item["pocket_id"]), int(item["landing_cell"]))),
    }


def _impact_fraction(env: PoolToolSinglePlayerEnv, system: object, phi: float, path_type: str) -> float:
    cue_xy = env._ball_xy(system, env.cue_ball_id)
    object_xy = env._ball_xy(system, "1")
    radians = math.radians(phi)
    direction = np.asarray([math.cos(radians), math.sin(radians)], dtype=np.float64)
    start = cue_xy
    if path_type.startswith("cue_bank:"):
        bank = path_type.split(":", 1)[1]
        reflected = direction.copy()
        if bank in {"left", "right"}:
            rail_x = 0.0 if bank == "left" else env.table_spec.play_width_x
            if abs(float(direction[0])) > 1e-12:
                t = (rail_x - float(cue_xy[0])) / float(direction[0])
                if t > 0.0:
                    start = cue_xy + t * direction
            reflected[0] *= -1.0
        elif bank in {"bottom", "top"}:
            rail_y = 0.0 if bank == "bottom" else env.table_spec.play_length_y
            if abs(float(direction[1])) > 1e-12:
                t = (rail_y - float(cue_xy[1])) / float(direction[1])
                if t > 0.0:
                    start = cue_xy + t * direction
            reflected[1] *= -1.0
        direction = reflected / max(float(np.linalg.norm(reflected)), 1e-12)
    object_from_cue = object_xy - start
    signed_distance = float(direction[0] * object_from_cue[1] - direction[1] * object_from_cue[0])
    return signed_distance / (2.0 * env.table_spec.ball_radius)


def _impact_bin(value: float) -> str:
    clipped = max(-1.25, min(1.25, value))
    lo = math.floor(clipped / 0.25) * 0.25
    hi = lo + 0.25
    return f"{lo:+.2f}:{hi:+.2f}"


def _impact_bin_sort_key(label: str) -> float:
    return float(label.split(":", 1)[0])


if __name__ == "__main__":
    main()
