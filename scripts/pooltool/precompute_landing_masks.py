#!/usr/bin/env python3
"""Precompute PoolTool reachable cue-ball landing masks after randomized breaks."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snooker_env.pooltool_high_level import CueLandingGrid, PoolToolSinglePlayerEnv, ShotAction  # noqa: E402
from snooker_env.pooltool_landing_cache import LandingMaskCache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-type", choices=("nineball", "example"), default="nineball")
    parser.add_argument("--legal-mode", choices=("any", "lowest"), default="any")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--break-target-ball", default="1")
    parser.add_argument("--break-speed", type=float, default=10.0)
    parser.add_argument("--break-speed-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=(8.0, 12.0))
    parser.add_argument("--break-phi-jitter-degrees", type=float, default=2.0)
    parser.add_argument("--landing-x-bins", type=int, default=8)
    parser.add_argument("--landing-y-bins", type=int, default=4)
    parser.add_argument("--cache", type=Path, default=Path("outputs/pooltool/landing_mask_cache.sqlite"))
    parser.add_argument("--speed-grid", type=float, nargs="+", default=(0.8, 1.2, 1.6, 2.0, 2.6, 3.2))
    parser.add_argument("--cut-offsets", type=float, nargs="+", default=(0.0, -0.75, 0.75, -1.5, 1.5))
    parser.add_argument("--side-spin-grid", type=float, nargs="+", default=(0.0, -0.3, 0.3, -0.6, 0.6))
    parser.add_argument("--top-spin-grid", type=float, nargs="+", default=(0.0, -0.4, 0.4, -0.8, 0.8))
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive.")
    if args.break_speed_range[0] > args.break_speed_range[1]:
        raise SystemExit("--break-speed-range MIN must be <= MAX.")

    rng = random.Random(args.seed)
    landing_grid = CueLandingGrid(x_bins=args.landing_x_bins, y_bins=args.landing_y_bins)
    env = PoolToolSinglePlayerEnv(
        game_type=args.game_type,
        legal_mode=args.legal_mode,
        landing_grid=landing_grid,
        random_seed=args.seed,
    )
    cache = LandingMaskCache(
        args.cache,
        landing_grid=landing_grid,
        speed_grid=tuple(float(value) for value in args.speed_grid),
        cut_offsets=tuple(float(value) for value in args.cut_offsets),
        side_spin_grid=tuple(float(value) for value in args.side_spin_grid),
        top_spin_grid=tuple(float(value) for value in args.top_spin_grid),
        shot_path_modes=env.shot_path_modes,
    )

    total_pairs = 0
    total_reachable_cells = 0
    try:
        for sample_idx in range(1, args.samples + 1):
            system = env.reset()
            if args.game_type != "example":
                speed = rng.uniform(float(args.break_speed_range[0]), float(args.break_speed_range[1]))
                base_phi = float(env.pt.aim.at_ball(system, args.break_target_ball))
                phi = base_phi + rng.uniform(-args.break_phi_jitter_degrees, args.break_phi_jitter_degrees)
                env.break_rack(speed=speed, target_ball_id=args.break_target_ball, phi=phi)
                system = env.system
            else:
                speed = args.break_speed
                phi = 0.0

            sample_pairs = 0
            sample_reachable_cells = 0
            for ball_id in env.legal_ball_ids(system):
                for pocket_id in env.pocket_ids(system):
                    action = ShotAction(ball_id, pocket_id)
                    if not env.is_geometrically_pottable(system, action):
                        continue
                    cells = cache.get_or_compute(
                        system,
                        action,
                        lambda action=action, system=system: env.reachable_landing_cells(
                            system,
                            action,
                            speed_grid=tuple(float(value) for value in args.speed_grid),
                            cut_offsets=tuple(float(value) for value in args.cut_offsets),
                            side_spin_grid=tuple(float(value) for value in args.side_spin_grid),
                            top_spin_grid=tuple(float(value) for value in args.top_spin_grid),
                        ),
                    )
                    sample_pairs += 1
                    sample_reachable_cells += len(cells)

            total_pairs += sample_pairs
            total_reachable_cells += sample_reachable_cells
            if args.log_interval > 0 and (sample_idx == 1 or sample_idx % args.log_interval == 0):
                stats = cache.stats
                print(
                    "precompute: "
                    f"sample={sample_idx}/{args.samples} "
                    f"break_speed={speed:.3f} "
                    f"break_phi={phi:.3f} "
                    f"pairs={sample_pairs} "
                    f"reachable_cells={sample_reachable_cells} "
                    f"cache_rows={cache.count_rows()} "
                    f"hits={stats.hits} "
                    f"misses={stats.misses} "
                    f"writes={stats.writes}",
                    flush=True,
                )
    finally:
        stats = cache.stats
        rows = cache.count_rows()
        cache.close()

    print(
        "done: "
        f"samples={args.samples} "
        f"total_pairs={total_pairs} "
        f"total_reachable_cells={total_reachable_cells} "
        f"cache_rows={rows} "
        f"hits={stats.hits} "
        f"misses={stats.misses} "
        f"writes={stats.writes} "
        f"cache={args.cache}",
        flush=True,
    )


if __name__ == "__main__":
    main()
