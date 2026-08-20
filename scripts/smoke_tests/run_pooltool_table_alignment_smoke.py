#!/usr/bin/env python3
"""Verify PoolTool high-level table geometry matches the shared project frame."""

from __future__ import annotations

import math

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.pooltool_high_level import PoolToolSinglePlayerEnv  # noqa: E402


def _assert_close(name: str, actual: float, expected: float, tol: float = 1e-6) -> None:
    if not math.isclose(actual, expected, abs_tol=tol):
        raise RuntimeError(f"{name}: expected {expected:.9f}, got {actual:.9f}")


def main() -> None:
    env = PoolToolSinglePlayerEnv(game_type="nineball", legal_mode="lowest", random_seed=42)
    system = env.reset()
    spec = env.table_spec

    print(f"pooltool_table_w={float(system.table.w):.6f} m")
    print(f"pooltool_table_l={float(system.table.l):.6f} m")
    _assert_close("table width", float(system.table.w), 1.27)
    _assert_close("table length", float(system.table.l), 2.54)

    pockets = env.pocket_world_centers(system)
    for pocket_id in sorted(pockets):
        x, y = pockets[pocket_id]
        print(f"pocket_{pocket_id}_world=({x:.6f}, {y:.6f}) m")

    expected_pockets = {
        "lb": (-0.675, -1.310),
        "lt": (-0.675, 1.310),
        "rb": (0.675, -1.310),
        "rt": (0.675, 1.310),
        "lc": (-0.717426, 0.0),
        "rc": (0.717426, 0.0),
    }
    for pocket_id, expected in expected_pockets.items():
        actual = pockets[pocket_id]
        if not np.allclose(actual, expected, atol=1e-6):
            raise RuntimeError(f"pocket {pocket_id}: expected {expected}, got {actual}")

    cue_ball = system.balls[env.cue_ball_id]
    print(f"ball_radius={float(cue_ball.params.R):.6f} m")
    print(f"ball_mass={float(cue_ball.params.m):.6f} kg")
    _assert_close("ball radius", float(cue_ball.params.R), 0.0285)
    _assert_close("ball mass", float(cue_ball.params.m), 0.165)

    cue_world = env.ball_world_xyz(env.cue_ball_id, system)
    print(f"cue_ball_world=({cue_world[0]:.6f}, {cue_world[1]:.6f}, {cue_world[2]:.6f}) m")
    _assert_close("cue ball world z", cue_world[2], spec.ball_center_z)

    pool_origin_world = env.pool_to_world_xy(np.asarray([0.0, 0.0], dtype=np.float64))
    print(f"pool_origin_world=({pool_origin_world[0]:.6f}, {pool_origin_world[1]:.6f}) m")
    if not np.allclose(pool_origin_world, (-0.635, -1.270), atol=1e-6):
        raise RuntimeError(f"PoolTool origin maps incorrectly: {pool_origin_world}")


if __name__ == "__main__":
    main()
