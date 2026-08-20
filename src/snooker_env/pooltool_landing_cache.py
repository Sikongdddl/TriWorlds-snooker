"""Persistent cache for PoolTool cue-ball landing masks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable

from snooker_env.pooltool_high_level import CueLandingGrid, ShotAction, ShotSolution


@dataclass(frozen=True)
class LandingMaskCacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0


class LandingMaskCache:
    """SQLite-backed reachable landing-cell cache.

    The cache is keyed by a grid-discretized PoolTool state and the shot/mask solver
    configuration. It is deliberately local and disposable; changing grid or
    sampling parameters naturally produces different keys.
    """

    def __init__(
        self,
        path: Path,
        *,
        landing_grid: CueLandingGrid,
        speed_grid: tuple[float, ...],
        cut_offsets: tuple[float, ...],
        side_spin_grid: tuple[float, ...],
        top_spin_grid: tuple[float, ...],
        shot_path_modes: tuple[str, ...] = ("direct",),
    ) -> None:
        self.path = path
        self.landing_grid = landing_grid
        self.speed_grid = speed_grid
        self.cut_offsets = cut_offsets
        self.side_spin_grid = side_spin_grid
        self.top_spin_grid = top_spin_grid
        self.shot_path_modes = shot_path_modes
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS landing_masks (
                key TEXT PRIMARY KEY,
                state_key TEXT NOT NULL,
                target_ball_id TEXT NOT NULL,
                target_pocket_id TEXT NOT NULL,
                cells TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS landing_solutions (
                key TEXT PRIMARY KEY,
                state_key TEXT NOT NULL,
                target_ball_id TEXT NOT NULL,
                target_pocket_id TEXT NOT NULL,
                solutions TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    @property
    def stats(self) -> LandingMaskCacheStats:
        return LandingMaskCacheStats(hits=self._hits, misses=self._misses, writes=self._writes)

    def close(self) -> None:
        self._conn.close()

    def get_or_compute(
        self,
        system: Any,
        action: ShotAction,
        compute: Callable[[], frozenset[int]],
    ) -> frozenset[int]:
        state_key = self.state_key(system)
        key = self.cache_key(state_key, action)
        row = self._conn.execute("SELECT cells FROM landing_masks WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self._hits += 1
            return self._decode_cells(str(row[0]))

        self._misses += 1
        cells = compute()
        self.put(state_key, action, cells)
        return cells

    def put(self, state_key: str, action: ShotAction, cells: frozenset[int]) -> None:
        key = self.cache_key(state_key, action)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO landing_masks
                (key, state_key, target_ball_id, target_pocket_id, cells, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                state_key,
                action.target_ball_id,
                action.target_pocket_id,
                self._encode_cells(cells),
                time.time(),
            ),
        )
        self._conn.commit()
        self._writes += 1

    def get_or_compute_solutions(
        self,
        system: Any,
        action: ShotAction,
        compute: Callable[[], dict[int, ShotSolution]],
    ) -> dict[int, ShotSolution]:
        state_key = self.state_key(system)
        key = self.cache_key(state_key, action)
        row = self._conn.execute("SELECT solutions FROM landing_solutions WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self._hits += 1
            return self._decode_solutions(str(row[0]))

        self._misses += 1
        solutions = compute()
        self.put_solutions(state_key, action, solutions)
        return solutions

    def put_solutions(self, state_key: str, action: ShotAction, solutions: dict[int, ShotSolution]) -> None:
        key = self.cache_key(state_key, action)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO landing_solutions
                (key, state_key, target_ball_id, target_pocket_id, solutions, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                state_key,
                action.target_ball_id,
                action.target_pocket_id,
                self._encode_solutions(solutions),
                time.time(),
            ),
        )
        self._conn.commit()
        self._writes += 1

    def state_key(self, system: Any) -> str:
        pieces = [f"table:{float(system.table.w):.6f}:{float(system.table.l):.6f}"]
        for ball_id, ball in sorted(system.balls.items()):
            state = int(ball.state.s)
            if state == 4:
                pieces.append(f"{ball_id}:pocketed")
                continue
            cell = self.landing_grid.encode_xy(system, ball.state.rvw[0, :2])
            pieces.append(f"{ball_id}:{state}:cell:{cell}")
        return "|".join(pieces)

    def cache_key(self, state_key: str, action: ShotAction) -> str:
        payload = "|".join(
            (
                state_key,
                f"ball:{action.target_ball_id}",
                f"pocket:{action.target_pocket_id}",
                f"grid:{self.landing_grid.x_bins}x{self.landing_grid.y_bins}",
                f"speeds:{self._float_tuple_key(self.speed_grid)}",
                f"offsets:{self._float_tuple_key(self.cut_offsets)}",
                f"side:{self._float_tuple_key(self.side_spin_grid)}",
                f"top:{self._float_tuple_key(self.top_spin_grid)}",
                f"paths:{','.join(self.shot_path_modes)}",
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def count_rows(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM landing_masks").fetchone()
        return int(row[0])

    def count_solution_rows(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM landing_solutions").fetchone()
        return int(row[0])

    @staticmethod
    def _encode_cells(cells: frozenset[int]) -> str:
        return ",".join(str(cell) for cell in sorted(cells))

    @staticmethod
    def _decode_cells(text: str) -> frozenset[int]:
        if not text:
            return frozenset()
        return frozenset(int(item) for item in text.split(","))

    @staticmethod
    def _encode_solutions(solutions: dict[int, ShotSolution]) -> str:
        payload = {
            str(cell): {
                "speed": solution.speed,
                "phi": solution.phi,
                "side_spin": solution.side_spin,
                "top_spin": solution.top_spin,
                "elevation": solution.elevation,
                "path_type": solution.path_type,
            }
            for cell, solution in sorted(solutions.items())
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _decode_solutions(text: str) -> dict[int, ShotSolution]:
        if not text:
            return {}
        payload = json.loads(text)
        return {
            int(cell): ShotSolution(
                speed=float(values["speed"]),
                phi=float(values["phi"]),
                side_spin=float(values.get("side_spin", 0.0)),
                top_spin=float(values.get("top_spin", 0.0)),
                elevation=float(values.get("elevation", 0.0)),
                path_type=str(values.get("path_type", "direct")),
            )
            for cell, values in payload.items()
        }

    @staticmethod
    def _float_tuple_key(values: tuple[float, ...]) -> str:
        return ",".join(f"{value:.6g}" for value in values)
