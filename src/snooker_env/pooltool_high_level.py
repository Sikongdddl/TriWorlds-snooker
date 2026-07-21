"""PoolTool-backed high-level shot selection environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from snooker_env.pooltool_runtime import require_pooltool


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ShotAction:
    """High-level action: choose a target ball, target pocket, and optional cue landing cell."""

    target_ball_id: str
    target_pocket_id: str
    cue_landing_cell: int | None = None


@dataclass(frozen=True)
class CueLandingGrid:
    """Regular table-plane grid used for high-level cue-ball landing commands."""

    x_bins: int = 8
    y_bins: int = 4

    @property
    def cell_count(self) -> int:
        return self.x_bins * self.y_bins

    def cell_center(self, system: Any, cell: int) -> FloatArray:
        if cell < 0 or cell >= self.cell_count:
            raise ValueError(f"Invalid cue landing cell {cell}; expected [0, {self.cell_count}).")
        x_idx = cell % self.x_bins
        y_idx = cell // self.x_bins
        return np.asarray(
            [
                (x_idx + 0.5) * float(system.table.w) / self.x_bins,
                (y_idx + 0.5) * float(system.table.l) / self.y_bins,
            ],
            dtype=np.float64,
        )

    def encode_xy(self, system: Any, xy: FloatArray) -> int:
        x = int(np.clip(np.floor((xy[0] / float(system.table.w)) * self.x_bins), 0, self.x_bins - 1))
        y = int(np.clip(np.floor((xy[1] / float(system.table.l)) * self.y_bins), 0, self.y_bins - 1))
        return y * self.x_bins + x


@dataclass(frozen=True)
class ShotSolution:
    """Cue parameters found by the internal PoolTool shot solver."""

    speed: float
    phi: float
    side_spin: float = 0.0
    top_spin: float = 0.0
    elevation: float = 0.0


@dataclass(frozen=True)
class ShotEvaluation:
    """Result of simulating one high-level shot action."""

    action: ShotAction
    score: float
    success: bool
    foul: bool
    reason: str
    solution: ShotSolution | None = None
    next_system: Any | None = field(default=None, compare=False, repr=False)
    cue_ball_xy: FloatArray | None = None
    cue_landing_cell: int | None = None
    cue_landing_distance: float | None = None
    pot_success: bool | None = None
    landing_success: bool | None = None
    events: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShotPlan:
    """Planner output with the selected shot and evaluated alternatives."""

    action: ShotAction
    evaluation: ShotEvaluation
    candidates: tuple[ShotEvaluation, ...]


@dataclass(frozen=True)
class BreakResult:
    """Summary of a scripted rack break."""

    speed: float
    phi: float
    target_ball_id: str
    cue_scratch: bool
    pocketed_ball_ids: tuple[str, ...]
    ball_spread: float
    events: tuple[str, ...]


class PoolToolSinglePlayerEnv:
    """Small PoolTool environment for single-player clearance planning.

    The external high-level action is ``(target_ball_id, target_pocket_id)`` or
    ``(target_ball_id, target_pocket_id, cue_landing_cell)``. The environment
    uses a minimal internal shot solver to find cue parameters that pot the
    requested ball and, when requested, leave the cue ball in the target landing
    cell.
    """

    def __init__(
        self,
        *,
        cue_ball_id: str = "cue",
        game_type: str = "nineball",
        legal_mode: str = "any",
        speed_grid: tuple[float, ...] = (0.8, 1.2, 1.6, 2.0, 2.6, 3.2),
        cut_offsets: tuple[float, ...] = (0.0, -1.0, 1.0, -2.0, 2.0),
        side_spin_grid: tuple[float, ...] = (0.0, -0.4, 0.4, -0.8, 0.8),
        top_spin_grid: tuple[float, ...] = (0.0, -0.5, 0.5, -0.9, 0.9),
        landing_grid: CueLandingGrid | None = None,
        landing_tolerance_scale: float = 0.5,
        max_events: int = 80,
        random_seed: int | None = 42,
    ) -> None:
        if legal_mode not in {"any", "lowest"}:
            raise ValueError("legal_mode must be 'any' or 'lowest'.")
        self.pt = require_pooltool()
        self.cue_ball_id = cue_ball_id
        self.game_type = game_type
        self.legal_mode = legal_mode
        self.speed_grid = speed_grid
        self.cut_offsets = cut_offsets
        self.side_spin_grid = side_spin_grid
        self.top_spin_grid = top_spin_grid
        self.landing_grid = CueLandingGrid() if landing_grid is None else landing_grid
        self.landing_tolerance_scale = landing_tolerance_scale
        self.max_events = max_events
        self.random_seed = random_seed
        self.system = self.create_initial_system()

    def create_initial_system(self) -> Any:
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        if self.game_type == "example":
            return self.pt.System.example()
        table = self.pt.Table.default()
        if self.game_type == "nineball":
            balls = self.pt.get_rack(self.pt.GameType.NINEBALL, table)
        else:
            raise ValueError(f"Unsupported PoolTool game_type: {self.game_type}")
        cue = self.pt.Cue(cue_ball_id=self.cue_ball_id)
        return self.pt.System(table=table, balls=balls, cue=cue)

    def reset(self, *, break_rack: bool = False, break_speed: float = 10.0, break_target_ball_id: str = "1") -> Any:
        self.system = self.create_initial_system()
        if break_rack:
            self.break_rack(speed=break_speed, target_ball_id=break_target_ball_id)
        return self.system

    def break_rack(self, *, speed: float = 10.0, target_ball_id: str = "1", phi: float | None = None) -> BreakResult:
        """Apply a strong scripted break shot to scatter the initial rack."""

        if target_ball_id not in self.system.balls:
            raise ValueError(f"Break target ball is missing: {target_ball_id}")
        if phi is None:
            phi = float(self.pt.aim.at_ball(self.system, target_ball_id))

        self.system.cue.set_state(
            V0=float(speed),
            phi=float(phi),
            a=0.0,
            b=0.0,
            theta=0.0,
            cue_ball_id=self.cue_ball_id,
        )
        self.pt.simulate(self.system, inplace=True, max_events=max(self.max_events, 120))
        pocketed = tuple(
            ball_id
            for ball_id, ball in sorted(self.system.balls.items())
            if ball_id != self.cue_ball_id and self.is_ball_pocketed(ball)
        )
        return BreakResult(
            speed=float(speed),
            phi=float(phi),
            target_ball_id=target_ball_id,
            cue_scratch=self.is_ball_pocketed(self.system.balls[self.cue_ball_id]),
            pocketed_ball_ids=pocketed,
            ball_spread=self._ball_spread(self.system),
            events=tuple(str(event.event_type) for event in self.system.events),
        )

    def legal_ball_ids(self, system: Any | None = None) -> tuple[str, ...]:
        system = self.system if system is None else system
        ids = tuple(
            ball_id
            for ball_id, ball in sorted(system.balls.items())
            if ball_id != self.cue_ball_id and not self.is_ball_pocketed(ball)
        )
        if self.legal_mode == "lowest" and ids:
            numeric = sorted(ids, key=lambda value: int(value) if value.isdigit() else value)
            return (numeric[0],)
        return ids

    def pocket_ids(self, system: Any | None = None) -> tuple[str, ...]:
        system = self.system if system is None else system
        return tuple(system.table.pockets.keys())

    def enumerate_actions(self, system: Any | None = None) -> tuple[ShotAction, ...]:
        system = self.system if system is None else system
        return tuple(
            ShotAction(ball_id, pocket_id)
            for ball_id in self.legal_ball_ids(system)
            for pocket_id in self.pocket_ids(system)
        )

    def enumerate_position_actions(self, system: Any | None = None) -> tuple[ShotAction, ...]:
        """Enumerate ball/pocket/cue-landing actions for position-play policies."""

        system = self.system if system is None else system
        return tuple(
            ShotAction(ball_id, pocket_id, landing_cell)
            for ball_id in self.legal_ball_ids(system)
            for pocket_id in self.pocket_ids(system)
            for landing_cell in range(self.landing_grid.cell_count)
        )

    def is_geometrically_pottable(self, system: Any, action: ShotAction) -> bool:
        """Return whether a direct-pot action has a clear ghost-ball geometry."""

        if action.target_ball_id not in self.legal_ball_ids(system):
            return False
        if action.target_pocket_id not in system.table.pockets:
            return False
        return self._ghost_ball_phi(system, action) is not None

    def evaluate_action(self, system: Any, action: ShotAction) -> ShotEvaluation:
        if action.target_ball_id not in self.legal_ball_ids(system):
            return ShotEvaluation(action, score=-1_000.0, success=False, foul=True, reason="illegal_target")
        if action.target_pocket_id not in system.table.pockets:
            return ShotEvaluation(action, score=-1_000.0, success=False, foul=True, reason="missing_pocket")

        base_phi = self._ghost_ball_phi(system, action)
        if base_phi is None:
            return ShotEvaluation(action, score=-500.0, success=False, foul=False, reason="blocked_or_bad_geometry")

        if action.cue_landing_cell is not None:
            self.landing_grid.cell_center(system, action.cue_landing_cell)

        best: ShotEvaluation | None = None
        for cut_offset in self.cut_offsets:
            phi = (base_phi + cut_offset) % 360.0
            for speed in self.speed_grid:
                for side_spin in self.side_spin_grid_for(action):
                    for top_spin in self.top_spin_grid_for(action):
                        if side_spin * side_spin + top_spin * top_spin >= 0.98:
                            continue
                        candidate = self._simulate_solution(
                            system,
                            action,
                            ShotSolution(speed=speed, phi=phi, side_spin=side_spin, top_spin=top_spin),
                        )
                        if best is None or candidate.score > best.score:
                            best = candidate
                        if candidate.success and not candidate.foul:
                            return candidate

        if best is None:
            return ShotEvaluation(action, score=-500.0, success=False, foul=False, reason="no_solution")
        return best

    def step(self, action: ShotAction) -> ShotEvaluation:
        evaluation = self.evaluate_action(self.system, action)
        if evaluation.next_system is not None:
            self.system = evaluation.next_system
        return evaluation

    def is_cleared(self, system: Any | None = None) -> bool:
        return not self.legal_ball_ids(self.system if system is None else system)

    def is_ball_pocketed(self, ball: Any) -> bool:
        return int(ball.state.s) == 4

    def _simulate_solution(self, system: Any, action: ShotAction, solution: ShotSolution) -> ShotEvaluation:
        shot = system.copy()
        shot.cue.set_state(
            V0=solution.speed,
            phi=solution.phi,
            a=solution.side_spin,
            b=solution.top_spin,
            theta=solution.elevation,
            cue_ball_id=self.cue_ball_id,
        )
        self.pt.simulate(shot, inplace=True, max_events=self.max_events)

        pot_success = self._target_pocketed(shot, action)
        cue_scratch = self.is_ball_pocketed(shot.balls[self.cue_ball_id])
        first_contact = self._first_ball_contact(shot)
        wrong_first_contact = first_contact is not None and first_contact != action.target_ball_id
        foul = cue_scratch or wrong_first_contact
        cue_ball_xy = np.asarray(shot.balls[self.cue_ball_id].state.rvw[0, :2], dtype=np.float64)
        landing_cell, landing_distance, landing_success = self._cue_landing_result(shot, action, cue_ball_xy)
        success = pot_success and (landing_success if action.cue_landing_cell is not None else True)

        score = self._base_score(system, shot, action, solution, success, foul)
        reason = "pot" if pot_success else "miss"
        if pot_success and action.cue_landing_cell is not None and not landing_success:
            reason = f"landing_miss:{landing_cell}"
        if cue_scratch:
            reason = "cue_scratch"
        elif wrong_first_contact:
            reason = f"wrong_first_contact:{first_contact}"

        return ShotEvaluation(
            action=action,
            score=score,
            success=success,
            foul=foul,
            reason=reason,
            solution=solution,
            next_system=shot,
            cue_ball_xy=cue_ball_xy,
            cue_landing_cell=landing_cell,
            cue_landing_distance=landing_distance,
            pot_success=pot_success,
            landing_success=landing_success,
            events=tuple(str(event.event_type) for event in shot.events),
        )

    def _base_score(
        self,
        before: Any,
        after: Any,
        action: ShotAction,
        solution: ShotSolution,
        success: bool,
        foul: bool,
    ) -> float:
        cue_xy = self._ball_xy(before, self.cue_ball_id)
        ball_xy = self._ball_xy(before, action.target_ball_id)
        pocket_xy = self._pocket_aim_xy(before, action.target_pocket_id)
        cue_to_ball = float(np.linalg.norm(ball_xy - cue_xy))
        ball_to_pocket = float(np.linalg.norm(pocket_xy - ball_xy))
        score = 0.0
        score += 1000.0 if success else 0.0
        score -= 1000.0 if foul else 0.0
        if action.cue_landing_cell is not None and not foul:
            target_xy = self.landing_grid.cell_center(after, action.cue_landing_cell)
            cue_after_xy = self._ball_xy(after, self.cue_ball_id)
            landing_distance = float(np.linalg.norm(cue_after_xy - target_xy))
            cell_radius = self._landing_cell_radius(after)
            score += 250.0 if landing_distance <= cell_radius else 0.0
            score -= 120.0 * landing_distance
        score -= 2.0 * cue_to_ball
        score -= 3.0 * ball_to_pocket
        score -= 0.05 * solution.speed
        if success and not foul:
            remaining_before = len(self.legal_ball_ids(before))
            remaining_after = len(self.legal_ball_ids(after))
            score += 50.0 * max(0, remaining_before - remaining_after)
        return score

    def _target_pocketed(self, system: Any, action: ShotAction) -> bool:
        for event in system.events:
            if str(event.event_type) != "ball_pocket":
                continue
            ids = tuple(str(item) for item in event.ids)
            if action.target_ball_id in ids and action.target_pocket_id in ids:
                return True
        return False

    def _first_ball_contact(self, system: Any) -> str | None:
        for event in system.events:
            if str(event.event_type) != "ball_ball":
                continue
            ids = tuple(str(item) for item in event.ids)
            if self.cue_ball_id not in ids:
                continue
            others = [item for item in ids if item != self.cue_ball_id]
            return others[0] if others else None
        return None

    def _ghost_ball_phi(self, system: Any, action: ShotAction) -> float | None:
        cue_xy = self._ball_xy(system, self.cue_ball_id)
        object_xy = self._ball_xy(system, action.target_ball_id)
        pocket_xy = self._pocket_aim_xy(system, action.target_pocket_id)
        radius = float(system.balls[action.target_ball_id].params.R)

        object_to_pocket = pocket_xy - object_xy
        object_to_pocket_norm = float(np.linalg.norm(object_to_pocket))
        if object_to_pocket_norm <= 2.0 * radius:
            return None
        pot_dir = object_to_pocket / object_to_pocket_norm
        ghost_xy = object_xy - 2.0 * radius * pot_dir
        cue_to_ghost = ghost_xy - cue_xy
        if float(np.linalg.norm(cue_to_ghost)) <= 1e-9:
            return None

        blockers = [
            self._ball_xy(system, ball_id)
            for ball_id in self.legal_ball_ids(system)
            if ball_id not in {action.target_ball_id, self.cue_ball_id}
        ]
        if self._path_blocked(object_xy, pocket_xy, blockers, 2.15 * radius):
            return None
        if self._path_blocked(cue_xy, ghost_xy, blockers + [object_xy], 2.05 * radius):
            return None

        angle = float(np.degrees(np.arctan2(cue_to_ghost[1], cue_to_ghost[0])) % 360.0)
        return angle

    def side_spin_grid_for(self, action: ShotAction) -> tuple[float, ...]:
        return self.side_spin_grid if action.cue_landing_cell is not None else (0.0,)

    def top_spin_grid_for(self, action: ShotAction) -> tuple[float, ...]:
        return self.top_spin_grid if action.cue_landing_cell is not None else (0.0,)

    def _cue_landing_result(
        self,
        system: Any,
        action: ShotAction,
        cue_ball_xy: FloatArray,
    ) -> tuple[int | None, float | None, bool | None]:
        if action.cue_landing_cell is None:
            return None, None, None
        if self.is_ball_pocketed(system.balls[self.cue_ball_id]):
            return None, None, False
        actual_cell = self.landing_grid.encode_xy(system, cue_ball_xy)
        target_xy = self.landing_grid.cell_center(system, action.cue_landing_cell)
        distance = float(np.linalg.norm(cue_ball_xy - target_xy))
        return actual_cell, distance, actual_cell == action.cue_landing_cell

    def _landing_cell_radius(self, system: Any) -> float:
        cell_w = float(system.table.w) / self.landing_grid.x_bins
        cell_l = float(system.table.l) / self.landing_grid.y_bins
        return 0.5 * min(cell_w, cell_l) * self.landing_tolerance_scale

    def _path_blocked(self, start: FloatArray, end: FloatArray, blockers: Iterable[FloatArray], clearance: float) -> bool:
        segment = end - start
        length2 = float(np.dot(segment, segment))
        if length2 <= 1e-12:
            return True
        for point in blockers:
            t = float(np.dot(point - start, segment) / length2)
            if t <= 0.0 or t >= 1.0:
                continue
            closest = start + t * segment
            if float(np.linalg.norm(point - closest)) < clearance:
                return True
        return False

    def _ball_xy(self, system: Any, ball_id: str) -> FloatArray:
        return np.asarray(system.balls[ball_id].state.rvw[0, :2], dtype=np.float64)

    def _pocket_aim_xy(self, system: Any, pocket_id: str) -> FloatArray:
        pocket = system.table.pockets[pocket_id]
        return np.asarray(pocket.center[:2], dtype=np.float64)

    def _ball_spread(self, system: Any) -> float:
        positions = [
            self._ball_xy(system, ball_id)
            for ball_id, ball in sorted(system.balls.items())
            if ball_id != self.cue_ball_id and not self.is_ball_pocketed(ball)
        ]
        if not positions:
            return 0.0
        array = np.asarray(positions, dtype=np.float64)
        center = np.mean(array, axis=0)
        return float(np.mean(np.linalg.norm(array - center, axis=1)))


class HeuristicClearancePlanner:
    """Depth-limited heuristic planner over PoolTool high-level actions."""

    def __init__(
        self,
        env: PoolToolSinglePlayerEnv,
        *,
        depth: int = 2,
        beam_width: int = 4,
        discount: float = 0.6,
    ) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive.")
        self.env = env
        self.depth = depth
        self.beam_width = beam_width
        self.discount = discount

    def choose_action(self, system: Any | None = None) -> ShotPlan:
        system = self.env.system if system is None else system
        candidates = self.rank_actions(system, depth=self.depth)
        if not candidates:
            raise RuntimeError("No high-level shot candidates are available.")
        best = candidates[0]
        return ShotPlan(action=best.action, evaluation=best, candidates=tuple(candidates))

    def rank_actions(self, system: Any, depth: int) -> list[ShotEvaluation]:
        evaluated = [self.env.evaluate_action(system, action) for action in self.env.enumerate_actions(system)]
        evaluated.sort(key=lambda item: item.score, reverse=True)
        evaluated = evaluated[: self.beam_width]
        if depth <= 1:
            return evaluated

        rescored: list[ShotEvaluation] = []
        for evaluation in evaluated:
            if not evaluation.success or evaluation.foul or evaluation.next_system is None:
                rescored.append(evaluation)
                continue
            if self.env.is_cleared(evaluation.next_system):
                rescored.append(
                    ShotEvaluation(
                        action=evaluation.action,
                        score=evaluation.score + 500.0,
                        success=evaluation.success,
                        foul=evaluation.foul,
                        reason=evaluation.reason + "+clear",
                        solution=evaluation.solution,
                        next_system=evaluation.next_system,
                        cue_ball_xy=evaluation.cue_ball_xy,
                        events=evaluation.events,
                    )
                )
                continue
            next_ranked = self.rank_actions(evaluation.next_system, depth=depth - 1)
            lookahead = next_ranked[0].score if next_ranked else -100.0
            rescored.append(
                ShotEvaluation(
                    action=evaluation.action,
                    score=evaluation.score + self.discount * lookahead,
                    success=evaluation.success,
                    foul=evaluation.foul,
                    reason=evaluation.reason + "+lookahead",
                    solution=evaluation.solution,
                    next_system=evaluation.next_system,
                    cue_ball_xy=evaluation.cue_ball_xy,
                    events=evaluation.events,
                )
            )
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored
