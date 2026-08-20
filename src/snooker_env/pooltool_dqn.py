"""DQN scaffold for PoolTool high-level ball/pocket/landing selection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from snooker_env.pooltool_high_level import CueLandingGrid, PoolToolSinglePlayerEnv, ShotAction, ShotEvaluation, ShotSolution
from snooker_env.pooltool_landing_cache import LandingMaskCache


MAX_OBJECT_BALLS = 9


@dataclass(frozen=True)
class DQNTransition:
    """One high-level DQN transition."""

    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    done: bool
    next_action_mask: tuple[bool, ...]


class ReplayBuffer:
    """Small replay buffer using plain Python containers."""

    def __init__(self, capacity: int, *, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.storage: deque[DQNTransition] = deque(maxlen=capacity)
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.storage)

    def append(self, transition: DQNTransition) -> None:
        self.storage.append(transition)

    def sample(self, batch_size: int) -> list[DQNTransition]:
        if batch_size > len(self.storage):
            raise ValueError("batch_size exceeds replay size.")
        return self.random.sample(list(self.storage), batch_size)


class QNetwork(nn.Module):
    """Plain MLP Q network for fixed ball/pocket actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PoolToolDQNEnv:
    """PoolTool environment wrapper with fixed discrete high-level actions."""

    def __init__(
        self,
        *,
        game_type: str = "nineball",
        legal_mode: str = "any",
        random_seed: int | None = 42,
        break_rack: bool = True,
        break_speed: float = 10.0,
        randomize_break: bool = False,
        break_speed_range: tuple[float, float] | None = None,
        break_phi_jitter_degrees: float = 0.0,
        break_target_ball_id: str = "1",
        ordered_pocket_actions: bool = False,
        ordered_landing_actions: bool = False,
        reset_max_attempts: int = 100,
        prune_blocked_actions: bool = True,
        include_cue_landing: bool = True,
        landing_x_bins: int = 8,
        landing_y_bins: int = 4,
        prune_unreachable_landing_actions: bool = True,
        mask_unreachable_landing_actions: bool = False,
        landing_mask_speed_grid: tuple[float, ...] = (0.8, 1.2, 1.6, 2.0, 2.6, 3.2),
        landing_mask_cut_offsets: tuple[float, ...] = (0.0, -0.75, 0.75, -1.5, 1.5),
        landing_mask_side_spin_grid: tuple[float, ...] = (0.0, -0.3, 0.3, -0.6, 0.6),
        landing_mask_top_spin_grid: tuple[float, ...] = (0.0, -0.4, 0.4, -0.8, 0.8),
        shot_path_modes: tuple[str, ...] = ("direct", "cue_bank", "object_bank"),
        fast_landing_solver: bool = True,
        fast_landing_max_trials: int = 160,
        landing_mask_cache_path: Path | None = Path("outputs/pooltool/landing_mask_cache.sqlite"),
        pot_reward: float = 10.0,
        clear_reward: float = 50.0,
        foul_penalty: float = -20.0,
        miss_penalty: float = -2.0,
        unsolved_penalty: float = -8.0,
        step_penalty: float = -0.1,
        speed_penalty: float = 0.02,
        landing_reward: float = 5.0,
        landing_distance_penalty: float = 2.0,
        next_pocket_reward: float = 1.5,
        no_next_shot_penalty: float = -3.0,
        max_position_reward: float = 6.0,
    ) -> None:
        landing_grid = CueLandingGrid(x_bins=landing_x_bins, y_bins=landing_y_bins)
        if ordered_landing_actions:
            ordered_pocket_actions = True
            include_cue_landing = True
        if ordered_pocket_actions and legal_mode != "lowest":
            legal_mode = "lowest"
        self.env = PoolToolSinglePlayerEnv(
            game_type=game_type,
            legal_mode=legal_mode,
            random_seed=random_seed,
            landing_grid=landing_grid,
            speed_grid=landing_mask_speed_grid,
            cut_offsets=landing_mask_cut_offsets,
            side_spin_grid=landing_mask_side_spin_grid,
            top_spin_grid=landing_mask_top_spin_grid,
            shot_path_modes=shot_path_modes,
        )
        self.break_rack_enabled = break_rack
        self.break_speed = break_speed
        self.randomize_break = randomize_break
        self.break_speed_range = break_speed_range
        self.break_phi_jitter_degrees = break_phi_jitter_degrees
        self.break_target_ball_id = break_target_ball_id
        self.random = random.Random(random_seed)
        self.ordered_pocket_actions = ordered_pocket_actions
        self.ordered_landing_actions = ordered_landing_actions
        self.reset_max_attempts = reset_max_attempts
        self.prune_blocked_actions = prune_blocked_actions
        self.include_cue_landing = include_cue_landing
        self.landing_grid = landing_grid
        self.prune_unreachable_landing_actions = prune_unreachable_landing_actions
        self.mask_unreachable_landing_actions = mask_unreachable_landing_actions
        self.landing_mask_speed_grid = landing_mask_speed_grid
        self.landing_mask_cut_offsets = landing_mask_cut_offsets
        self.landing_mask_side_spin_grid = landing_mask_side_spin_grid
        self.landing_mask_top_spin_grid = landing_mask_top_spin_grid
        self.shot_path_modes = shot_path_modes
        self.fast_landing_solver = fast_landing_solver
        self.fast_landing_max_trials = fast_landing_max_trials
        self.landing_mask_cache = (
            LandingMaskCache(
                landing_mask_cache_path,
                landing_grid=landing_grid,
                speed_grid=landing_mask_speed_grid,
                cut_offsets=landing_mask_cut_offsets,
                side_spin_grid=landing_mask_side_spin_grid,
                top_spin_grid=landing_mask_top_spin_grid,
                shot_path_modes=self.env.shot_path_modes,
            )
            if landing_mask_cache_path is not None and include_cue_landing and prune_unreachable_landing_actions
            else None
        )
        self.pot_reward = pot_reward
        self.clear_reward = clear_reward
        self.foul_penalty = foul_penalty
        self.miss_penalty = miss_penalty
        self.unsolved_penalty = unsolved_penalty
        self.step_penalty = step_penalty
        self.speed_penalty = speed_penalty
        self.landing_reward = landing_reward
        self.landing_distance_penalty = landing_distance_penalty
        self.next_pocket_reward = next_pocket_reward
        self.no_next_shot_penalty = no_next_shot_penalty
        self.max_position_reward = max_position_reward
        self.object_ball_ids = tuple(str(idx) for idx in range(1, MAX_OBJECT_BALLS + 1))
        self.pocket_ids = tuple(self.env.pocket_ids(self.env.create_initial_system()))
        if len(self.pocket_ids) != 6:
            raise ValueError(f"Expected 6 pockets for fixed DQN action space, got {self.pocket_ids}")
        self.actions = self._build_actions()

    @property
    def state_dim(self) -> int:
        return 2 + MAX_OBJECT_BALLS * 3

    @property
    def action_dim(self) -> int:
        return len(self.actions)

    def reset(self) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        max_attempts = self.reset_max_attempts if self.ordered_landing_actions and self.randomize_break else 1
        last_observation: tuple[tuple[float, ...], tuple[bool, ...]] | None = None
        for _attempt in range(max_attempts):
            self.env.reset()
            if self.break_rack_enabled:
                speed = self.break_speed
                phi = None
                if self.randomize_break:
                    if self.break_speed_range is not None:
                        lo, hi = self.break_speed_range
                        if lo > hi:
                            raise ValueError("break_speed_range lower bound exceeds upper bound.")
                        speed = self.random.uniform(lo, hi)
                    if self.break_phi_jitter_degrees > 0.0:
                        base_phi = float(self.env.pt.aim.at_ball(self.env.system, self.break_target_ball_id))
                        phi = base_phi + self.random.uniform(-self.break_phi_jitter_degrees, self.break_phi_jitter_degrees)
                self.env.break_rack(speed=speed, target_ball_id=self.break_target_ball_id, phi=phi)
            self.prewarm_landing_cache(self.env.system)
            last_observation = self.observe()
            if any(last_observation[1]) or not self.ordered_landing_actions:
                return last_observation
        if last_observation is None:
            raise RuntimeError("PoolTool DQN reset failed before producing an observation.")
        raise RuntimeError(
            "PoolTool DQN reset produced no valid ordered landing actions after "
            f"{max_attempts} randomized break attempts."
        )

    def observe(self) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        return self.encode_state(self.env.system), self.action_mask(self.env.system)

    def step(self, action_index: int) -> tuple[tuple[float, ...], float, bool, tuple[bool, ...], dict[str, Any]]:
        if action_index < 0 or action_index >= self.action_dim:
            raise ValueError(f"Invalid action index: {action_index}")
        action = self.actions[action_index]
        action = self._resolve_action(action)
        before = self.env.system
        mask = self.action_mask(before)
        if not mask[action_index]:
            state, next_mask = self.observe()
            return state, self.foul_penalty, True, next_mask, {
                "action": action,
                "success": False,
                "foul": True,
                "reason": "masked_action",
            }
        pair_action = ShotAction(action.target_ball_id, action.target_pocket_id)
        cached_solution = (
            self._landing_solutions(before, pair_action).get(action.cue_landing_cell)
            if action.cue_landing_cell is not None and self.fast_landing_solver
            else None
        )
        masked_landing = action.cue_landing_cell is not None and cached_solution is None
        if masked_landing:
            state, next_mask = self.observe()
            return state, self.unsolved_penalty, True, next_mask, {
                "action": action,
                "success": False,
                "foul": False,
                "reason": "masked_landing",
                "position_reward": 0.0,
                "next_valid_pockets": None,
                "pot_success": None,
                "landing_success": False,
                "cue_landing_cell": None,
                "cue_landing_distance": None,
                "dead_position": False,
                "remaining_balls": self.env.legal_ball_ids(self.env.system),
            }

        if cached_solution is not None:
            evaluation = self.env.evaluate_solution(before, action, cached_solution)
            if evaluation.next_system is not None:
                self.env.system = evaluation.next_system
        else:
            evaluation = self.env.step(action)
        next_state, next_mask = self.observe()
        cue_ball_displacement: float | None = None
        before_cue_xy = np.asarray(self._ball_xy(before, self.env.cue_ball_id), dtype=np.float64)
        after_cue_xy = np.asarray(self._ball_xy(self.env.system, self.env.cue_ball_id), dtype=np.float64)
        if np.all(np.isfinite(before_cue_xy)) and np.all(np.isfinite(after_cue_xy)):
            cue_ball_displacement = float(np.linalg.norm(after_cue_xy - before_cue_xy))
        dead_position = not any(next_mask) and not self.env.is_cleared()
        done = bool(evaluation.foul or not evaluation.success or self.env.is_cleared() or dead_position)
        reward = self.reward(before, evaluation, self.env.system)
        position_reward, next_valid_pockets = self.position_reward(evaluation, self.env.system)
        return next_state, reward, done, next_mask, {
            "action": action,
            "success": evaluation.success,
            "foul": evaluation.foul,
            "reason": evaluation.reason,
            "solution": evaluation.solution,
            "position_reward": position_reward,
            "next_valid_pockets": next_valid_pockets,
            "pot_success": evaluation.pot_success,
            "landing_success": evaluation.landing_success,
            "cue_landing_cell": evaluation.cue_landing_cell,
            "cue_landing_distance": evaluation.cue_landing_distance,
            "cue_ball_displacement": cue_ball_displacement,
            "shot_path_type": None if evaluation.solution is None else evaluation.solution.path_type,
            "dead_position": dead_position,
            "remaining_balls": self.env.legal_ball_ids(self.env.system),
        }

    def action_mask(self, system: Any) -> tuple[bool, ...]:
        legal = set(self.env.legal_ball_ids(system))
        if self.ordered_landing_actions:
            return self._ordered_landing_action_mask(system, legal)
        if self.ordered_pocket_actions:
            return self._ordered_pocket_action_mask(system, legal)
        if self.include_cue_landing and self.prune_unreachable_landing_actions and self.mask_unreachable_landing_actions:
            return self._position_action_mask(system, legal)

        mask = []
        for action in self.actions:
            ok = action.target_ball_id in legal
            if ok and self.prune_blocked_actions:
                ok = self.env.is_geometrically_pottable(system, action)
            mask.append(ok)
        if not any(mask):
            for idx, action in enumerate(self.actions):
                if action.target_ball_id in legal:
                    mask[idx] = True
        return tuple(mask)

    def landing_mask_cache_stats(self) -> dict[str, int] | None:
        if self.landing_mask_cache is None:
            return None
        stats = self.landing_mask_cache.stats
        return {
            "hits": stats.hits,
            "misses": stats.misses,
            "writes": stats.writes,
            "rows": self.landing_mask_cache.count_rows(),
            "solution_rows": self.landing_mask_cache.count_solution_rows(),
        }

    def prewarm_landing_cache(self, system: Any) -> dict[str, int]:
        if not self.include_cue_landing or self.landing_mask_cache is None:
            return {"pairs": 0, "reachable_cells": 0}
        legal = set(self.env.legal_ball_ids(system))
        pairs = 0
        reachable_cells = 0
        ball_ids = (self._current_ordered_target(system),) if self.ordered_landing_actions else self.object_ball_ids
        for ball_id in ball_ids:
            if ball_id is None or ball_id not in legal:
                continue
            for pocket_id in self.pocket_ids:
                action = ShotAction(ball_id, pocket_id)
                if self.prune_blocked_actions and not self.env.is_geometrically_pottable(system, action):
                    continue
                cells = self._reachable_landing_cells(system, action)
                pairs += 1
                reachable_cells += len(cells)
        return {"pairs": pairs, "reachable_cells": reachable_cells}

    def _position_action_mask(self, system: Any, legal: set[str]) -> tuple[bool, ...]:
        reachable_by_pair: dict[tuple[str, str], frozenset[int]] = {}
        for ball_id in self.object_ball_ids:
            if ball_id not in legal:
                continue
            for pocket_id in self.pocket_ids:
                pair_action = ShotAction(ball_id, pocket_id)
                if self.prune_blocked_actions and not self.env.is_geometrically_pottable(system, pair_action):
                    reachable_by_pair[(ball_id, pocket_id)] = frozenset()
                    continue
                reachable_by_pair[(ball_id, pocket_id)] = self._reachable_landing_cells(system, pair_action)

        mask = []
        for action in self.actions:
            if action.target_ball_id not in legal:
                mask.append(False)
                continue
            reachable = reachable_by_pair.get((action.target_ball_id, action.target_pocket_id), frozenset())
            mask.append(action.cue_landing_cell in reachable)

        if not any(mask):
            fallback_cells: dict[tuple[str, str], int] = {}
            for (ball_id, pocket_id), reachable in reachable_by_pair.items():
                if reachable:
                    continue
                fallback = self.env.evaluate_action(system, ShotAction(ball_id, pocket_id))
                if fallback.success and not fallback.foul and fallback.cue_ball_xy is not None:
                    fallback_cells[(ball_id, pocket_id)] = self.landing_grid.encode_xy(
                        fallback.next_system,
                        fallback.cue_ball_xy,
                    )
            for idx, action in enumerate(self.actions):
                if action.target_ball_id not in legal:
                    continue
                fallback_cell = fallback_cells.get((action.target_ball_id, action.target_pocket_id))
                mask[idx] = action.cue_landing_cell == fallback_cell
        if not any(mask):
            for idx, action in enumerate(self.actions):
                if action.target_ball_id in legal:
                    mask[idx] = True
        return tuple(mask)

    def _ordered_pocket_action_mask(self, system: Any, legal: set[str]) -> tuple[bool, ...]:
        target_ball = self._current_ordered_target(system)
        if target_ball is None:
            return tuple(False for _ in self.actions)
        mask = []
        for action in self.actions:
            resolved = ShotAction(target_ball, action.target_pocket_id)
            ok = target_ball in legal
            if ok and self.prune_blocked_actions:
                ok = self.env.is_geometrically_pottable(system, resolved)
            mask.append(ok)
        if not any(mask):
            for idx, _action in enumerate(self.actions):
                mask[idx] = True
        return tuple(mask)

    def _ordered_landing_action_mask(self, system: Any, legal: set[str]) -> tuple[bool, ...]:
        target_ball = self._current_ordered_target(system)
        if target_ball is None or target_ball not in legal:
            return tuple(False for _ in self.actions)
        reachable_by_pocket: dict[str, frozenset[int]] = {}
        for pocket_id in self.pocket_ids:
            pair_action = ShotAction(target_ball, pocket_id)
            if self.prune_blocked_actions and not self.env.is_geometrically_pottable(system, pair_action):
                reachable_by_pocket[pocket_id] = frozenset()
                continue
            reachable_by_pocket[pocket_id] = self._reachable_landing_cells(system, pair_action)
        return tuple(
            action.cue_landing_cell in reachable_by_pocket.get(action.target_pocket_id, frozenset())
            for action in self.actions
        )

    def _reachable_landing_cells(self, system: Any, action: ShotAction) -> frozenset[int]:
        if self.fast_landing_solver:
            return frozenset(self._landing_solutions(system, action).keys())

        def compute() -> frozenset[int]:
            return self.env.reachable_landing_cells(
                system,
                action,
                speed_grid=self.landing_mask_speed_grid,
                cut_offsets=self.landing_mask_cut_offsets,
                side_spin_grid=self.landing_mask_side_spin_grid,
                top_spin_grid=self.landing_mask_top_spin_grid,
            )

        if self.landing_mask_cache is None:
            return compute()
        return self.landing_mask_cache.get_or_compute(system, action, compute)

    def _landing_solutions(self, system: Any, action: ShotAction) -> dict[int, ShotSolution]:
        def compute() -> dict[int, ShotSolution]:
            return self.env.fast_landing_solutions(
                system,
                action,
                speed_grid=self.landing_mask_speed_grid,
                cut_offsets=self.landing_mask_cut_offsets,
                side_spin_grid=self.landing_mask_side_spin_grid,
                top_spin_grid=self.landing_mask_top_spin_grid,
                max_trials=self.fast_landing_max_trials,
            )

        if self.landing_mask_cache is None:
            return compute()
        return self.landing_mask_cache.get_or_compute_solutions(system, action, compute)

    def _is_unreachable_landing_action(self, system: Any, action: ShotAction) -> bool:
        if action.cue_landing_cell is None:
            return False
        cells = self._reachable_landing_cells(system, ShotAction(action.target_ball_id, action.target_pocket_id))
        return action.cue_landing_cell not in cells

    def encode_state(self, system: Any) -> tuple[float, ...]:
        cue_xy = self._ball_xy(system, self.env.cue_ball_id)
        cue_world_xy = self.env.pool_to_world_xy(cue_xy)
        values: list[float] = [cue_world_xy[0], cue_world_xy[1]]
        for ball_id in self.object_ball_ids:
            ball = system.balls.get(ball_id)
            if ball is None or self.env.is_ball_pocketed(ball):
                values.extend([0.0, 0.0, 1.0])
                continue
            xy = self._ball_xy(system, ball_id)
            world_xy = self.env.pool_to_world_xy(xy)
            values.extend([world_xy[0], world_xy[1], 0.0])
        return tuple(float(v) for v in values)

    def reward(self, before: Any, evaluation: ShotEvaluation, after: Any) -> float:
        if evaluation.foul:
            return self.foul_penalty
        reward = self.step_penalty
        if evaluation.success:
            before_count = len(self.env.legal_ball_ids(before))
            after_count = len(self.env.legal_ball_ids(after))
            reward += self.pot_reward * max(1, before_count - after_count)
        elif evaluation.pot_success and evaluation.action.cue_landing_cell is not None:
            reward += self.unsolved_penalty
        else:
            reward += self.miss_penalty
        if evaluation.action.cue_landing_cell is not None:
            if evaluation.landing_success:
                reward += self.landing_reward
            elif evaluation.cue_landing_distance is not None:
                reward -= self.landing_distance_penalty * evaluation.cue_landing_distance
        if self.env.is_cleared(after):
            reward += self.clear_reward
        reward += self.position_reward(evaluation, after)[0]
        if evaluation.solution is not None:
            reward -= self.speed_penalty * evaluation.solution.speed
        return float(reward)

    def position_reward(self, evaluation: ShotEvaluation, after: Any) -> tuple[float, int | None]:
        if not self.ordered_pocket_actions or not evaluation.success or evaluation.foul or self.env.is_cleared(after):
            return 0.0, None
        next_target = self._current_ordered_target(after)
        if next_target is None:
            return 0.0, None
        valid_pockets = sum(
            1
            for pocket_id in self.pocket_ids
            if self.env.is_geometrically_pottable(after, ShotAction(next_target, pocket_id))
        )
        if valid_pockets <= 0:
            return float(self.no_next_shot_penalty), 0
        return float(min(self.max_position_reward, self.next_pocket_reward * valid_pockets)), valid_pockets

    def _build_actions(self) -> tuple[ShotAction, ...]:
        if self.ordered_landing_actions:
            return tuple(
                ShotAction("__current__", pocket_id, landing_cell)
                for pocket_id in self.pocket_ids
                for landing_cell in range(self.landing_grid.cell_count)
            )
        if self.ordered_pocket_actions:
            return tuple(ShotAction("__current__", pocket_id) for pocket_id in self.pocket_ids)
        if not self.include_cue_landing:
            return tuple(
                ShotAction(ball_id, pocket_id)
                for ball_id in self.object_ball_ids
                for pocket_id in self.pocket_ids
            )
        return tuple(
            ShotAction(ball_id, pocket_id, landing_cell)
            for ball_id in self.object_ball_ids
            for pocket_id in self.pocket_ids
            for landing_cell in range(self.landing_grid.cell_count)
        )

    def _resolve_action(self, action: ShotAction) -> ShotAction:
        if not self.ordered_pocket_actions:
            return action
        target_ball = self._current_ordered_target(self.env.system)
        if target_ball is None:
            return action
        return ShotAction(target_ball, action.target_pocket_id, action.cue_landing_cell)

    def _current_ordered_target(self, system: Any) -> str | None:
        legal = self.env.legal_ball_ids(system)
        return legal[0] if legal else None

    def _ball_xy(self, system: Any, ball_id: str) -> tuple[float, float]:
        xy = np.asarray(system.balls[ball_id].state.rvw[0, :2], dtype=np.float64)
        return float(xy[0]), float(xy[1])


class TwoPlayerPoolToolDQNEnv(PoolToolDQNEnv):
    """Two-player turn-taking high-level DQN environment.

    The environment keeps the action space at six pocket choices. The active
    target is the lowest remaining ball, both agents see the same state encoding,
    and a trainer can attach one Q network/replay buffer per player.
    """

    def __init__(
        self,
        *,
        players: int = 2,
        switch_on_miss: bool = True,
        **kwargs: Any,
    ) -> None:
        if players != 2:
            raise ValueError("TwoPlayerPoolToolDQNEnv currently supports exactly two players.")
        kwargs["ordered_pocket_actions"] = True
        kwargs["ordered_landing_actions"] = False
        kwargs["include_cue_landing"] = False
        kwargs["legal_mode"] = "lowest"
        kwargs["shot_path_modes"] = ("direct",)
        super().__init__(**kwargs)
        self.players = players
        self.switch_on_miss = switch_on_miss
        self.current_player = 0
        self.player_scores = [0 for _ in range(players)]
        self.player_turns = [0 for _ in range(players)]

    def reset(self) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        self.current_player = 0
        self.player_scores = [0 for _ in range(self.players)]
        self.player_turns = [0 for _ in range(self.players)]
        return super().reset()

    def step(self, action_index: int) -> tuple[tuple[float, ...], float, bool, tuple[bool, ...], dict[str, Any]]:
        acting_player = self.current_player
        next_state, reward, single_player_done, next_mask, info = super().step(action_index)
        success = bool(info.get("success", False))
        foul = bool(info.get("foul", False))
        cleared = self.env.is_cleared()
        self.player_turns[acting_player] += 1
        if success and not foul:
            self.player_scores[acting_player] += 1
        switch_turn = bool((foul or not success) and self.switch_on_miss and not cleared)
        cue_ball_restored = self._restore_cue_ball_if_pocketed()
        if switch_turn:
            self.current_player = (self.current_player + 1) % self.players
            next_state, next_mask = self.observe()
            dead_position = not any(next_mask) and not cleared
            done = bool(dead_position)
            info["dead_position"] = dead_position
        else:
            done = bool(cleared or single_player_done)
        info.update(
            {
                "acting_player": acting_player,
                "current_player": self.current_player,
                "switch_turn": switch_turn,
                "cue_ball_restored": cue_ball_restored,
                "player_scores": tuple(self.player_scores),
                "player_turns": tuple(self.player_turns),
            }
        )
        return next_state, reward, done, next_mask, info

    def _restore_cue_ball_if_pocketed(self) -> bool:
        cue_ball = self.env.system.balls[self.env.cue_ball_id]
        if not self.env.is_ball_pocketed(cue_ball) and np.all(np.isfinite(cue_ball.state.rvw[0])):
            return False
        radius = self.env.table_spec.ball_radius
        candidates = (
            (0.25 * float(self.env.system.table.w), 0.25 * float(self.env.system.table.l)),
            (0.50 * float(self.env.system.table.w), 0.25 * float(self.env.system.table.l)),
            (0.75 * float(self.env.system.table.w), 0.25 * float(self.env.system.table.l)),
            (0.50 * float(self.env.system.table.w), 0.50 * float(self.env.system.table.l)),
        )
        occupied = [
            np.asarray(ball.state.rvw[0, :2], dtype=np.float64)
            for ball_id, ball in self.env.system.balls.items()
            if ball_id != self.env.cue_ball_id and not self.env.is_ball_pocketed(ball)
        ]
        selected = candidates[0]
        for xy in candidates:
            point = np.asarray(xy, dtype=np.float64)
            if all(float(np.linalg.norm(point - other)) > 2.2 * radius for other in occupied):
                selected = xy
                break
        cue_ball.state.rvw[:] = 0.0
        cue_ball.state.rvw[0] = [float(selected[0]), float(selected[1]), radius]
        cue_ball.state.s = 0
        return True
