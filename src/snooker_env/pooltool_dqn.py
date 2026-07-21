"""DQN scaffold for PoolTool high-level ball/pocket selection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from snooker_env.pooltool_high_level import PoolToolSinglePlayerEnv, ShotAction, ShotEvaluation


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
    """PoolTool environment wrapper with fixed 54-way ball/pocket actions."""

    def __init__(
        self,
        *,
        game_type: str = "nineball",
        legal_mode: str = "any",
        random_seed: int | None = 42,
        break_rack: bool = True,
        break_speed: float = 10.0,
        break_target_ball_id: str = "1",
        prune_blocked_actions: bool = True,
        pot_reward: float = 10.0,
        clear_reward: float = 50.0,
        foul_penalty: float = -20.0,
        miss_penalty: float = -2.0,
        step_penalty: float = -0.1,
        speed_penalty: float = 0.02,
    ) -> None:
        self.env = PoolToolSinglePlayerEnv(game_type=game_type, legal_mode=legal_mode, random_seed=random_seed)
        self.break_rack_enabled = break_rack
        self.break_speed = break_speed
        self.break_target_ball_id = break_target_ball_id
        self.prune_blocked_actions = prune_blocked_actions
        self.pot_reward = pot_reward
        self.clear_reward = clear_reward
        self.foul_penalty = foul_penalty
        self.miss_penalty = miss_penalty
        self.step_penalty = step_penalty
        self.speed_penalty = speed_penalty
        self.object_ball_ids = tuple(str(idx) for idx in range(1, MAX_OBJECT_BALLS + 1))
        self.pocket_ids = tuple(self.env.pocket_ids(self.env.create_initial_system()))
        if len(self.pocket_ids) != 6:
            raise ValueError(f"Expected 6 pockets for fixed DQN action space, got {self.pocket_ids}")
        self.actions = tuple(
            ShotAction(ball_id, pocket_id)
            for ball_id in self.object_ball_ids
            for pocket_id in self.pocket_ids
        )

    @property
    def state_dim(self) -> int:
        return 2 + MAX_OBJECT_BALLS * 3

    @property
    def action_dim(self) -> int:
        return len(self.actions)

    def reset(self) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        self.env.reset()
        if self.break_rack_enabled:
            self.env.break_rack(speed=self.break_speed, target_ball_id=self.break_target_ball_id)
        return self.observe()

    def observe(self) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        return self.encode_state(self.env.system), self.action_mask(self.env.system)

    def step(self, action_index: int) -> tuple[tuple[float, ...], float, bool, tuple[bool, ...], dict[str, Any]]:
        if action_index < 0 or action_index >= self.action_dim:
            raise ValueError(f"Invalid action index: {action_index}")
        action = self.actions[action_index]
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

        evaluation = self.env.step(action)
        done = bool(evaluation.foul or not evaluation.success or self.env.is_cleared())
        next_state, next_mask = self.observe()
        reward = self.reward(before, evaluation, self.env.system)
        return next_state, reward, done, next_mask, {
            "action": action,
            "success": evaluation.success,
            "foul": evaluation.foul,
            "reason": evaluation.reason,
            "solution": evaluation.solution,
            "remaining_balls": self.env.legal_ball_ids(self.env.system),
        }

    def action_mask(self, system: Any) -> tuple[bool, ...]:
        legal = set(self.env.legal_ball_ids(system))
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

    def encode_state(self, system: Any) -> tuple[float, ...]:
        table_w = float(system.table.w)
        table_l = float(system.table.l)
        cue_xy = self._ball_xy(system, self.env.cue_ball_id)
        values: list[float] = [cue_xy[0] / table_w, cue_xy[1] / table_l]
        for ball_id in self.object_ball_ids:
            ball = system.balls.get(ball_id)
            if ball is None or self.env.is_ball_pocketed(ball):
                values.extend([0.0, 0.0, 1.0])
                continue
            xy = self._ball_xy(system, ball_id)
            values.extend([xy[0] / table_w, xy[1] / table_l, 0.0])
        return tuple(float(v) for v in values)

    def reward(self, before: Any, evaluation: ShotEvaluation, after: Any) -> float:
        if evaluation.foul:
            return self.foul_penalty
        reward = self.step_penalty
        if evaluation.success:
            before_count = len(self.env.legal_ball_ids(before))
            after_count = len(self.env.legal_ball_ids(after))
            reward += self.pot_reward * max(1, before_count - after_count)
        else:
            reward += self.miss_penalty
        if self.env.is_cleared(after):
            reward += self.clear_reward
        if evaluation.solution is not None:
            reward -= self.speed_penalty * evaluation.solution.speed
        return float(reward)

    def _ball_xy(self, system: Any, ball_id: str) -> tuple[float, float]:
        xy = np.asarray(system.balls[ball_id].state.rvw[0, :2], dtype=np.float64)
        return float(xy[0]), float(xy[1])
