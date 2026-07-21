"""Discrete-state PoolTool policies for high-level shot selection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from snooker_env.pooltool_high_level import PoolToolSinglePlayerEnv, ShotAction, ShotEvaluation


DiscreteState = tuple[int, ...]
DiscreteAction = tuple[str, str]


@dataclass(frozen=True)
class DiscreteTransition:
    """One deterministic transition in the discretized high-level MDP."""

    state: DiscreteState
    action: DiscreteAction
    next_state: DiscreteState
    reward: float
    terminal: bool
    evaluation: ShotEvaluation


@dataclass(frozen=True)
class ValueIterationResult:
    """Result of fitting a tabular value policy."""

    states: tuple[DiscreteState, ...]
    values: dict[DiscreteState, float]
    policy: dict[DiscreteState, DiscreteAction]
    transitions: dict[tuple[DiscreteState, DiscreteAction], DiscreteTransition]
    iterations: int
    max_delta: float


class PoolTableDiscretizer:
    """Encode PoolTool ball positions into a fixed table grid.

    The state layout is:

    ``cue_cell, ball_1_cell, ball_2_cell, ...``

    A pocketed ball is encoded as ``-1``. Non-pocketed balls are encoded as a
    row-major table cell index in ``[0, x_bins * y_bins)``.
    """

    def __init__(self, *, x_bins: int = 8, y_bins: int = 4, cue_ball_id: str = "cue") -> None:
        if x_bins <= 0 or y_bins <= 0:
            raise ValueError("x_bins and y_bins must be positive.")
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.cue_ball_id = cue_ball_id

    def encode(self, env: PoolToolSinglePlayerEnv, system: Any) -> DiscreteState:
        ball_ids = [self.cue_ball_id] + [
            ball_id for ball_id in sorted(system.balls) if ball_id != self.cue_ball_id
        ]
        return tuple(self._ball_cell(env, system, ball_id) for ball_id in ball_ids)

    def _ball_cell(self, env: PoolToolSinglePlayerEnv, system: Any, ball_id: str) -> int:
        ball = system.balls[ball_id]
        if env.is_ball_pocketed(ball):
            return -1
        xy = np.asarray(ball.state.rvw[0, :2], dtype=np.float64)
        if not np.all(np.isfinite(xy)):
            return -1
        x = int(np.clip(np.floor((xy[0] / float(system.table.w)) * self.x_bins), 0, self.x_bins - 1))
        y = int(np.clip(np.floor((xy[1] / float(system.table.l)) * self.y_bins), 0, self.y_bins - 1))
        return y * self.x_bins + x


class DiscreteValueIterationPolicy:
    """Tabular value-iteration policy over PoolTool high-level actions."""

    def __init__(
        self,
        env: PoolToolSinglePlayerEnv,
        discretizer: PoolTableDiscretizer,
        *,
        gamma: float = 0.9,
        pot_reward: float = 10.0,
        clear_reward: float = 50.0,
        foul_penalty: float = -20.0,
        miss_penalty: float = -2.0,
        speed_penalty: float = 0.02,
    ) -> None:
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must be in [0, 1).")
        self.env = env
        self.discretizer = discretizer
        self.gamma = gamma
        self.pot_reward = pot_reward
        self.clear_reward = clear_reward
        self.foul_penalty = foul_penalty
        self.miss_penalty = miss_penalty
        self.speed_penalty = speed_penalty
        self.result: ValueIterationResult | None = None
        self.representatives: dict[DiscreteState, Any] = {}

    def fit(
        self,
        initial_system: Any,
        *,
        max_depth: int | None = None,
        max_states: int | None = None,
        action_prune: int | None = None,
        prune_blocked_actions: bool = False,
        iterations: int = 80,
        tolerance: float = 1e-4,
        log_interval: int = 25,
    ) -> ValueIterationResult:
        if max_depth is not None and max_depth <= 0:
            raise ValueError("max_depth must be positive when set.")
        if max_states is not None and max_states <= 0:
            raise ValueError("max_states must be positive when set.")

        transitions = self._expand_reachable_graph(
            initial_system,
            max_depth=max_depth,
            max_states=max_states,
            action_prune=action_prune,
            prune_blocked_actions=prune_blocked_actions,
            log_interval=log_interval,
        )
        states = tuple(self.representatives)
        values = {state: 0.0 for state in states}
        policy: dict[DiscreteState, DiscreteAction] = {}
        max_delta = float("inf")

        actions_by_state: dict[DiscreteState, list[DiscreteTransition]] = {}
        for transition in transitions.values():
            actions_by_state.setdefault(transition.state, []).append(transition)

        completed_iterations = 0
        for completed_iterations in range(1, iterations + 1):
            max_delta = 0.0
            next_values = values.copy()
            for state, candidates in actions_by_state.items():
                q_values = [
                    (
                        transition.reward
                        if transition.terminal
                        else transition.reward + self.gamma * values.get(transition.next_state, 0.0),
                        transition.action,
                    )
                    for transition in candidates
                ]
                best_value, best_action = max(q_values, key=lambda item: item[0])
                next_values[state] = best_value
                policy[state] = best_action
                max_delta = max(max_delta, abs(best_value - values.get(state, 0.0)))
            values = next_values
            if log_interval > 0 and (completed_iterations == 1 or completed_iterations % log_interval == 0):
                print(
                    "value_iteration: "
                    f"iteration={completed_iterations} "
                    f"states={len(states)} "
                    f"max_delta={max_delta:.6f}",
                    flush=True,
                )
            if max_delta < tolerance:
                break

        self.result = ValueIterationResult(
            states=states,
            values=values,
            policy=policy,
            transitions=transitions,
            iterations=completed_iterations,
            max_delta=max_delta,
        )
        return self.result

    def choose_action(self, system: Any) -> ShotAction:
        if self.result is None:
            raise RuntimeError("Policy has not been fit yet.")
        state = self.discretizer.encode(self.env, system)
        action = self.result.policy.get(state)
        if action is None:
            raise RuntimeError(f"No learned action for discretized state: {state}")
        return ShotAction(target_ball_id=action[0], target_pocket_id=action[1])

    def _expand_reachable_graph(
        self,
        initial_system: Any,
        *,
        max_depth: int | None,
        max_states: int | None,
        action_prune: int | None,
        prune_blocked_actions: bool,
        log_interval: int,
    ) -> dict[tuple[DiscreteState, DiscreteAction], DiscreteTransition]:
        self.representatives = {}
        transitions: dict[tuple[DiscreteState, DiscreteAction], DiscreteTransition] = {}
        initial_state = self.discretizer.encode(self.env, initial_system)
        self.representatives[initial_state] = initial_system.copy()
        queue: deque[tuple[DiscreteState, int]] = deque([(initial_state, 0)])

        expanded = 0
        while queue:
            state, depth = queue.popleft()
            system = self.representatives[state]
            if (max_depth is not None and depth >= max_depth) or self.env.is_cleared(system):
                continue

            actions = self.env.enumerate_actions(system)
            if prune_blocked_actions:
                actions = tuple(action for action in actions if self.env.is_geometrically_pottable(system, action))
            evaluations = [self.env.evaluate_action(system, action) for action in actions]
            if action_prune is not None:
                evaluations.sort(key=lambda item: item.score, reverse=True)
                evaluations = evaluations[:action_prune]

            for evaluation in evaluations:
                if evaluation.next_system is None:
                    next_system = system
                else:
                    next_system = evaluation.next_system
                next_state = self.discretizer.encode(self.env, next_system)
                terminal = evaluation.foul or self.env.is_cleared(next_system)
                reward = self._reward(system, evaluation, next_system)
                action_key = (evaluation.action.target_ball_id, evaluation.action.target_pocket_id)
                transitions[(state, action_key)] = DiscreteTransition(
                    state=state,
                    action=action_key,
                    next_state=next_state,
                    reward=reward,
                    terminal=terminal,
                    evaluation=evaluation,
                )
                if (
                    not terminal
                    and next_state not in self.representatives
                    and (max_states is None or len(self.representatives) < max_states)
                ):
                    self.representatives[next_state] = next_system.copy()
                    queue.append((next_state, depth + 1))

            expanded += 1
            if log_interval > 0 and (expanded == 1 or expanded % log_interval == 0):
                print(
                    "expand_graph: "
                    f"expanded={expanded} "
                    f"states={len(self.representatives)} "
                    f"transitions={len(transitions)} "
                    f"frontier={len(queue)} "
                    f"depth={depth}",
                    flush=True,
                )

        return transitions

    def _reward(self, before: Any, evaluation: ShotEvaluation, after: Any) -> float:
        if evaluation.foul:
            return self.foul_penalty
        reward = 0.0
        if evaluation.success:
            remaining_before = len(self.env.legal_ball_ids(before))
            remaining_after = len(self.env.legal_ball_ids(after))
            reward += self.pot_reward * max(1, remaining_before - remaining_after)
        else:
            reward += self.miss_penalty
        if self.env.is_cleared(after):
            reward += self.clear_reward
        if evaluation.solution is not None:
            reward -= self.speed_penalty * evaluation.solution.speed
        return reward
