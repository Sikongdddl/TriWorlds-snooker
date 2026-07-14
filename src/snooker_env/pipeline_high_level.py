"""High-level game strategy policies."""

from __future__ import annotations

from typing import Protocol

from snooker_env.pipeline_types import SceneState, ShotIntent, SkillCommand, SkillId


class HighLevelPolicy(Protocol):
    """Selects a sequence of mid-level shot skills from table state."""

    def plan(self, state: SceneState) -> tuple[SkillCommand, ...]:
        """Return the next mid-level skill sequence."""


class GeometricShotPlanner(HighLevelPolicy):
    """Minimal deterministic high-level strategy for early integration tests."""

    def __init__(self, cue_ball_name: str = "cue_ball", default_speed: float = 0.8) -> None:
        self.cue_ball_name = cue_ball_name
        self.default_speed = default_speed

    def plan(self, state: SceneState) -> tuple[SkillCommand, ...]:
        if self.cue_ball_name not in state.balls:
            raise KeyError(f"Missing cue ball state: {self.cue_ball_name}")

        object_names = sorted(name for name in state.balls if name != self.cue_ball_name)
        if not object_names:
            raise ValueError("High-level planner needs at least one object ball.")

        intent = ShotIntent(
            cue_ball_name=self.cue_ball_name,
            object_ball_name=object_names[0],
            target_pocket="demo_forward",
            target_direction=None,
            desired_cue_ball_position=None,
            target_speed=self.default_speed,
            risk_preference=0.5,
        )
        return (SkillCommand(SkillId.POT_SHOT, intent),)


class SimpleRLHighLevelPolicy(HighLevelPolicy):
    """Adapter slot for a future game-strategy RL policy."""

    def plan(self, state: SceneState) -> tuple[SkillCommand, ...]:
        raise NotImplementedError(
            "SimpleRLHighLevelPolicy is an interface placeholder. "
            "Use GeometricShotPlanner until a trained high-level checkpoint exists."
        )


class VLMHighLevelPolicy(HighLevelPolicy):
    """Adapter slot for VLM table reasoning."""

    def plan(self, state: SceneState) -> tuple[SkillCommand, ...]:
        if state.vision is None:
            raise ValueError("VLMHighLevelPolicy requires SceneState.vision.")
        raise NotImplementedError(
            "VLMHighLevelPolicy is a placeholder for image/table-state prompting."
        )
