"""Pipeline composition for high/mid/low level policy stacks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snooker_env.pipeline_high_level import GeometricShotPlanner, HighLevelPolicy
from snooker_env.pipeline_low_level import DualArmImpedanceController, LowLevelPolicy
from snooker_env.pipeline_mid_level import MidLevelSkill, default_skill_registry
from snooker_env.pipeline_system import DefaultRecoveryPlanner, GeometricBodyPositioningPlanner
from snooker_env.pipeline_types import PipelineOutput, SceneState, SkillId


@dataclass
class SnookerPipeline:
    """Thin orchestration layer mirroring the booster_gym hierarchy."""

    high_level: HighLevelPolicy
    mid_level_skills: dict[SkillId, MidLevelSkill]
    low_level: LowLevelPolicy
    body_positioning: GeometricBodyPositioningPlanner
    recovery: DefaultRecoveryPlanner

    def plan_once(self, state: SceneState) -> PipelineOutput:
        skills = self.high_level.plan(state)
        cue_trajectories = []
        cue_commands = []
        joint_actions = []
        for command in skills:
            skill = self.mid_level_skills.get(command.skill_id)
            if skill is None:
                raise KeyError(f"No mid-level skill registered for {command.skill_id}")
            trajectory = skill.rollout(command, state)
            cue_trajectories.append(trajectory)
            for cue_command in trajectory:
                cue_commands.append(cue_command)
                joint_actions.append(self.low_level.act(cue_command, state))

        body_position = None
        if cue_commands:
            first_intent = skills[0].intent
            stroke_direction = next(
                (command.linear_velocity for command in cue_commands if np.linalg.norm(command.linear_velocity) > 1e-9),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
            )
            body_position = self.body_positioning.plan(
                cue_commands[0].pose,
                first_intent.object_ball_name or first_intent.cue_ball_name,
                stroke_direction,
                state,
            )

        return PipelineOutput(
            skills=skills,
            cue_trajectories=tuple(cue_trajectories),
            cue_commands=tuple(cue_commands),
            joint_actions=tuple(joint_actions),
            body_position=body_position,
            recovery=self.recovery.plan(state),
        )


def build_default_pipeline() -> SnookerPipeline:
    """Build the initial scripted stack used by smoke tests."""

    return SnookerPipeline(
        high_level=GeometricShotPlanner(),
        mid_level_skills=default_skill_registry(),
        low_level=DualArmImpedanceController(),
        body_positioning=GeometricBodyPositioningPlanner(),
        recovery=DefaultRecoveryPlanner(),
    )
