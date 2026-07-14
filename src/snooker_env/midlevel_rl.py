"""Curriculum-learning interfaces for mid-level shot policies.

The high-level policy still calls semantic shot skills such as ``pot_shot``.
This module exposes the internal training curriculum used to learn each skill:

1. impact parameter inference
2. cue setup trajectory generation
3. stroke trajectory generation

No RL framework is imported here. Training code can wrap these dataclasses and
protocols with its own replay buffer, optimizer, vectorization, or logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol

import numpy as np

from snooker_env.midlevel_env import MidLevelCueEnv, MidLevelRolloutResult
from snooker_env.pipeline_mid_level import PotShotPolicy, _cue_pose_from_plan, _shot_direction
from snooker_env.pipeline_types import CueCommand, Pose3D, SceneState, SkillCommand, SkillId


class CurriculumStage(str, Enum):
    """Mid-level curriculum stages trained from easier to harder."""

    IMPACT_PARAMETER_INFERENCE = "impact_parameter_inference"
    CUE_SETUP_TRAJECTORY_GENERATION = "cue_setup_trajectory_generation"
    STROKE_TRAJECTORY_GENERATION = "stroke_trajectory_generation"


@dataclass(frozen=True)
class ImpactParameters:
    """Learned shot parameters before generating a cue trajectory."""

    cue_direction: np.ndarray
    cue_speed: float
    contact_offset_yz: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    cue_elevation: float = 0.0
    spin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    tolerance: float = 0.02


@dataclass(frozen=True)
class CueSetupPlan:
    """Pre-stroke cue setup generated from impact parameters."""

    setup_pose: Pose3D
    align_pose: Pose3D
    stroke_start_pose: Pose3D


@dataclass(frozen=True)
class StrokeTrajectoryPlan:
    """Final cue-command sequence executed in MuJoCo."""

    commands: tuple[CueCommand, ...]


@dataclass(frozen=True)
class StageObservation:
    """Observation passed to a stage policy."""

    stage: CurriculumStage
    skill_id: SkillId
    command: SkillCommand
    scene_state: SceneState
    impact: ImpactParameters | None = None
    setup: CueSetupPlan | None = None


@dataclass(frozen=True)
class StageAction:
    """Action produced by a stage policy.

    Exactly one field is expected for the matching stage.
    """

    impact: ImpactParameters | None = None
    setup: CueSetupPlan | None = None
    stroke: StrokeTrajectoryPlan | None = None


@dataclass(frozen=True)
class StageResult:
    """Reward and diagnostics for one curriculum stage."""

    reward: float
    terminated: bool
    metrics: Mapping[str, float]
    rollout: MidLevelRolloutResult | None = None


class ImpactParameterPolicy(Protocol):
    """Policy for stage 1: infer impact parameters from shot intent."""

    def act(self, observation: StageObservation) -> ImpactParameters:
        """Return cue direction, speed, contact offset, elevation, and spin."""


class CueSetupPolicy(Protocol):
    """Policy for stage 2: generate pre-stroke setup poses."""

    def act(self, observation: StageObservation) -> CueSetupPlan:
        """Return setup/alignment/stroke-start cue poses."""


class StrokeTrajectoryPolicy(Protocol):
    """Policy for stage 3: generate executable cue commands."""

    def act(self, observation: StageObservation) -> StrokeTrajectoryPlan:
        """Return the final cue-command trajectory."""


class ScriptedMidLevelCurriculumPolicy(ImpactParameterPolicy, CueSetupPolicy, StrokeTrajectoryPolicy):
    """Scripted baseline implementing all three curriculum stages.

    This is not meant to be the final policy. It gives training code a stable
    reference implementation and makes each stage executable before RL is wired
    in.
    """

    def __init__(self, skill_id: SkillId = SkillId.POT_SHOT) -> None:
        self.skill_id = skill_id
        self._shot_policy = PotShotPolicy()

    def act(self, observation: StageObservation) -> ImpactParameters | CueSetupPlan | StrokeTrajectoryPlan:
        if observation.stage == CurriculumStage.IMPACT_PARAMETER_INFERENCE:
            return self.infer_impact(observation.command, observation.scene_state)
        if observation.stage == CurriculumStage.CUE_SETUP_TRAJECTORY_GENERATION:
            if observation.impact is None:
                raise ValueError("Cue setup stage requires observation.impact.")
            return self.generate_setup(observation.command, observation.scene_state, observation.impact)
        if observation.stage == CurriculumStage.STROKE_TRAJECTORY_GENERATION:
            if observation.impact is None or observation.setup is None:
                raise ValueError("Stroke stage requires observation.impact and observation.setup.")
            return self.generate_stroke(observation.command, observation.impact, observation.setup)
        raise ValueError(f"Unknown curriculum stage: {observation.stage}")

    def infer_impact(self, command: SkillCommand, state: SceneState) -> ImpactParameters:
        direction = _shot_direction(command, state)
        default_speed = self._shot_policy.default_speed if command.intent.target_speed is None else command.intent.target_speed
        speed = float(command.params.get("target_speed", default_speed))
        return ImpactParameters(cue_direction=direction, cue_speed=speed)

    def generate_setup(self, command: SkillCommand, state: SceneState, impact: ImpactParameters) -> CueSetupPlan:
        params = dict(command.params)
        params["target_speed"] = impact.cue_speed
        staged_command = SkillCommand(command.skill_id, command.intent, params=params)
        return CueSetupPlan(
            setup_pose=_cue_pose_from_plan(staged_command, state, backoff=self._shot_policy.setup_backoff),
            align_pose=_cue_pose_from_plan(staged_command, state, backoff=self._shot_policy.align_backoff),
            stroke_start_pose=_cue_pose_from_plan(staged_command, state, backoff=self._shot_policy.stroke_backoff),
        )

    def generate_stroke(
        self,
        command: SkillCommand,
        impact: ImpactParameters,
        setup: CueSetupPlan,
    ) -> StrokeTrajectoryPlan:
        direction = impact.cue_direction / max(float(np.linalg.norm(impact.cue_direction)), 1e-9)
        stroke = CueCommand(
            pose=setup.stroke_start_pose,
            linear_velocity=direction * impact.cue_speed,
            angular_velocity=np.zeros(3, dtype=np.float64),
            debug_label=f"{command.skill_id.value}:stroke",
        )
        stroke_start = CueCommand(
            pose=setup.stroke_start_pose,
            linear_velocity=np.zeros(3, dtype=np.float64),
            angular_velocity=np.zeros(3, dtype=np.float64),
            debug_label=f"{command.skill_id.value}:stroke_start",
        )
        follow_pose = Pose3D(
            position=setup.stroke_start_pose.position + direction * (self._shot_policy.stroke_backoff + self._shot_policy.follow_through),
            quat_wxyz=setup.stroke_start_pose.quat_wxyz,
        )
        follow = CueCommand(
            pose=follow_pose,
            linear_velocity=np.zeros(3, dtype=np.float64),
            angular_velocity=np.zeros(3, dtype=np.float64),
            debug_label=f"{command.skill_id.value}:follow_through",
        )
        setup_command = CueCommand(
            pose=setup.setup_pose,
            linear_velocity=np.zeros(3, dtype=np.float64),
            angular_velocity=np.zeros(3, dtype=np.float64),
            debug_label=f"{command.skill_id.value}:setup",
        )
        align_command = CueCommand(
            pose=setup.align_pose,
            linear_velocity=np.zeros(3, dtype=np.float64),
            angular_velocity=np.zeros(3, dtype=np.float64),
            debug_label=f"{command.skill_id.value}:align",
        )
        return StrokeTrajectoryPlan(commands=(setup_command, align_command, stroke_start, stroke, follow))


class MidLevelCurriculumEnv:
    """Small non-Gym adapter for stage-wise mid-level RL training."""

    def __init__(self, cue_env: MidLevelCueEnv | None = None) -> None:
        self.cue_env = cue_env or MidLevelCueEnv()
        self.scene_state = self.cue_env.reset()

    def reset(self, command: SkillCommand, stage: CurriculumStage) -> StageObservation:
        self.scene_state = self.cue_env.reset()
        return StageObservation(stage=stage, skill_id=command.skill_id, command=command, scene_state=self.scene_state)

    def evaluate_impact(self, impact: ImpactParameters) -> StageResult:
        speed_penalty = abs(float(impact.cue_speed))
        norm_error = abs(float(np.linalg.norm(impact.cue_direction)) - 1.0)
        reward = 1.0 - norm_error - 0.02 * speed_penalty
        return StageResult(
            reward=reward,
            terminated=True,
            metrics={"direction_norm_error": norm_error, "cue_speed": float(impact.cue_speed)},
        )

    def evaluate_setup(self, setup: CueSetupPlan) -> StageResult:
        setup_to_align = float(np.linalg.norm(setup.align_pose.position - setup.setup_pose.position))
        align_to_stroke = float(np.linalg.norm(setup.stroke_start_pose.position - setup.align_pose.position))
        reward = 1.0 - 0.1 * abs(setup_to_align) - 0.1 * abs(align_to_stroke)
        return StageResult(
            reward=reward,
            terminated=True,
            metrics={"setup_to_align": setup_to_align, "align_to_stroke": align_to_stroke},
        )

    def evaluate_stroke(self, stroke: StrokeTrajectoryPlan) -> StageResult:
        rollout = self.cue_env.execute(stroke.commands)
        object_speed = float(np.linalg.norm(rollout.object_ball_final_velocity))
        cue_speed = float(np.linalg.norm(rollout.cue_ball_final_velocity))
        contact_bonus = 1.0 if rollout.first_cue_ball_contact_time is not None else -1.0
        ball_contact_bonus = 1.0 if rollout.first_ball_ball_contact_time is not None else 0.0
        stability_penalty = 2.0 if rollout.has_nan or rollout.exploded else 0.0
        reward = contact_bonus + ball_contact_bonus + object_speed + 0.2 * cue_speed - stability_penalty
        return StageResult(
            reward=reward,
            terminated=True,
            metrics={
                "object_ball_speed": object_speed,
                "cue_ball_speed": cue_speed,
                "cue_ball_contact": float(rollout.first_cue_ball_contact_time is not None),
                "ball_ball_contact": float(rollout.first_ball_ball_contact_time is not None),
                "stable": float(not (rollout.has_nan or rollout.exploded)),
                "constraint_projection_count": float(rollout.constraint_projection_count),
                "min_cue_table_clearance": float(rollout.min_cue_table_clearance),
            },
            rollout=rollout,
        )
