"""Smoke test the three-stage mid-level curriculum interface."""

from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_rl import (  # noqa: E402
    CurriculumStage,
    MidLevelCurriculumEnv,
    ScriptedMidLevelCurriculumPolicy,
    StageObservation,
)
from snooker_env.pipeline_types import ShotIntent, SkillCommand, SkillId  # noqa: E402


def main() -> None:
    command = SkillCommand(
        skill_id=SkillId.POT_SHOT,
        intent=ShotIntent(
            cue_ball_name="cue_ball",
            object_ball_name="object_ball_0",
            target_pocket="demo_forward",
            target_speed=1.1,
        ),
    )
    env = MidLevelCurriculumEnv()
    policy = ScriptedMidLevelCurriculumPolicy(skill_id=SkillId.POT_SHOT)

    impact_obs = env.reset(command, CurriculumStage.IMPACT_PARAMETER_INFERENCE)
    impact = policy.infer_impact(command, impact_obs.scene_state)
    impact_result = env.evaluate_impact(impact)

    setup_obs = StageObservation(
        stage=CurriculumStage.CUE_SETUP_TRAJECTORY_GENERATION,
        skill_id=command.skill_id,
        command=command,
        scene_state=impact_obs.scene_state,
        impact=impact,
    )
    setup = policy.generate_setup(command, setup_obs.scene_state, impact)
    setup_result = env.evaluate_setup(setup)

    stroke_obs = StageObservation(
        stage=CurriculumStage.STROKE_TRAJECTORY_GENERATION,
        skill_id=command.skill_id,
        command=command,
        scene_state=setup_obs.scene_state,
        impact=impact,
        setup=setup,
    )
    stroke = policy.generate_stroke(command, impact, setup)
    stroke_result = env.evaluate_stroke(stroke)

    print(f"impact stage reward: {impact_result.reward:.4f} metrics={dict(impact_result.metrics)}")
    print(f"setup stage reward: {setup_result.reward:.4f} metrics={dict(setup_result.metrics)}")
    print(f"stroke stage reward: {stroke_result.reward:.4f} metrics={dict(stroke_result.metrics)}")
    print(f"stroke commands: {len(stroke.commands)}")

    if stroke_result.rollout is None:
        raise RuntimeError("Stroke stage did not produce a rollout result.")
    if stroke_result.rollout.first_cue_ball_contact_time is None:
        raise RuntimeError("Curriculum smoke failed: cue did not contact cue ball.")
    if stroke_result.rollout.has_nan or stroke_result.rollout.exploded:
        raise RuntimeError("Curriculum smoke failed: unstable simulation.")


if __name__ == "__main__":
    main()
