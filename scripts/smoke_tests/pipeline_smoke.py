"""Smoke test for the hierarchical snooker policy pipeline."""

from __future__ import annotations

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.pipeline import build_default_pipeline
from snooker_env.pipeline_mid_level import default_skill_registry
from snooker_env.pipeline_types import BallState, SceneState, ShotIntent, SkillCommand, SkillId, TableState


def _demo_state() -> SceneState:
    return SceneState(
        time=0.0,
        table=TableState(),
        balls={
            "cue_ball": BallState(
                name="cue_ball",
                position=np.array([-0.45, 0.0, 0.82], dtype=np.float64),
            ),
            "object_ball_0": BallState(
                name="object_ball_0",
                position=np.array([0.15, 0.08, 0.82], dtype=np.float64),
            ),
        },
    )


def main() -> None:
    pipeline = build_default_pipeline()
    output = pipeline.plan_once(_demo_state())

    print("High-level skill sequence:")
    for idx, command in enumerate(output.skills):
        intent = command.intent
        print(
            f"  {idx}: {command.skill_id.value} "
            f"object={intent.object_ball_name} pocket={intent.target_pocket} "
            f"speed={intent.target_speed}"
        )

    print("\nMid-level cue command trajectories:")
    offset = 0
    for skill_idx, trajectory in enumerate(output.cue_trajectories):
        print(f"  skill {skill_idx}: commands={len(trajectory)}")
        for local_idx, command in enumerate(trajectory):
            print(
                f"    {offset + local_idx}: label={command.debug_label or '-'} "
                f"pose={np.round(command.pose.position, 4).tolist()} "
                f"vel={np.round(command.linear_velocity, 4).tolist()}"
            )
        offset += len(trajectory)

    print("\nLow-level joint actions:")
    for idx, action in enumerate(output.joint_actions):
        torque_norm = float(np.linalg.norm(action.torque_targets))
        print(f"  {idx}: joints={len(action.joint_names)} torque_norm={torque_norm:.4f}")

    if output.body_position is not None:
        base = output.body_position.base
        print(
            "\nBody positioning: "
            f"mode={output.body_position.requested_mode.value} "
            f"vxy={np.round(base.linear_velocity_xy, 4).tolist()} "
            f"wz={base.angular_velocity_z:.4f} "
            f"bend={output.body_position.bend_preparation:.2f}"
        )

    if output.recovery is not None:
        print(
            "Recovery: "
            f"mode={output.recovery.requested_mode.value} "
            f"settle_time={output.recovery.settle_time:.2f}"
        )

    if len(output.skills) != 1:
        raise RuntimeError(f"Expected 1 high-level skill call, got {len(output.skills)}")
    if len(output.cue_commands) < 4:
        raise RuntimeError(f"Expected at least 4 cue commands, got {len(output.cue_commands)}")
    if not output.joint_actions:
        raise RuntimeError("Expected low-level joint actions.")

    print("\nRegistered mid-level policies:")
    registry = default_skill_registry()
    demo_state = _demo_state()
    for skill_id in SkillId:
        skill = registry[skill_id]
        command = SkillCommand(
            skill_id=skill_id,
            intent=ShotIntent(cue_ball_name="cue_ball", object_ball_name="object_ball_0"),
        )
        trajectory = skill.rollout(command, demo_state)
        print(f"  {skill_id.value}: commands={len(trajectory)}")
        if not trajectory:
            raise RuntimeError(f"{skill_id.value} produced an empty trajectory.")


if __name__ == "__main__":
    main()
