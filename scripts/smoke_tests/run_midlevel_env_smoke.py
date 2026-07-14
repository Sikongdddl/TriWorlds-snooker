"""Run a semantic mid-level shot policy in the two-ball cue environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL, MidLevelCueEnv
from snooker_env.pipeline_mid_level import PotShotPolicy
from snooker_env.pipeline_types import ShotIntent, SkillCommand, SkillId


def main() -> None:
    env = MidLevelCueEnv(DEFAULT_MIDLEVEL_MODEL)
    state = env.reset()
    command = SkillCommand(
        skill_id=SkillId.POT_SHOT,
        intent=ShotIntent(
            cue_ball_name="cue_ball",
            object_ball_name="object_ball_0",
            target_pocket="demo_forward",
            target_speed=1.1,
        ),
    )
    cue_commands = PotShotPolicy().rollout(command, state)
    result = env.execute(cue_commands)

    print(f"model={Path(DEFAULT_MIDLEVEL_MODEL)}")
    print(f"commands={len(cue_commands)}")
    print(f"action_repeat={env.action_repeat} command_dt={env.command_dt:.4f}s")
    for idx, cue_command in enumerate(cue_commands):
        print(
            f"  {idx}: label={cue_command.debug_label or '-'} "
            f"pose={np.round(cue_command.pose.position, 4).tolist()} "
            f"vel={np.round(cue_command.linear_velocity, 4).tolist()}"
        )
    print(f"first cue_tip/cue_ball contact: {result.first_cue_ball_contact_time}")
    print(f"first cue_ball/object_ball contact: {result.first_ball_ball_contact_time}")
    print(f"constraint projection count: {result.constraint_projection_count}")
    print(f"minimum cue/table clearance: {result.min_cue_table_clearance}")
    print(f"cue ball final pos: {result.cue_ball_final_position}")
    print(f"cue ball final vel: {result.cue_ball_final_velocity}")
    print(f"object ball final pos: {result.object_ball_final_position}")
    print(f"object ball final vel: {result.object_ball_final_velocity}")
    print(f"NaN/Inf detected: {result.has_nan}")
    print(f"Numerical explosion detected: {result.exploded}")

    if result.first_cue_ball_contact_time is None:
        raise RuntimeError("Midlevel smoke failed: cue did not contact cue ball.")
    if np.linalg.norm(result.cue_ball_final_velocity) < 1e-3:
        raise RuntimeError("Midlevel smoke failed: cue ball did not move.")
    if result.has_nan or result.exploded:
        raise RuntimeError("Midlevel smoke failed: unstable simulation.")
    if result.min_cue_table_clearance < -1e-6:
        raise RuntimeError("Midlevel smoke failed: cue trajectory penetrated table/cushion proxies after projection.")


if __name__ == "__main__":
    main()
