"""Check terminal reward gating and the required outcome ordering."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_ppo_env import compute_terminal_reward  # noqa: E402
from snooker_env.midlevel_two_ball import TwoBallShotResult  # noqa: E402


def main() -> None:
    target = np.array([0.1, -0.2], dtype=np.float64)
    base = TwoBallShotResult(
        target_pocket="pocket_middle_posx",
        shot_direction=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        cue_speed=0.8,
        elapsed_time=2.0,
        cue_ball_final_position=np.array([target[0], target[1], 1.0785]),
        object_ball_final_position=np.array([0.72, 0.0, 1.0]),
        first_ball_contact_time=0.2,
        first_cushion_contact_time=None,
        object_pocket="pocket_middle_posx",
        cue_pocket=None,
        min_object_pocket_distance=0.0,
        initial_object_pocket_distance=0.5,
        stopped=True,
        timed_out=False,
        numerical_failure=False,
        cushion_before_object=False,
        object_cushion_before_pocket=False,
        any_cushion_contact=False,
        contact_events=(),
    )
    good = compute_terminal_reward(base, target)
    bad_position = compute_terminal_reward(
        replace(
            base,
            cue_ball_final_position=np.array([target[0] + 0.25, target[1], 1.0785]),
        ),
        target,
    )
    close_miss = compute_terminal_reward(
        replace(base, object_pocket=None, min_object_pocket_distance=0.01), target
    )
    wrong_pocket = compute_terminal_reward(
        replace(base, object_pocket="pocket_middle_negx"), target
    )
    scratch = compute_terminal_reward(
        replace(base, cue_pocket="pocket_middle_posx", stopped=False), target
    )

    print(
        f"good={good.total:.4f} bad_position={bad_position.total:.4f} "
        f"close_miss={close_miss.total:.4f} wrong={wrong_pocket.total:.4f} "
        f"scratch={scratch.total:.4f}"
    )
    if not good.total > bad_position.total > close_miss.total > wrong_pocket.total:
        raise RuntimeError("Terminal reward ordering is incorrect.")
    if scratch.total >= close_miss.total:
        raise RuntimeError("A scratch must score below a near-pocket miss.")
    if not good.joint_success or bad_position.joint_success:
        raise RuntimeError("Joint-success gating does not use the 5 cm stop threshold.")
    if close_miss.position_reward != 0.0:
        raise RuntimeError("Position reward leaked into a non-pot outcome.")


if __name__ == "__main__":
    main()
