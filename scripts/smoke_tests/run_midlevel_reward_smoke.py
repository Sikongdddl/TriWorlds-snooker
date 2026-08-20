"""Check terminal reward gating and the required outcome ordering."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_ppo_env import (  # noqa: E402
    MAX_TERMINAL_REWARD,
    compute_terminal_reward,
)
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
        replace(
            base,
            object_pocket=None,
            object_ball_final_position=np.array([0.82, 0.0, 1.0]),
            min_object_pocket_distance=0.01,
        ),
        target,
    )
    wrong_pocket = compute_terminal_reward(
        replace(base, object_pocket="pocket_middle_negx"), target
    )
    scratch = compute_terminal_reward(
        replace(base, cue_pocket="pocket_middle_posx", stopped=False),
        target,
    )
    timeout = compute_terminal_reward(
        replace(
            base,
            object_pocket=None,
            stopped=False,
            timed_out=True,
            object_ball_final_position=np.array([0.72, 0.0, 1.0]),
        ),
        target,
    )

    print(
        f"good={good.total:.4f} bad_position={bad_position.total:.4f} "
        f"close_miss={close_miss.total:.4f} wrong={wrong_pocket.total:.4f} "
        f"scratch={scratch.total:.4f}"
    )
    if not 0.0 <= close_miss.object_ball_reward < 1.0:
        raise RuntimeError("Object-ball distance reward is outside [0, 1).")
    if good.object_ball_reward != 1.0 or wrong_pocket.object_ball_reward != 0.0:
        raise RuntimeError("Only the requested pocket may receive the pot reward.")
    if not good.cue_position_reward > bad_position.cue_position_reward > 0.0:
        raise RuntimeError("Cue-position reward does not decrease with distance.")
    if close_miss.cue_position_reward != 0.0:
        raise RuntimeError("Cue-position reward leaked into a non-pot outcome.")
    if scratch.total != 0.0:
        raise RuntimeError("A scratch must override all reward components to zero.")
    if timeout.total != 0.0:
        raise RuntimeError("A timed-out moving state must not be treated as a stop point.")
    if not good.joint_success or bad_position.joint_success:
        raise RuntimeError("Joint-success gating does not use the 5 cm stop threshold.")
    if good.total != MAX_TERMINAL_REWARD:
        raise RuntimeError("An exact joint success does not receive the maximum reward.")
    if good.joint_success_bonus <= 0.0 or bad_position.joint_success_bonus != 0.0:
        raise RuntimeError("The 5 cm joint-success bonus is not gated correctly.")


if __name__ == "__main__":
    main()
