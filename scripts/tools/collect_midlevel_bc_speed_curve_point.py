"""Collect one independently persisted BC-centered speed-curve point."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from collect_midlevel_bc_speed_curves import _run_curve_point  # noqa: E402
from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.midlevel_mujoco_warp_vec_env import (  # noqa: E402
    MJWarpMidLevelVecEnv,
)
from snooker_env.midlevel_ppo import generated_behavior_cloning_data  # noqa: E402
from snooker_env.midlevel_sac_her import SingleStepTD3BC  # noqa: E402
from snooker_env.midlevel_tasks import TwoBallTaskDataset  # noqa: E402
from snooker_env.midlevel_two_ball import (  # noqa: E402
    MAX_CUE_SPEED,
    MIN_CUE_SPEED,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--offset-mps", type=float, required=True)
    parser.add_argument("--batch-start", type=int, required=True)
    parser.add_argument("--num-worlds", type=int, default=4096)
    parser.add_argument("--physics-device", default="cuda:0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-steps", type=int, default=64)
    parser.add_argument("--check-interval-steps", type=int, default=8192)
    parser.add_argument("--max-shot-time", type=float, default=8.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not np.isfinite(args.offset_mps):
        raise ValueError("--offset-mps must be finite.")
    if args.batch_start < 0 or args.batch_start % args.num_worlds != 0:
        raise ValueError("--batch-start must be an execution-batch boundary.")
    dataset = TwoBallTaskDataset.load(args.tasks, validate_model=False)
    stop = args.batch_start + args.num_worlds
    if stop > len(dataset):
        raise ValueError("Curve point requires a complete task batch.")
    task_indices = np.arange(args.batch_start, stop, dtype=np.int64)
    observations, generated_actions = generated_behavior_cloning_data(dataset)
    observations = observations[task_indices]
    generated_actions = generated_actions[task_indices].copy()
    generated_actions[:, 0] = 0.0
    policy = SingleStepTD3BC.load(args.checkpoint, device=args.device)
    bc_actions, _ = policy.predict(observations, deterministic=True)
    bc_actions = np.asarray(bc_actions, dtype=np.float32)
    bc_actions[:, 0] = 0.0
    speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    actions = bc_actions.copy()
    actions[:, 1] += np.float32(args.offset_mps / speed_half_range)
    if np.any(actions[:, 1] < -1.0) or np.any(actions[:, 1] > 1.0):
        raise ValueError("Curve point exceeds the physical action range.")

    environment = MJWarpMidLevelVecEnv(
        dataset,
        args.model,
        num_envs=args.num_worlds,
        device=args.physics_device,
        chunk_steps=args.chunk_steps,
        check_interval_steps=args.check_interval_steps,
        max_time=args.max_shot_time,
    )
    try:
        print(f"speed_curve_offset_mps={args.offset_mps:+.3f}", flush=True)
        point = _run_curve_point(environment, task_indices, actions)
    finally:
        environment.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        checkpoint=np.asarray(str(args.checkpoint)),
        task_library=np.asarray(str(args.tasks)),
        batch_start=np.asarray(args.batch_start, dtype=np.int64),
        num_worlds=np.asarray(args.num_worlds, dtype=np.int64),
        offset_mps=np.asarray(args.offset_mps, dtype=np.float64),
        task_indices=task_indices,
        bc_actions=bc_actions,
        generated_actions=generated_actions,
        **point,
    )
    print(f"curve_point_output={args.output}", flush=True)


if __name__ == "__main__":
    main()
