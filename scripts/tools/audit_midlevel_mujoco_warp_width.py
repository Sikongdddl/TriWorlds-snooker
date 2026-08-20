"""Compare a canonical 4096-world rollout with a narrower MJWarp width."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from audit_midlevel_mujoco_warp_slots import (  # noqa: E402
    _comparison_report,
    _execute,
    _outcome_report,
)
from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.midlevel_mujoco_warp_vec_env import (  # noqa: E402
    MJWarpMidLevelVecEnv,
)
from snooker_env.midlevel_tasks import TwoBallTaskDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "outputs/tasks/midlevel_two_ball_train.npz",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument(
        "--reference",
        type=Path,
        default=(
            ROOT / "outputs/diagnostics/midlevel_world_slot_audit.npz"
        ),
    )
    parser.add_argument("--physics-device", default="cuda:0")
    parser.add_argument("--num-worlds", type=int, default=1024)
    parser.add_argument("--chunk-steps", type=int, default=64)
    parser.add_argument("--check-interval-steps", type=int, default=8192)
    parser.add_argument("--max-shot-time", type=float, default=8.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT / "outputs/diagnostics/midlevel_world_width_audit.json"
        ),
    )
    args = parser.parse_args()

    if args.num_worlds <= 0:
        raise ValueError("--num-worlds must be positive.")
    dataset = TwoBallTaskDataset.load(args.tasks, validate_model=False)
    with np.load(args.reference, allow_pickle=False) as archive:
        task_indices = np.asarray(
            archive["canonical_task_indices"],
            dtype=np.int64,
        )
        actions = np.asarray(archive["canonical_actions"], dtype=np.float32)
        prefix = "canonical_repeat_0__"
        reference = {
            name[len(prefix) :]: np.asarray(archive[name])
            for name in archive.files
            if name.startswith(prefix)
        }
    if len(task_indices) % args.num_worlds != 0:
        raise ValueError(
            "Reference task count must be divisible by the comparison width."
        )
    if actions.shape != (len(task_indices), 2):
        raise ValueError("Reference actions do not match its task indices.")

    environment = MJWarpMidLevelVecEnv(
        dataset,
        args.model,
        num_envs=args.num_worlds,
        device=args.physics_device,
        chunk_steps=args.chunk_steps,
        check_interval_steps=args.check_interval_steps,
        max_time=args.max_shot_time,
    )
    batches: list[dict[str, np.ndarray]] = []
    try:
        for start in range(0, len(task_indices), args.num_worlds):
            stop = start + args.num_worlds
            print(
                f"width_audit={args.num_worlds} tasks={start}:{stop}",
                flush=True,
            )
            batches.append(
                _execute(
                    environment,
                    task_indices[start:stop],
                    actions[start:stop],
                )
            )
    finally:
        environment.close()
    candidate = {
        name: np.concatenate([batch[name] for batch in batches], axis=0)
        for name in batches[0]
    }
    comparison = _comparison_report(reference, candidate)
    passed = bool(
        comparison["cue_final_delta_max_m"] <= 1.0e-7
        and comparison["object_final_delta_max_m"] <= 1.0e-7
        and comparison["reward_delta_max"] <= 1.0e-7
        and comparison["any_outcome_flag_mismatch_rate"] == 0.0
    )
    report = {
        "audit_version": "mujoco-warp-world-width-v1",
        "reference": str(args.reference),
        "reference_num_worlds": int(len(task_indices)),
        "comparison_num_worlds": args.num_worlds,
        "task_count": int(len(task_indices)),
        "comparison": comparison,
        "outcome": _outcome_report(candidate),
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        **{f"comparison__{name}": value for name, value in candidate.items()},
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    if not passed:
        raise RuntimeError("MJWarp result changed with parallel world width.")


if __name__ == "__main__":
    main()
