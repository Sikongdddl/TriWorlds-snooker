"""Select expanded speed under one fixed reference angle on development only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--details-dir",
        type=Path,
        default=Path("outputs/evaluations"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        with args.output.open(encoding="utf-8") as source:
            report = json.load(source)
        print(json.dumps(report, sort_keys=True))
        return

    candidates = []
    development_hash: str | None = None
    for seed in (0, 1, 2):
        path = (
            args.details_dir
            / f"midlevel_v14_speed_candidate_reference_angle_s{seed}.npz"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            correct_pot = np.asarray(archive["correct_pot"], dtype=np.bool_)
            joint_success = np.asarray(
                archive["joint_success"],
                dtype=np.bool_,
            )
            scratch = np.asarray(archive["cue_scratch"], dtype=np.bool_)
            wrong = np.asarray(archive["wrong_pocket"], dtype=np.bool_)
        if metadata.get("stochastic"):
            raise ValueError("Speed-candidate evaluation is stochastic.")
        if metadata.get("policy_device") != "cuda:0":
            raise ValueError(
                "Speed candidate was not evaluated with CUDA policy inference."
            )
        if int(metadata.get("parallel_num_envs", -1)) != 4096:
            raise ValueError(
                "Speed candidate was not evaluated in its fixed "
                "4096-world/slot layout."
            )
        if int(metadata.get("chunk_steps", -1)) != 64 or int(
            metadata.get("check_interval_steps", -1)
        ) != 8192:
            raise ValueError("Speed-candidate evaluation settings differ.")
        expected_angle = (
            "outputs/checkpoints/"
            "midlevel_two_ball_td3_her_v10_canonical_e800_b2048_s2."
            "bc_only.zip"
        )
        if metadata.get("angle_action_source") != expected_angle:
            raise ValueError(
                "Speed candidates were not evaluated with the fixed best-BC angle."
            )
        if metadata.get("speed_action_checkpoint") != metadata.get("checkpoint"):
            raise ValueError("Speed-candidate checkpoint metadata is inconsistent.")
        current_hash = str(metadata["task_library_content_sha256"])
        if development_hash is None:
            development_hash = current_hash
        elif current_hash != development_hash:
            raise ValueError("Speed candidates used different development sets.")
        checkpoint = Path(str(metadata["checkpoint"]))
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        metrics = {
            "seed": seed,
            "checkpoint": str(checkpoint),
            "correct_pot_rate": float(np.mean(correct_pot)),
            "joint_success_rate": float(np.mean(joint_success)),
            "scratch_rate": float(np.mean(scratch)),
            "wrong_pocket_rate": float(np.mean(wrong)),
        }
        metrics["safety_eligible"] = bool(
            metrics["correct_pot_rate"] >= 0.90
            and metrics["scratch_rate"] <= 0.02
            and metrics["wrong_pocket_rate"] <= 0.01
        )
        candidates.append(metrics)
    eligible = [candidate for candidate in candidates if candidate["safety_eligible"]]
    if not eligible:
        raise RuntimeError("No fixed-angle expanded speed candidate is safety-eligible.")
    selected = max(
        eligible,
        key=lambda candidate: (
            candidate["joint_success_rate"],
            candidate["correct_pot_rate"],
            -candidate["scratch_rate"],
        ),
    )
    report = {
        "version": "midlevel-v14-fixed-angle-speed-initializer-v2",
        "scope": "development_only_fresh_test_not_evaluated",
        "development_task_library_content_sha256": development_hash,
        "candidates": candidates,
        "selected_seed": selected["seed"],
        "selected_checkpoint": selected["checkpoint"],
        "selection_metric": (
            "fixed_reference_angle_joint_success_then_pot_then_lower_scratch"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
