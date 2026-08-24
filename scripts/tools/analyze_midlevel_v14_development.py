"""Aggregate v14 development runs and controls without opening fresh test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_two_ball import (  # noqa: E402
    MAX_ANGLE_RESIDUAL,
    MAX_CUE_SPEED,
    MIN_CUE_SPEED,
)


CONFIGURATIONS = ("canonical", "structured_reference", "structured_zero")
CONTROL_MODES = ("reference", "zero")
SEEDS = (0, 1, 2)
MINIMUM_ROBUST_JOINT_IMPROVEMENT = 0.0025


def detail_path(root: Path, configuration: str, seed: int) -> Path:
    if configuration == "canonical":
        name = f"midlevel_v14_canonical_208896_s{seed}.npz"
    elif configuration == "structured_reference":
        name = f"midlevel_v14_structured_208896_reference_s{seed}.npz"
    elif configuration == "structured_zero":
        name = f"midlevel_v14_structured_208896_zero_s{seed}.npz"
    else:
        raise ValueError(configuration)
    return root / name


def control_detail_path(root: Path, mode: str) -> Path:
    if mode not in CONTROL_MODES:
        raise ValueError(mode)
    return root / f"midlevel_v14_structured_control_{mode}.npz"


def load_details(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    metadata = json.loads(str(arrays.pop("metadata").item()))
    required = {
        "task_index",
        "pocket_index",
        "action",
        "correct_pot",
        "cue_scratch",
        "wrong_pocket",
        "stopped",
        "timed_out",
        "numerical_failure",
        "joint_success",
        "stop_error_m",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    return {"path": path, "metadata": metadata, **arrays}


def rates(details: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "task_count": int(len(details["task_index"])),
        "correct_pot_rate": float(np.mean(details["correct_pot"])),
        "joint_success_rate": float(np.mean(details["joint_success"])),
        "scratch_rate": float(np.mean(details["cue_scratch"])),
        "wrong_pocket_rate": float(np.mean(details["wrong_pocket"])),
        "stopped_rate": float(np.mean(details["stopped"])),
        "timeout_rate": float(np.mean(details["timed_out"])),
        "numerical_failure_rate": float(
            np.mean(details["numerical_failure"])
        ),
        "mean_stop_error_m": float(np.mean(details["stop_error_m"])),
    }
    per_pocket: dict[str, dict[str, float | int]] = {}
    for pocket in range(6):
        selected = details["pocket_index"] == pocket
        per_pocket[str(pocket)] = {
            "task_count": int(np.count_nonzero(selected)),
            "correct_pot_rate": float(np.mean(details["correct_pot"][selected])),
            "joint_success_rate": float(
                np.mean(details["joint_success"][selected])
            ),
            "scratch_rate": float(np.mean(details["cue_scratch"][selected])),
        }
    report["per_pocket"] = per_pocket
    return report


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def paired_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float | int]:
    if not np.array_equal(baseline["task_index"], candidate["task_index"]):
        raise ValueError("Paired development details have different task order.")
    baseline_success = baseline["joint_success"]
    candidate_success = candidate["joint_success"]
    paired_joint_delta = (
        candidate_success.astype(np.float64)
        - baseline_success.astype(np.float64)
    )
    paired_standard_error = float(
        np.std(paired_joint_delta, ddof=1) / np.sqrt(len(paired_joint_delta))
    )
    paired_mean = float(np.mean(paired_joint_delta))
    event_fields = (
        "correct_pot",
        "joint_success",
        "cue_scratch",
        "wrong_pocket",
        "stopped",
        "timed_out",
        "numerical_failure",
    )
    changed = np.zeros(len(baseline_success), dtype=np.bool_)
    for field in event_fields:
        changed |= baseline[field] != candidate[field]
    action_delta = candidate["action"] - baseline["action"]
    speed_half_range = 0.5 * (MAX_CUE_SPEED - MIN_CUE_SPEED)
    return {
        "joint_gain_count": int(
            np.count_nonzero(~baseline_success & candidate_success)
        ),
        "joint_loss_count": int(
            np.count_nonzero(baseline_success & ~candidate_success)
        ),
        "joint_net_count": int(
            np.count_nonzero(candidate_success)
            - np.count_nonzero(baseline_success)
        ),
        "joint_rate_difference": paired_mean,
        "joint_rate_difference_standard_error": paired_standard_error,
        "joint_rate_difference_lower_95": (
            paired_mean - 1.96 * paired_standard_error
        ),
        "joint_rate_difference_upper_95": (
            paired_mean + 1.96 * paired_standard_error
        ),
        "any_terminal_event_change_count": int(np.count_nonzero(changed)),
        "mean_absolute_angle_change_deg": float(
            np.mean(np.abs(action_delta[:, 0]))
            * np.rad2deg(MAX_ANGLE_RESIDUAL)
        ),
        "mean_absolute_speed_change_mps": float(
            np.mean(np.abs(action_delta[:, 1])) * speed_half_range
        ),
        "p95_absolute_speed_change_mps": float(
            np.percentile(np.abs(action_delta[:, 1]), 95) * speed_half_range
        ),
        "maximum_absolute_speed_change_mps": float(
            np.max(np.abs(action_delta[:, 1])) * speed_half_range
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--details-dir",
        type=Path,
        default=Path("outputs/evaluations"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluations/midlevel_v14_development_summary.json"),
    )
    args = parser.parse_args()

    runs: dict[str, dict[int, dict[str, Any]]] = {}
    development_hash: str | None = None
    task_index: np.ndarray | None = None
    for configuration in CONFIGURATIONS:
        runs[configuration] = {}
        for seed in SEEDS:
            details = load_details(detail_path(args.details_dir, configuration, seed))
            metadata = details["metadata"]
            if metadata.get("stochastic"):
                raise ValueError("Development selection requires deterministic runs.")
            if metadata.get("policy_device") != "cuda:0":
                raise ValueError(
                    "Development run did not use CUDA policy inference."
                )
            if int(metadata.get("parallel_num_envs", -1)) != 4096:
                raise ValueError(
                    "Development run was not evaluated in the fixed "
                    "4096-world/slot layout."
                )
            if int(metadata.get("chunk_steps", -1)) != 64 or int(
                metadata.get("check_interval_steps", -1)
            ) != 8192:
                raise ValueError("Development evaluation settings differ.")
            current_hash = str(metadata["task_library_content_sha256"])
            if development_hash is None:
                development_hash = current_hash
                task_index = details["task_index"]
            elif current_hash != development_hash or not np.array_equal(
                task_index,
                details["task_index"],
            ):
                raise ValueError("Development runs do not use one fixed task set.")
            runs[configuration][seed] = details

    controls: dict[str, dict[str, Any]] = {}
    for mode in CONTROL_MODES:
        details = load_details(control_detail_path(args.details_dir, mode))
        metadata = details["metadata"]
        if metadata.get("stochastic"):
            raise ValueError("Initializer control evaluation is stochastic.")
        if metadata.get("policy_device") != "cuda:0":
            raise ValueError(
                "Initializer control did not use CUDA policy inference."
            )
        if int(metadata.get("parallel_num_envs", -1)) != 4096:
            raise ValueError(
                "Initializer control was not evaluated in the fixed "
                "4096-world/slot layout."
            )
        if int(metadata.get("chunk_steps", -1)) != 64 or int(
            metadata.get("check_interval_steps", -1)
        ) != 8192:
            raise ValueError("Initializer control evaluation settings differ.")
        if str(metadata["task_library_content_sha256"]) != development_hash or not (
            np.array_equal(task_index, details["task_index"])
        ):
            raise ValueError("Initializer control used a different development set.")
        controls[mode] = details

    run_reports = {
        configuration: {
            str(seed): {
                **rates(runs[configuration][seed]),
                "checkpoint": runs[configuration][seed]["metadata"]["checkpoint"],
            }
            for seed in SEEDS
        }
        for configuration in CONFIGURATIONS
    }
    control_reports = {
        mode: {
            **rates(controls[mode]),
            "speed_action_checkpoint": controls[mode]["metadata"].get(
                "speed_action_checkpoint"
            ),
            "angle_action_source": controls[mode]["metadata"].get(
                "angle_action_source"
            ),
        }
        for mode in CONTROL_MODES
    }
    group_reports: dict[str, dict[str, Any]] = {}
    for configuration in CONFIGURATIONS:
        group_reports[configuration] = {}
        for metric in (
            "correct_pot_rate",
            "joint_success_rate",
            "scratch_rate",
            "wrong_pocket_rate",
        ):
            group_reports[configuration][metric] = distribution(
                [
                    float(run_reports[configuration][str(seed)][metric])
                    for seed in SEEDS
                ]
            )

    paired = {
        "structured_reference_vs_canonical": {
            str(seed): paired_comparison(
                runs["canonical"][seed],
                runs["structured_reference"][seed],
            )
            for seed in SEEDS
        },
        "structured_zero_vs_canonical": {
            str(seed): paired_comparison(
                runs["canonical"][seed],
                runs["structured_zero"][seed],
            )
            for seed in SEEDS
        },
        "zero_vs_reference_controlled_angle_ab": {
            str(seed): paired_comparison(
                runs["structured_reference"][seed],
                runs["structured_zero"][seed],
            )
            for seed in SEEDS
        },
        "structured_reference_vs_untrained_reference_control": {
            str(seed): paired_comparison(
                controls["reference"],
                runs["structured_reference"][seed],
            )
            for seed in SEEDS
        },
        "structured_zero_vs_untrained_zero_control": {
            str(seed): paired_comparison(
                controls["zero"],
                runs["structured_zero"][seed],
            )
            for seed in SEEDS
        },
        "untrained_zero_vs_reference_controlled_angle_ab": paired_comparison(
            controls["reference"],
            controls["zero"],
        ),
    }

    # The exact-zero experiment is intended to isolate angle behavior.  Its
    # speed Actor is trained with the same seed, examples, ordering, and loss
    # as the reference-angle variant.  Abort selection if numerical drift has
    # made speed a confounder, or if a supposedly zero-angle Actor is nonzero.
    controlled_angle_ab: dict[str, dict[str, float | bool]] = {}
    for seed in SEEDS:
        reference_action = runs["structured_reference"][seed]["action"]
        zero_action = runs["structured_zero"][seed]["action"]
        maximum_speed_delta = float(
            np.max(np.abs(reference_action[:, 1] - zero_action[:, 1]))
        )
        maximum_zero_angle = float(np.max(np.abs(zero_action[:, 0])))
        speed_control_passed = maximum_speed_delta <= 1.0e-6
        zero_angle_passed = maximum_zero_angle == 0.0
        controlled_angle_ab[str(seed)] = {
            "maximum_normalized_speed_delta": maximum_speed_delta,
            "maximum_zero_variant_angle_action": maximum_zero_angle,
            "speed_control_passed": speed_control_passed,
            "zero_angle_passed": zero_angle_passed,
        }
        if not speed_control_passed or not zero_angle_passed:
            raise RuntimeError(
                "Reference/zero angle A/B is confounded for seed "
                f"{seed}: {controlled_angle_ab[str(seed)]}"
            )
    control_speed_delta = float(
        np.max(
            np.abs(
                controls["reference"]["action"][:, 1]
                - controls["zero"]["action"][:, 1]
            )
        )
    )
    control_zero_angle = float(
        np.max(np.abs(controls["zero"]["action"][:, 0]))
    )
    if control_speed_delta != 0.0 or control_zero_angle != 0.0:
        raise RuntimeError(
            "Untrained angle controls are not exact: "
            f"speed_delta={control_speed_delta}, zero_angle={control_zero_angle}."
        )

    eligible = []
    for configuration in CONFIGURATIONS:
        group = group_reports[configuration]
        if (
            group["correct_pot_rate"]["mean"] >= 0.90
            and group["scratch_rate"]["mean"] <= 0.02
            and group["wrong_pocket_rate"]["mean"] <= 0.01
        ):
            eligible.append(configuration)
    if not eligible:
        raise RuntimeError("No v14 configuration meets the safety constraints.")
    raw_best = max(
        eligible,
        key=lambda name: group_reports[name]["joint_success_rate"]["mean"],
    )
    selected_configuration = raw_best
    selection_reason = "highest_three_seed_mean_joint_success"
    if raw_best != "canonical" and "canonical" in eligible:
        improvement = (
            group_reports[raw_best]["joint_success_rate"]["mean"]
            - group_reports["canonical"]["joint_success_rate"]["mean"]
        )
        pair_name = f"{raw_best}_vs_canonical"
        seed_win_count = sum(
            int(paired[pair_name][str(seed)]["joint_net_count"] > 0)
            for seed in SEEDS
        )
        if (
            improvement < MINIMUM_ROBUST_JOINT_IMPROVEMENT
            or seed_win_count < 2
        ):
            selected_configuration = "canonical"
            selection_reason = (
                "structured_gain_below_0.25pp_or_not_positive_in_two_seeds"
            )
    selected_seed = max(
        SEEDS,
        key=lambda seed: float(
            run_reports[selected_configuration][str(seed)]["joint_success_rate"]
        ),
    )
    selected_checkpoint = run_reports[selected_configuration][str(selected_seed)][
        "checkpoint"
    ]
    best_control_mode = max(
        CONTROL_MODES,
        key=lambda mode: float(control_reports[mode]["joint_success_rate"]),
    )
    best_control_joint = float(
        control_reports[best_control_mode]["joint_success_rate"]
    )
    provisional_joint = float(
        run_reports[selected_configuration][str(selected_seed)][
            "joint_success_rate"
        ]
    )
    report = {
        "version": "midlevel-v14-development-selection-v2",
        "scope": "development_only_fresh_test_not_evaluated",
        "development_task_library_content_sha256": development_hash,
        "minimum_robust_joint_improvement": (
            MINIMUM_ROBUST_JOINT_IMPROVEMENT
        ),
        "runs": run_reports,
        "untrained_initializer_controls": control_reports,
        "three_seed_groups": group_reports,
        "paired_comparisons": paired,
        "controlled_angle_ab_validation": controlled_angle_ab,
        "initializer_control_advisory": {
            "best_mode": best_control_mode,
            "joint_success_rate": best_control_joint,
            "difference_from_provisional_selected_run": (
                best_control_joint - provisional_joint
            ),
            "outperforms_provisional_selected_run": (
                best_control_joint > provisional_joint
            ),
            "selection_status": (
                "diagnostic_only_requires_deployable_composed_checkpoint"
            ),
        },
        "provisional_selection": {
            "configuration": selected_configuration,
            "seed": selected_seed,
            "checkpoint": selected_checkpoint,
            "reason": selection_reason,
            "requires_review_before_one_time_test": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
