"""Check that held-out physical outcomes gate Critic-guided Actor updates."""

from __future__ import annotations

import sys

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from train_midlevel_two_ball_sac_her import critic_actor_gate_report  # noqa: E402


def report(diagnostics: dict[str, object]) -> dict[str, object]:
    return critic_actor_gate_report(
        diagnostics,
        minimum_pairwise_agreement=0.0,
        minimum_candidate_count=128,
        minimum_improvement_precision=0.75,
        minimum_improvement_precision_lower_95=0.65,
        minimum_reward_improvement=0.002,
        minimum_safe_improvement=0.0,
        minimum_joint_success_improvement=0.0,
        maximum_failure_increase=0.0,
    )


def main() -> None:
    accepted = {
        "pairwise_both_critics_agreement": 0.72,
        "candidate_nonzero_selection_count": 256,
        "candidate_nonzero_true_improvement_precision": 0.80,
        "candidate_nonzero_true_improvement_precision_lower_95": 0.74,
        "candidate_selected_physical_reward_improvement_mean": 0.02,
        "candidate_selected_physical_safe_improvement_mean": 0.01,
        "candidate_selected_physical_joint_success_improvement_mean": 0.03,
        "candidate_selected_physical_failure_increase_mean": -0.01,
    }
    rejected = dict(accepted)
    rejected.update(
        candidate_nonzero_true_improvement_precision=0.45,
        candidate_nonzero_true_improvement_precision_lower_95=0.40,
        candidate_selected_physical_joint_success_improvement_mean=-0.001,
    )
    accepted_report = report(accepted)
    rejected_report = report(rejected)
    if not bool(accepted_report["passed"]):
        raise RuntimeError("A physically beneficial Critic was rejected.")
    if bool(rejected_report["passed"]):
        raise RuntimeError("An unreliable Critic was allowed to update the Actor.")
    rejected_checks = rejected_report["checks"]
    expected_failures = {
        "candidate_nonzero_true_improvement_precision",
        "candidate_nonzero_true_improvement_precision_lower_95",
        "candidate_selected_physical_joint_success_improvement_mean",
    }
    actual_failures = {
        name
        for name, check in rejected_checks.items()
        if not bool(check["passed"])
    }
    if actual_failures != expected_failures:
        raise RuntimeError("Critic-to-Actor gate reported unexpected failures.")
    print(
        "accepted=True rejected=True "
        "coverage_precision_reward_safety_joint_failure_checked=True"
    )


if __name__ == "__main__":
    main()
