"""Check pure supervised hindsight BC, pocket heads, and controlled angle A/B."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from _bootstrap import add_src_to_path

add_src_to_path()

from run_midlevel_offline_curve_td3_smoke import (  # noqa: E402
    _OneStepEnv,
    _synthetic_curves,
)
from snooker_env.midlevel_sac_her import (  # noqa: E402
    ConservativeResidualTD3Policy,
    MidLevelGeometricFeatures,
    SingleStepTD3BC,
    StructuredSpeedTD3Policy,
    behavior_clone_structured_speed_policy,
)


def _model_kwargs() -> dict[str, object]:
    return {
        "device": "cpu",
        "verbose": 0,
        "learning_rate": 3.0e-4,
        "actor_candidate_supervision_weight": 0.0,
        "actor_physical_probe_supervision_weight": 0.0,
        "buffer_size": 1,
        "learning_starts": 0,
        "batch_size": 1,
        "gamma": 0.0,
        "train_freq": (1, "step"),
        "gradient_steps": 0,
    }


def main() -> None:
    curves = _synthetic_curves()
    legal_sensitivity = curves.legal_local_sensitivity_m_per_mps()
    altered_cue_final = curves.cue_final.copy()
    altered_cue_final[~curves.safe, :2] += 100.0
    altered_curves = replace(curves, cue_final=altered_cue_final)
    if not np.array_equal(
        legal_sensitivity,
        altered_curves.legal_local_sensitivity_m_per_mps(),
    ):
        raise RuntimeError("Failed curve endpoints leaked into legal sensitivity.")
    if not np.all(np.isfinite(legal_sensitivity)):
        raise RuntimeError("Legal local sensitivity contains non-finite values.")
    environment = DummyVecEnv([_OneStepEnv])
    set_random_seed(301)
    source = SingleStepTD3BC(
        ConservativeResidualTD3Policy,
        environment,
        seed=301,
        policy_kwargs={
            "net_arch": [32, 32],
            "features_extractor_class": MidLevelGeometricFeatures,
        },
        **_model_kwargs(),
    )
    source_state = {
        name: value.detach().clone()
        for name, value in source.policy.actor.state_dict().items()
    }

    def structured(angle_mode: str) -> SingleStepTD3BC:
        set_random_seed(302)
        model = SingleStepTD3BC(
            StructuredSpeedTD3Policy,
            environment,
            seed=302,
            policy_kwargs={
                "net_arch": [32, 32],
                "features_extractor_class": MidLevelGeometricFeatures,
                "angle_mode": angle_mode,
                "pocket_head_count": 6,
                "angle_reference_net_arch": (32, 32),
            },
            **_model_kwargs(),
        )
        model.policy.install_angle_reference(source_state)
        model.policy.install_speed_reference(source_state)
        return model

    reference = structured("reference")
    zero = structured("zero")
    source_actions, _ = source.predict(curves.observation, deterministic=True)
    reference_initial, _ = reference.predict(
        curves.observation,
        deterministic=True,
    )
    zero_initial, _ = zero.predict(curves.observation, deterministic=True)
    if not np.array_equal(reference_initial[:, 0], source_actions[:, 0]):
        raise RuntimeError("Structured Actor changed its frozen reference angle.")
    if not np.allclose(
        reference_initial[:, 1],
        source_actions[:, 1],
        atol=5.0e-7,
        rtol=0.0,
    ):
        raise RuntimeError("Pocket heads did not reproduce their speed initializer.")
    if np.any(zero_initial[:, 0] != 0.0) or not np.array_equal(
        reference_initial[:, 1],
        zero_initial[:, 1],
    ):
        raise RuntimeError("Initial angle A/B changed the paired speed output.")
    frozen_states = []
    for model in (reference, zero):
        frozen_states.append(
            {
                name: value.detach().clone()
                for name, value in model.policy.actor.state_dict().items()
                if name.startswith("features_extractor.")
                or name.startswith("speed_trunk.")
            }
        )

    reports = []
    for model in (reference, zero):
        reports.append(
            behavior_clone_structured_speed_policy(
                model.policy,
                curves,
                epochs=2,
                batch_size=64,
                learning_rate=3.0e-4,
                final_learning_rate=1.0e-4,
                speed_weight=1.0,
                speed_error_scale_mps=0.03,
                canonical_anchor_weight=4.0,
                middle_pocket_weight=2.0,
                sensitivity_loss_weight=0.25,
                freeze_speed_trunk=True,
                seed=303,
            )
        )
    expected_legal = int(np.count_nonzero(curves.safe))
    if any(report.legal_curve_sample_count != expected_legal for report in reports):
        raise RuntimeError("Structured BC skipped a legal curve outcome.")
    reference_final, _ = reference.predict(
        curves.observation,
        deterministic=True,
    )
    zero_final, _ = zero.predict(curves.observation, deterministic=True)
    if not np.array_equal(reference_final[:, 1], zero_final[:, 1]):
        raise RuntimeError("Paired supervised training changed A/B speed output.")
    if np.any(zero_final[:, 0] != 0.0):
        raise RuntimeError("Zero-angle A/B emitted a nonzero angle.")
    for model, expected_state in zip(
        (reference, zero),
        frozen_states,
        strict=True,
    ):
        actual_state = model.policy.actor.state_dict()
        if any(
            not value.equal(actual_state[name])
            for name, value in expected_state.items()
        ):
            raise RuntimeError("Frozen speed feature/trunk parameters changed.")
    if not all(report.speed_trunk_frozen for report in reports):
        raise RuntimeError("Structured BC report did not record its frozen trunk.")

    with tempfile.TemporaryDirectory(prefix="structured-speed-bc-") as directory:
        checkpoint = Path(directory) / "model.zip"
        reference.save(checkpoint)
        restored = SingleStepTD3BC.load(checkpoint, device="cpu")
        restored_actions, _ = restored.predict(
            curves.observation,
            deterministic=True,
        )
        if not np.array_equal(reference_final, restored_actions):
            raise RuntimeError("Structured speed checkpoint changed Actor output.")
    environment.close()
    print(
        f"tasks={curves.task_count} legal={expected_legal} "
        f"updates={reports[0].gradient_updates} pocket_heads=6 "
        "critic_updates=0 angle_ab_controlled=True trunk_frozen=True "
        "legal_sensitivity=True save_load=True"
    )


if __name__ == "__main__":
    main()
