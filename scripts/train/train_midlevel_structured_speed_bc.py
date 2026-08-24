"""Train pocket-headed speed BC from every legal seven-point curve outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_offline_curves import (  # noqa: E402
    OfflineSpeedCurveDataset,
)
from snooker_env.midlevel_ppo import (  # noqa: E402
    MIDLEVEL_TRAINING_MANIFEST_VERSION,
    generated_behavior_cloning_data,
)
from snooker_env.midlevel_ppo_env import (  # noqa: E402
    CUE_POSITION_REWARD_DISTANCE_SCALE,
    CUE_POSITION_REWARD_WEIGHT,
    JOINT_SUCCESS_REWARD_BONUS,
    MAX_TERMINAL_REWARD,
    MIDLEVEL_REWARD_VERSION,
    OBJECT_BALL_REWARD_WEIGHT,
    OBJECT_POCKET_REWARD_DISTANCE_SCALE,
)
from snooker_env.midlevel_sac_her import (  # noqa: E402
    MIDLEVEL_GEOMETRIC_FEATURE_DIM,
    MIDLEVEL_GEOMETRIC_FEATURE_VERSION,
    SINGLE_STEP_HER_VERSION,
    SINGLE_STEP_TD3_VERSION,
    STRUCTURED_SPEED_ANGLE_MODES,
    STRUCTURED_SPEED_BC_VERSION,
    MidLevelGeometricFeatures,
    SingleStepTD3BC,
    StructuredSpeedTD3Policy,
    behavior_clone_structured_speed_policy,
    td3_behavior_cloning_metrics,
)
from snooker_env.midlevel_tasks import TwoBallTaskDataset  # noqa: E402
from snooker_env.midlevel_two_ball import SHOT_EXECUTION_VERSION  # noqa: E402


class OfflineSpaceEnv(gym.Env):
    """Expose spaces only; supervised BC never calls physics or ``step``."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=(8,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=(2,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        return np.zeros(8, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        raise RuntimeError("Pure supervised BC must not execute environment steps.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_development_dataset(
    training: TwoBallTaskDataset,
    development: TwoBallTaskDataset,
) -> None:
    for name in (
        "xml_hash",
        "model_hash",
        "physics_backend",
        "backend_hash",
        "execution_max_time",
        "stop_speed",
        "stop_hold_time",
    ):
        if getattr(training, name) != getattr(development, name):
            raise ValueError(
                f"Development tasks have mismatched physics field {name!r}."
            )


def build_manifest(
    args: argparse.Namespace,
    dataset: TwoBallTaskDataset,
    curves: OfflineSpeedCurveDataset,
) -> dict[str, object]:
    return {
        "manifest_version": MIDLEVEL_TRAINING_MANIFEST_VERSION,
        "shot_execution_version": SHOT_EXECUTION_VERSION,
        "physics": {
            "xml_sha256": dataset.xml_hash,
            "model_sha256": dataset.model_hash,
            "backend": dataset.physics_backend,
            "backend_sha256": dataset.backend_hash,
        },
        "task_dataset": {
            "content_sha256": dataset.content_sha256(),
            "task_count": len(dataset),
            "generation_seed": dataset.generation_seed,
            "execution_max_time": dataset.execution_max_time,
            "stop_speed": dataset.stop_speed,
            "stop_hold_time": dataset.stop_hold_time,
        },
        "environment": {
            "backend": "mujoco-warp",
            "num_envs": 0,
            "max_shot_time": dataset.execution_max_time,
            "training_physics_rollouts": 0,
        },
        "reward": {
            "version": MIDLEVEL_REWARD_VERSION,
            "object_pocket_distance_scale": (
                OBJECT_POCKET_REWARD_DISTANCE_SCALE
            ),
            "cue_position_distance_scale": CUE_POSITION_REWARD_DISTANCE_SCALE,
            "object_ball_weight": OBJECT_BALL_REWARD_WEIGHT,
            "cue_position_weight": CUE_POSITION_REWARD_WEIGHT,
            "joint_success_bonus": JOINT_SUCCESS_REWARD_BONUS,
            "maximum_terminal_reward": MAX_TERMINAL_REWARD,
            "scratch_reward": 0.0,
        },
        "algorithm": {
            "name": "SingleStepTD3BC",
            "version": SINGLE_STEP_TD3_VERSION,
            "policy": "StructuredSpeedTD3Policy",
            "supervised_version": STRUCTURED_SPEED_BC_VERSION,
            "training_objective": "pure_supervised_hindsight_bc",
            "critic_used_for_training_or_selection": False,
            "critic_gradient_updates": 0,
            "gamma": 0.0,
            "deterministic_actor": True,
            "actor_q_objective": None,
            "net_arch": list(args.net_arch),
            "pocket_head_count": 6,
            "angle_mode": args.angle_mode,
            "speed_trunk_frozen": args.freeze_speed_trunk,
            "features": {
                "version": MIDLEVEL_GEOMETRIC_FEATURE_VERSION,
                "dimension": MIDLEVEL_GEOMETRIC_FEATURE_DIM,
            },
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "final_learning_rate": args.final_learning_rate,
            "speed_weight": args.speed_weight,
            "speed_error_scale_mps": args.speed_error_scale_mps,
            "canonical_anchor_weight": args.canonical_anchor_weight,
            "middle_pocket_weight": args.middle_pocket_weight,
            "sensitivity_weight_minimum": (
                args.sensitivity_weight_minimum
            ),
            "sensitivity_weight_maximum": (
                args.sensitivity_weight_maximum
            ),
            "sensitivity_loss_weight": args.sensitivity_loss_weight,
            "sensitivity_estimator": "nearest_legal_center_slope_v1",
            "sensitivity_distance_scale_m": (
                args.sensitivity_distance_scale_m
            ),
        },
        "hindsight_replay": {
            "version": SINGLE_STEP_HER_VERSION,
            "enabled": False,
            "replay_buffer_used": False,
            "supervised_hindsight_enabled": True,
            "target_pocket_relabelled": False,
            "target_stop_relabelled": True,
            "relabel_source": "all_legal_measured_seven_point_curve_outcomes",
            "eligible_supervised_record_count": int(
                np.count_nonzero(curves.safe)
            ),
            "eligibility": (
                "correct_pot_and_no_scratch_and_no_wrong_pocket_and_stopped_"
                "and_not_timed_out_and_no_numerical_failure"
            ),
            "offline_curve_content_sha256": file_sha256(curves.path),
        },
        "angle_reference": {
            "checkpoint": str(args.angle_reference),
            "checkpoint_sha256": file_sha256(args.angle_reference),
            "frozen": True,
            "ab_zero_angle": args.angle_mode == "zero",
        },
        "speed_reference": {
            "checkpoint": str(args.speed_reference),
            "checkpoint_sha256": file_sha256(args.speed_reference),
            "expanded_data_initializer": (
                args.speed_reference != args.angle_reference
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "outputs/tasks/midlevel_two_ball_train.npz",
    )
    parser.add_argument(
        "--offline-speed-curves",
        type=Path,
        default=(
            ROOT
            / "outputs/diagnostics/midlevel_speed_perturbations_208896.npz"
        ),
    )
    parser.add_argument(
        "--development-tasks",
        type=Path,
        default=ROOT / "outputs/tasks/midlevel_two_ball_validation.npz",
    )
    parser.add_argument(
        "--angle-reference",
        type=Path,
        default=(
            ROOT
            / "outputs/checkpoints/"
            "midlevel_two_ball_td3_her_v10_canonical_e800_b2048_s2."
            "bc_only.zip"
        ),
    )
    parser.add_argument(
        "--angle-mode",
        choices=STRUCTURED_SPEED_ANGLE_MODES,
        default="reference",
    )
    parser.add_argument(
        "--speed-reference",
        type=Path,
        help=(
            "Canonical BC whose speed trunk initializes all six heads; "
            "defaults to --angle-reference."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-task-count", type=int, default=208_896)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--final-learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--speed-weight", type=float, default=1.0)
    parser.add_argument("--speed-error-scale-mps", type=float, default=0.005)
    parser.add_argument("--canonical-anchor-weight", type=float, default=64.0)
    parser.add_argument("--middle-pocket-weight", type=float, default=2.0)
    parser.add_argument(
        "--sensitivity-weight-minimum",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--sensitivity-weight-maximum",
        type=float,
        default=4.0,
    )
    parser.add_argument("--sensitivity-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--sensitivity-distance-scale-m",
        type=float,
        default=0.05,
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--freeze-speed-trunk",
        action="store_true",
        help=(
            "Keep the selected canonical feature/trunk mapping fixed and "
            "train only the six pocket-specific speed heads."
        ),
    )
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=(512, 512, 256),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speed_reference is None:
        args.speed_reference = args.angle_reference
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    for path in (
        args.tasks,
        args.offline_speed_curves,
        args.development_tasks,
        args.angle_reference,
        args.speed_reference,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    dataset = TwoBallTaskDataset.load(args.tasks, validate_model=False)
    if len(dataset) != args.expected_task_count:
        raise ValueError(
            f"Structured round requires exactly {args.expected_task_count} "
            f"training tasks, got {len(dataset)}."
        )
    observations, actions = generated_behavior_cloning_data(dataset)
    curves = OfflineSpeedCurveDataset.load(
        args.offline_speed_curves,
        task_dataset=dataset,
        reference_observations=observations,
        reference_actions=actions,
    )
    development = TwoBallTaskDataset.load(
        args.development_tasks,
        validate_model=False,
    )
    validate_development_dataset(dataset, development)
    print(
        "structured_speed_inputs="
        + json.dumps(
            {
                "tasks": len(dataset),
                "task_hash": dataset.content_sha256(),
                "curves": curves.report(),
                "development_tasks": len(development),
                "angle_mode": args.angle_mode,
                "angle_reference": str(args.angle_reference),
                "speed_reference": str(args.speed_reference),
                "seed": args.seed,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    set_random_seed(args.seed)
    environment = DummyVecEnv([OfflineSpaceEnv])
    try:
        model = SingleStepTD3BC(
            StructuredSpeedTD3Policy,
            environment,
            seed=args.seed,
            device=args.device,
            verbose=1,
            learning_rate=3.0e-4,
            actor_candidate_supervision_weight=0.0,
            actor_physical_probe_supervision_weight=0.0,
            buffer_size=1,
            learning_starts=0,
            batch_size=1,
            gamma=0.0,
            train_freq=(1, "step"),
            gradient_steps=0,
            policy_kwargs={
                "net_arch": list(args.net_arch),
                "n_critics": 2,
                "features_extractor_class": MidLevelGeometricFeatures,
                "angle_mode": args.angle_mode,
                "pocket_head_count": 6,
                "angle_reference_net_arch": (512, 512, 256),
            },
        )
        angle_source = SingleStepTD3BC.load(
            str(args.angle_reference),
            device=args.device,
        )
        source_actor = angle_source.policy.actor
        source_state = {
            name: value.detach().clone()
            for name, value in source_actor.state_dict().items()
        }
        model.policy.install_angle_reference(source_state)
        del angle_source, source_state

        speed_source = SingleStepTD3BC.load(
            str(args.speed_reference),
            device=args.device,
        )
        speed_manifest = getattr(
            speed_source,
            "midlevel_training_manifest",
            None,
        )
        speed_task = (
            speed_manifest.get("task_dataset", {})
            if isinstance(speed_manifest, dict)
            else {}
        )
        if speed_task.get("content_sha256") != dataset.content_sha256():
            raise ValueError(
                "Speed initializer was not trained on the active task library."
            )
        speed_state = {
            name: value.detach().clone()
            for name, value in speed_source.policy.actor.state_dict().items()
        }
        model.policy.install_speed_reference(speed_state)
        del speed_source, speed_state

        report = behavior_clone_structured_speed_policy(
            model.policy,
            curves,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            final_learning_rate=args.final_learning_rate,
            speed_weight=args.speed_weight,
            speed_error_scale_mps=args.speed_error_scale_mps,
            canonical_anchor_weight=args.canonical_anchor_weight,
            middle_pocket_weight=args.middle_pocket_weight,
            sensitivity_weight_minimum=args.sensitivity_weight_minimum,
            sensitivity_weight_maximum=args.sensitivity_weight_maximum,
            sensitivity_loss_weight=args.sensitivity_loss_weight,
            sensitivity_distance_scale_m=(
                args.sensitivity_distance_scale_m
            ),
            freeze_speed_trunk=args.freeze_speed_trunk,
            seed=args.seed,
            max_grad_norm=args.max_grad_norm,
        )
        development_metrics = td3_behavior_cloning_metrics(
            model.policy,
            development,
            batch_size=args.batch_size,
            angle_weight=1.0,
            speed_weight=args.speed_weight,
        )
        report_data: dict[str, object] = report.as_dict()
        report_data["development_reconstruction"] = development_metrics
        manifest = build_manifest(args, dataset, curves)
        model.midlevel_training_manifest = manifest
        model.midlevel_behavior_cloning_report = report_data
        model.policy.actor_target.load_state_dict(
            model.policy.actor.state_dict()
        )
        model.policy.snapshot_reference_actor()

        output = (
            args.output
            if args.output.suffix == ".zip"
            else Path(f"{args.output}.zip")
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output))
        base = Path(str(output)[: -len(".zip")])
        Path(f"{base}.manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        Path(f"{base}.bc.json").write_text(
            json.dumps(report_data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "structured_speed_bc="
            + json.dumps(report_data, sort_keys=True),
            flush=True,
        )
        print(f"structured_speed_checkpoint={output}", flush=True)
    finally:
        environment.close()


if __name__ == "__main__":
    main()
