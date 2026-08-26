"""Train one deterministic mid-level policy with direct behavior cloning."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.midlevel_bc import (  # noqa: E402
    DEFAULT_HIDDEN_SIZES,
    DIRECT_BC_ALGORITHM_VERSION,
    DIRECT_BC_CHECKPOINT_VERSION,
    MIDLEVEL_GEOMETRIC_FEATURE_DIM,
    MIDLEVEL_GEOMETRIC_FEATURE_VERSION,
    DirectBCPolicy,
    behavior_cloning_metrics,
    behavior_cloning_metrics_by_difficulty,
    require_compatible_task_physics,
    task_physics_manifest,
    train_direct_behavior_cloning,
    write_json,
)
from snooker_env.midlevel_env import DEFAULT_MIDLEVEL_MODEL  # noqa: E402
from snooker_env.midlevel_tasks import (  # noqa: E402
    CPU_PHYSICS_BACKEND,
    MUJOCO_WARP_PHYSICS_BACKEND,
    TwoBallTaskDataset,
    require_balanced_task_difficulty,
    require_complete_task_difficulty,
)
from snooker_env.midlevel_two_ball import (  # noqa: E402
    MAX_ANGLE_RESIDUAL,
    MAX_CUE_SPEED,
    MIN_CUE_SPEED,
    SHOT_EXECUTION_VERSION,
    TwoBallShotSimulator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "outputs/tasks/midlevel_two_ball_train.npz",
    )
    parser.add_argument(
        "--validation-tasks",
        type=Path,
        default=ROOT / "outputs/tasks/midlevel_two_ball_validation.npz",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument(
        "--allow-task-fingerprint-mismatch",
        action="store_true",
        help=(
            "Reuse task arrays when only their stored XML/model/backend hashes "
            "differ from the active implementation. Backend type and shot "
            "timing must still match."
        ),
    )
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--final-learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--angle-weight", type=float, default=1.0)
    parser.add_argument("--speed-weight", type=float, default=8.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_HIDDEN_SIZES,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/checkpoints/midlevel_bc.pt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing checkpoint and its JSON reports.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if not args.hidden_sizes or any(size <= 0 for size in args.hidden_sizes):
        raise ValueError("--hidden-sizes must contain positive integers.")
    for name in (
        "learning_rate",
        "final_learning_rate",
        "angle_weight",
        "speed_weight",
        "max_grad_norm",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.final_learning_rate > args.learning_rate:
        raise ValueError("--final-learning-rate cannot exceed --learning-rate.")
    if args.output.suffix != ".pt":
        raise ValueError("--output must use the .pt extension.")
    for path in (args.tasks, args.validation_tasks):
        if not path.is_file():
            raise FileNotFoundError(path)


def _artifact_paths(checkpoint: Path) -> tuple[Path, Path]:
    base = checkpoint.with_suffix("")
    return (
        Path(f"{base}.manifest.json"),
        Path(f"{base}.bc.json"),
    )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    manifest_path, report_path = _artifact_paths(args.output)
    existing = [
        path
        for path in (args.output, manifest_path, report_path)
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Direct BC output already exists; pass --overwrite to replace: "
            + ", ".join(map(str, existing))
        )

    simulator = TwoBallShotSimulator(args.model)
    training_tasks = TwoBallTaskDataset.load(
        args.tasks,
        validate_model=False,
    )
    validation_tasks = TwoBallTaskDataset.load(
        args.validation_tasks,
        validate_model=False,
    )
    require_compatible_task_physics(
        training_tasks,
        validation_tasks,
        context="Validation",
    )
    require_complete_task_difficulty(training_tasks, context="Training")
    require_balanced_task_difficulty(validation_tasks, context="Validation")
    training_content_sha256 = training_tasks.content_sha256()
    validation_content_sha256 = validation_tasks.content_sha256()
    if training_tasks.physics_backend == CPU_PHYSICS_BACKEND:
        active_backend_hash = simulator.model_hash
    elif training_tasks.physics_backend == MUJOCO_WARP_PHYSICS_BACKEND:
        from snooker_env.midlevel_mujoco_warp_vec_env import (
            active_mujoco_warp_backend_sha256,
        )

        _, _, active_backend_hash = active_mujoco_warp_backend_sha256(args.model)
    else:
        raise ValueError(
            "Unsupported task physics backend: "
            f"{training_tasks.physics_backend!r}."
        )
    task_physics = task_physics_manifest(training_tasks)
    active_physics: dict[str, object] = {
        "xml_sha256": simulator.xml_hash,
        "model_sha256": simulator.model_hash,
        "backend": training_tasks.physics_backend,
        "backend_sha256": active_backend_hash,
        "execution_max_time": simulator.max_time,
        "stop_speed": simulator.stop_speed,
        "stop_hold_time": simulator.stop_hold_time,
    }
    semantic_fields = (
        "backend",
        "execution_max_time",
        "stop_speed",
        "stop_hold_time",
    )
    semantic_differences = [
        name
        for name in semantic_fields
        if task_physics[name] != active_physics[name]
    ]
    if semantic_differences:
        raise ValueError(
            "Task dataset execution semantics differ from the active model: "
            + ", ".join(semantic_differences)
        )
    fingerprint_fields = (
        "xml_sha256",
        "model_sha256",
        "backend_sha256",
    )
    fingerprint_differences = [
        name
        for name in fingerprint_fields
        if task_physics[name] != active_physics[name]
    ]
    if fingerprint_differences and not args.allow_task_fingerprint_mismatch:
        raise ValueError(
            "Task dataset fingerprints differ from the active implementation: "
            + ", ".join(fingerprint_differences)
            + ". Regenerate the libraries or pass "
            "--allow-task-fingerprint-mismatch after verifying physical reuse."
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    policy = DirectBCPolicy(args.hidden_sizes, device=args.device)

    def progress(
        epoch: int,
        epochs: int,
        learning_rate: float,
        loss: float,
    ) -> None:
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"direct_bc_epoch={epoch}/{epochs} "
                f"learning_rate={learning_rate:.8g} loss={loss:.8g}",
                flush=True,
            )

    training_report = train_direct_behavior_cloning(
        policy,
        training_tasks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        angle_weight=args.angle_weight,
        speed_weight=args.speed_weight,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        progress=progress,
    )
    validation_metrics = behavior_cloning_metrics(
        policy,
        validation_tasks,
        batch_size=min(args.batch_size, len(validation_tasks)),
        angle_weight=args.angle_weight,
        speed_weight=args.speed_weight,
    )
    validation_by_difficulty = behavior_cloning_metrics_by_difficulty(
        policy,
        validation_tasks,
        batch_size=min(args.batch_size, len(validation_tasks)),
        angle_weight=args.angle_weight,
        speed_weight=args.speed_weight,
    )
    if training_tasks.content_sha256() != training_content_sha256:
        raise RuntimeError("Direct BC mutated the training task dataset in memory.")
    if validation_tasks.content_sha256() != validation_content_sha256:
        raise RuntimeError("Direct BC mutated the validation task dataset in memory.")
    report: dict[str, object] = {
        "training": training_report.as_dict(),
        "validation": validation_metrics.as_dict(),
        "validation_by_difficulty": validation_by_difficulty,
    }
    manifest: dict[str, object] = {
        "checkpoint_version": DIRECT_BC_CHECKPOINT_VERSION,
        "shot_execution_version": SHOT_EXECUTION_VERSION,
        "algorithm": {
            "name": "DirectBehaviorCloning",
            "version": DIRECT_BC_ALGORITHM_VERSION,
            "training_stage_count": 1,
            "training_seed_count": 1,
            "objective": "weighted_action_reconstruction",
            "deterministic_actor": True,
        },
        "physics": task_physics,
        "active_physics_at_training": active_physics,
        "task_reuse": {
            "fingerprint_mismatch_allowed": (
                args.allow_task_fingerprint_mismatch
            ),
            "fingerprint_mismatch_fields": fingerprint_differences,
        },
        "task_dataset": {
            "path": str(args.tasks),
            "content_sha256": training_content_sha256,
            "task_count": len(training_tasks),
            "generation_seed": training_tasks.generation_seed,
            "difficulty_profile": training_tasks.difficulty_profile(),
        },
        "validation_dataset": {
            "path": str(args.validation_tasks),
            "content_sha256": validation_content_sha256,
            "task_count": len(validation_tasks),
            "generation_seed": validation_tasks.generation_seed,
            "difficulty_profile": validation_tasks.difficulty_profile(),
        },
        "observation": {
            "dimension": 8,
            "geometric_feature_version": MIDLEVEL_GEOMETRIC_FEATURE_VERSION,
            "geometric_feature_dimension": MIDLEVEL_GEOMETRIC_FEATURE_DIM,
        },
        "action": {
            "dimension": 2,
            "maximum_angle_residual_rad": MAX_ANGLE_RESIDUAL,
            "minimum_cue_speed_mps": MIN_CUE_SPEED,
            "maximum_cue_speed_mps": MAX_CUE_SPEED,
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "final_learning_rate": args.final_learning_rate,
            "angle_weight": args.angle_weight,
            "speed_weight": args.speed_weight,
            "max_grad_norm": args.max_grad_norm,
            "hidden_sizes": list(args.hidden_sizes),
            "device": args.device,
        },
    }
    policy.manifest = manifest
    policy.training_report = report

    policy.save(args.output)
    write_json(manifest_path, manifest)
    write_json(report_path, report)
    print("direct_bc=" + json.dumps(report, sort_keys=True), flush=True)
    print(f"checkpoint={args.output}", flush=True)
    print(f"manifest={manifest_path}", flush=True)
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
