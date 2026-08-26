"""Exercise direct BC fitting, checkpoint round-trip, and pipeline rollout."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_bc import (  # noqa: E402
    DIRECT_BC_ALGORITHM_VERSION,
    BCCheckpointMidLevelPolicy,
    DirectBCPolicy,
    behavior_cloning_metrics_by_difficulty,
    generated_behavior_cloning_data,
    task_physics_manifest,
    train_direct_behavior_cloning,
    validate_policy_for_dataset,
)
from snooker_env.midlevel_tasks import (  # noqa: E402
    EVENT_FLAG_NAMES,
    CPU_PHYSICS_BACKEND,
    TwoBallTaskDataset,
)
from snooker_env.midlevel_two_ball import (  # noqa: E402
    POCKET_NAMES,
    POCKET_POSITIONS,
    ghost_ball_direction,
)
from snooker_env.pipeline_types import (  # noqa: E402
    BallState,
    SceneState,
    ShotIntent,
    SkillCommand,
    SkillId,
    TableState,
)


def _dataset() -> TwoBallTaskDataset:
    count = 48
    cue_positions: list[np.ndarray] = []
    object_positions: list[np.ndarray] = []
    target_positions: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    speeds: list[float] = []
    pocket_indices = np.arange(count, dtype=np.int64) % len(POCKET_NAMES)
    for index, pocket_index in enumerate(pocket_indices):
        pocket = POCKET_POSITIONS[POCKET_NAMES[int(pocket_index)]][:2]
        outward = pocket / max(float(np.linalg.norm(pocket)), 1.0e-12)
        tangent = np.array([-outward[1], outward[0]], dtype=np.float64)
        object_position = (
            pocket
            - outward * (0.22 + 0.01 * (index % 4))
            + tangent * (index % 3 - 1) * 0.01
        )
        cue_position = (
            object_position
            - outward * (0.38 + 0.015 * (index % 5))
            + tangent * (index % 7 - 3) * 0.015
        )
        target_position = np.array(
            [
                0.42 * np.sin(0.31 * index),
                0.82 * np.cos(0.23 * index),
            ],
            dtype=np.float64,
        )
        cue_positions.append(cue_position)
        object_positions.append(object_position)
        target_positions.append(target_position)
        directions.append(
            ghost_ball_direction(cue_position, object_position, pocket)
        )
        speeds.append(
            0.55
            + 0.20 * float(np.linalg.norm(target_position - cue_position))
            + 0.03 * int(pocket_index)
        )
    event_flags = np.ones((count, len(EVENT_FLAG_NAMES)), dtype=np.bool_)
    return TwoBallTaskDataset(
        cue_positions=np.stack(cue_positions),
        object_positions=np.stack(object_positions),
        pocket_indices=pocket_indices.astype(np.int8),
        target_stop_positions=np.stack(target_positions),
        generated_directions=np.stack(directions),
        generated_speeds=np.asarray(speeds, dtype=np.float64),
        candidate_seeds=np.arange(count, dtype=np.uint64),
        elapsed_times=np.full(count, 1.0, dtype=np.float64),
        min_object_pocket_distances=np.full(count, 0.01, dtype=np.float64),
        event_flags=event_flags,
        xml_hash="1" * 64,
        model_hash="2" * 64,
        physics_backend=CPU_PHYSICS_BACKEND,
        backend_hash="3" * 64,
        generation_seed=7,
        execution_max_time=8.0,
        stop_speed=0.01,
        stop_hold_time=0.2,
    )


def main() -> None:
    dataset = _dataset()
    dataset_hash = dataset.content_sha256()
    observations, targets = generated_behavior_cloning_data(dataset)
    if observations.shape != (len(dataset), 8) or targets.shape != (
        len(dataset),
        2,
    ):
        raise RuntimeError("Direct BC data contract is malformed.")
    if dataset.content_sha256() != dataset_hash:
        raise RuntimeError("Direct BC preprocessing mutated its task dataset.")

    np.random.seed(5)
    torch.manual_seed(5)
    policy = DirectBCPolicy((32, 32), device="cpu")
    report = train_direct_behavior_cloning(
        policy,
        dataset,
        epochs=80,
        batch_size=16,
        learning_rate=3.0e-3,
        final_learning_rate=3.0e-4,
        angle_weight=1.0,
        speed_weight=8.0,
        seed=5,
    )
    if report.final.loss >= report.initial.loss:
        raise RuntimeError("Direct BC loss did not decrease.")
    if report.final.speed_mae_mps >= report.initial.speed_mae_mps:
        raise RuntimeError("Direct BC speed error did not decrease.")
    difficulty_metrics = behavior_cloning_metrics_by_difficulty(
        policy,
        dataset,
        batch_size=16,
        angle_weight=1.0,
        speed_weight=8.0,
    )
    reported_samples = sum(
        int(metrics["sample_count"])
        for metrics in difficulty_metrics["cells"].values()
    )
    if reported_samples != len(dataset):
        raise RuntimeError(
            "Per-difficulty BC metrics do not cover the complete dataset."
        )
    if dataset.content_sha256() != dataset_hash:
        raise RuntimeError("Direct BC training or metrics mutated its task dataset.")

    policy.manifest = {
        "algorithm": {
            "name": "DirectBehaviorCloning",
            "version": DIRECT_BC_ALGORITHM_VERSION,
        },
        "physics": task_physics_manifest(dataset),
    }
    policy.training_report = report.as_dict()
    validate_policy_for_dataset(policy, dataset)

    with tempfile.TemporaryDirectory(prefix="midlevel-direct-bc-smoke-") as directory:
        checkpoint = Path(directory) / "policy.pt"
        policy.save(checkpoint)
        restored = DirectBCPolicy.load(checkpoint, device="cpu")
        expected = policy.predict(observations[:8])
        actual = restored.predict(observations[:8])
        if not np.array_equal(expected, actual):
            raise RuntimeError("Direct BC checkpoint changed Actor predictions.")

        adapter = BCCheckpointMidLevelPolicy(checkpoint)
        task = dataset[0]
        state = SceneState(
            time=0.0,
            table=TableState(),
            balls={
                "cue_ball": BallState(
                    name="cue_ball",
                    position=np.r_[task.cue_position, 1.0785],
                ),
                "object_ball_0": BallState(
                    name="object_ball_0",
                    position=np.r_[task.object_position, 1.0785],
                ),
            },
        )
        command = SkillCommand(
            skill_id=SkillId.POSITION_SHOT,
            intent=ShotIntent(
                cue_ball_name="cue_ball",
                object_ball_name="object_ball_0",
                target_pocket=task.pocket_name,
                desired_cue_ball_position=np.r_[task.target_stop_position, 1.0785],
            ),
        )
        rollout = adapter.rollout(command, state)
        if len(rollout) != 3:
            raise RuntimeError("Direct BC adapter returned the wrong trajectory length.")
        if not all(
            np.all(np.isfinite(cue_command.linear_velocity))
            for cue_command in rollout
        ):
            raise RuntimeError("Direct BC adapter returned non-finite commands.")

    print(
        "midlevel_direct_bc=PASS "
        f"initial_loss={report.initial.loss:.6g} "
        f"final_loss={report.final.loss:.6g} "
        f"speed_mae_mps={report.final.speed_mae_mps:.6g}"
    )


if __name__ == "__main__":
    main()
