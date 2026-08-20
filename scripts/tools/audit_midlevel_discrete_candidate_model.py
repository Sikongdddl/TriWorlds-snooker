"""Audit a discrete multi-head candidate model on real BC speed curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_sac_her import (  # noqa: E402
    MidLevelGeometricFeatures,
    SingleStepTD3BC,
)


class DiscreteCandidateModel(nn.Module):
    """Predict reward, safety, improvement, and endpoint per speed bin."""

    def __init__(self, input_dim: int, candidate_count: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.reward = nn.Linear(256, candidate_count)
        self.safe = nn.Linear(256, candidate_count)
        self.improvement = nn.Linear(256, candidate_count)
        self.best = nn.Linear(256, candidate_count)
        self.cue_endpoint = nn.Linear(256, 2 * candidate_count)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        latent = self.trunk(inputs)
        return (
            self.reward(latent),
            self.safe(latent),
            self.improvement(latent),
            self.best(latent),
            self.cue_endpoint(latent),
        )


def holdout_mask(task_indices: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    indices = np.asarray(task_indices, dtype=np.uint64)
    mixed = indices + np.uint64(seed + 1)
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    threshold = int(fraction * 1_000_000)
    return (mixed % np.uint64(1_000_000)) < np.uint64(threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-seed", type=int, default=20_000)
    parser.add_argument("--maximum-offset-mps", type=float, default=0.03)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with np.load(args.curve, allow_pickle=False) as archive:
        curve = {name: np.asarray(archive[name]) for name in archive.files}
    selected_offsets = np.flatnonzero(
        np.abs(curve["offsets_mps"]) <= args.maximum_offset_mps + 1.0e-12
    )
    offsets = curve["offsets_mps"][selected_offsets].astype(np.float32)
    observations = curve["observation"][0].astype(np.float32)
    rewards = curve["reward"][selected_offsets].T.astype(np.float32)
    safe = (
        curve["correct_pot"][selected_offsets]
        & ~curve["cue_scratch"][selected_offsets]
        & ~curve["wrong_pocket"][selected_offsets]
        & curve["stopped"][selected_offsets]
        & ~curve["timed_out"][selected_offsets]
        & ~curve["numerical_failure"][selected_offsets]
    ).T
    cue_endpoints = curve["cue_final"][selected_offsets, :, :2].transpose(
        1,
        0,
        2,
    )
    cue_endpoints = cue_endpoints.astype(np.float32)
    cue_endpoints[:, :, 0] /= 0.75
    cue_endpoints[:, :, 1] /= 1.40
    center_index = int(np.flatnonzero(offsets == 0.0)[0])
    safe_rewards = np.where(safe, rewards, -np.inf)
    maximum_safe_reward = np.max(safe_rewards, axis=1)
    has_safe = np.isfinite(maximum_safe_reward)
    tied_best = safe & np.isclose(
        rewards,
        maximum_safe_reward[:, None],
        atol=1e-7,
        rtol=0.0,
    )
    best_indices = np.argmin(
        np.where(tied_best, np.abs(offsets)[None, :], np.inf),
        axis=1,
    )
    best_indices[~has_safe] = center_index
    improvement = (
        safe
        & (rewards >= rewards[:, center_index : center_index + 1] + 0.05)
    )

    checkpoint = SingleStepTD3BC.load(args.checkpoint, device=args.device)
    observation_tensor = torch.as_tensor(
        observations,
        dtype=torch.float32,
        device=args.device,
    )
    extractor = MidLevelGeometricFeatures(
        gym.spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
    ).to(args.device)
    with torch.no_grad():
        geometric = extractor(observation_tensor)
        baseline_speed = checkpoint.actor(observation_tensor)[:, 1:2]
        inputs = torch.cat((geometric, baseline_speed), dim=1)

    task_indices = curve["task_indices"].astype(np.int64)
    held_out = holdout_mask(
        task_indices,
        args.holdout_fraction,
        args.holdout_seed,
    )
    training_indices = np.flatnonzero(~held_out)
    held_out_indices = np.flatnonzero(held_out)
    reward_targets = torch.as_tensor(rewards, device=args.device)
    safe_targets = torch.as_tensor(
        safe.astype(np.float32),
        device=args.device,
    )
    improvement_targets = torch.as_tensor(
        improvement.astype(np.float32),
        device=args.device,
    )
    best_targets = torch.as_tensor(
        best_indices,
        dtype=torch.long,
        device=args.device,
    )
    endpoint_targets = torch.as_tensor(cue_endpoints, device=args.device)
    positive_count = np.count_nonzero(improvement[training_indices])
    negative_count = improvement[training_indices].size - positive_count
    improvement_positive_weight = torch.as_tensor(
        negative_count / max(positive_count, 1),
        dtype=torch.float32,
        device=args.device,
    )

    models: list[DiscreteCandidateModel] = []
    final_losses: list[float] = []
    rng = np.random.default_rng(0)
    for model_index in range(2):
        torch.manual_seed(100 + model_index)
        model = DiscreteCandidateModel(inputs.shape[1], len(offsets)).to(
            args.device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_value = 0.0
        for _ in range(args.updates):
            indices = rng.choice(
                training_indices,
                size=min(args.batch_size, len(training_indices)),
                replace=False,
            )
            index_tensor = torch.as_tensor(
                indices,
                dtype=torch.long,
                device=args.device,
            )
            (
                predicted_reward,
                predicted_safe,
                predicted_improvement,
                predicted_best,
                predicted_endpoint,
            ) = model(inputs[index_tensor])
            predicted_endpoint = predicted_endpoint.reshape(
                len(index_tensor),
                len(offsets),
                2,
            )
            batch_safe = safe_targets[index_tensor]
            endpoint_error = torch.sum(
                torch.square(
                    predicted_endpoint - endpoint_targets[index_tensor]
                ),
                dim=2,
            )
            endpoint_loss = torch.sum(endpoint_error * batch_safe) / torch.clamp(
                torch.sum(batch_safe),
                min=1.0,
            )
            loss = (
                F.smooth_l1_loss(
                    predicted_reward / 3.5,
                    reward_targets[index_tensor] / 3.5,
                )
                + 0.5
                * F.binary_cross_entropy_with_logits(
                    predicted_safe,
                    batch_safe,
                )
                + F.binary_cross_entropy_with_logits(
                    predicted_improvement,
                    improvement_targets[index_tensor],
                    pos_weight=improvement_positive_weight,
                )
                + F.cross_entropy(predicted_best, best_targets[index_tensor])
                + 0.5 * endpoint_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.item())
        model.eval()
        models.append(model)
        final_losses.append(loss_value)

    held_out_tensor = torch.as_tensor(
        held_out_indices,
        dtype=torch.long,
        device=args.device,
    )
    with torch.no_grad():
        predictions = [model(inputs[held_out_tensor]) for model in models]
    predicted_rewards = torch.stack(
        [prediction[0] for prediction in predictions]
    ).cpu().numpy()
    predicted_safe = torch.sigmoid(
        torch.stack([prediction[1] for prediction in predictions])
    ).cpu().numpy()
    predicted_improvement = torch.sigmoid(
        torch.stack([prediction[2] for prediction in predictions])
    ).cpu().numpy()
    predicted_best = torch.stack(
        [prediction[3] for prediction in predictions]
    ).cpu().numpy()
    held_rewards = rewards[held_out_indices]
    held_safe = safe[held_out_indices]
    held_improvement = improvement[held_out_indices]
    held_best = best_indices[held_out_indices]
    held_rows = np.arange(len(held_out_indices))

    best_reward = held_rewards[held_rows, held_best]
    eligible = best_reward[:, None] >= held_rewards + 0.05
    q_agreements = []
    for model_index in range(2):
        best_prediction = predicted_rewards[
            model_index,
            held_rows,
            held_best,
        ]
        q_agreements.append(
            best_prediction[:, None] > predicted_rewards[model_index]
        )
    both_ranking = q_agreements[0] & q_agreements[1]

    threshold_sweep: list[dict[str, float | int]] = []
    center_rewards = held_rewards[:, center_index]
    for improvement_threshold in (0.5, 0.7, 0.8, 0.9):
        for safety_threshold in (0.5, 0.7, 0.9):
            approved = (
                np.all(
                    predicted_improvement >= improvement_threshold,
                    axis=0,
                )
                & np.all(predicted_safe >= safety_threshold, axis=0)
            )
            approved[:, center_index] = True
            score = np.min(predicted_best, axis=0)
            score = np.where(approved, score, -np.inf)
            selected = np.argmax(score, axis=1)
            nonzero = selected != center_index
            selected_reward = held_rewards[held_rows, selected]
            selected_safe = held_safe[held_rows, selected]
            true_improvement = (
                selected_safe & (selected_reward >= center_rewards + 0.05)
            )
            threshold_sweep.append(
                {
                    "improvement_probability": improvement_threshold,
                    "safety_probability": safety_threshold,
                    "nonzero_count": int(np.count_nonzero(nonzero)),
                    "nonzero_rate": float(np.mean(nonzero)),
                    "true_improvement_precision": (
                        float(np.mean(true_improvement[nonzero]))
                        if np.any(nonzero)
                        else float("nan")
                    ),
                    "reward_improvement_mean": float(
                        np.mean(selected_reward - center_rewards)
                    ),
                }
            )

    report = {
        "model": "twin-discrete-candidate-multitask-v1",
        "training_task_count": int(len(training_indices)),
        "held_out_task_count": int(len(held_out_indices)),
        "updates_per_model": args.updates,
        "final_losses": final_losses,
        "pairwise_both_reward_heads_agreement": float(
            np.mean(both_ranking[eligible])
        ),
        "best_classification_exact_q1": float(
            np.mean(np.argmax(predicted_best[0], axis=1) == held_best)
        ),
        "best_classification_exact_q2": float(
            np.mean(np.argmax(predicted_best[1], axis=1) == held_best)
        ),
        "improvement_positive_rate": float(np.mean(held_improvement)),
        "threshold_sweep": threshold_sweep,
    }
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
