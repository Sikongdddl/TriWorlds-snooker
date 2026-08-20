"""Check deterministic TD3 residuals, delayed learning, and resume state."""

from __future__ import annotations

from pathlib import Path
import tempfile

import gymnasium as gym
import numpy as np
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from snooker_env.midlevel_ppo_env import MAX_TERMINAL_REWARD  # noqa: E402
from snooker_env.midlevel_sac_her import (  # noqa: E402
    ConservativeResidualTD3Policy,
    SingleStepCuePositionHerReplayBuffer,
    SingleStepTD3BC,
    replay_buffer_path,
)


class _OneStepEnv(gym.Env[np.ndarray, np.ndarray]):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        return np.zeros(8, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        return np.zeros(8, dtype=np.float32), 0.0, True, False, {}


def main() -> None:
    environment = DummyVecEnv([_OneStepEnv for _ in range(4)])
    model = SingleStepTD3BC(
        ConservativeResidualTD3Policy,
        environment,
        seed=19,
        device="cpu",
        learning_rate=3.0e-4,
        actor_learning_rate=1.0e-5,
        critic_learning_rate=3.0e-4,
        actor_update_interval=4,
        actor_learning_starts=8,
        residual_l2_weight=0.02,
        buffer_size=64,
        batch_size=4,
        learning_starts=0,
        gamma=0.0,
        train_freq=(1, "step"),
        gradient_steps=4,
        replay_buffer_class=SingleStepCuePositionHerReplayBuffer,
        replay_buffer_kwargs={
            "her_ratio": 0.0,
            "success_ratio": 0.25,
            "failure_ratio": 0.25,
        },
        policy_kwargs={"net_arch": [32, 32]},
    )
    model.set_logger(configure(format_strings=[]))
    observations = np.zeros((4, 8), dtype=np.float32)
    observations[:, 0] = np.linspace(-0.3, 0.3, 4)
    baseline, _ = model.predict(observations, deterministic=True)
    certified = np.asarray(baseline, dtype=np.float32)
    certified[:, 0] = 0.0

    model.configure_conservative_speed_residual(
        max_speed_residual_mps=0.12,
        exploration_initial_std=0.35,
        exploration_final_std=0.05,
        exploration_decay_timesteps=16,
    )
    model.configure_discrete_candidate_ranking((-0.12, 0.0, 0.12))
    model.configure_behavior_cloning_reference(
        observations,
        certified,
        initial_weight=1.0,
        final_weight=1.0,
        decay_actor_updates=1,
        batch_size=4,
        angle_weight=4.0,
        speed_weight=4.0,
        residual_weight=0.25,
    )
    actor_state = {
        name: value.detach().clone()
        for name, value in model.actor.state_dict().items()
    }
    output_layer = next(
        layer
        for layer in reversed(model.actor.mu)
        if isinstance(layer, torch.nn.Linear)
    )
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.zero_()
        output_layer.bias[1] = 1.0
        residual_value = float(model.actor(torch.as_tensor(observations))[0, 1])
        observed_anchor = float(model._behavior_cloning_regularization_loss())
    expected_anchor = 0.25 * residual_value * residual_value
    if not np.isclose(observed_anchor, expected_anchor, atol=1.0e-7):
        raise RuntimeError("Online BC anchor was not measured in residual units.")
    model.actor.load_state_dict(actor_state)
    deterministic_action, _ = model.predict(observations, deterministic=True)
    stochastic_action, _ = model.predict(observations, deterministic=False)
    if hasattr(model.actor, "log_std"):
        raise RuntimeError("TD3-style actor still exposes a stochastic log-std head.")
    if not np.array_equal(deterministic_action[:, 0], np.zeros(4)):
        raise RuntimeError("Residual policy did not hard-lock the angle action.")
    if not np.allclose(deterministic_action[:, 1], certified[:, 1], atol=1e-7):
        raise RuntimeError("Zero residual did not reproduce the frozen BC speed.")
    max_normalized_residual = model.speed_residual_action_scale
    if np.any(
        np.abs(stochastic_action[:, 1] - certified[:, 1])
        > max_normalized_residual + 1e-7
    ):
        raise RuntimeError("Exploration exceeded the hard speed residual bound.")
    repeated_observations = np.repeat(observations[:1], 4096, axis=0)
    repeated_baseline = np.repeat(certified[:1, 1], 4096)
    model.num_timesteps = 0
    initial_noisy, _ = model.predict(repeated_observations, deterministic=False)
    model.num_timesteps = 16
    final_noisy, _ = model.predict(repeated_observations, deterministic=False)
    initial_std = float(np.std(initial_noisy[:, 1] - repeated_baseline))
    final_std = float(np.std(final_noisy[:, 1] - repeated_baseline))
    if not initial_std > 3.0 * final_std > 0.0:
        raise RuntimeError(
            "Bounded Gaussian exploration did not decay with timesteps."
        )
    model.num_timesteps = 0

    infos = [
        {
            "correct_pot": index % 2 == 0,
            "cue_scratch": index == 1,
            "wrong_pocket": False,
            "stopped": index % 2 == 0,
            "timed_out": index == 3,
            "numerical_failure": False,
            "cue_ball_final_position": np.zeros(3, dtype=np.float64),
        }
        for index in range(4)
    ]
    model.replay_buffer.add(
        observations,
        observations.copy(),
        deterministic_action,
        np.array([MAX_TERMINAL_REWARD, 0.0, 1.0, 0.0], dtype=np.float32),
        np.ones(4, dtype=np.bool_),
        infos,
    )
    actor_before = [parameter.detach().clone() for parameter in model.actor.parameters()]
    reference_before = [
        parameter.detach().clone()
        for parameter in model.policy.reference_actor.parameters()
    ]
    model.train(gradient_steps=4, batch_size=4)
    if model._actor_updates != 0:
        raise RuntimeError("Actor updated before local exploration warmup completed.")
    if any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, model.actor.parameters())
    ):
        raise RuntimeError("Deferred actor changed during critic-only online warmup.")

    model.num_timesteps = 8
    model.train(gradient_steps=4, batch_size=4)
    if model._actor_updates != 1:
        raise RuntimeError("Conservative actor did not honor its update interval.")
    if any(
        not torch.equal(before, after)
        for before, after in zip(
            reference_before,
            model.policy.reference_actor.parameters(),
        )
    ):
        raise RuntimeError("Actor training changed the frozen BC reference.")

    with tempfile.TemporaryDirectory(
        prefix="midlevel-conservative-residual-td3-"
    ) as directory:
        checkpoint = Path(directory) / "residual_smoke"
        model.save(checkpoint)
        model.save_replay_buffer(replay_buffer_path(checkpoint))
        restored = SingleStepTD3BC.load(
            checkpoint,
            env=environment,
            device="cpu",
        )
        restored.load_replay_buffer(replay_buffer_path(checkpoint))
        restored_action, _ = restored.predict(observations, deterministic=True)
        if not restored.residual_policy_enabled:
            raise RuntimeError("Checkpoint lost residual-policy configuration.")
        if not np.allclose(
            restored_action,
            model.predict(observations, deterministic=True)[0],
        ):
            raise RuntimeError("Checkpoint changed the effective residual action.")
        if any(
            parameter.requires_grad
            for parameter in restored.policy.reference_actor.parameters()
        ):
            raise RuntimeError("Restored BC reference is no longer frozen.")
        if any(
            parameter.requires_grad
            for parameter in restored.actor_target.parameters()
        ):
            raise RuntimeError("Unused TD3 actor target is no longer frozen.")

    environment.close()
    print(
        f"actor_updates={model._actor_updates} angle_locked=True "
        f"max_speed_residual_mps={model.max_speed_residual_mps:.3f} "
        f"noise_std_initial={initial_std:.5f} noise_std_final={final_std:.5f} "
        "deterministic_actor=True residual_bc_anchor=True "
        "reference_frozen=True resume=True"
    )


if __name__ == "__main__":
    main()
