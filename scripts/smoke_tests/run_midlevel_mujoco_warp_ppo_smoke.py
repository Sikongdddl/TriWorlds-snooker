"""Exercise batched MJWarp rollout, PPO update, checkpoint, and resume."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from run_midlevel_two_ball_ppo_env_smoke import _fixture_task  # noqa: E402
from snooker_env.midlevel_mujoco_warp_vec_env import (  # noqa: E402
    MJWarpMidLevelVecEnv,
    active_mujoco_warp_backend_sha256,
)
from snooker_env.midlevel_ppo import BoundedActorCriticPolicy  # noqa: E402
from snooker_env.midlevel_tasks import (  # noqa: E402
    MUJOCO_WARP_PHYSICS_BACKEND,
    TwoBallTaskDataset,
)
from snooker_env.midlevel_two_ball import (  # noqa: E402
    TwoBallShotSimulator,
    encode_speed_action,
    ghost_ball_direction,
    quantize_cue_speed,
)


def main() -> None:
    simulator = TwoBallShotSimulator()
    fixture = _fixture_task(simulator)
    cue_position = np.array([0.10, 0.0], dtype=np.float64)
    generated_speed = quantize_cue_speed(1.0)
    task = replace(
        fixture,
        cue_position=cue_position,
        generated_direction=ghost_ball_direction(
            cue_position,
            fixture.object_position,
            fixture.pocket_position,
        ),
        generated_speed=generated_speed,
    )
    dataset = TwoBallTaskDataset.from_tasks(
        [task],
        simulator,
        generation_seed=7,
        physics_backend=MUJOCO_WARP_PHYSICS_BACKEND,
        backend_hash=active_mujoco_warp_backend_sha256()[2],
    )
    with tempfile.TemporaryDirectory(
        prefix="midlevel-mujoco-warp-ppo-smoke-"
    ) as directory:
        checkpoint = Path(directory) / "ppo_smoke"
        base_env = MJWarpMidLevelVecEnv(
            dataset,
            num_envs=2,
            seed=17,
            chunk_steps=2,
            check_interval_steps=512,
            max_time=0.60,
            # Accelerate only the terminal detector so the smoke covers the
            # non-truncated stopped-shot and gated-position-reward path.
            stop_speed=10.0,
            stop_hold_time=0.40,
            validate_task_execution=False,
        )
        base_env.seed(17)
        first_observation = base_env.reset()
        base_env.seed(17)
        repeated_observation = base_env.reset()
        if not np.array_equal(first_observation, repeated_observation):
            raise RuntimeError("MJWarp vector reset is not seed deterministic.")

        generated_action = np.array(
            [0.0, encode_speed_action(task.generated_speed)],
            dtype=np.float32,
        )
        actions = np.tile(generated_action, (base_env.num_envs, 1))
        _, rewards, dones, infos = base_env.step(actions)
        if rewards.shape != (2,) or dones.shape != (2,) or not np.all(dones):
            raise RuntimeError("MJWarp VecEnv returned an invalid terminal batch.")
        if not np.all(np.isfinite(rewards)):
            raise RuntimeError("MJWarp VecEnv produced a non-finite reward.")
        if not all(info["physics_backend"] == "mujoco_warp" for info in infos):
            raise RuntimeError("MJWarp backend identity is missing from infos.")
        if not all(bool(info["legal_first_contact"]) for info in infos):
            raise RuntimeError("The generated action did not contact the object ball.")
        if not all(
            bool(info["stopped"]) and not bool(info["timed_out"])
            for info in infos
        ):
            raise RuntimeError("The generated batch did not take the stopped-shot path.")

        environment = VecMonitor(base_env)
        policy = PPO(
            BoundedActorCriticPolicy,
            environment,
            seed=5,
            device="cpu",
            n_steps=1,
            batch_size=2,
            n_epochs=1,
            gamma=1.0,
            gae_lambda=1.0,
            learning_rate=3e-4,
            policy_kwargs={"net_arch": [32, 32], "log_std_init": -1.0},
        )
        policy.learn(total_timesteps=2)
        policy.save(checkpoint)
        reloaded = PPO.load(checkpoint, env=environment, device="cpu")
        reloaded.learn(total_timesteps=2, reset_num_timesteps=False)
        observation = environment.reset()
        action, _ = reloaded.predict(observation, deterministic=True)
        repeated_observation = np.repeat(observation[:1], 4096, axis=0)
        observation_tensor, _ = reloaded.policy.obs_to_tensor(
            repeated_observation
        )
        with torch.no_grad():
            sampled_actions, _, sampled_log_prob = reloaded.policy(
                observation_tensor,
                deterministic=False,
            )
        bounded_samples = bool(
            torch.all(sampled_actions >= -1.0)
            and torch.all(sampled_actions <= 1.0)
            and torch.all(torch.isfinite(sampled_log_prob))
        )

        parameters_finite = all(
            bool(np.all(np.isfinite(parameter.detach().cpu().numpy())))
            for parameter in reloaded.policy.parameters()
        )
        rollout_finite = all(
            bool(np.all(np.isfinite(values)))
            for values in (
                reloaded.rollout_buffer.rewards,
                reloaded.rollout_buffer.advantages,
                reloaded.rollout_buffer.returns,
            )
        )
        throughput = base_env.last_world_steps_per_second
        environment.close()

        timeout_base_env = MJWarpMidLevelVecEnv(
            dataset,
            num_envs=2,
            seed=17,
            chunk_steps=1,
            check_interval_steps=32,
            max_time=0.001,
            validate_task_execution=False,
        )
        timeout_environment = VecMonitor(timeout_base_env)
        timeout_policy = PPO(
            BoundedActorCriticPolicy,
            timeout_environment,
            seed=5,
            device="cpu",
            n_steps=1,
            batch_size=2,
            n_epochs=1,
            gamma=1.0,
            gae_lambda=1.0,
            learning_rate=3e-4,
            policy_kwargs={"net_arch": [32, 32], "log_std_init": -1.0},
        )
        timeout_policy.learn(total_timesteps=2)
        timeout_rewards_unbootstrapped = bool(
            np.allclose(
                timeout_policy.rollout_buffer.rewards[0],
                timeout_base_env.last_terminal_rewards,
                atol=1e-7,
                rtol=0.0,
            )
        )
        timeout_environment.close()

        overflow_forced_failure = False
        overflow_env = None
        try:
            overflow_env = MJWarpMidLevelVecEnv(
                dataset,
                num_envs=1,
                seed=17,
                chunk_steps=1,
                check_interval_steps=32,
                nconmax=1,
                njmax=8,
                max_time=0.001,
            )
            overflow_env.reset()
            overflow_env.step(actions[:1])
        except (ValueError, RuntimeError) as error:
            overflow_forced_failure = "capacity overflow" in str(error)
            overflow_forced_failure |= "capacity is below" in str(error)
        finally:
            if overflow_env is not None:
                overflow_env.close()

    print(
        f"timesteps={reloaded.num_timesteps} "
        f"rewards={np.round(rewards, 4).tolist()} "
        f"action={np.round(action[0], 4).tolist()} "
        f"world_steps_per_second={throughput:.0f} "
        f"rollout_finite={rollout_finite} "
        f"parameters_finite={parameters_finite} "
        f"bounded_samples={bounded_samples} "
        f"timeout_rewards_unbootstrapped={timeout_rewards_unbootstrapped} "
        f"overflow_forced_failure={overflow_forced_failure}"
    )
    if reloaded.num_timesteps != 4:
        raise RuntimeError("PPO checkpoint did not resume its timestep count.")
    if not rollout_finite or not parameters_finite:
        raise RuntimeError("MJWarp PPO produced non-finite training state.")
    if not bounded_samples:
        raise RuntimeError("PPO sampled an action outside the executed action Box.")
    if not timeout_rewards_unbootstrapped:
        raise RuntimeError("A timeout reward was altered by TimeLimit value bootstrapping.")
    if not overflow_forced_failure:
        raise RuntimeError("MJWarp capacity overflow did not force a failure.")
    if not np.all(np.isfinite(action)):
        raise RuntimeError("Reloaded PPO produced a non-finite action.")


if __name__ == "__main__":
    main()
