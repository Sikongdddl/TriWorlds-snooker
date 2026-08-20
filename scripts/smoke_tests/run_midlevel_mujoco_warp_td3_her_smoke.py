"""Exercise MJWarp rollout, TD3+BC+HER, checkpoint, and resume."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
from stable_baselines3.common.vec_env import VecMonitor

from _bootstrap import add_src_to_path

add_src_to_path()

from run_midlevel_two_ball_ppo_env_smoke import _fixture_task  # noqa: E402
from snooker_env.midlevel_ppo import generated_behavior_cloning_data  # noqa: E402
from snooker_env.midlevel_mujoco_warp_vec_env import (  # noqa: E402
    MJWarpMidLevelVecEnv,
    active_mujoco_warp_backend_sha256,
)
from snooker_env.midlevel_sac_her import (  # noqa: E402
    ConservativeResidualTD3Policy,
    HINDSIGHT_SUCCESS_REWARD,
    MidLevelGeometricFeatures,
    TD3CheckpointMidLevelPolicy,
    SingleStepTD3BC,
    SingleStepCuePositionHerReplayBuffer,
    behavior_clone_td3_policy,
    collect_local_speed_probes,
    prefill_certified_replay_buffer,
    replay_buffer_path,
)
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
    task = replace(
        fixture,
        cue_position=cue_position,
        generated_direction=ghost_ball_direction(
            cue_position,
            fixture.object_position,
            fixture.pocket_position,
        ),
        generated_speed=quantize_cue_speed(1.0),
    )
    second_task = replace(task, candidate_seed=task.candidate_seed + 1)
    dataset = TwoBallTaskDataset.from_tasks(
        [task, second_task],
        simulator,
        generation_seed=7,
        physics_backend=MUJOCO_WARP_PHYSICS_BACKEND,
        backend_hash=active_mujoco_warp_backend_sha256()[2],
    )
    with tempfile.TemporaryDirectory(
        prefix="midlevel-mujoco-warp-td3-her-smoke-"
    ) as directory:
        checkpoint = Path(directory) / "td3_her_smoke"
        base_env = MJWarpMidLevelVecEnv(
            dataset,
            num_envs=2,
            seed=17,
            chunk_steps=2,
            check_interval_steps=512,
            max_time=0.60,
            stop_speed=10.0,
            stop_hold_time=0.40,
            validate_task_execution=False,
        )
        base_env.seed(17)
        initial_observation = base_env.reset()
        generated_action = np.array(
            [0.0, encode_speed_action(task.generated_speed)],
            dtype=np.float32,
        )
        actions = np.tile(generated_action, (base_env.num_envs, 1))
        _, rewards, dones, infos = base_env.step(actions)
        if not np.all(dones) or not all(bool(info["correct_pot"]) for info in infos):
            raise RuntimeError("Generated actions did not produce successful terminal shots.")

        environment = VecMonitor(base_env)
        policy = SingleStepTD3BC(
            ConservativeResidualTD3Policy,
            environment,
            seed=5,
            device="cuda:0",
            learning_rate=3e-4,
            actor_learning_rate=1e-5,
            critic_learning_rate=3e-4,
            actor_update_interval=4,
            actor_learning_starts=0,
            residual_l2_weight=0.02,
            buffer_size=64,
            learning_starts=0,
            batch_size=2,
            tau=0.005,
            gamma=0.0,
            train_freq=(1, "step"),
            gradient_steps=1,
            replay_buffer_class=SingleStepCuePositionHerReplayBuffer,
            replay_buffer_kwargs={
                "her_ratio": 1.0,
                "success_ratio": 0.0,
                "failure_ratio": 0.0,
            },
            policy_kwargs={
                "net_arch": [32, 32],
                "features_extractor_class": MidLevelGeometricFeatures,
            },
        )
        bc_report = behavior_clone_td3_policy(
            policy.policy,
            dataset,
            epochs=50,
            batch_size=2,
            learning_rate=1e-3,
            final_learning_rate=1e-4,
            angle_weight=4.0,
            speed_weight=4.0,
            seed=5,
        )
        bc_observations, bc_actions = generated_behavior_cloning_data(dataset)
        policy.configure_conservative_speed_residual(
            max_speed_residual_mps=0.12,
            exploration_initial_std=0.35,
            exploration_final_std=0.05,
            exploration_decay_timesteps=4,
        )
        policy.configure_discrete_candidate_ranking((-0.02, 0.0, 0.02))
        policy.configure_behavior_cloning_reference(
            bc_observations,
            bc_actions,
            initial_weight=1.0,
            final_weight=0.1,
            decay_actor_updates=4,
            batch_size=2,
            angle_weight=4.0,
            speed_weight=4.0,
        )
        prefill_report = prefill_certified_replay_buffer(
            policy,
            dataset,
            bc_observations,
            bc_actions,
        )
        probe_report = collect_local_speed_probes(
            policy,
            environment,
            dataset,
            bc_observations,
            bc_actions,
            task_count=2,
            offsets_mps=(-0.02, 0.0, 0.02),
            seed=5,
        )
        if policy.replay_buffer.local_probe_transition_count != 6:
            raise RuntimeError("GPU local speed probes were not stored in replay.")
        if policy.replay_buffer.local_probe_group_count != 6:
            raise RuntimeError("GPU speed probes were not grouped in replay.")
        warmup_report = policy.warmup_critic(2, batch_size=2)
        terminal_observations = np.stack(
            [info["terminal_observation"] for info in infos]
        ).astype(np.float32)
        policy.replay_buffer.add(
            initial_observation,
            terminal_observations,
            actions,
            rewards,
            dones,
            infos,
        )
        if policy.replay_buffer.eligible_transition_count < 4:
            raise RuntimeError("Successful GPU transitions were not admitted to HER.")
        hindsight = policy.replay_buffer.sample(16)
        if not np.allclose(
            hindsight.rewards.cpu().numpy(),
            HINDSIGHT_SUCCESS_REWARD,
        ):
            raise RuntimeError("GPU hindsight samples do not have maximum reward.")

        policy.learn(total_timesteps=2)
        policy.save(checkpoint)
        policy.save_replay_buffer(replay_buffer_path(checkpoint))
        reloaded = SingleStepTD3BC.load(
            checkpoint,
            env=environment,
            device="cuda:0",
        )
        reloaded.load_replay_buffer(replay_buffer_path(checkpoint))
        reloaded.learn(total_timesteps=2, reset_num_timesteps=False)
        observation = environment.reset()
        action, _ = reloaded.predict(observation, deterministic=True)
        parameters_finite = all(
            bool(np.all(np.isfinite(parameter.detach().cpu().numpy())))
            for parameter in reloaded.policy.parameters()
        )
        replay_sample = reloaded.replay_buffer.sample(16)
        replay_finite = bool(
            np.all(np.isfinite(replay_sample.rewards.cpu().numpy()))
            and np.all(np.isfinite(replay_sample.actions.cpu().numpy()))
        )
        bounded_action = bool(np.all(action >= -1.0) and np.all(action <= 1.0))
        angle_locked = bool(np.array_equal(action[:, 0], np.zeros(len(action))))
        adapter = TD3CheckpointMidLevelPolicy(checkpoint)
        predicted = adapter.predict(
            task.cue_position,
            task.object_position,
            task.pocket_position,
            task.target_stop_position,
        )
        throughput = base_env.last_world_steps_per_second
        environment.close()

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
        f"timesteps={reloaded.num_timesteps} rewards={np.round(rewards, 4).tolist()} "
        f"her_eligible={reloaded.replay_buffer.eligible_transition_count} "
        f"bc_initial={bc_report.initial_loss:.6g} bc_final={bc_report.final_loss:.6g} "
        f"prefill={prefill_report['stored_count']} "
        f"local_probes={probe_report['stored_transition_count']} "
        f"warmup_final={warmup_report['final_loss']:.6g} "
        f"world_steps_per_second={throughput:.0f} replay_finite={replay_finite} "
        f"parameters_finite={parameters_finite} bounded_action={bounded_action} "
        f"angle_locked={angle_locked} "
        f"overflow_forced_failure={overflow_forced_failure}"
    )
    if reloaded.num_timesteps != 4:
        raise RuntimeError("TD3 checkpoint did not resume its timestep count.")
    if bc_report.final_loss >= bc_report.initial_loss:
        raise RuntimeError("TD3 behavior cloning did not reduce reconstruction loss.")
    if not replay_finite or not parameters_finite or not bounded_action or not angle_locked:
        raise RuntimeError("TD3 produced non-finite or unbounded training state.")
    if not np.all(np.isfinite(predicted.normalized_action)):
        raise RuntimeError("TD3 checkpoint adapter produced a non-finite action.")
    if not overflow_forced_failure:
        raise RuntimeError("MJWarp capacity overflow did not force a failure.")


if __name__ == "__main__":
    main()
