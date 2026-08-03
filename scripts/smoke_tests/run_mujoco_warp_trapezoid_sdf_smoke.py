"""Validate calibrated MuJoCo Warp trapezoid-SDF cushion contacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

import run_mujoco_warp_parity_smoke as parity
from snooker_env.table_geometry import BALL_RADIUS


@dataclass(frozen=True)
class ContactComparison:
    cpu_restitution: float
    warp_restitution: float
    cpu_position: np.ndarray
    warp_position: np.ndarray
    cpu_linear_velocity: np.ndarray
    warp_linear_velocity: np.ndarray
    cpu_angular_velocity: np.ndarray
    warp_angular_velocity: np.ndarray

    @property
    def restitution_error(self) -> float:
        return abs(self.cpu_restitution - self.warp_restitution)

    @property
    def position_error(self) -> float:
        return float(np.linalg.norm(self.cpu_position - self.warp_position))

    @property
    def linear_velocity_error(self) -> float:
        return float(
            np.linalg.norm(self.cpu_linear_velocity - self.warp_linear_velocity)
        )

    @property
    def angular_velocity_error(self) -> float:
        return float(
            np.linalg.norm(self.cpu_angular_velocity - self.warp_angular_velocity)
        )

    @property
    def angular_surface_velocity_error(self) -> float:
        return BALL_RADIUS * self.angular_velocity_error

    @property
    def warp_angular_surface_speed(self) -> float:
        return BALL_RADIUS * float(np.linalg.norm(self.warp_angular_velocity))


def _scenario(
    model_path: Path,
    speed: float,
    rail_y: float,
) -> parity.Scenario:
    scenario = replace(
        parity._trapezoid_cushion_scenario(model_path),
        duration=0.10,
        initial_speed=speed,
    )
    parity._set_ball(
        scenario.data,
        scenario.cue_qpos,
        scenario.cue_dof,
        (0.6542 - speed * 0.05, rail_y, parity.BALL_CENTER_Z),
        (speed, 0.0, 0.0),
    )
    mujoco.mj_forward(scenario.model, scenario.data)
    return scenario


def _motion_scenario(
    model_path: Path,
    linear_velocity: tuple[float, float, float],
    angular_velocity: tuple[float, float, float],
    rail_y: float = -0.40,
) -> parity.Scenario:
    """Place the ball 50 ms before an oblique/spinning rail impact."""

    velocity_x, velocity_y, _ = linear_velocity
    if velocity_x <= 0.0:
        raise ValueError("Trapezoid motion cases must travel toward the +X rail.")
    scenario = replace(
        parity._trapezoid_cushion_scenario(model_path),
        duration=0.10,
        initial_speed=velocity_x,
    )
    parity._set_ball(
        scenario.data,
        scenario.cue_qpos,
        scenario.cue_dof,
        (
            0.6542 - velocity_x * 0.05,
            rail_y - velocity_y * 0.05,
            parity.BALL_CENTER_Z,
        ),
        linear_velocity,
        angular_velocity,
    )
    mujoco.mj_forward(scenario.model, scenario.data)
    return scenario


def _full_table_scenario(
    model_path: Path,
    speed: float,
    rail_y: float,
) -> parity.Scenario:
    model = parity._load_model(model_path)
    data = mujoco.MjData(model)
    parity._park_cue(model, data)
    cue_qpos, cue_dof = parity._joint_addresses(model, "cue_ball_free")
    object_qpos, object_dof = parity._joint_addresses(model, "object_ball_0_free")
    parity._set_ball(
        data,
        cue_qpos,
        cue_dof,
        (0.35, rail_y, parity.BALL_CENTER_Z),
        (speed, 0.0, 0.0),
    )
    parity._set_ball(
        data,
        object_qpos,
        object_dof,
        (-0.30, 0.80, parity.BALL_CENTER_Z),
        (0.0, 0.0, 0.0),
    )
    mujoco.mj_forward(model, data)
    return parity.Scenario(
        name="full_table_transition",
        model=model,
        data=data,
        duration=0.22,
        cue_qpos=cue_qpos,
        cue_dof=cue_dof,
        initial_speed=speed,
    )


def _compare_scenarios(
    cpu_scenario: parity.Scenario,
    warp_scenario: parity.Scenario,
) -> ContactComparison:
    cpu_rollout = parity._run_cpu(cpu_scenario, pocket_sample_steps=100)
    warp_rollout = parity._run_warp(warp_scenario, pocket_sample_steps=100)
    return ContactComparison(
        cpu_restitution=parity._restitution(cpu_scenario, cpu_rollout),
        warp_restitution=parity._restitution(warp_scenario, warp_rollout),
        cpu_position=cpu_rollout.qpos[
            cpu_scenario.cue_qpos:cpu_scenario.cue_qpos + 3
        ],
        warp_position=warp_rollout.qpos[
            warp_scenario.cue_qpos:warp_scenario.cue_qpos + 3
        ],
        cpu_linear_velocity=cpu_rollout.qvel[
            cpu_scenario.cue_dof:cpu_scenario.cue_dof + 3
        ],
        warp_linear_velocity=warp_rollout.qvel[
            warp_scenario.cue_dof:warp_scenario.cue_dof + 3
        ],
        cpu_angular_velocity=cpu_rollout.qvel[
            cpu_scenario.cue_dof + 3:cpu_scenario.cue_dof + 6
        ],
        warp_angular_velocity=warp_rollout.qvel[
            warp_scenario.cue_dof + 3:warp_scenario.cue_dof + 6
        ],
    )


def _compare(
    model_path: Path,
    speed: float,
    rail_y: float,
) -> ContactComparison:
    return _compare_scenarios(
        _scenario(model_path, speed, rail_y),
        _scenario(model_path, speed, rail_y),
    )


def _compare_full_table(
    model_path: Path,
    speed: float,
    rail_y: float,
) -> ContactComparison:
    return _compare_scenarios(
        _full_table_scenario(model_path, speed, rail_y),
        _full_table_scenario(model_path, speed, rail_y),
    )


def _compare_motion(
    model_path: Path,
    linear_velocity: tuple[float, float, float],
    angular_velocity: tuple[float, float, float],
) -> ContactComparison:
    return _compare_scenarios(
        _motion_scenario(model_path, linear_velocity, angular_velocity),
        _motion_scenario(model_path, linear_velocity, angular_velocity),
    )


def _format_vector(value: np.ndarray) -> str:
    return np.array2string(value, precision=5, separator=",", suppress_small=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=parity.DEFAULT_MIDLEVEL_MODEL)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("The trapezoid-SDF smoke test requires a CUDA device.")
    wp.set_device(device)
    registration_model = parity._load_model(args.model)
    parity.register_mujoco_billiards_sdf(registration_model)

    speed_comparisons: list[ContactComparison] = []
    for speed in (0.3, 0.5, 1.0, 1.5, 2.0, 2.5):
        comparison = _compare(args.model, speed, -0.40)
        speed_comparisons.append(comparison)
        print(
            f"speed={speed:.1f} "
            f"restitution_cpu={comparison.cpu_restitution:.6f} "
            f"restitution_warp={comparison.warp_restitution:.6f} "
            f"restitution_error={comparison.restitution_error:.6f} "
            f"linear_error={comparison.linear_velocity_error:.6f} "
            f"angular_error={comparison.angular_velocity_error:.6f}"
        )

    rail_points = (
        -1.290,
        -1.275,
        -1.260,
        -1.250,
        -1.240,
        -1.230,
        -1.225,
        -1.200,
        -1.100,
        -0.800,
        -0.580,
        -0.400,
        -0.180,
        -0.130,
        -0.120,
        -0.110,
        -0.100,
        -0.090,
        -0.080,
        -0.070,
        -0.055,
        -0.040,
    )
    rail_comparisons: list[ContactComparison] = []
    face_comparisons: list[ContactComparison] = []
    for rail_y in rail_points:
        comparison = _compare(args.model, 2.0, rail_y)
        rail_comparisons.append(comparison)
        if -1.225 <= rail_y <= -0.100:
            face_comparisons.append(comparison)
        print(
            f"rail_y={rail_y:.3f} position_error={comparison.position_error:.6f} "
            f"linear_error={comparison.linear_velocity_error:.6f} "
            f"angular_error={comparison.angular_velocity_error:.6f} "
            f"angular_surface_error="
            f"{comparison.angular_surface_velocity_error:.6f} "
            f"linear_cpu={_format_vector(comparison.cpu_linear_velocity)} "
            f"linear_warp={_format_vector(comparison.warp_linear_velocity)}"
        )

    full_table_comparisons: list[ContactComparison] = []
    for rail_y in (-0.140, -0.080, -0.040):
        comparison = _compare_full_table(args.model, 2.0, rail_y)
        full_table_comparisons.append(comparison)
        print(
            f"full_table_y={rail_y:.3f} "
            f"position_error={comparison.position_error:.6f} "
            f"linear_error={comparison.linear_velocity_error:.6f} "
            f"angular_error={comparison.angular_velocity_error:.6f} "
            f"position_cpu={_format_vector(comparison.cpu_position)} "
            f"position_warp={_format_vector(comparison.warp_position)}"
        )

    motion_cases = (
        ("oblique_2_1", (2.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
        ("oblique_1_2", (1.0, 2.0, 0.0), (0.0, 0.0, 0.0)),
        ("spin_positive", (2.0, 0.5, 0.0), (0.0, 0.0, 20.0)),
        ("spin_negative", (2.0, 0.5, 0.0), (0.0, 0.0, -20.0)),
    )
    motion_comparisons: dict[str, ContactComparison] = {}
    for name, linear_velocity, angular_velocity in motion_cases:
        comparison = _compare_motion(
            args.model,
            linear_velocity,
            angular_velocity,
        )
        motion_comparisons[name] = comparison
        print(
            f"motion={name} position_error={comparison.position_error:.6f} "
            f"linear_error={comparison.linear_velocity_error:.6f} "
            f"angular_error={comparison.angular_velocity_error:.6f} "
            f"angular_surface_error="
            f"{comparison.angular_surface_velocity_error:.6f} "
            f"angular_cpu={_format_vector(comparison.cpu_angular_velocity)} "
            f"angular_warp={_format_vector(comparison.warp_angular_velocity)}"
        )

    failures: list[str] = []
    if max(item.restitution_error for item in speed_comparisons) > 0.07:
        failures.append("speed-sweep restitution error > 0.07")
    if max(item.linear_velocity_error for item in speed_comparisons) > 0.15:
        failures.append("speed-sweep linear-velocity error > 0.15 m/s")
    if max(item.position_error for item in rail_comparisons) > 0.015:
        failures.append("isolated rail position error > 0.015 m")
    if max(item.linear_velocity_error for item in rail_comparisons) > 0.17:
        failures.append("isolated rail linear-velocity error > 0.17 m/s")
    if max(item.angular_velocity_error for item in face_comparisons) > 4.0:
        failures.append("isolated rail-face angular-velocity error > 4 rad/s")
    # CPU MuJoCo emits a seed-dependent multi-contact cloud at each sharp end
    # cap and jumps by more than 7 rad/s between adjacent 1 cm samples. Judge
    # those locations in surface-speed units and independently bound MJWarp's
    # representative-manifold spin instead of fitting it to that discontinuity.
    # The flat rail face remains covered by the stricter 4 rad/s check above.
    if max(item.angular_surface_velocity_error for item in rail_comparisons) > 0.50:
        failures.append("isolated rail angular surface-speed error > 0.50 m/s")
    if max(item.warp_angular_surface_speed for item in rail_comparisons) > 0.25:
        failures.append("isolated rail MJWarp angular surface speed > 0.25 m/s")
    if max(item.position_error for item in full_table_comparisons) > 0.020:
        failures.append("full-table transition position error > 0.020 m")
    if max(item.linear_velocity_error for item in full_table_comparisons) > 0.35:
        failures.append("full-table transition linear-velocity error > 0.35 m/s")
    if (
        max(
            item.angular_surface_velocity_error
            for item in full_table_comparisons
        )
        > 0.36
    ):
        failures.append("full-table transition angular surface-speed error > 0.36 m/s")
    oblique_comparisons = (
        motion_comparisons["oblique_2_1"],
        motion_comparisons["oblique_1_2"],
    )
    spin_comparisons = (
        motion_comparisons["spin_positive"],
        motion_comparisons["spin_negative"],
    )
    if max(item.position_error for item in oblique_comparisons) > 0.008:
        failures.append("oblique-impact position error > 0.008 m")
    if max(item.linear_velocity_error for item in oblique_comparisons) > 0.12:
        failures.append("oblique-impact linear-velocity error > 0.12 m/s")
    if max(item.angular_velocity_error for item in oblique_comparisons) > 5.0:
        failures.append("oblique-impact angular-velocity error > 5 rad/s")
    if max(item.position_error for item in spin_comparisons) > 0.008:
        failures.append("spinning-impact position error > 0.008 m")
    if max(item.linear_velocity_error for item in spin_comparisons) > 0.10:
        failures.append("spinning-impact linear-velocity error > 0.10 m/s")
    if max(item.angular_surface_velocity_error for item in spin_comparisons) > 0.25:
        failures.append("spinning-impact angular surface-speed error > 0.25 m/s")
    if failures:
        raise RuntimeError("Trapezoid-SDF calibration failed: " + "; ".join(failures))
    print("mujoco_warp_trapezoid_sdf=PASS")


if __name__ == "__main__":
    main()
