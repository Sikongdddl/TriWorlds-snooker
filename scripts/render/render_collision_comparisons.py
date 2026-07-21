"""Render baseline-versus-calibrated billiards collision comparisons."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.contact_events import CollisionEventMonitor  # noqa: E402
from snooker_env.init_pose import set_lift_grip_ready_pose  # noqa: E402


@dataclass
class Panel:
    label: str
    model: mujoco.MjModel
    data: mujoco.MjData
    renderer: mujoco.Renderer
    camera: mujoco.MjvCamera
    monitor: CollisionEventMonitor
    ball_dofs: tuple[int, ...]
    ball_bodies: tuple[int, ...]
    physics_summary: str
    rack_summary: str
    peak_ball_speed: float = 0.0


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Missing {object_type.name}: {name}")
    return object_id


def _geom_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or f"geom_{index}"
        for index in range(model.ngeom)
    )


def _configure_baseline(model: mujoco.MjModel) -> None:
    """Recreate the pre-calibration collision settings in memory."""

    model.opt.timestep = 0.001
    for geom_id, name in enumerate(_geom_names(model)):
        if name == "cue_ball_geom" or name.startswith("object_ball_") and name.endswith("_geom"):
            model.geom_priority[geom_id] = 2
            model.geom_condim[geom_id] = 6
            model.geom_friction[geom_id] = (0.18, 0.004, 0.00008)
            model.geom_solref[geom_id] = (0.0025, 1.0)
            model.geom_solimp[geom_id] = (0.95, 0.99, 0.001, 0.5, 2.0)
        elif name == "playfield_collision":
            model.geom_priority[geom_id] = 0
            model.geom_condim[geom_id] = 3
            model.geom_friction[geom_id] = (0.18, 0.004, 0.00008)
            model.geom_solref[geom_id] = (0.003, 1.0)
        elif name.startswith("cushion_nose_"):
            # Keep the compiled rail collision active, but give it the old
            # low-priority settings so the ball geom overrides the contact.
            model.geom_contype[geom_id] = 1
            model.geom_conaffinity[geom_id] = 2
            model.geom_priority[geom_id] = 0
            model.geom_condim[geom_id] = 3
            model.geom_friction[geom_id] = (0.35, 0.006, 0.0001)
            model.geom_solref[geom_id] = (0.02, 1.0)
        elif name.startswith("pocket_catch_") or name.startswith("pocket_wall_"):
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
        elif name.startswith("cushion_"):
            model.geom_contype[geom_id] = 1
            model.geom_conaffinity[geom_id] = 2
            model.geom_priority[geom_id] = 0
            model.geom_condim[geom_id] = 3
            model.geom_friction[geom_id] = (0.35, 0.006, 0.0001)
            model.geom_solref[geom_id] = (0.02, 1.0)
        elif name == "cue_shaft":
            model.geom_contype[geom_id] = 4
            model.geom_conaffinity[geom_id] = 2
            model.geom_condim[geom_id] = 6
            model.geom_friction[geom_id] = (0.25, 0.006, 0.0001)
            model.geom_solref[geom_id] = (0.0025, 1.0)
        elif name == "cue_tip":
            model.geom_solref[geom_id] = (0.002, 1.0)


def _camera(scenario: str) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = -90.0
    if scenario == "ball_ball":
        camera.distance, camera.elevation = 1.55, -55.0
        camera.lookat[:] = (-0.02, 0.0, 0.78)
    elif scenario == "cushion":
        camera.distance, camera.elevation = 1.55, -52.0
        camera.lookat[:] = (-0.35, 0.20, 0.78)
    elif scenario == "pocket":
        camera.distance, camera.elevation = 1.35, -25.0
        camera.lookat[:] = (0.0, 0.55, 0.73)
    else:
        camera.distance, camera.elevation = 2.05, -72.0
        camera.lookat[:] = (0.58, 0.0, 0.78)
    return camera


def _set_free_joint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos = int(model.jnt_qposadr[joint_id])
    dof = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos:qpos + 3] = position
    data.qpos[qpos + 3:qpos + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[dof:dof + 3] = velocity
    data.qvel[dof + 3:dof + 6] = 0.0


def _setup(model: mujoco.MjModel, data: mujoco.MjData, scenario: str) -> None:
    if scenario == "ball_ball":
        _set_free_joint(model, data, "cue_ball_free", (-0.42, 0.0, 0.789575), (3.0, 0.0, 0.0))
        _set_free_joint(model, data, "object_ball_0_free", (0.02, 0.0, 0.789575))
    elif scenario == "cushion":
        _set_free_joint(model, data, "cue_ball_free", (-0.40, 0.20, 0.789575), (0.45, 1.50, 0.0))
        _set_free_joint(model, data, "object_ball_0_free", (0.70, -0.20, 0.789575))
    elif scenario == "pocket":
        _set_free_joint(model, data, "cue_ball_free", (0.0, 0.32, 0.789575), (0.0, 0.80, 0.0))
        _set_free_joint(model, data, "object_ball_0_free", (0.70, -0.20, 0.789575))
    elif scenario == "break":
        _set_free_joint(model, data, "cue_ball_free", (0.20, 0.0, 0.789575), (3.0, 0.0, 0.0))
    else:
        raise ValueError(scenario)
    mujoco.mj_forward(model, data)


def _ball_ids(model: mujoco.MjModel) -> tuple[tuple[int, ...], tuple[int, ...]]:
    dofs: list[int] = []
    bodies: list[int] = []
    for body_name in ("cue_ball",) + tuple(f"object_ball_{index}" for index in range(16)):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{body_name}_free")
        if body_id >= 0 and joint_id >= 0:
            bodies.append(body_id)
            dofs.append(int(model.jnt_dofadr[joint_id]))
    return tuple(dofs), tuple(bodies)


def _physics_summary(model: mujoco.MjModel) -> str:
    """Expose the compiled contact parameters used by this exact panel."""

    ball_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_ball_geom")
    friction = model.geom_friction[ball_geom]
    solref = model.geom_solref[ball_geom]
    return (
        f"dt={model.opt.timestep * 1000.0:.2f}ms  "
        f"ball condim={int(model.geom_condim[ball_geom])}  "
        f"mu={friction[0]:.3f}  solref=({solref[0]:.2g},{solref[1]:.2g})"
    )


def _rack_summary(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    """Report the minimum surface clearance within the object-ball rack."""

    geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"object_ball_{index}_geom")
        for index in range(16)
    ]
    geom_ids = [geom_id for geom_id in geom_ids if geom_id >= 0]
    clearances: list[float] = []
    for first_index, first_geom in enumerate(geom_ids):
        first_body = int(model.geom_bodyid[first_geom])
        first_radius = float(model.geom_size[first_geom, 0])
        for second_geom in geom_ids[first_index + 1:]:
            second_body = int(model.geom_bodyid[second_geom])
            second_radius = float(model.geom_size[second_geom, 0])
            clearance = float(
                np.linalg.norm(data.xpos[first_body] - data.xpos[second_body])
                - first_radius
                - second_radius
            )
            clearances.append(clearance)
    minimum = min(clearances) if clearances else float("nan")
    return f"rack minimum gap={minimum * 1000.0:.3f}mm"


def _make_panel(model_path: Path, label: str, baseline: bool, scenario: str, width: int, height: int) -> Panel:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if baseline:
        _configure_baseline(model)
    # The decorative ceiling lamp sits between a top-down camera and the
    # table. Move only its non-colliding visual geom to a hidden render group.
    ceiling_light = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "ceiling_light_low_light_mat_0_visual",
    )
    if ceiling_light >= 0:
        model.geom_group[ceiling_light] = 4
    data = mujoco.MjData(model)
    if scenario == "break":
        set_lift_grip_ready_pose(model, data)
    _setup(model, data, scenario)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, width=width, height=height)
    option = getattr(renderer, "scene_option", getattr(renderer, "_scene_option", None))
    if option is not None:
        option.geomgroup[3] = 0
        option.geomgroup[4] = 0
        option.sitegroup[3] = 0
    dofs, bodies = _ball_ids(model)
    return Panel(
        label,
        model,
        data,
        renderer,
        _camera(scenario),
        CollisionEventMonitor(model),
        dofs,
        bodies,
        _physics_summary(model),
        _rack_summary(model, data),
    )


def _advance(panel: Panel, target_time: float) -> None:
    while panel.data.time + 0.5 * panel.model.opt.timestep < target_time:
        mujoco.mj_step(panel.model, panel.data)
        panel.monitor.scan(panel.data)
        panel.peak_ball_speed = max(
            panel.peak_ball_speed,
            max(float(np.linalg.norm(panel.data.qvel[dof:dof + 3])) for dof in panel.ball_dofs),
        )


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(path, size=size) if path.exists() else ImageFont.load_default()


def _annotate(frame: np.ndarray, panel: Panel, time: float) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 116), fill=(8, 8, 10))
    kinds = [event.kind for event in panel.monitor.events]
    ball_z = min(float(panel.data.xpos[body, 2]) for body in panel.ball_bodies)
    draw.text((14, 7), panel.label, font=_font(19), fill=(255, 255, 255))
    draw.text(
        (14, 32),
        panel.physics_summary,
        font=_font(12),
        fill=(180, 225, 255),
    )
    draw.text(
        (14, 53),
        panel.rack_summary,
        font=_font(12),
        fill=(190, 255, 190),
    )
    draw.text(
        (14, 75),
        f"t={time:.2f}s  peak={panel.peak_ball_speed:.2f}m/s  min_z={ball_z:.3f}m",
        font=_font(12),
        fill=(185, 220, 255),
    )
    draw.text(
        (14, 96),
        f"ball-ball={kinds.count('ball_ball')}  rail={kinds.count('ball_cushion')}  pocket={kinds.count('ball_pocket')}",
        font=_font(12),
        fill=(255, 210, 120),
    )
    return np.asarray(image)


def render_scenario(
    scenario: str,
    output: Path,
    width: int,
    height: int,
    fps: int,
    seconds: float,
    break_model: Path,
) -> None:
    model_path = break_model if scenario == "break" else ROOT / "models" / "midlevel_train_scene.xml"
    print(f"{scenario}: loading current model from {model_path.resolve()}")
    panels = (
        _make_panel(model_path, "LEGACY: in-memory parameter override", True, scenario, width, height),
        _make_panel(model_path, "CURRENT: compiled working-tree XML", False, scenario, width, height),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=9) as writer:
        for frame_index in range(int(round(seconds * fps))):
            time = frame_index / fps
            frames: list[np.ndarray] = []
            for panel in panels:
                _advance(panel, time)
                panel.renderer.update_scene(panel.data, camera=panel.camera)
                frames.append(_annotate(panel.renderer.render(), panel, time))
            writer.append_data(np.concatenate(frames, axis=1))

    for panel in panels:
        counts = {kind: sum(event.kind == kind for event in panel.monitor.events) for kind in (
            "ball_ball", "ball_cushion", "ball_pocket"
        )}
        minimum_z = min(float(panel.data.xpos[body, 2]) for body in panel.ball_bodies)
        print(
            f"{scenario}:{panel.label}: peak={panel.peak_ball_speed:.6f} "
            f"minimum_z={minimum_z:.6f} events={counts} "
            f"physics=[{panel.physics_summary}] {panel.rack_summary}"
        )
        panel.renderer.close()
    print(f"wrote={output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("ball_ball", "cushion", "pocket", "break", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "collision_validation")
    parser.add_argument(
        "--break-model",
        type=Path,
        default=ROOT / "models" / "scene_pool_asset.xml",
        help="Current working-tree model used by the break comparison.",
    )
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    durations = {"ball_ball": 1.8, "cushion": 2.2, "pocket": 1.8, "break": 2.6}
    scenarios = tuple(durations) if args.scenario == "all" else (args.scenario,)
    for scenario in scenarios:
        render_scenario(
            scenario,
            args.output_dir / f"collision_{scenario}_before_after.mp4",
            args.panel_width,
            args.height,
            args.fps,
            durations[scenario],
            args.break_model,
        )


if __name__ == "__main__":
    main()
