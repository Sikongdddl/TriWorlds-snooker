"""Render the imported Gento role-aware IK/PPO rollout from two viewpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from snooker_env.gento_ik_env import GentoRoleIKEnv  # noqa: E402


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    suffix = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / suffix
    return ImageFont.truetype(path, size=size) if path.exists() else ImageFont.load_default()


def _configure_renderer(renderer: mujoco.Renderer) -> None:
    option = getattr(renderer, "scene_option", None)
    if option is None:
        option = getattr(renderer, "_scene_option", None)
    if option is not None:
        option.geomgroup[4] = 0
        option.sitegroup[:] = 0


def _annotate(
    frame: np.ndarray,
    *,
    title: str,
    elapsed: float,
    peak_ball_speed: float,
    max_cue_table_penetration: float,
    contacted: bool,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 78), fill=(7, 10, 14))
    draw.text((14, 8), title, font=_font(19, bold=True), fill=(245, 248, 255))
    draw.text(
        (14, 37),
        f"t={elapsed:4.2f}s  ball peak={peak_ball_speed:.3f} m/s  contact={'yes' if contacted else 'no'}",
        font=_font(13),
        fill=(181, 226, 255),
    )
    draw.text(
        (14, 57),
        f"cue/table max penetration={1e3 * max_cue_table_penetration:.3f} mm",
        font=_font(12),
        fill=(187, 255, 194),
    )
    return np.asarray(image)


def _metrics_template() -> dict[str, float]:
    return {
        "max_robot_table_penetration": 0.0,
        "max_cue_palm_penetration": 0.0,
        "max_cue_table_penetration": 0.0,
        "support_direction_error": 0.0,
        "front_support_error": 0.0,
        "rear_grip_error": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "assets" / "policies" / "gento_role_ik_residual_final.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "gento_dev_midlevel" / "gento_ik_dual_view.mp4",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=2.20)
    args = parser.parse_args()
    if args.width % 2:
        raise ValueError("--width must be even so both camera panels have equal width.")

    policy = PPO.load(args.checkpoint, device="cpu")
    env = GentoRoleIKEnv()
    observation, _ = env.reset(seed=7)
    cue_ball_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "cue_ball",
    )
    start_ball = env.data.xpos[cue_ball_id].copy()

    panel_width = args.width // 2
    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, panel_width)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, args.height)
    closeup_renderer = mujoco.Renderer(
        env.model,
        width=panel_width,
        height=args.height,
    )
    top_renderer = mujoco.Renderer(
        env.model,
        width=panel_width,
        height=args.height,
    )
    _configure_renderer(closeup_renderer)
    _configure_renderer(top_renderer)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    poster_path = args.output.with_suffix(".png")
    metrics_path = args.output.with_suffix(".json")
    maxima = _metrics_template()
    last_info: dict[str, object] = {}
    done = False
    poster_written = False
    sim_timestep = float(env.model.opt.timestep)

    with imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=9) as writer:
        for frame_index in range(int(round(args.seconds * args.fps))):
            target_time = frame_index / args.fps
            while env.data.time + 0.5 * sim_timestep < target_time:
                if not done:
                    action, _ = policy.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, last_info = env.step(action)
                    done = terminated or truncated
                    for key in maxima:
                        maxima[key] = max(maxima[key], float(last_info[key]))
                else:
                    mujoco.mj_step(env.model, env.data)

            closeup_renderer.update_scene(env.data, camera="gento_grip_closeup")
            closeup = _annotate(
                closeup_renderer.render(),
                title="SIDE GRASP  |  right hand: support / direction",
                elapsed=float(env.data.time),
                peak_ball_speed=float(env._peak_cue_ball_speed),
                max_cue_table_penetration=maxima["max_cue_table_penetration"],
                contacted=env._first_cue_ball_contact_time is not None,
            )
            top_renderer.update_scene(env.data, camera="gento_shot_top")
            top = _annotate(
                top_renderer.render(),
                title="TOP VIEW  |  left hand: stroke velocity",
                elapsed=float(env.data.time),
                peak_ball_speed=float(env._peak_cue_ball_speed),
                max_cue_table_penetration=maxima["max_cue_table_penetration"],
                contacted=env._first_cue_ball_contact_time is not None,
            )
            combined = np.concatenate((closeup, top), axis=1)
            writer.append_data(combined)
            if env._first_cue_ball_contact_time is not None and not poster_written:
                imageio.imwrite(poster_path, combined)
                poster_written = True

    end_ball = env.data.xpos[cue_ball_id].copy()
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "video": str(args.output.resolve()),
        "success": bool(last_info.get("success", False)),
        "episode_finished": done,
        "first_cue_ball_contact_time_s": env._first_cue_ball_contact_time,
        "peak_cue_ball_speed_m_per_s": env._peak_cue_ball_speed,
        "cue_ball_displacement_xyz_m": (end_ball - start_ball).tolist(),
        **maxima,
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    closeup_renderer.close()
    top_renderer.close()
    env.close()

    print(json.dumps(summary, indent=2))
    print(f"wrote_video={args.output}")
    print(f"wrote_contact_frame={poster_path}")
    print(f"wrote_metrics={metrics_path}")
    if not done or not summary["success"]:
        raise RuntimeError("Gento PPO rollout did not complete successfully.")
    if maxima["max_robot_table_penetration"] > 0.003:
        raise RuntimeError("Rendered rollout penetrated the robot/table collision body.")
    if maxima["max_cue_palm_penetration"] > 0.002:
        raise RuntimeError("Rendered rollout let the cue pass through a palm guard.")
    if maxima["max_cue_table_penetration"] > 0.001:
        raise RuntimeError("Rendered rollout penetrated the source table geometry.")


if __name__ == "__main__":
    main()
