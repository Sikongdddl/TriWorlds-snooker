#!/usr/bin/env python3
"""Render a saved PoolTool planner rollout as a top-down MP4 video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snooker_env.pooltool_runtime import require_pooltool  # noqa: E402
from snooker_env.pooltool_visualization import BALL_COLORS  # noqa: E402


STATE_NAMES = {
    0: "stationary",
    1: "spinning",
    2: "sliding",
    3: "rolling",
    4: "pocketed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, default=Path("outputs/pooltool/clearance_rollout.msgpack"))
    parser.add_argument("--plan", type=Path, default=Path("outputs/pooltool/clearance_plan.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/pooltool/clearance_rollout.mp4"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.02, help="PoolTool continuous trajectory sampling interval.")
    parser.add_argument("--seconds-per-shot-cap", type=float, default=5.5)
    parser.add_argument("--pause-seconds", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pt = require_pooltool()
    if not args.rollout.exists():
        raise SystemExit(f"Rollout file does not exist: {args.rollout}")

    multisystem = pt.MultiSystem.load(args.rollout)
    if len(multisystem) == 0:
        raise SystemExit(f"Rollout file contains no systems: {args.rollout}")

    plan = _load_plan(args.plan)
    shot_records = _rollout_records(plan, len(multisystem))
    systems = []
    for system in multisystem:
        shot = system.copy()
        if shot.simulated and not shot.continuized:
            pt.continuize(shot, dt=args.dt, inplace=True)
        systems.append(shot)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=9) as writer:
        for idx, system in enumerate(systems):
            record = shot_records[idx]
            duration = max(_system_duration(system), 0.2)
            playback_duration = min(duration, args.seconds_per_shot_cap)
            frame_count = max(1, int(round(playback_duration * args.fps)))
            for frame_idx in range(frame_count):
                alpha = frame_idx / max(frame_count - 1, 1)
                time = alpha * duration
                frame = _draw_frame(system, record, idx, len(systems), time, args.width, args.height)
                writer.append_data(np.asarray(frame))

            pause_count = int(round(args.pause_seconds * args.fps))
            for _ in range(pause_count):
                frame = _draw_frame(system, record, idx, len(systems), duration, args.width, args.height)
                writer.append_data(np.asarray(frame))

    print(f"wrote={args.output}")
    print(f"shots={len(systems)} fps={args.fps} size={args.width}x{args.height}")


def _load_plan(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _rollout_records(plan: dict[str, Any], count: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if plan.get("break") is not None:
        break_record = dict(plan["break"])
        break_record.update(
            {
                "label": "scripted break",
                "target_ball_id": break_record.get("target_ball_id", "1"),
                "target_pocket_id": "-",
                "success": not bool(break_record.get("cue_scratch", False)),
                "foul": bool(break_record.get("cue_scratch", False)),
                "reason": "break",
                "score": float(break_record.get("ball_spread", 0.0)),
                "solution": {
                    "speed": break_record.get("speed"),
                    "phi": break_record.get("phi"),
                    "side_spin": 0.0,
                    "top_spin": 0.0,
                    "elevation": 0.0,
                },
            }
        )
        records.append(break_record)
    for idx, shot in enumerate(plan.get("shots", [])):
        records.append({"label": f"planner shot {idx}", **shot})
    while len(records) < count:
        records.append({"label": f"shot {len(records)}"})
    return records[:count]


def _system_duration(system: Any) -> float:
    times = [float(system.t)]
    for ball in system.balls.values():
        if not ball.history_cts.empty:
            _rvw, _ss, ts = ball.history_cts.vectorize()
            if len(ts):
                times.append(float(np.nanmax(ts)))
    return max(times)


def _draw_frame(system: Any, record: dict[str, Any], shot_idx: int, shot_count: int, time: float, width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), "#edf4ef")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()

    margin = 34
    sidebar_w = min(390, max(320, int(width * 0.31)))
    table_box = (margin, margin, width - sidebar_w - margin, height - margin)
    projector = _Projector(system, table_box)
    _draw_table(draw, system, projector)

    for ball_id, ball in sorted(system.balls.items()):
        samples = _trajectory_samples(ball, time)
        if len(samples) > 1:
            points = [projector.xy(point) for point in samples if np.all(np.isfinite(point))]
            if len(points) > 1:
                draw.line(points, fill=_rgb(BALL_COLORS.get(ball_id, "#9aa6b2")), width=3)

    for ball_id, ball in sorted(system.balls.items()):
        pos, state = _ball_at(ball, time)
        if pos is None:
            continue
        _draw_ball(draw, projector, ball_id, pos, int(state), fonts["tiny"])

    _draw_sidebar(draw, record, system, shot_idx, shot_count, time, width, height, sidebar_w, fonts)
    return image


def _draw_table(draw: ImageDraw.ImageDraw, system: Any, projector: _Projector) -> None:
    x0, y0, x1, y1 = projector.box
    rail_pad = 18
    draw.rounded_rectangle((x0 - rail_pad, y0 - rail_pad, x1 + rail_pad, y1 + rail_pad), radius=20, fill="#5a3424")

    felt_box = (
        int(round(projector.origin_x)),
        int(round(projector.origin_y)),
        int(round(projector.origin_x + projector.draw_w)),
        int(round(projector.origin_y + projector.draw_h)),
    )
    draw.rectangle(felt_box, fill="#0b6b4b")

    for pocket in system.table.pockets.values():
        px, py = projector.xy(pocket.center[:2])
        pr = max(10, int(round(float(pocket.radius) * projector.scale * 1.28)))
        draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill="#050806")

    cushion_width = max(7, int(round(0.018 * projector.scale)))
    for cushion in system.table.cushion_segments.linear.values():
        p1 = projector.xy(cushion.p1[:2])
        p2 = projector.xy(cushion.p2[:2])
        draw.line((p1, p2), fill="#123d30", width=cushion_width)


def _draw_ball(draw: ImageDraw.ImageDraw, projector: _Projector, ball_id: str, pos: np.ndarray, state: int, font: ImageFont.ImageFont) -> None:
    x, y = projector.xy(pos[:2])
    r = max(8, int(projector.ball_radius))
    fill = BALL_COLORS.get(ball_id, "#9aa6b2")
    outline = "#111111" if ball_id == "cue" else "#26352d"
    if state == 4:
        draw.ellipse((x - r, y - r, x + r, y + r), outline=outline, width=2)
        return
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=2)
    text_fill = "#111111" if ball_id in {"cue", "1", "9"} else "#ffffff"
    bbox = draw.textbbox((0, 0), ball_id, font=font)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2 - 1), ball_id, fill=text_fill, font=font)


def _draw_sidebar(
    draw: ImageDraw.ImageDraw,
    record: dict[str, Any],
    system: Any,
    shot_idx: int,
    shot_count: int,
    time: float,
    width: int,
    height: int,
    sidebar_w: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    x0 = width - sidebar_w + 18
    y = 34
    draw.text((x0, y), "PoolTool planner rollout", fill="#17211b", font=fonts["title"])
    y += 38
    draw.text((x0, y), f"{record.get('label', f'shot {shot_idx}')}   {shot_idx + 1}/{shot_count}", fill="#33443a", font=fonts["body"])
    y += 28
    draw.text((x0, y), f"t = {time:5.2f}s", fill="#5f6f65", font=fonts["body"])
    y += 34

    solution = record.get("solution") or {}
    lines = [
        ("target", f"{record.get('target_ball_id', '-')}/{record.get('target_pocket_id', '-')}"),
        ("success", str(record.get("success", "-"))),
        ("foul", str(record.get("foul", "-"))),
        ("score", _fmt(record.get("score"))),
        ("reason", str(record.get("reason", "-"))),
        ("V0", _fmt(solution.get("speed"))),
        ("phi", _fmt(solution.get("phi"))),
    ]
    for key, value in lines:
        draw.text((x0, y), key, fill="#6a796f", font=fonts["small"])
        draw.text((x0 + 95, y), value, fill="#17211b", font=fonts["small"])
        y += 22

    y += 12
    draw.text((x0, y), "live ball states", fill="#17211b", font=fonts["body_bold"])
    y += 24
    for ball_id, ball in sorted(system.balls.items())[:10]:
        pos, state = _ball_at(ball, time)
        if pos is None:
            continue
        color = _rgb(BALL_COLORS.get(ball_id, "#9aa6b2"))
        draw.ellipse((x0, y + 4, x0 + 10, y + 14), fill=color, outline="#1a241e")
        draw.text((x0 + 18, y), f"{ball_id:>3} {STATE_NAMES.get(int(state), str(state))}", fill="#34453b", font=fonts["tiny"])
        y += 18

    y += 12
    draw.text((x0, y), "first events", fill="#17211b", font=fonts["body_bold"])
    y += 24
    for event in list(system.events)[:9]:
        label = f"{float(event.time):4.2f} {str(event.event_type)} {'/'.join(str(item) for item in event.ids)}"
        draw.text((x0, y), label[:48], fill="#34453b", font=fonts["tiny"])
        y += 18
        if y > height - 32:
            break


def _trajectory_samples(ball: Any, time: float) -> list[np.ndarray]:
    if ball.history_cts.empty:
        state = np.asarray(ball.state.rvw[0, :2], dtype=float)
        return [state] if np.all(np.isfinite(state)) else []
    rvw, _ss, ts = ball.history_cts.vectorize()
    mask = ts <= time
    if not np.any(mask):
        return []
    points = rvw[mask, 0, :2]
    stride = max(1, len(points) // 180)
    return [np.asarray(point, dtype=float) for point in points[::stride]]


def _ball_at(ball: Any, time: float) -> tuple[np.ndarray | None, int]:
    if ball.history_cts.empty:
        pos = np.asarray(ball.state.rvw[0, :2], dtype=float)
        return (pos if np.all(np.isfinite(pos)) else None), int(ball.state.s)
    rvw, ss, ts = ball.history_cts.vectorize()
    idx = int(np.searchsorted(ts, time, side="right") - 1)
    idx = max(0, min(idx, len(ts) - 1))
    pos = np.asarray(rvw[idx, 0, :2], dtype=float)
    return (pos if np.all(np.isfinite(pos)) else None), int(ss[idx])


class _Projector:
    def __init__(self, system: Any, box: tuple[int, int, int, int]) -> None:
        self.system = system
        self.box = box
        x0, y0, x1, y1 = box
        table_w = float(system.table.w)
        table_l = float(system.table.l)
        self.scale = min((x1 - x0) / table_w, (y1 - y0) / table_l)
        self.draw_w = table_w * self.scale
        self.draw_h = table_l * self.scale
        self.origin_x = x0 + ((x1 - x0) - self.draw_w) / 2.0
        self.origin_y = y0 + ((y1 - y0) - self.draw_h) / 2.0
        self.ball_radius = float(next(iter(system.balls.values())).params.R) * self.scale

    def xy(self, point: Any) -> tuple[int, int]:
        arr = np.asarray(point, dtype=float)
        x = self.origin_x + float(arr[0]) * self.scale
        y = self.origin_y + (float(self.system.table.l) - float(arr[1])) * self.scale
        return int(round(x)), int(round(y))


def _fonts() -> dict[str, ImageFont.ImageFont]:
    def load(size: int, bold: bool = False) -> ImageFont.ImageFont:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
        for path in candidates:
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        return ImageFont.load_default()

    return {
        "title": load(24, True),
        "body": load(17),
        "body_bold": load(17, True),
        "small": load(15),
        "tiny": load(13),
    }


def _rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (np.floating, np.integer)):
        return _fmt(value.item())
    return str(value)


if __name__ == "__main__":
    main()
