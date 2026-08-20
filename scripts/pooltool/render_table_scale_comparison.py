#!/usr/bin/env python3
"""Render PoolTool default vs project-aligned table with one metric scale."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snooker_env.pooltool_high_level import DEFAULT_POOL_TABLE_SPEC  # noqa: E402


class TableDrawSpec(NamedTuple):
    label: str
    width: float
    length: float
    corner_abs_x: float
    corner_abs_y: float
    side_abs_x: float
    ball_radius: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/pooltool/table_scale_comparison.png"))
    parser.add_argument("--pixels-per-meter", type=float, default=230.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default = TableDrawSpec(
        label="PoolTool default 7ft",
        width=0.9906,
        length=1.9812,
        corner_abs_x=0.9906 / 2.0 + 0.0417 / 2.0**0.5,
        corner_abs_y=1.9812 / 2.0 + 0.0417 / 2.0**0.5,
        side_abs_x=0.9906 / 2.0 + 0.0685,
        ball_radius=0.028575,
    )
    aligned_spec = DEFAULT_POOL_TABLE_SPEC
    aligned = TableDrawSpec(
        label="Project aligned 9ft",
        width=aligned_spec.play_width_x,
        length=aligned_spec.play_length_y,
        corner_abs_x=aligned_spec.corner_pocket_world_abs_x,
        corner_abs_y=aligned_spec.corner_pocket_world_abs_y,
        side_abs_x=aligned_spec.middle_pocket_world_abs_x,
        ball_radius=aligned_spec.ball_radius,
    )

    scale = float(args.pixels_per_meter)
    panel_gap = 80
    margin = 54
    title_h = 60
    footer_h = 82
    max_panel_w = int(round(max(default.width, aligned.width) * scale)) + 2 * margin
    max_panel_h = int(round(max(default.length, aligned.length) * scale)) + title_h + footer_h
    width_px = 2 * max_panel_w + panel_gap + 2 * margin
    height_px = max_panel_h + 2 * margin

    image = Image.new("RGB", (width_px, height_px), "#f4f1e8")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()

    _draw_panel(draw, default, margin, margin, max_panel_w, max_panel_h, scale, fonts)
    _draw_panel(draw, aligned, margin + max_panel_w + panel_gap, margin, max_panel_w, max_panel_h, scale, fonts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"wrote={args.output}")
    print(f"scale={scale:.1f}px/m")


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    spec: TableDrawSpec,
    left: int,
    top: int,
    panel_w: int,
    panel_h: int,
    scale: float,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    center_x = left + panel_w // 2
    table_w_px = int(round(spec.width * scale))
    table_l_px = int(round(spec.length * scale))
    table_left = center_x - table_w_px // 2
    table_top = top + 58
    table_right = table_left + table_w_px
    table_bottom = table_top + table_l_px

    draw.text((left, top), spec.label, fill="#1f2a24", font=fonts["title"])
    draw.text(
        (left, top + 28),
        f"play area {spec.width:.4f} m x {spec.length:.4f} m",
        fill="#46554d",
        font=fonts["body"],
    )

    cushion = max(10, int(round(0.05 * scale)))
    draw.rounded_rectangle(
        (table_left - cushion, table_top - cushion, table_right + cushion, table_bottom + cushion),
        radius=16,
        fill="#6f4b2f",
        outline="#4a3323",
        width=2,
    )
    draw.rectangle((table_left, table_top, table_right, table_bottom), fill="#2e7a57", outline="#17472f", width=2)

    pocket_r = max(8, int(round(0.060 * scale)))
    pocket_centers = [
        (-spec.corner_abs_x, -spec.corner_abs_y),
        (-spec.corner_abs_x, spec.corner_abs_y),
        (spec.corner_abs_x, -spec.corner_abs_y),
        (spec.corner_abs_x, spec.corner_abs_y),
        (-spec.side_abs_x, 0.0),
        (spec.side_abs_x, 0.0),
    ]
    for wx, wy in pocket_centers:
        px = center_x + int(round(wx * scale))
        py = table_top + table_l_px // 2 - int(round(wy * scale))
        draw.ellipse((px - pocket_r, py - pocket_r, px + pocket_r, py + pocket_r), fill="#111111")

    ball_r = max(3, int(round(spec.ball_radius * scale)))
    ball_positions = _sample_ball_positions(spec)
    colors = ["#ffffff", "#f4d44d", "#2768c5", "#d43a34", "#5c2f91", "#ef7d22"]
    for idx, (wx, wy) in enumerate(ball_positions):
        px = center_x + int(round(wx * scale))
        py = table_top + table_l_px // 2 - int(round(wy * scale))
        draw.ellipse((px - ball_r, py - ball_r, px + ball_r, py + ball_r), fill=colors[idx], outline="#1b1b1b", width=1)

    x0, y0 = table_left, table_bottom + 28
    meter_px = int(round(scale))
    draw.line((x0, y0, x0 + meter_px, y0), fill="#222222", width=3)
    draw.line((x0, y0 - 6, x0, y0 + 6), fill="#222222", width=2)
    draw.line((x0 + meter_px, y0 - 6, x0 + meter_px, y0 + 6), fill="#222222", width=2)
    draw.text((x0, y0 + 10), "1 m", fill="#222222", font=fonts["body"])

    rel = spec.width / (2.0 * spec.ball_radius)
    draw.text(
        (table_left, y0 + 34),
        f"short side = {rel:.1f} ball diameters",
        fill="#46554d",
        font=fonts["body"],
    )


def _sample_ball_positions(spec: TableDrawSpec) -> list[tuple[float, float]]:
    x = -0.25 * spec.width
    y = -0.28 * spec.length
    spacing = 2.2 * spec.ball_radius
    return [
        (x, y),
        (0.18 * spec.width, 0.05 * spec.length),
        (0.18 * spec.width + spacing, 0.05 * spec.length + spacing),
        (0.18 * spec.width - spacing, 0.05 * spec.length + spacing),
        (0.18 * spec.width + 2.0 * spacing, 0.05 * spec.length + 2.0 * spacing),
        (0.18 * spec.width, 0.05 * spec.length + 2.0 * spacing),
    ]


def _fonts() -> dict[str, ImageFont.ImageFont]:
    try:
        title = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        body = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        title = ImageFont.load_default()
        body = ImageFont.load_default()
    return {"title": title, "body": body}


if __name__ == "__main__":
    main()
