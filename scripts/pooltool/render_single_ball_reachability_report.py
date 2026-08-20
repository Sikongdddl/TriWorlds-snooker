#!/usr/bin/env python3
"""Render a compact visual report for single-ball landing reachability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/pooltool/single_ball_landing_reachability_contact_offsets.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pooltool/single_ball_landing_reachability_contact_offsets.png"),
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path("outputs/pooltool/single_ball_landing_reachability_contact_offsets.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text())
    families: list[dict[str, Any]] = list(data["families"])

    width = 1900
    height = 1220
    image = Image.new("RGB", (width, height), "#f5f2e8")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()

    draw.text((48, 34), "Single Ball Landing Reachability", fill="#1e2924", font=fonts["title"])
    draw.text(
        (48, 72),
        f"cue={data['cue_world_xy']}  object={data['object_world_xy']}  grid=8x4 cells  max=6 pockets x 32 cells = 192 combos",
        fill="#4a5a50",
        font=fonts["body"],
    )

    _draw_family_summary(draw, families, 48, 120, 1550, 300, fonts)
    _draw_aim_offset_panel(draw, families, 48, 470, 760, 660, fonts)
    _draw_impact_panel(draw, families, 880, 470, 760, 660, fonts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    args.table_output.write_text(_markdown_table(data) + "\n")
    print(f"wrote={args.output}")
    print(f"wrote={args.table_output}")


def _draw_family_summary(
    draw: ImageDraw.ImageDraw,
    families: list[dict[str, Any]],
    x: int,
    y: int,
    w: int,
    h: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    draw.text((x, y), "A. Reachable (pocket, landing-cell) combos by solver family", fill="#1e2924", font=fonts["subtitle"])
    chart_x = x + 360
    chart_y = y + 48
    chart_w = w - 420
    row_h = 34
    max_combo = 192
    for idx, family in enumerate(families):
        yy = chart_y + idx * row_h
        label = _short_family_name(family["name"])
        combos = int(family["reachable_combo_count"])
        shots = int(family["successful_shot_count"])
        trials = int(family["trial_count"])
        bar_w = int(round(chart_w * combos / max_combo))
        draw.text((x, yy - 2), label, fill="#27352f", font=fonts["small"])
        draw.rectangle((chart_x, yy, chart_x + chart_w, yy + 18), fill="#dfd8c9")
        draw.rectangle((chart_x, yy, chart_x + bar_w, yy + 18), fill="#2e7a57")
        draw.text((chart_x + chart_w + 12, yy - 2), f"{combos:>2}/192  shots={shots}  trials={trials}", fill="#27352f", font=fonts["small"])
    draw.text(
        (x, y + h - 42),
        "Important: combo count is not pocket count. One pocket can contribute multiple landing cells.",
        fill="#7a3d2d",
        font=fonts["body_bold"],
    )


def _draw_aim_offset_panel(
    draw: ImageDraw.ImageDraw,
    families: list[dict[str, Any]],
    x: int,
    y: int,
    w: int,
    h: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    draw.text((x, y), "B. Contribution by commanded aim offset", fill="#1e2924", font=fonts["subtitle"])
    selected = [families[3], families[4], families[5]]
    colors = ["#2e7a57", "#d18f30", "#466fb4"]
    offsets = sorted({float(stat["cut_offset"]) for fam in selected for stat in fam["aim_offset_stats"]})
    offsets = [value for value in offsets if any(_offset_combo_count(fam, value) for fam in selected)]
    chart_x = x + 54
    chart_y = y + 62
    chart_w = w - 100
    chart_h = h - 220
    max_count = max(1, max(_offset_combo_count(fam, offset) for fam in selected for offset in offsets))
    group_w = max(24, chart_w // max(1, len(offsets)))
    bar_w = max(5, min(14, group_w // 5))

    draw.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill="#38463f", width=2)
    draw.line((chart_x, chart_y, chart_x, chart_y + chart_h), fill="#38463f", width=2)
    for idx, offset in enumerate(offsets):
        gx = chart_x + idx * group_w + group_w // 2
        for fam_idx, fam in enumerate(selected):
            count = _offset_combo_count(fam, offset)
            bh = int(round(chart_h * count / max_count))
            bx = gx + (fam_idx - 1) * (bar_w + 2)
            draw.rectangle((bx, chart_y + chart_h - bh, bx + bar_w, chart_y + chart_h), fill=colors[fam_idx])
        if idx % 2 == 0:
            draw.text((gx - 20, chart_y + chart_h + 8), f"{offset:g}", fill="#38463f", font=fonts["tiny"])
    draw.text((chart_x, chart_y + chart_h + 34), "aim offset in degrees", fill="#38463f", font=fonts["small"])
    _legend(draw, [(colors[i], _short_family_name(f["name"])) for i, f in enumerate(selected)], x + 20, chart_y + chart_h + 70, fonts)


def _draw_impact_panel(
    draw: ImageDraw.ImageDraw,
    families: list[dict[str, Any]],
    x: int,
    y: int,
    w: int,
    h: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    draw.text((x, y), "C. Contribution by estimated contact thickness", fill="#1e2924", font=fonts["subtitle"])
    selected = [families[4], families[5]]
    colors = ["#d18f30", "#466fb4"]
    bins = sorted({stat["impact_bin"] for fam in selected for stat in fam["impact_fraction_stats"]}, key=_bin_key)
    chart_x = x + 54
    chart_y = y + 62
    chart_w = w - 100
    chart_h = h - 220
    max_count = max(1, max(_impact_combo_count(fam, label) for fam in selected for label in bins))
    group_w = max(30, chart_w // max(1, len(bins)))
    bar_w = max(8, min(18, group_w // 4))

    draw.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill="#38463f", width=2)
    draw.line((chart_x, chart_y, chart_x, chart_y + chart_h), fill="#38463f", width=2)
    for idx, label in enumerate(bins):
        gx = chart_x + idx * group_w + group_w // 2
        for fam_idx, fam in enumerate(selected):
            count = _impact_combo_count(fam, label)
            bh = int(round(chart_h * count / max_count))
            bx = gx + (fam_idx - 0.5) * (bar_w + 2)
            draw.rectangle((bx, chart_y + chart_h - bh, bx + bar_w, chart_y + chart_h), fill=colors[fam_idx])
        draw.text((gx - 22, chart_y + chart_h + 8), label.split(":", 1)[0], fill="#38463f", font=fonts["tiny"])
    draw.text((chart_x, chart_y + chart_h + 34), "signed impact fraction bins; |1| is thin contact", fill="#38463f", font=fonts["small"])
    _legend(draw, [(colors[i], _short_family_name(f["name"])) for i, f in enumerate(selected)], x + 20, chart_y + chart_h + 70, fonts)


def _legend(
    draw: ImageDraw.ImageDraw,
    items: list[tuple[str, str]],
    x: int,
    y: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    xx = x
    for color, label in items:
        draw.rectangle((xx, y, xx + 14, y + 14), fill=color)
        draw.text((xx + 20, y - 2), label, fill="#38463f", font=fonts["small"])
        xx += 210


def _offset_combo_count(family: dict[str, Any], offset: float) -> int:
    for stat in family["aim_offset_stats"]:
        if float(stat["cut_offset"]) == offset:
            return int(stat["reachable_combo_count"])
    return 0


def _impact_combo_count(family: dict[str, Any], label: str) -> int:
    for stat in family["impact_fraction_stats"]:
        if stat["impact_bin"] == label:
            return int(stat["reachable_combo_count"])
    return 0


def _bin_key(label: str) -> float:
    return float(label.split(":", 1)[0])


def _short_family_name(name: str) -> str:
    replacements = {
        "direct_center_fixed_speed": "direct fixed speed",
        "direct_narrow_aim_offsets": "direct narrow aim",
        "direct_wide_aim_offsets": "direct wide aim",
        "direct_wide_aim_plus_speed": "direct + speed",
        "direct_wide_aim_speed_top_back": "direct + speed + top/back",
        "direct_and_bank_wide_aim_speed_top_back": "direct/bank + speed + top/back",
    }
    return replacements.get(name, name)


def _markdown_table(data: dict[str, Any]) -> str:
    lines = [
        "# Single Ball Landing Reachability",
        "",
        f"- Cue world xy: `{data['cue_world_xy']}`",
        f"- Object world xy: `{data['object_world_xy']}`",
        f"- Landing grid: `{data['landing_grid']['x_bins']} x {data['landing_grid']['y_bins']}`",
        "",
        "| Solver family | Trials | Successful shots | Reachable combos | Non-zero aim offsets | Impact bins |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for family in data["families"]:
        offsets = [
            f"{stat['cut_offset']}: {stat['reachable_combo_count']}"
            for stat in family["aim_offset_stats"]
            if int(stat["reachable_combo_count"]) > 0
        ]
        bins = [
            f"{stat['impact_bin']}: {stat['reachable_combo_count']}"
            for stat in family["impact_fraction_stats"]
            if int(stat["reachable_combo_count"]) > 0
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    _short_family_name(str(family["name"])),
                    str(family["trial_count"]),
                    str(family["successful_shot_count"]),
                    f"{family['reachable_combo_count']} / {family['possible_combo_count']}",
                    "; ".join(offsets),
                    "; ".join(bins),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _fonts() -> dict[str, ImageFont.ImageFont]:
    try:
        return {
            "title": ImageFont.truetype("DejaVuSans-Bold.ttf", 30),
            "subtitle": ImageFont.truetype("DejaVuSans-Bold.ttf", 20),
            "body_bold": ImageFont.truetype("DejaVuSans-Bold.ttf", 16),
            "body": ImageFont.truetype("DejaVuSans.ttf", 16),
            "small": ImageFont.truetype("DejaVuSans.ttf", 13),
            "tiny": ImageFont.truetype("DejaVuSans.ttf", 10),
        }
    except OSError:
        default = ImageFont.load_default()
        return {key: default for key in ("title", "subtitle", "body_bold", "body", "small", "tiny")}


if __name__ == "__main__":
    main()
