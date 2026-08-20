#!/usr/bin/env python3
"""Render two-player PoolTool DQN training diagnostics from a JSON summary."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "ink": (35, 43, 36),
    "muted": (83, 88, 80),
    "paper": (247, 245, 237),
    "grid": (218, 216, 205),
    "border": (128, 130, 120),
    "blue": (64, 101, 181),
    "orange": (213, 141, 35),
    "green": (45, 126, 91),
    "red": (166, 70, 58),
    "purple": (145, 60, 181),
    "cyan": (39, 137, 151),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--title", default="Unmasked Two-player DQN Training")
    return parser.parse_args()


def moving_average(values: list[float], window: int) -> list[float]:
    result: list[float] = []
    queue: list[float] = []
    running = 0.0
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        result.append(running / len(queue))
    return result


def finite(values: Iterable[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


class Plotter:
    def __init__(self, image: Image.Image, episode_count: int, clone_episodes: list[int]) -> None:
        self.image = image
        self.draw = ImageDraw.Draw(image)
        self.episode_count = episode_count
        self.clone_episodes = clone_episodes
        font_dir = Path("/usr/share/fonts/truetype/dejavu")
        self.title_font = ImageFont.truetype(str(font_dir / "DejaVuSans-Bold.ttf"), 38)
        self.heading_font = ImageFont.truetype(str(font_dir / "DejaVuSans-Bold.ttf"), 23)
        self.font = ImageFont.truetype(str(font_dir / "DejaVuSans.ttf"), 16)
        self.small_font = ImageFont.truetype(str(font_dir / "DejaVuSans.ttf"), 13)

    def line_panel(
        self,
        box: tuple[int, int, int, int],
        title: str,
        series: list[tuple[str, list[float], tuple[int, int, int]]],
        *,
        y_range: tuple[float, float] | None = None,
        draw_clones: bool = True,
    ) -> None:
        x, y, width, height = box
        self.draw.text((x, y - 34), title, fill=COLORS["ink"], font=self.heading_font)
        left, top = x + 62, y + 14
        right, bottom = x + width - 20, y + height - 42
        self.draw.rectangle((left, top, right, bottom), outline=COLORS["border"])

        all_values = finite(value for _name, values, _color in series for value in values)
        if y_range is None:
            low, high = min(all_values), max(all_values)
            if abs(high - low) < 1e-12:
                high = low + 1.0
            pad = 0.08 * (high - low)
            low, high = low - pad, high + pad
        else:
            low, high = y_range

        for tick in range(5):
            yy = top + (bottom - top) * tick / 4
            value = high - (high - low) * tick / 4
            self.draw.line((left, yy, right, yy), fill=COLORS["grid"])
            self.draw.text((x + 2, yy - 8), f"{value:.2g}", fill=COLORS["muted"], font=self.small_font)

        count = max(len(values) for _name, values, _color in series)
        for name, values, color in series:
            points = []
            for index, value in enumerate(values):
                px = left + (right - left) * index / max(1, count - 1)
                py = bottom - (bottom - top) * (value - low) / (high - low)
                points.append((px, py))
            if len(points) > 1:
                self.draw.line(points, fill=color, width=3)

        if draw_clones:
            for episode in self.clone_episodes:
                xx = left + (right - left) * episode / max(1, self.episode_count)
                self.draw.line((xx, top, xx, bottom), fill=COLORS["red"], width=2)

        legend_x, legend_y = left + 10, top + 9
        for name, _values, color in series:
            self.draw.rectangle((legend_x, legend_y, legend_x + 13, legend_y + 13), fill=color)
            self.draw.text((legend_x + 19, legend_y - 2), name, fill=COLORS["ink"], font=self.small_font)
            legend_y += 19
        self.draw.text((left, bottom + 8), "episode 1", fill=COLORS["muted"], font=self.small_font)
        right_label = f"episode {self.episode_count}"
        label_width = self.draw.textbbox((0, 0), right_label, font=self.small_font)[2]
        self.draw.text((right - label_width, bottom + 8), right_label, fill=COLORS["muted"], font=self.small_font)

    def bar_panel(
        self,
        box: tuple[int, int, int, int],
        title: str,
        items: list[tuple[str, int]],
    ) -> None:
        x, y, width, height = box
        self.draw.text((x, y - 34), title, fill=COLORS["ink"], font=self.heading_font)
        left, top = x + 180, y + 14
        right, bottom = x + width - 25, y + height - 36
        maximum = max(value for _name, value in items)
        row_height = (bottom - top) / len(items)
        for index, (name, value) in enumerate(items):
            yy = top + index * row_height
            bar_right = left + (right - left) * value / maximum
            color = COLORS["green"] if name == "pot" else COLORS["red"] if "blocked" in name else COLORS["orange"]
            self.draw.text((x + 5, yy + 2), name[:22], fill=COLORS["ink"], font=self.small_font)
            self.draw.rectangle((left, yy + 2, bar_right, yy + row_height - 6), fill=color)
            self.draw.text((bar_right + 7, yy + 2), str(value), fill=COLORS["muted"], font=self.small_font)


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text())
    records = data.get("records", [])
    if not records:
        raise SystemExit(f"No records found in {args.input}")

    episodes = len(records)
    clone_events = data.get("clone_events", [])
    clones = [int(event["episode"]) for event in clone_events]
    window = args.window

    clear = [float(bool(record["cleared"])) for record in records]
    p0_win = [float(record.get("winner") == 0) for record in records]
    p1_win = [float(record.get("winner") == 1) for record in records]
    no_winner = [float(record.get("winner") is None) for record in records]
    turns = [float(record["turns"]) for record in records]
    p0_return = [float(record["returns"][0]) for record in records]
    p1_return = [float(record["returns"][1]) for record in records]
    updates = [float(record["updates"][0]) for record in records]
    epsilon = [float(record["epsilon"]) for record in records]
    losses = []
    previous_loss = 0.0
    for record in records:
        value = record["last_loss"][0]
        if value is not None and math.isfinite(float(value)):
            previous_loss = float(value)
        losses.append(previous_loss)

    p0_success: list[float] = []
    p1_success: list[float] = []
    foul_rate: list[float] = []
    reasons: Counter[str] = Counter()
    paths: Counter[str] = Counter()
    valid_counts: Counter[int] = Counter()
    zero_displacements = 0
    for record in records:
        trajectory = record.get("trajectory", [])
        for turn in trajectory:
            reasons[str(turn.get("reason"))] += 1
            paths[str(turn.get("shot_path_type"))] += 1
            if "valid_action_count" in turn:
                valid_counts[int(turn["valid_action_count"])] += 1
            displacement = turn.get("cue_ball_displacement")
            if displacement is not None and float(displacement) < 1e-8:
                zero_displacements += 1
        for player, destination in ((0, p0_success), (1, p1_success)):
            own = [turn for turn in trajectory if turn.get("player") == player]
            destination.append(sum(bool(turn.get("success")) for turn in own) / len(own) if own else 0.0)
        foul_rate.append(sum(bool(turn.get("foul")) for turn in trajectory) / len(trajectory) if trajectory else 0.0)

    image = Image.new("RGB", (2400, 1660), COLORS["paper"])
    plotter = Plotter(image, episodes, clones)
    plotter.draw.text((45, 28), args.title, fill=COLORS["ink"], font=plotter.title_font)
    subtitle = (
        f"{episodes} episodes | {int(updates[-1])} active-player updates | MA{window} | "
        "all six pocket actions exposed"
    )
    plotter.draw.text((47, 76), subtitle, fill=COLORS["muted"], font=plotter.font)
    plotter.draw.text((1870, 77), "red vertical lines = opponent clone", fill=COLORS["red"], font=plotter.small_font)

    boxes = [
        (45, 150, 1120, 315), (1230, 150, 1120, 315),
        (45, 535, 1120, 315), (1230, 535, 1120, 315),
        (45, 920, 1120, 315), (1230, 920, 1120, 315),
    ]
    plotter.line_panel(boxes[0], "1. Clearance and winner rates", [
        ("clear", moving_average(clear, window), COLORS["green"]),
        ("p0 win", moving_average(p0_win, window), COLORS["blue"]),
        ("p1 win", moving_average(p1_win, window), COLORS["orange"]),
        ("no winner", moving_average(no_winner, window), COLORS["red"]),
    ], y_range=(0.0, 1.0))
    plotter.line_panel(boxes[1], "2. Episode returns", [
        ("p0 return", moving_average(p0_return, window), COLORS["blue"]),
        ("p1 return", moving_average(p1_return, window), COLORS["orange"]),
        ("p0-p1", moving_average([a - b for a, b in zip(p0_return, p1_return)], window), COLORS["green"]),
    ])
    plotter.line_panel(boxes[2], "3. Turns per episode", [
        ("turns", moving_average(turns, window), COLORS["cyan"]),
    ])
    plotter.line_panel(boxes[3], "4. DQN TD loss", [
        ("loss", moving_average(losses, window), COLORS["purple"]),
    ])
    plotter.line_panel(boxes[4], "5. Per-shot outcomes", [
        ("p0 success", moving_average(p0_success, window), COLORS["blue"]),
        ("p1 success", moving_average(p1_success, window), COLORS["orange"]),
        ("foul", moving_average(foul_rate, window), COLORS["red"]),
    ], y_range=(0.0, 0.65))
    plotter.line_panel(boxes[5], "6. Learning schedule", [
        ("updates / 3000", [value / 3000.0 for value in updates], COLORS["green"]),
        ("epsilon x 10", [value * 10.0 for value in epsilon], COLORS["orange"]),
    ])

    bottom_y = 1320
    top_reasons = reasons.most_common(6)
    plotter.bar_panel((45, bottom_y, 1120, 275), "7. Shot outcome counts", top_reasons)

    plotter.draw.text((1230, bottom_y - 34), "8. Final diagnostics", fill=COLORS["ink"], font=plotter.heading_font)
    plotter.draw.rectangle((1230, bottom_y + 14, 2350, bottom_y + 239), outline=COLORS["border"])
    last_window = records[-1000:]
    final_clear = sum(bool(record["cleared"]) for record in last_window) / len(last_window)
    final_p0 = sum(record.get("winner") == 0 for record in last_window) / len(last_window)
    final_p1 = sum(record.get("winner") == 1 for record in last_window) / len(last_window)
    final_turns = sum(float(record["turns"]) for record in last_window) / len(last_window)
    valid_text = ", ".join(f"{key}: {value}" for key, value in sorted(valid_counts.items()))
    diagnostics = [
        f"Last 1000: clear={final_clear:.3f}, p0 win={final_p0:.3f}, p1 win={final_p1:.3f}",
        f"Last 1000: mean turns={final_turns:.2f}, loss MA={moving_average(losses, window)[-1]:.4f}",
        f"Final updates={int(updates[-1])}, final epsilon={epsilon[-1]:.3f}",
        f"Opponent clones={len(clones)}, latest={clones[-1] if clones else 'none'}",
        f"Shot paths: direct={paths.get('direct', 0)}, best-effort={paths.get('best_effort_direct', 0)}",
        f"Near-zero cue displacement (<1e-8 m)={zero_displacements}; exact zero=0",
        f"Valid action count distribution: {valid_text}",
    ]
    for index, line in enumerate(diagnostics):
        plotter.draw.text((1255, bottom_y + 30 + index * 30), line, fill=COLORS["ink"], font=plotter.font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"wrote={args.output}")
    print(f"episodes={episodes} updates={int(updates[-1])} clones={clones}")


if __name__ == "__main__":
    main()
