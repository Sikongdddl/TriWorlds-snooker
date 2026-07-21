"""Static visualization helpers for PoolTool planner rollouts."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np


BALL_COLORS: dict[str, str] = {
    "cue": "#f8f8f2",
    "1": "#f4d03f",
    "2": "#2e86de",
    "3": "#e74c3c",
    "4": "#7d3c98",
    "5": "#f39c12",
    "6": "#27ae60",
    "7": "#8e3b22",
    "8": "#111111",
    "9": "#f7dc6f",
}


def event_summary(event: Any) -> dict[str, Any]:
    """Return a compact, JSON-safe PoolTool event summary."""

    return {
        "time": float(getattr(event, "time", 0.0)),
        "type": str(getattr(event, "event_type", type(event).__name__)),
        "ids": tuple(str(item) for item in getattr(event, "ids", ())),
    }


def ball_summary(ball: Any) -> dict[str, Any]:
    """Return final state information for one ball."""

    rvw = np.asarray(ball.state.rvw, dtype=float)
    return {
        "position": rvw[0].tolist(),
        "linear_velocity": rvw[1].tolist(),
        "angular_velocity": rvw[2].tolist(),
        "motion_state": int(ball.state.s),
        "time": float(ball.state.t),
    }


def ensure_continuized(pt: Any, system: Any, *, dt: float) -> Any:
    """Return a copy with continuous ball histories for trajectory drawing."""

    shot = system.copy()
    if shot.simulated and not shot.continuized:
        pt.continuize(shot, dt=dt, inplace=True)
    return shot


def write_static_rollout_report(
    *,
    pt: Any,
    path: Path,
    systems: list[Any],
    records: list[dict[str, Any]],
    title: str,
    trajectory_dt: float = 0.02,
) -> None:
    """Write a self-contained HTML/SVG report for a PoolTool rollout."""

    path.parent.mkdir(parents=True, exist_ok=True)
    visual_systems = [ensure_continuized(pt, system, dt=trajectory_dt) for system in systems]
    body = "\n".join(_shot_section(system, record, idx) for idx, (system, record) in enumerate(zip(visual_systems, records)))
    payload = html.escape(json.dumps(records, indent=2, sort_keys=True), quote=False)
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17211b;
      --muted: #5f6f65;
      --felt: #0b6b4b;
      --rail: #5b3425;
      --line: #d9e6dd;
      --panel: #f7faf8;
      --border: #d7e2da;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #eef4f0;
    }}
    header {{
      padding: 28px 32px 20px;
      border-bottom: 1px solid var(--border);
      background: white;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; letter-spacing: 0; }}
    h3 {{ margin: 18px 0 8px; font-size: 15px; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); }}
    main {{ padding: 24px 32px 40px; display: grid; gap: 22px; }}
    section {{
      background: white;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
    }}
    .grid {{ display: grid; grid-template-columns: minmax(420px, 1.4fr) minmax(320px, 0.8fr); gap: 18px; align-items: start; }}
    .table-wrap {{ width: 100%; overflow-x: auto; }}
    svg {{ display: block; width: 100%; min-width: 420px; height: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; background: var(--panel); }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    details {{ margin-top: 14px; }}
    summary {{ cursor: pointer; color: var(--muted); }}
    pre {{ white-space: pre-wrap; overflow-x: auto; background: #101815; color: #edf7f0; padding: 14px; border-radius: 8px; }}
    .metric-row {{ display: grid; grid-template-columns: 140px 1fr; gap: 8px; margin: 5px 0; font-size: 14px; }}
    .metric-row span:first-child {{ color: var(--muted); }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px 12px; margin-top: 10px; font-size: 12px; color: var(--muted); }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; border: 1px solid rgba(0,0,0,.35); }}
    @media (max-width: 900px) {{ main, header {{ padding-left: 16px; padding-right: 16px; }} .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>Top-down PoolTool rollout report. The SVG paths are sampled from continuous ball histories; the serialized <code>.msgpack</code> file can be opened in PoolTool GUI for 3D playback.</p>
  </header>
  <main>
{body}
    <section>
      <h2>Raw Rollout Metadata</h2>
      <details>
        <summary>Show JSON</summary>
        <pre>{payload}</pre>
      </details>
    </section>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def _shot_section(system: Any, record: dict[str, Any], idx: int) -> str:
    label = html.escape(str(record.get("label", f"shot {idx}")))
    metrics = _metrics_table(record)
    candidates = _candidates_table(record.get("candidates", ()))
    events = _events_table([event_summary(event) for event in system.events[:30]])
    legend = _legend(system)
    return f"""    <section>
      <h2>{label}</h2>
      <div class="grid">
        <div>
          <div class="table-wrap">{_table_svg(system)}</div>
          {legend}
        </div>
        <div>
          <h3>Decision</h3>
          {metrics}
          <h3>Top Candidates</h3>
          {candidates}
          <h3>First Events</h3>
          {events}
        </div>
      </div>
    </section>"""


def _metrics_table(record: dict[str, Any]) -> str:
    solution = record.get("solution") or {}
    rows = [
        ("target", f"{record.get('target_ball_id', '-')}/{record.get('target_pocket_id', '-')}"),
        ("success", str(record.get("success", "-"))),
        ("foul", str(record.get("foul", "-"))),
        ("score", _fmt(record.get("score"))),
        ("reason", str(record.get("reason", "-"))),
        ("cue V0", _fmt(solution.get("speed"))),
        ("cue phi", _fmt(solution.get("phi"))),
        ("remaining", ", ".join(record.get("remaining_balls", ())) or "none"),
    ]
    return "\n".join(
        f'          <div class="metric-row"><span>{html.escape(key)}</span><span><code>{html.escape(value)}</code></span></div>'
        for key, value in rows
    )


def _candidates_table(candidates: Any) -> str:
    rows = list(candidates)[:8]
    if not rows:
        return "<p>No candidate list recorded for this step.</p>"
    body = "\n".join(
        "<tr>"
        f"<td><code>{html.escape(str(row.get('target_ball_id', '-')))}</code></td>"
        f"<td><code>{html.escape(str(row.get('target_pocket_id', '-')))}</code></td>"
        f"<td>{html.escape(_fmt(row.get('score')))}</td>"
        f"<td>{html.escape(str(row.get('success', '-')))}</td>"
        f"<td>{html.escape(str(row.get('reason', '-')))}</td>"
        "</tr>"
        for row in rows
    )
    return f"<table><thead><tr><th>ball</th><th>pocket</th><th>score</th><th>ok</th><th>reason</th></tr></thead><tbody>{body}</tbody></table>"


def _events_table(events: list[dict[str, Any]]) -> str:
    if not events:
        return "<p>No events recorded.</p>"
    body = "\n".join(
        "<tr>"
        f"<td>{html.escape(_fmt(event['time']))}</td>"
        f"<td><code>{html.escape(str(event['type']))}</code></td>"
        f"<td>{html.escape(', '.join(event['ids']))}</td>"
        "</tr>"
        for event in events
    )
    return f"<table><thead><tr><th>t</th><th>type</th><th>ids</th></tr></thead><tbody>{body}</tbody></table>"


def _table_svg(system: Any) -> str:
    table = system.table
    width = float(table.w)
    length = float(table.l)
    view_w = 1000.0
    margin = 60.0
    scale = (view_w - 2.0 * margin) / width
    view_h = length * scale + 2.0 * margin
    mask_id = f"pocket-cutouts-{abs(id(system))}"

    def xy(point: Any) -> tuple[float, float]:
        arr = np.asarray(point, dtype=float)
        return margin + float(arr[0]) * scale, margin + (length - float(arr[1])) * scale

    radius = max(4.0, float(next(iter(system.balls.values())).params.R) * scale)
    felt_x = margin
    felt_y = margin
    felt_w = width * scale
    felt_h = length * scale
    pockets = "\n".join(
        f'<circle cx="{xy(pocket.center)[0]:.2f}" cy="{xy(pocket.center)[1]:.2f}" r="{max(16.0, float(pocket.radius) * scale * 1.28):.2f}" fill="#050806" opacity="0.96" />'
        for pocket in system.table.pockets.values()
    )
    mask_pockets = "\n".join(
        f'<circle cx="{xy(pocket.center)[0]:.2f}" cy="{xy(pocket.center)[1]:.2f}" r="{max(16.0, float(pocket.radius) * scale * 1.22):.2f}" fill="black" />'
        for pocket in system.table.pockets.values()
    )
    cushion_lines = _cushion_svg(system, xy, scale)
    paths = []
    balls = []
    for ball_id, ball in sorted(system.balls.items()):
        color = BALL_COLORS.get(ball_id, "#9aa6b2")
        if not ball.history_cts.empty:
            rvw, _ss, _ts = ball.history_cts.vectorize()
            points = [xy(state[0, :2]) for state in rvw if np.all(np.isfinite(state[0, :2]))]
            if len(points) > 1:
                point_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in points[:: max(1, len(points) // 260)])
                paths.append(f'<polyline points="{point_str}" fill="none" stroke="{color}" stroke-width="3.2" opacity="0.68" />')
        pos = np.asarray(ball.state.rvw[0, :2], dtype=float)
        if not np.all(np.isfinite(pos)):
            continue
        cx, cy = xy(pos)
        stroke = "#111" if ball_id == "cue" else "rgba(0,0,0,.45)"
        balls.append(
            f'<g><circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}" fill="{color}" stroke="{stroke}" stroke-width="2" />'
            f'<text x="{cx:.2f}" y="{cy + 4:.2f}" text-anchor="middle" font-size="{max(10.0, radius):.1f}" fill="{"#111" if ball_id in {"cue", "1", "9"} else "#fff"}">{html.escape(ball_id)}</text></g>'
        )

    return f"""<svg viewBox="0 0 {view_w:.0f} {view_h:.0f}" role="img" aria-label="PoolTool top-down trajectory">
  <rect x="0" y="0" width="{view_w:.0f}" height="{view_h:.0f}" rx="24" fill="#5b3425"/>
  <mask id="{mask_id}">
    <rect x="{felt_x:.2f}" y="{felt_y:.2f}" width="{felt_w:.2f}" height="{felt_h:.2f}" fill="white"/>
    {mask_pockets}
  </mask>
  <rect x="{felt_x:.2f}" y="{felt_y:.2f}" width="{felt_w:.2f}" height="{felt_h:.2f}" fill="#0b6b4b" mask="url(#{mask_id})"/>
  {pockets}
  {cushion_lines}
  {"".join(paths)}
  {"".join(balls)}
</svg>"""


def _cushion_svg(system: Any, xy: Any, scale: float) -> str:
    stroke_width = max(7.0, 0.018 * scale)
    lines = []
    for cushion in system.table.cushion_segments.linear.values():
        x1, y1 = xy(cushion.p1[:2])
        x2, y2 = xy(cushion.p2[:2])
        lines.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="#123d30" stroke-width="{stroke_width:.2f}" stroke-linecap="round" />'
        )
    return "\n  ".join(lines)


def _legend(system: Any) -> str:
    entries = []
    for ball_id in sorted(system.balls):
        color = BALL_COLORS.get(ball_id, "#9aa6b2")
        entries.append(f'<span><i class="dot" style="background:{color}"></i><code>{html.escape(ball_id)}</code></span>')
    return '<div class="legend">' + "".join(entries) + "</div>"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (np.floating, np.integer)):
        return _fmt(value.item())
    return str(value)
