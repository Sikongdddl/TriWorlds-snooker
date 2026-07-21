#!/usr/bin/env python3
"""Reproduce a PoolTool billiards simulation example.

This script intentionally stays outside the MuJoCo stack. It gives us a
realistic event-based billiards simulator to study high-level strategy before
deciding how to connect it to the robot/cue-control layers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snooker_env.pooltool_runtime import require_pooltool  # noqa: E402


def _require_pooltool() -> Any:
    try:
        return require_pooltool()
    except RuntimeError as exc:
        raise SystemExit(
            "PoolTool is not installed. Install the optional dependency with:\n\n"
            "  python -m pip install -r requirements-pooltool.txt\n"
        ) from exc


def _to_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _event_summary(event: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "type": str(getattr(event, "event_type", getattr(event, "type", type(event).__name__))),
        "time": _to_list(getattr(event, "time", getattr(event, "t", None))),
    }
    ids = getattr(event, "ids", None)
    if ids is not None:
        summary["ids"] = _to_list(ids)
    agents = getattr(event, "agents", None)
    if agents is not None:
        summary["agents"] = [
            {
                "id": getattr(agent, "id", None),
                "type": str(getattr(agent, "agent_type", None)),
            }
            for agent in agents
        ]
    return summary


def _ball_summary(ball: Any) -> dict[str, Any]:
    state = ball.state
    rvw = np.asarray(state.rvw, dtype=float)
    return {
        "position": rvw[0].tolist(),
        "linear_velocity": rvw[1].tolist(),
        "angular_velocity": rvw[2].tolist(),
        "motion_state": int(state.s),
        "time": float(state.t),
    }


def _build_system(pt: Any, args: argparse.Namespace) -> Any:
    if args.example_system:
        system = pt.System.example()
    else:
        table = pt.Table.default()
        balls = pt.get_rack(pt.GameType.NINEBALL, table)
        cue = pt.Cue(cue_ball_id="cue")
        system = pt.System(table=table, balls=balls, cue=cue)

    if args.phi is None:
        object_ball_id = args.object_ball
        if object_ball_id not in system.balls:
            raise ValueError(f"Object ball {object_ball_id!r} is not in PoolTool system balls: {sorted(system.balls)}")
        phi = pt.aim.at_ball(system, object_ball_id)
    else:
        phi = args.phi

    system.cue.set_state(
        V0=args.speed,
        phi=phi,
        a=args.side_spin,
        b=args.top_spin,
        theta=args.elevation,
    )
    return system


def run(args: argparse.Namespace) -> dict[str, Any]:
    pt = _require_pooltool()
    system = _build_system(pt, args)

    pt.simulate(
        system,
        inplace=True,
        continuous=args.continuous,
        dt=args.dt,
        t_final=args.t_final,
        max_events=args.max_events,
    )

    summary = {
        "pooltool_version": getattr(pt, "__version__", "unknown"),
        "simulated": bool(system.simulated),
        "time": float(system.t),
        "cue": {
            "V0": float(system.cue.V0),
            "phi": float(system.cue.phi),
            "a": float(system.cue.a),
            "b": float(system.cue.b),
            "theta": float(system.cue.theta),
            "cue_ball_id": system.cue.cue_ball_id,
        },
        "events": [_event_summary(event) for event in system.events],
        "balls": {ball_id: _ball_summary(ball) for ball_id, ball in system.balls.items()},
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if args.show:
        pt.show(system)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=8.0, help="Cue impact speed V0 in m/s.")
    parser.add_argument("--phi", type=float, default=None, help="Cue horizontal direction in PoolTool degrees. Defaults to aim.at_ball().")
    parser.add_argument("--side-spin", type=float, default=0.0, help="PoolTool cue parameter a: +left, -right.")
    parser.add_argument("--top-spin", type=float, default=0.0, help="PoolTool cue parameter b: +top, -bottom.")
    parser.add_argument("--elevation", type=float, default=0.0, help="PoolTool cue inclination theta in degrees.")
    parser.add_argument("--object-ball", default="1", help="Ball ID used by aim.at_ball when --phi is omitted.")
    parser.add_argument("--example-system", action="store_true", help="Use pt.System.example() instead of a 9-ball rack.")
    parser.add_argument("--continuous", action="store_true", help="Continuize ball trajectories for visualization.")
    parser.add_argument("--dt", type=float, default=0.01, help="Continuous history timestep when --continuous is set.")
    parser.add_argument("--t-final", type=float, default=None, help="Optional simulation cutoff time.")
    parser.add_argument("--max-events", type=int, default=0, help="Optional maximum event count; 0 means PoolTool default.")
    parser.add_argument("--output", type=Path, default=Path("outputs/pooltool/pooltool_example_summary.json"))
    parser.add_argument("--show", action="store_true", help="Open the PoolTool GUI after simulation.")
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(f"pooltool_version={summary['pooltool_version']}")
    print(f"simulated={summary['simulated']} t={summary['time']:.6f}s events={len(summary['events'])}")
    print("cue=" + json.dumps(summary["cue"], sort_keys=True))
    print(f"balls={sorted(summary['balls'])}")
    if summary["events"]:
        print("first_events=" + json.dumps(summary["events"][:5], sort_keys=True))


if __name__ == "__main__":
    main()
