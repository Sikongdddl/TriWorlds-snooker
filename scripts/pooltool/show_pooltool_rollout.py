#!/usr/bin/env python3
"""Open a saved PoolTool rollout in the PoolTool GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", nargs="?", type=Path, default=Path("outputs/pooltool/clearance_rollout.msgpack"))
    parser.add_argument("--title", default="PoolTool planner rollout: n/p switches shots, Enter toggles parallel view")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.rollout.exists():
        raise SystemExit(f"Rollout file does not exist: {args.rollout}")

    pt = _require_pooltool()
    multisystem = pt.MultiSystem.load(args.rollout)
    if len(multisystem) == 0:
        raise SystemExit(f"Rollout file contains no systems: {args.rollout}")
    print(f"opening {args.rollout} with {len(multisystem)} PoolTool shots")
    pt.show(multisystem, title=args.title)


if __name__ == "__main__":
    main()
