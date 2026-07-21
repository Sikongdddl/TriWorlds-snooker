"""Utilities for using PoolTool from this repository."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def require_pooltool() -> Any:
    """Import PoolTool with local writable runtime directories configured."""

    runtime_dir = ROOT / "outputs" / "pooltool" / "runtime"
    home_dir = runtime_dir / "home"
    cache_dir = runtime_dir / "numba_cache"
    home_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # PoolTool 0.6.0 writes to Path.home()/.config/pooltool during import, and
    # Numba needs a writable cache directory. Keep those local to ignored output
    # artifacts so the script works in sandboxed workspaces.
    os.environ["HOME"] = str(home_dir)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))

    try:
        import pooltool as pt
    except ImportError as exc:
        raise RuntimeError(
            "PoolTool is not installed. Install the optional dependency with:\n\n"
            "  python -m pip install -r requirements-pooltool.txt\n"
        ) from exc
    return pt
