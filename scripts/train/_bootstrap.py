"""Local import helper for scripts nested one level below scripts/."""

from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root

