"""Scene loading helpers."""

from __future__ import annotations

from pathlib import Path

import mujoco


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model_path() -> Path:
    return project_root() / "models" / "scene_pool_asset.xml"


def load_model(model_path: str | Path | None = None) -> mujoco.MjModel:
    path = Path(model_path) if model_path is not None else default_model_path()
    if not path.exists():
        raise FileNotFoundError(f"MuJoCo model does not exist: {path}")
    try:
        return mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load MuJoCo model from {path}: {exc}") from exc
