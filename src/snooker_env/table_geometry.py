"""Geometry constants and lookup helpers for the original mujoco-billiards table."""

from __future__ import annotations

import mujoco
import numpy as np


TABLE_LENGTH = 2.54
TABLE_WIDTH = 1.27
TABLE_SURFACE_Z = 1.05
BALL_RADIUS = 0.0285
BALL_CENTER_Z = TABLE_SURFACE_Z + BALL_RADIUS
POCKET_ENTRY_Z = 1.035


def _surface_material_id(model: mujoco.MjModel) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "mat-surface")


def cloth_geom_ids(model: mujoco.MjModel) -> tuple[int, ...]:
    """Return source table box geoms covered by the cloth material."""

    material_id = _surface_material_id(model)
    return tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_matid[geom_id]) == material_id
        and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX)
    )


def cushion_geom_ids(model: mujoco.MjModel) -> tuple[int, ...]:
    """Return source SDF and cylindrical cushion/pocket-rim geoms."""

    material_id = _surface_material_id(model)
    cushion_types = {int(mujoco.mjtGeom.mjGEOM_SDF), int(mujoco.mjtGeom.mjGEOM_CYLINDER)}
    return tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_matid[geom_id]) == material_id
        and int(model.geom_type[geom_id]) in cushion_types
    )


def central_cloth_geom_id(model: mujoco.MjModel) -> int:
    """Find the largest cloth box, which is the central playing surface."""

    candidates = cloth_geom_ids(model)
    if not candidates:
        raise ValueError("Model does not contain the mujoco-billiards cloth geoms.")
    return max(candidates, key=lambda geom_id: float(np.prod(model.geom_size[geom_id, :2])))


def nearest_cushion_geom_id(model: mujoco.MjModel, target_position: np.ndarray) -> int:
    """Find the source cylindrical cushion nose nearest a world-frame location."""

    candidates = tuple(
        geom_id
        for geom_id in cushion_geom_ids(model)
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    )
    if not candidates:
        raise ValueError("Model does not contain the mujoco-billiards cushion geoms.")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    target = np.asarray(target_position, dtype=np.float64)
    return min(candidates, key=lambda geom_id: float(np.linalg.norm(data.geom_xpos[geom_id] - target)))
