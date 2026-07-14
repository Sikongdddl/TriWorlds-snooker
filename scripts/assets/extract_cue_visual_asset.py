"""Extract the Sketchfab cue mesh as a local visual asset for MuJoCo."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_OBJ = ROOT / "assets/table/pool_table_traditional/mujoco_full/pool_cue_cue_mat_0.obj"
SOURCE_TEXTURE = ROOT / "assets/table/pool_table_traditional/textures/cue_mat_baseColor.png"
OUT_DIR = ROOT / "assets/cue/sketchfab_pool_table_traditional"
OUT_OBJ = OUT_DIR / "pool_cue_visual_local.obj"
OUT_TEXTURE = OUT_DIR / "cue_mat_baseColor.png"
OUT_README = OUT_DIR / "README.md"

CUE_LENGTH = 1.45
# The extracted Sketchfab cue mesh points from tip to butt after PCA alignment.
# MuJoCo cue convention is local +X from butt to tip, so flip the visual mesh.
FLIP_LOCAL_X = True


def _parse_obj(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    vertices: list[list[float]] = []
    for line in lines:
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    return lines, np.asarray(vertices, dtype=np.float64)


def _localize_vertices(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0)
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    x_axis = vectors[:, int(np.argmax(values))]
    if x_axis[0] < 0:
        x_axis *= -1.0

    z_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = np.cross(z_hint, x_axis)
    if np.linalg.norm(y_axis) < 1e-8:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    basis = np.column_stack([x_axis, y_axis, z_axis])
    local = centered @ basis
    length = float(local[:, 0].max() - local[:, 0].min())
    if length <= 0.0:
        raise ValueError("Cue mesh has invalid zero length.")
    local *= CUE_LENGTH / length
    if FLIP_LOCAL_X:
        local[:, 0] *= -1.0
    local[:, 0] -= 0.5 * (local[:, 0].max() + local[:, 0].min())
    local[:, 1] -= 0.5 * (local[:, 1].max() + local[:, 1].min())
    local[:, 2] -= 0.5 * (local[:, 2].max() + local[:, 2].min())
    return local


def extract() -> None:
    if not SOURCE_OBJ.exists():
        raise FileNotFoundError(SOURCE_OBJ)
    if not SOURCE_TEXTURE.exists():
        raise FileNotFoundError(SOURCE_TEXTURE)

    lines, vertices = _parse_obj(SOURCE_OBJ)
    local_vertices = _localize_vertices(vertices)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_TEXTURE, OUT_TEXTURE)

    vertex_index = 0
    output_lines: list[str] = [
        "# Localized visual cue mesh extracted from Sketchfab Pool Table Traditional.",
        "# Physics remains in models/cue_physics.xml primitive geoms.",
    ]
    for line in lines:
        if line.startswith("mtllib ") or line.startswith("usemtl "):
            continue
        if line.startswith("v "):
            vertex = local_vertices[vertex_index]
            output_lines.append(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}")
            vertex_index += 1
        else:
            output_lines.append(line)
    OUT_OBJ.write_text("\n".join(output_lines) + "\n")

    OUT_README.write_text(
        "# Sketchfab Cue Visual Asset\n\n"
        "This directory contains the localized visual mesh for the cue extracted "
        "from the downloaded Sketchfab `Pool Table Traditional` glTF asset.\n\n"
        "- `pool_cue_visual_local.obj`: visual mesh centered on the MuJoCo cue body.\n"
        "- `cue_mat_baseColor.png`: base color texture used by `models/cue_physics.xml`.\n\n"
        "The active physics model is still defined by primitive geoms in "
        "`models/cue_physics.xml`: `cue_shaft` and `cue_tip`. The mesh here is "
        "visual-only and has no collision or mass.\n"
    )

    mins = local_vertices.min(axis=0)
    maxs = local_vertices.max(axis=0)
    print(f"wrote={OUT_OBJ}")
    print(f"wrote={OUT_TEXTURE}")
    print(f"local_bounds_min={mins}")
    print(f"local_bounds_max={maxs}")


if __name__ == "__main__":
    extract()
