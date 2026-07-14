from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = ROOT / "assets" / "table" / "pool_table_traditional"
GLTF_PATH = ASSET_ROOT / "scene.gltf"
OUT_DIR = ASSET_ROOT / "mujoco_full"
MODEL_PATH = ROOT / "models" / "pool_table_asset.xml"
SCENE_PATH = ROOT / "models" / "scene_pool_asset.xml"


MATERIALS = {
    "pool_table_mat": {
        "texture": "../assets/table/pool_table_traditional/textures/pool_table_mat_baseColor.png",
        "rgba": "1 1 1 1",
    },
    "cue_mat": {
        "texture": "../assets/table/pool_table_traditional/textures/cue_mat_baseColor.png",
        "rgba": "1 1 1 1",
    },
    "billiard_ball_mat": {
        "texture": "../assets/table/pool_table_traditional/textures/billiard_ball_mat_baseColor.png",
        "rgba": "1 1 1 1",
    },
    "light_mat": {
        "texture": "../assets/table/pool_table_traditional/textures/light_mat_baseColor.png",
        "rgba": "1 1 1 1",
        "emission": "0.2",
    },
}


def _clean_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return cleaned.strip("_").lower()


def _material_for_node(name: str) -> str:
    if "pool_table_mat" in name:
        return "pool_table_mat"
    if "cue_mat" in name:
        return "cue_mat"
    if "billiard_ball_mat" in name:
        return "billiard_ball_mat"
    if "light_mat" in name:
        return "light_mat"
    return "pool_table_mat"


def _is_replaced_by_physics(name: str) -> bool:
    return "billiard_ball" in name or "pool_cue" in name


def _map_axes(vertices: np.ndarray) -> np.ndarray:
    # trimesh loads the Sketchfab glTF in meters with X=table width, Y=up,
    # Z=table length. Convert to project convention: X=length, Y=width, Z=up.
    mapped = np.empty_like(vertices)
    mapped[:, 0] = vertices[:, 2]
    mapped[:, 1] = vertices[:, 0]
    mapped[:, 2] = vertices[:, 1]
    return mapped


def _write_xml(path: Path, root: ET.Element) -> None:
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="unicode", xml_declaration=False)
    text = path.read_text()
    path.write_text(text + ("\n" if not text.endswith("\n") else ""))


def convert() -> None:
    if not GLTF_PATH.exists():
        raise FileNotFoundError(GLTF_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded = trimesh.load(GLTF_PATH)
    if not isinstance(loaded, trimesh.Scene):
        raise TypeError(f"Expected a trimesh.Scene from {GLTF_PATH}, got {type(loaded)!r}")

    exported: list[tuple[str, str, str]] = []
    all_bounds: list[np.ndarray] = []
    table_bounds: np.ndarray | None = None

    for node_name in sorted(loaded.graph.nodes_geometry):
        transform, geom_name = loaded.graph[node_name]
        mesh = loaded.geometry[geom_name].copy()
        mesh.apply_transform(transform)
        mesh.vertices = _map_axes(mesh.vertices)
        all_bounds.append(mesh.bounds)
        if node_name.startswith("pooltable_low"):
            table_bounds = mesh.bounds.copy()

        mesh_name = _clean_name(node_name)
        obj_path = OUT_DIR / f"{mesh_name}.obj"
        mesh.export(obj_path)
        exported.append((mesh_name, obj_path.relative_to(ROOT).as_posix(), _material_for_node(node_name)))

    combined_bounds = np.array(all_bounds)
    mins = combined_bounds[:, 0, :].min(axis=0)
    maxs = combined_bounds[:, 1, :].max(axis=0)
    if table_bounds is None:
        raise RuntimeError("Could not find pooltable_low mesh in the glTF scene.")
    center_xy = 0.5 * (table_bounds[0, :2] + table_bounds[1, :2])
    z_offset = -table_bounds[0, 2]

    mujoco = ET.Element("mujoco")
    asset = ET.SubElement(mujoco, "asset")
    for material_name, spec in MATERIALS.items():
        texture_name = f"{material_name}_texture"
        ET.SubElement(
            asset,
            "texture",
            {
                "name": texture_name,
                "type": "2d",
                "file": spec["texture"],
            },
        )
        mat_attrs = {
            "name": material_name,
            "texture": texture_name,
            "rgba": spec["rgba"],
            "reflectance": "0.12",
        }
        if "emission" in spec:
            mat_attrs["emission"] = spec["emission"]
        ET.SubElement(asset, "material", mat_attrs)

    for mesh_name, rel_path, _ in exported:
        if _is_replaced_by_physics(mesh_name):
            continue
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": f"../{rel_path}"})

    worldbody = ET.SubElement(mujoco, "worldbody")
    root_body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "pool_table_traditional_asset_root",
            "pos": f"{-center_xy[0]:.8f} {-center_xy[1]:.8f} {z_offset:.8f}",
        },
    )
    for mesh_name, _, material_name in exported:
        if _is_replaced_by_physics(mesh_name):
            continue
        ET.SubElement(
            root_body,
            "geom",
            {
                "name": f"{mesh_name}_visual",
                "type": "mesh",
                "mesh": mesh_name,
                "material": material_name,
                "contype": "0",
                "conaffinity": "0",
            },
        )

    # Hidden physics proxies. They follow the zip asset dimensions but are not
    # rendered by default in render_scene.py because group 3 is hidden.
    ET.SubElement(
        root_body,
        "geom",
        {
            "name": "pool_asset_playfield_proxy",
            "type": "box",
            "pos": "0 0 0.654",
            "size": "1.18 0.60 0.025",
            "group": "3",
            "rgba": "0 0.4 1 0.08",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    _write_xml(MODEL_PATH, mujoco)

    scene = ET.Element("mujoco", {"model": "pool_table_traditional_asset_scene"})
    ET.SubElement(scene, "compiler", {"angle": "radian", "coordinate": "local", "autolimits": "true"})
    ET.SubElement(
        scene,
        "option",
        {
            "timestep": "0.001",
            "gravity": "0 0 -9.81",
            "integrator": "RK4",
            "solver": "Newton",
            "iterations": "80",
        },
    )
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "headlight", {"ambient": "0.45 0.45 0.45", "diffuse": "0.8 0.8 0.8", "specular": "0.25 0.25 0.25"})
    world = ET.SubElement(scene, "worldbody")
    ET.SubElement(world, "light", {"name": "key_light", "pos": "0 -2.5 3.0", "dir": "0 0 -1", "diffuse": "0.9 0.9 0.86"})
    ET.SubElement(world, "camera", {"name": "asset_overview", "pos": "0 -3.2 1.7", "xyaxes": "1 0 0 0 0.48 0.88"})
    ET.SubElement(scene, "include", {"file": "pool_table_asset.xml"})
    ET.SubElement(scene, "include", {"file": "table_physics.xml"})
    ET.SubElement(scene, "include", {"file": "balls_physics.xml"})
    ET.SubElement(scene, "include", {"file": "cue_physics.xml"})
    ET.SubElement(scene, "include", {"file": "lift_articulated.xml"})
    ET.SubElement(scene, "include", {"file": "grip_constraints.xml"})
    _write_xml(SCENE_PATH, scene)

    print(f"exported_meshes={len(exported)}")
    print(f"asset_bounds_min={mins}")
    print(f"asset_bounds_max={maxs}")
    print(f"wrote={MODEL_PATH}")
    print(f"wrote={SCENE_PATH}")


if __name__ == "__main__":
    convert()
