#!/usr/bin/env python3
"""Convert the imported Gento URDF into the MuJoCo side-grasp robot asset.

The source URDF has two seven-axis arms but stops at each Link_7.  This
converter therefore preserves the imported robot kinematics and appends a
small physical two-finger tool to each wrist.  The tool is intentionally made
from primitive collision bodies: it closes vertically around the cue while a
solid palm block prevents the shaft from passing through the wrist.

The imported base STL contains more than MuJoCo's 200k-face STL limit.  All
source meshes are converted losslessly to OBJ in a generated asset directory;
the original import is never modified.
"""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "gento"
    / "urdf"
    / "gento_skye.urdf"
)
DEFAULT_MESH_DIR = REPO_ROOT / "assets" / "robots" / "gento" / "meshes"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "gento_side_grasp_articulated.xml"


def _numbers(text: str | None, count: int, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    values = np.asarray([float(value) for value in text.split()], dtype=float)
    if values.size != count:
        raise ValueError(f"expected {count} values, got {text!r}")
    return values


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return value or "mesh"


def _fmt(values: np.ndarray | tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0])
    unit = axis / norm
    half = angle / 2.0
    return np.r_[math.cos(half), unit * math.sin(half)]


def _rpy_quat(rpy: np.ndarray) -> np.ndarray:
    """Return URDF fixed-axis roll/pitch/yaw as a MuJoCo wxyz quaternion."""
    roll, pitch, yaw = rpy
    qx = _axis_angle_quat(np.asarray([1.0, 0.0, 0.0]), roll)
    qy = _axis_angle_quat(np.asarray([0.0, 1.0, 0.0]), pitch)
    qz = _axis_angle_quat(np.asarray([0.0, 0.0, 1.0]), yaw)
    quat = _quat_mul(_quat_mul(qz, qy), qx)
    return quat / np.linalg.norm(quat)


def _origin(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if element is None:
        return np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])
    return (
        _numbers(element.get("xyz"), 3, (0.0, 0.0, 0.0)),
        _rpy_quat(_numbers(element.get("rpy"), 3, (0.0, 0.0, 0.0))),
    )


def _joint_name(name: str) -> str:
    return f"gento_{_slug(name)}"


def _body_name(name: str) -> str:
    return f"gento_{_slug(name)}"


def _add_inertial(body: ET.Element, link: ET.Element) -> None:
    source = link.find("inertial")
    if source is None or source.find("mass") is None or source.find("inertia") is None:
        return
    pos, quat = _origin(source.find("origin"))
    if not np.allclose(quat, (1.0, 0.0, 0.0, 0.0), atol=1e-8):
        raise ValueError(f"rotated inertial frames are not supported for {link.get('name')}")
    mass = float(source.find("mass").get("value", "0"))
    inertia = source.find("inertia")
    values = [
        float(inertia.get("ixx", "0")),
        float(inertia.get("iyy", "0")),
        float(inertia.get("izz", "0")),
        float(inertia.get("ixy", "0")),
        float(inertia.get("ixz", "0")),
        float(inertia.get("iyz", "0")),
    ]
    ET.SubElement(
        body,
        "inertial",
        pos=_fmt(pos),
        mass=f"{mass:.9g}",
        fullinertia=_fmt(values),
    )


def _add_gripper(body: ET.Element, side: str) -> list[tuple[str, float, float, float]]:
    """Append one vertical-closing, palm-guarded physical gripper."""
    mount = ET.SubElement(
        body,
        "body",
        name=f"gento_{side}_gripper_mount",
        pos="0.095 0 0",
    )
    ET.SubElement(
        mount,
        "geom",
        name=f"gento_{side}_palm_guard",
        type="box",
        pos="0.025 0 0",
        size="0.025 0.048 0.048",
        rgba="0.16 0.20 0.25 1",
        contype="80",
        conaffinity="40",
        condim="4",
        priority="3",
        friction="0.45 0.005 0.001",
        solref="0.003 1",
        solimp="0.94 0.99 0.001",
    )
    ET.SubElement(
        mount,
        "geom",
        name=f"gento_{side}_palm_face",
        type="box",
        pos="0.052 0 0",
        size="0.004 0.042 0.040",
        rgba="0.18 0.65 0.82 1",
        contype="0",
        conaffinity="0",
        density="0",
    )

    finger_specs = (
        ("upper", (0.090, 0.0, 0.044), (0.0, 0.0, 1.0), (-0.030, 0.0), -0.026),
        ("lower", (0.090, 0.0, -0.044), (0.0, 0.0, 1.0), (0.0, 0.030), 0.026),
    )
    actuators: list[tuple[str, float, float, float]] = []
    color = "0.10 0.82 0.95 1" if side == "left" else "1 0.38 0.12 1"
    # Robot-right is the forward guide: low tangential friction lets the cue
    # slide along its fixed direction. Robot-left is the rear speed hand and
    # keeps high friction so its axial motion actually drives the shaft.
    pad_friction = "1.15 0.012 0.0015" if side == "left" else "0.10 0.002 0.0002"
    for label, pos, axis, limits, closed in finger_specs:
        finger = ET.SubElement(
            mount,
            "body",
            name=f"gento_{side}_{label}_finger",
            pos=_fmt(pos),
            gravcomp="1",
        )
        joint_name = f"gento_{side}_{label}_finger_joint"
        ET.SubElement(
            finger,
            "joint",
            name=joint_name,
            type="slide",
            axis=_fmt(axis),
            range=_fmt(limits),
            damping="10",
            armature="0.003",
        )
        ET.SubElement(
            finger,
            "geom",
            name=f"gento_{side}_{label}_finger_pad",
            type="box",
            size="0.040 0.022 0.008",
            mass="0.09",
            rgba=color,
            contype="80",
            conaffinity="40",
            condim="4",
            priority="4",
            friction=pad_friction,
            solref="0.002 1",
            solimp="0.95 0.995 0.001",
        )
        actuators.append((joint_name, limits[0], limits[1], closed))

    ET.SubElement(
        mount,
        "site",
        name=f"gento_{side}_gripper_tcp",
        pos="0.090 0 0",
        size="0.014",
        rgba=color,
    )
    ET.SubElement(
        mount,
        "site",
        name=f"gento_{side}_approach_axis",
        type="capsule",
        fromto="0.045 0 0 0.125 0 0",
        size="0.003",
        rgba="1 0.9 0.1 0.8",
    )
    return actuators


def convert(urdf_path: Path, output_path: Path, mesh_output_dir: Path) -> None:
    urdf_path = urdf_path.resolve()
    urdf_root = ET.parse(urdf_path).getroot()
    package_root = urdf_path.parent.parent

    links = {link.get("name"): link for link in urdf_root.findall("link")}
    child_joints: dict[str, list[ET.Element]] = defaultdict(list)
    child_links: set[str] = set()
    for joint in urdf_root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        child_joints[parent].append(joint)
        child_links.add(child)
    root_links = [name for name in links if name not in child_links]
    if root_links != ["base_link"]:
        raise ValueError(f"expected base_link root, got {root_links}")

    mesh_output_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for stale_mesh in mesh_output_dir.glob("*.obj"):
        stale_mesh.unlink()

    mesh_assets: dict[str, tuple[str, np.ndarray, np.ndarray, Path]] = {}
    converted_mesh_count = 0
    for link in links.values():
        visual_mesh = link.find("visual/geometry/mesh")
        if visual_mesh is None:
            continue
        uri = visual_mesh.get("filename")
        prefix = "package://00_轮式升降机器人Marvin Max - 3/"
        if not uri.startswith(prefix):
            raise ValueError(f"unsupported mesh URI {uri!r}")
        source = package_root / uri[len(prefix) :]
        key = _slug(link.get("name"))
        mesh = trimesh.load(source, force="mesh", process=False)
        if len(mesh.faces) > 200_000:
            # MuJoCo's STL decoder rejects the imported 209,967-face base.
            # OBJ has no such limit; the remaining source STLs can be used
            # directly without duplicating another ~80 MB of mesh data.
            asset_path = mesh_output_dir / f"{key}.obj"
            mesh.export(asset_path)
            converted_mesh_count += 1
        else:
            asset_path = source
        mesh_assets[link.get("name")] = (
            key,
            mesh.bounds.copy(),
            mesh.extents.copy(),
            asset_path,
        )

    model = ET.Element("mujoco", model="gento_side_grasp_robot")
    ET.SubElement(
        model,
        "compiler",
        # dev-midlevel's source billiards table is authored in degrees. Body
        # transforms below use quaternions, and hinge ranges are converted to
        # degrees before serialization so this include shares that compiler.
        angle="degree",
        coordinate="local",
        autolimits="true",
        balanceinertia="true",
    )
    asset = ET.SubElement(model, "asset")
    ET.SubElement(asset, "material", name="gento_body_dark", rgba="0.34 0.38 0.44 1")
    ET.SubElement(asset, "material", name="gento_body_light", rgba="0.79 0.82 0.93 1")
    ET.SubElement(asset, "material", name="gento_wheel", rgba="0.09 0.10 0.12 1")
    model_dir = output_path.parent.resolve()
    for _, (mesh_name, _, _, mesh_path) in sorted(mesh_assets.items()):
        ET.SubElement(
            asset,
            "mesh",
            name=f"gento_mesh_{mesh_name}",
            file=str(mesh_path.relative_to(model_dir, walk_up=True)),
        )

    worldbody = ET.SubElement(model, "worldbody")
    # The imported ready pose was authored for a +X shot. Rotate the complete
    # robot another +90 degrees so the cue follows dev-midlevel's +Y table
    # convention while the mobile base stays outside the negative-Y rail.
    root_body = ET.SubElement(
        worldbody,
        "body",
        name="gento_root",
        pos="0.82 -1.50 0",
        quat="0 0 0 1",
    )

    actuator_specs: list[tuple[str, float, float, float, float]] = []
    gripper_actuators: list[tuple[str, float, float, float]] = []

    def add_link(link_name: str, parent: ET.Element, incoming: ET.Element | None = None) -> None:
        if incoming is None:
            body = parent
        else:
            pos, quat = _origin(incoming.find("origin"))
            joint_type = incoming.get("type", "fixed")
            limit = incoming.find("limit")
            lower = float(limit.get("lower", "0")) if limit is not None else 0.0
            upper = float(limit.get("upper", "0")) if limit is not None else 0.0
            axis = _numbers(
                incoming.find("axis").get("xyz") if incoming.find("axis") is not None else None,
                3,
                (0.0, 0.0, 1.0),
            )
            is_wheel = "wheel_joint" in incoming.get("name", "")
            fixed_revolute = joint_type in {"revolute", "continuous"} and (
                is_wheel or abs(upper - lower) < 1e-10
            )
            if fixed_revolute:
                fixed_angle = lower if abs(upper - lower) < 1e-10 else 0.0
                quat = _quat_mul(quat, _axis_angle_quat(axis, fixed_angle))
            body = ET.SubElement(
                parent,
                "body",
                name=_body_name(link_name),
                pos=_fmt(pos),
                quat=_fmt(quat / np.linalg.norm(quat)),
                gravcomp="1",
            )
            # SolidWorks exported this lift as fixed even though it carries a
            # valid 0..0.86 m prismatic limit.  Recover the intended joint.
            if incoming.get("name") == "waist_jiont_1_prismatic":
                joint_type = "prismatic"
                axis = np.asarray([0.0, 0.0, 1.0])
                fixed_revolute = False
            if joint_type != "fixed" and not fixed_revolute:
                name = _joint_name(incoming.get("name"))
                mj_type = "slide" if joint_type == "prismatic" else "hinge"
                attributes = dict(
                    name=name,
                    type=mj_type,
                    axis=_fmt(axis),
                    damping="5" if mj_type == "hinge" else "50",
                    armature="0.025" if mj_type == "hinge" else "0.08",
                )
                if joint_type != "continuous":
                    serialized_range = (
                        (math.degrees(lower), math.degrees(upper))
                        if mj_type == "hinge"
                        else (lower, upper)
                    )
                    attributes["range"] = _fmt(serialized_range)
                ET.SubElement(body, "joint", **attributes)
                effort = float(limit.get("effort", "80")) if limit is not None else 80.0
                effort = max(effort, 22.0)
                if incoming.get("name") == "waist_jiont_1_prismatic":
                    # The exporter marks the lift fixed and gives it zero
                    # effort.  Its 30+ kg upper assembly needs a realistic
                    # lift force budget once the intended slide is restored.
                    effort = 300.0
                home = min(max(0.0, lower), upper)
                actuator_specs.append((name, lower, upper, effort, home))

        link = links[link_name]
        if incoming is not None:
            _add_inertial(body, link)
        if link_name in mesh_assets:
            mesh_name, bounds, extents, _ = mesh_assets[link_name]
            if "wheel" in link_name:
                material = "gento_wheel"
            elif link_name in {"base_link", "waist_Link_1_prismatic", "waist_Link_2"}:
                material = "gento_body_dark"
            else:
                material = "gento_body_light"
            ET.SubElement(
                body,
                "geom",
                name=f"gento_visual_{mesh_name}",
                type="mesh",
                mesh=f"gento_mesh_{mesh_name}",
                material=material,
                contype="0",
                conaffinity="0",
                density="0",
            )
            if link_name.startswith("arm "):
                # A conservative local bounding proxy prevents the imported
                # high-poly arm from entering the solid table/rail volumes.
                center = (bounds[0] + bounds[1]) / 2.0
                half_size = np.maximum(extents * 0.43, 0.012)
                ET.SubElement(
                    body,
                    "geom",
                    name=f"gento_table_proxy_{mesh_name}",
                    type="box",
                    pos=_fmt(center),
                    size=_fmt(half_size),
                    rgba="0.12 0.68 1 0",
                    group="4",
                    contype="16",
                    conaffinity="8",
                    condim="4",
                    friction="0.70 0.01 0.001",
                    solref="0.004 1",
                    solimp="0.92 0.99 0.002",
                    density="0",
                )

        if link_name == "arm left_Link_7":
            gripper_actuators.extend(_add_gripper(body, "left"))
        elif link_name == "arm right_Link_7":
            gripper_actuators.extend(_add_gripper(body, "right"))

        for child_joint in child_joints.get(link_name, []):
            add_link(child_joint.find("child").get("link"), body, child_joint)

    add_link("base_link", root_body)

    actuators = ET.SubElement(model, "actuator")
    for name, lower, upper, effort, _ in actuator_specs:
        if "waist_jiont_1_prismatic" in name:
            stiffness = "6000"
        elif "arm_" in name:
            stiffness = "2000"
        elif "waist_joint" in name:
            stiffness = "2000"
        else:
            stiffness = "180"
        ET.SubElement(
            actuators,
            "position",
            name=f"{name}_pos",
            joint=name,
            kp=stiffness,
            forcerange=_fmt((-effort, effort)),
            ctrlrange=_fmt((lower, upper)),
        )
    for name, lower, upper, _ in gripper_actuators:
        ET.SubElement(
            actuators,
            "position",
            name=f"{name}_pos",
            joint=name,
            # With a 3.5 mm commanded preload this yields roughly 4.2 N per
            # pad, enough for the two lower fingers to support the 0.53 kg cue
            # under gravity while remaining comfortably below the force cap.
            kp="1200",
            forcerange="-100 100",
            ctrlrange=_fmt((lower, upper)),
        )

    ET.indent(model, space="  ")
    ET.ElementTree(model).write(output_path, encoding="unicode", xml_declaration=False)
    print(f"wrote {output_path.relative_to(REPO_ROOT)}")
    print(
        f"converted {converted_mesh_count} oversized mesh into "
        f"{mesh_output_dir.relative_to(REPO_ROOT)}"
    )
    print(f"created {len(actuator_specs)} imported joints and {len(gripper_actuators)} finger joints")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mesh-output-dir", type=Path, default=DEFAULT_MESH_DIR)
    args = parser.parse_args()
    convert(args.urdf, args.output, args.mesh_output_dir)


if __name__ == "__main__":
    main()
