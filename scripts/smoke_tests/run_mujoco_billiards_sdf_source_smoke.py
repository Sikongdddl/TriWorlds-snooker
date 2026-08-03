"""Validate the rebuilt mujoco-billiards SDF plugin at its C callback boundary."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


_VOID_POINTER_FIELDS = (
    "nstate",
    "nsensordata",
    "init",
    "destroy",
    "copy",
    "reset",
    "compute",
    "advance",
    "visualize",
    "actuator_act_dot",
    "sdf_distance",
    "sdf_gradient",
    "sdf_staticdistance",
    "sdf_attribute",
    "sdf_aabb",
)


class _MjpPlugin(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("nattribute", ctypes.c_int),
        ("attributes", ctypes.POINTER(ctypes.c_char_p)),
        ("capabilityflags", ctypes.c_int),
        ("needstage", ctypes.c_int),
        *((name, ctypes.c_void_p) for name in _VOID_POINTER_FIELDS),
    ]


_StaticDistance = ctypes.CFUNCTYPE(
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
)
_RuntimeGradient = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
    ctypes.c_int,
)
_Aabb = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
)


@dataclass(frozen=True)
class _Shape:
    plugin_name: str
    instance_name: str
    attributes: tuple[float, ...]
    attribute_names: tuple[str, ...]


_SHAPES = (
    _Shape(
        "mujoco.sdf.trapezoid",
        "test_trapezoid",
        (0.8, 1.0, 0.4, 0.6),
        ("basewidth", "topwidth", "height", "depth"),
    ),
    _Shape(
        "mujoco.sdf.hollow_cylinder",
        "test_hollow",
        (0.25, 0.35, 0.08),
        ("halfheight", "radius", "halfthickness"),
    ),
    _Shape(
        "mujoco.sdf.half_hollow_cylinder",
        "test_half_hollow",
        (0.25, 0.35, 0.08),
        ("halfheight", "radius", "halfthickness"),
    ),
)

_ALL_PLUGIN_NAMES = (
    "mujoco.sdf.bolt",
    "mujoco.sdf.bowl",
    "mujoco.sdf.chopped_cylinder",
    "mujoco.sdf.cone",
    "mujoco.sdf.gear",
    "mujoco.sdf.half_hollow_cylinder",
    "mujoco.sdf.hollow_cylinder",
    "mujoco.sdf.nut",
    "mujoco.sdf.torus",
    "mujoco.sdf.trapezoid",
    "mujoco.sdf.vertical_capped_cylinder",
)


def _load_plugin_api() -> tuple[ctypes.CDLL, Path]:
    package_dir = Path(mujoco.__file__).resolve().parent
    plugin_path = package_dir / "plugin" / "libsdf_plugin.so"
    if not plugin_path.is_file():
        raise FileNotFoundError(
            f"{plugin_path} is missing; run the SDF plugin build script first."
        )
    mujoco.mj_loadPluginLibrary(str(plugin_path))
    libraries = sorted(package_dir.glob("libmujoco.so.*"))
    if not libraries:
        raise FileNotFoundError(f"No libmujoco shared library under {package_dir}")
    library = ctypes.CDLL(str(libraries[-1]))
    library.mjp_getPlugin.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    library.mjp_getPlugin.restype = ctypes.POINTER(_MjpPlugin)
    return library, plugin_path


def _plugin(library: ctypes.CDLL, name: str) -> _MjpPlugin:
    slot = ctypes.c_int(-1)
    pointer = library.mjp_getPlugin(name.encode(), ctypes.byref(slot))
    if not pointer:
        raise RuntimeError(f"SDF plugin {name!r} is not registered.")
    plugin = pointer.contents
    if plugin.name.decode() != name or not plugin.sdf_staticdistance:
        raise RuntimeError(f"SDF plugin {name!r} has an invalid callback table.")
    return plugin


def _as_c(values: np.ndarray | tuple[float, ...]) -> ctypes.Array[ctypes.c_double]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return (ctypes.c_double * flat.size)(*flat)


def _static_distance(
    plugin: _MjpPlugin,
    point: np.ndarray,
    attributes: tuple[float, ...],
) -> float:
    callback = _StaticDistance(plugin.sdf_staticdistance)
    return float(callback(_as_c(point), _as_c(attributes)))


def _extruded_distance(planar: float, axial: float) -> float:
    outside = np.linalg.norm((max(planar, 0.0), max(axial, 0.0)))
    return min(max(planar, axial), 0.0) + float(outside)


def _trapezoid_reference(point: np.ndarray, attributes: tuple[float, ...]) -> float:
    base, top, height, depth = attributes
    vertices = np.array(
        [
            (-0.5 * base, -0.5 * height),
            (0.5 * base, -0.5 * height),
            (0.5 * top, 0.5 * height),
            (-0.5 * top, 0.5 * height),
        ],
        dtype=np.float64,
    )
    query = point[[0, 2]]
    distances: list[float] = []
    inside = True
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        edge = end - start
        parameter = np.clip(np.dot(query - start, edge) / np.dot(edge, edge), 0, 1)
        distances.append(float(np.linalg.norm(query - (start + parameter * edge))))
        cross = edge[0] * (query[1] - start[1]) - edge[1] * (
            query[0] - start[0]
        )
        inside &= bool(cross >= -1.0e-14)
    planar = min(distances) * (-1.0 if inside else 1.0)
    return _extruded_distance(planar, abs(float(point[1])) - 0.5 * depth)


def _hollow_reference(point: np.ndarray, attributes: tuple[float, ...]) -> float:
    halfheight, radius, halfthickness = attributes
    planar = abs(float(np.linalg.norm(point[:2])) - radius) - halfthickness
    return _extruded_distance(planar, abs(float(point[2])) - halfheight)


def _half_hollow_reference(
    point: np.ndarray,
    attributes: tuple[float, ...],
) -> float:
    halfheight, radius, halfthickness = attributes
    inner = radius - halfthickness
    outer = radius + halfthickness
    query = point[:2]
    radial = float(np.linalg.norm(query))
    candidates: list[np.ndarray] = []
    for arc_radius in (inner, outer):
        if query[0] >= 0.0 and radial > 1.0e-12:
            candidates.append(query * (arc_radius / radial))
        else:
            candidates.extend(
                (
                    np.array((0.0, arc_radius)),
                    np.array((0.0, -arc_radius)),
                )
            )
    candidates.extend(
        (
            np.array((0.0, np.clip(query[1], inner, outer))),
            np.array((0.0, np.clip(query[1], -outer, -inner))),
        )
    )
    planar = min(float(np.linalg.norm(query - candidate)) for candidate in candidates)
    inside = query[0] >= -1.0e-14 and inner - 1.0e-14 <= radial <= outer + 1.0e-14
    if inside:
        planar = -planar
    return _extruded_distance(planar, abs(float(point[2])) - halfheight)


def _model_xml() -> str:
    extensions: list[str] = []
    meshes: list[str] = []
    geoms: list[str] = []
    for index, shape in enumerate(_SHAPES):
        configs = "".join(
            f'<config key="{name}" value="{value:.17g}"/>'
            for name, value in zip(shape.attribute_names, shape.attributes)
        )
        extensions.append(
            f'<plugin plugin="{shape.plugin_name}">'
            f'<instance name="{shape.instance_name}">{configs}</instance>'
            "</plugin>"
        )
        meshes.append(
            f'<mesh name="shape_{index}"><plugin instance="{shape.instance_name}"/></mesh>'
        )
        geoms.append(
            f'<geom type="sdf" mesh="shape_{index}" pos="{2 * index} 0 0">'
            f'<plugin instance="{shape.instance_name}"/></geom>'
        )
    return (
        "<mujoco><extension>"
        + "".join(extensions)
        + "</extension><asset>"
        + "".join(meshes)
        + "</asset><worldbody>"
        + "".join(geoms)
        + "</worldbody></mujoco>"
    )


def _check_distances_and_aabbs(library: ctypes.CDLL) -> None:
    references = (
        _trapezoid_reference,
        _hollow_reference,
        _half_hollow_reference,
    )
    expected_aabbs = (
        np.array((0.0, 0.0, 0.0, 0.5, 0.3, 0.2)),
        np.array((0.0, 0.0, 0.0, 0.43, 0.43, 0.25)),
        np.array((0.215, 0.0, 0.0, 0.215, 0.43, 0.25)),
    )
    rng = np.random.default_rng(20260723)
    for shape, reference, expected_aabb in zip(
        _SHAPES,
        references,
        expected_aabbs,
    ):
        plugin = _plugin(library, shape.plugin_name)
        callback = _Aabb(plugin.sdf_aabb)
        actual_aabb = _as_c(np.zeros(6))
        callback(actual_aabb, _as_c(shape.attributes))
        np.testing.assert_allclose(actual_aabb[:], expected_aabb, atol=1.0e-14)

        maximum_error = 0.0
        for point in rng.uniform(-0.75, 0.75, size=(1_000, 3)):
            actual = _static_distance(plugin, point, shape.attributes)
            expected = reference(point, shape.attributes)
            maximum_error = max(maximum_error, abs(actual - expected))
        if maximum_error > 2.0e-12:
            raise AssertionError(
                f"{shape.plugin_name} distance error {maximum_error:.3e}"
            )
        print(f"{shape.instance_name}_distance_max_error={maximum_error:.3e}")


def _check_runtime_gradients(library: ctypes.CDLL) -> None:
    model = mujoco.MjModel.from_xml_string(_model_xml())
    data = mujoco.MjData(model)
    rng = np.random.default_rng(8675309)
    epsilon = 1.0e-6
    for instance, shape in enumerate(_SHAPES):
        plugin = _plugin(library, shape.plugin_name)
        gradient_callback = _RuntimeGradient(plugin.sdf_gradient)
        maximum_error = 0.0
        checked = 0
        for point in rng.uniform(-0.7, 0.7, size=(600, 3)):
            numerical = np.zeros(3, dtype=np.float64)
            for axis in range(3):
                delta = np.zeros(3, dtype=np.float64)
                delta[axis] = epsilon
                numerical[axis] = (
                    _static_distance(plugin, point + delta, shape.attributes)
                    - _static_distance(plugin, point - delta, shape.attributes)
                ) / (2.0 * epsilon)
            norm = float(np.linalg.norm(numerical))
            if norm < 0.999 or norm > 1.001:
                continue
            output = _as_c(np.zeros(3))
            gradient_callback(output, _as_c(point), data._address, instance)
            analytic = np.asarray(output[:])
            error = float(np.linalg.norm(analytic - numerical))
            maximum_error = max(maximum_error, error)
            checked += 1
        if checked < 300:
            raise AssertionError(
                f"Only {checked} differentiable samples passed for {shape.plugin_name}."
            )
        if maximum_error > 2.0e-5:
            raise AssertionError(
                f"{shape.plugin_name} gradient error {maximum_error:.3e}"
            )
        print(
            f"{shape.instance_name}_gradient_samples={checked} "
            f"max_error={maximum_error:.3e}"
        )


def _invalid_trapezoid_xml(value: str, key: str = "height") -> str:
    values = {
        "basewidth": "0.8",
        "topwidth": "1.0",
        "height": "0.4",
        "depth": "0.6",
    }
    values[key] = value
    configs = "".join(
        f'<config key="{name}" value="{attribute}"/>'
        for name, attribute in values.items()
    )
    return (
        '<mujoco><extension><plugin plugin="mujoco.sdf.trapezoid">'
        f'<instance name="invalid">{configs}</instance>'
        '</plugin></extension><asset><mesh name="invalid_mesh">'
        '<plugin instance="invalid"/></mesh></asset><worldbody>'
        '<geom type="sdf" mesh="invalid_mesh"><plugin instance="invalid"/></geom>'
        "</worldbody></mujoco>"
    )


def _invalid_plugin_xml(plugin_name: str, key: str, value: str) -> str:
    return (
        f'<mujoco><extension><plugin plugin="{plugin_name}">'
        f'<instance name="invalid"><config key="{key}" value="{value}"/></instance>'
        '</plugin></extension><asset><mesh name="invalid_mesh">'
        '<plugin instance="invalid"/></mesh></asset><worldbody>'
        '<geom type="sdf" mesh="invalid_mesh"><plugin instance="invalid"/></geom>'
        "</worldbody></mujoco>"
    )


def _check_invalid_attributes() -> None:
    for value, key in (
        ("0", "height"),
        ("-0.1", "height"),
        ("nan", "height"),
        ("inf", "height"),
        ("1e5000", "height"),
        ("0", "basewidth"),
    ):
        try:
            mujoco.MjModel.from_xml_string(_invalid_trapezoid_xml(value, key))
        except ValueError:
            continue
        raise AssertionError(f"Invalid trapezoid {key}={value!r} was accepted.")
    invalid_domains = (
        ("mujoco.sdf.bolt", "radius", "0"),
        ("mujoco.sdf.bowl", "height", "2"),
        ("mujoco.sdf.chopped_cylinder", "halfthickness", "2"),
        ("mujoco.sdf.cone", "height", "0"),
        ("mujoco.sdf.gear", "teeth", "2.5"),
        ("mujoco.sdf.half_hollow_cylinder", "halfthickness", "2"),
        ("mujoco.sdf.hollow_cylinder", "halfthickness", "2"),
        ("mujoco.sdf.nut", "radius", "0"),
        ("mujoco.sdf.torus", "radius1", "0"),
        ("mujoco.sdf.vertical_capped_cylinder", "radius", "0"),
    )
    for plugin_name, key, value in invalid_domains:
        try:
            mujoco.MjModel.from_xml_string(
                _invalid_plugin_xml(plugin_name, key, value)
            )
        except ValueError:
            continue
        raise AssertionError(
            f"Invalid {plugin_name} {key}={value!r} was accepted."
        )
    print("invalid_attribute_rejection=PASS")


def _check_all_defaults_compile(library: ctypes.CDLL) -> None:
    extensions: list[str] = []
    meshes: list[str] = []
    geoms: list[str] = []
    for index, plugin_name in enumerate(_ALL_PLUGIN_NAMES):
        _plugin(library, plugin_name)
        instance = f"default_{index}"
        mesh = f"default_mesh_{index}"
        extensions.append(
            f'<plugin plugin="{plugin_name}"><instance name="{instance}"/></plugin>'
        )
        meshes.append(f'<mesh name="{mesh}"><plugin instance="{instance}"/></mesh>')
        geoms.append(
            f'<geom type="sdf" mesh="{mesh}" pos="{3 * index} 0 0">'
            f'<plugin instance="{instance}"/></geom>'
        )
    xml = (
        "<mujoco><extension>"
        + "".join(extensions)
        + "</extension><asset>"
        + "".join(meshes)
        + "</asset><worldbody>"
        + "".join(geoms)
        + "</worldbody></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    if model.nplugin != len(_ALL_PLUGIN_NAMES):
        raise AssertionError(
            f"Expected {len(_ALL_PLUGIN_NAMES)} plugins, got {model.nplugin}."
        )
    if not np.isfinite(model.geom_aabb).all() or np.any(model.geom_aabb[:, 3:] <= 0):
        raise AssertionError("A default SDF plugin produced an invalid geom AABB.")
    print(f"default_plugin_compile_count={model.nplugin}")


def main() -> None:
    library, plugin_path = _load_plugin_api()
    _check_distances_and_aabbs(library)
    _check_runtime_gradients(library)
    _check_invalid_attributes()
    _check_all_defaults_compile(library)
    print(f"plugin={plugin_path}")
    print("mujoco_billiards_sdf_source=PASS")


if __name__ == "__main__":
    main()
