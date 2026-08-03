"""MuJoCo Warp implementations of the active table's analytical SDF plugins."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src.types import MJ_MINMU, vec_pluginattr
import numpy as np
import warp as wp


@dataclass(frozen=True)
class SDFPluginTypes:
    """Global MuJoCo plugin type IDs used by the compiled table model."""

    hollow_cylinder: int
    half_hollow_cylinder: int
    trapezoid: int


TRAPEZOID_SDF_DAMPING = 4_400.0
TRAPEZOID_MANIFOLD_CONTACTS = 5
TRAPEZOID_MANIFOLD_RAIL_SPAN_RATIO = 0.03
TRAPEZOID_MANIFOLD_DEPTH_SPAN_RATIO = 0.01
TRAPEZOID_MANIFOLD_SATELLITE_LEVER_RATIO = 0.25
MUJOCO_WARP_NCONMAX = 128
MUJOCO_WARP_NJMAX = 1_024


def normalize_zero_friction_contact_dims(model: mujoco.MjModel) -> tuple[int, ...]:
    """Remove zero-width friction dimensions that make MJWarp's solver singular.

    MuJoCo permits contacts such as ``condim=3, friction=0 0 0``. They are
    mathematically equivalent to a normal-only ``condim=1`` contact, but
    MJWarp warns that the zero-width friction constraints can produce NaNs.
    Dimensions with no physical width are removed. If rolling resistance is
    requested without a torsional dimension, the missing coefficient is raised
    only to MJWarp's minimum positive value because MuJoCo has no ``condim=5``.
    Explicit contact pairs are normalized with the same rules.
    """

    changed: list[int] = []
    for geom_id in range(model.ngeom):
        condim = int(model.geom_condim[geom_id])
        sliding, torsional, rolling = model.geom_friction[geom_id]
        replacement = condim
        friction_changed = False
        if condim > 1 and sliding < MJ_MINMU:
            replacement = 1
        elif condim == 6 and torsional < MJ_MINMU and rolling < MJ_MINMU:
            replacement = 3
        elif condim == 6 and rolling < MJ_MINMU:
            replacement = 4
        elif condim == 6 and torsional < MJ_MINMU:
            model.geom_friction[geom_id, 1] = MJ_MINMU
            friction_changed = True
        elif condim == 4 and torsional < MJ_MINMU:
            replacement = 3
        if replacement != condim:
            model.geom_condim[geom_id] = replacement
        if replacement != condim or friction_changed:
            changed.append(geom_id)

    for pair_id in range(model.npair):
        condim = int(model.pair_dim[pair_id])
        friction = model.pair_friction[pair_id]
        replacement = condim
        if condim > 1 and friction[0] < MJ_MINMU:
            replacement = 1
        elif condim == 4:
            if friction[1] < MJ_MINMU:
                friction[1] = MJ_MINMU
            if friction[2] < MJ_MINMU:
                replacement = 3
        elif condim == 6:
            if friction[1] < MJ_MINMU:
                friction[1] = MJ_MINMU
            torsional = friction[2]
            rolling_1 = friction[3]
            rolling_2 = friction[4]
            if torsional < MJ_MINMU and rolling_1 < MJ_MINMU and rolling_2 < MJ_MINMU:
                replacement = 3
            elif rolling_1 < MJ_MINMU and rolling_2 < MJ_MINMU:
                replacement = 4
            else:
                if torsional < MJ_MINMU:
                    friction[2] = MJ_MINMU
                if rolling_1 < MJ_MINMU:
                    friction[3] = MJ_MINMU
                if rolling_2 < MJ_MINMU:
                    friction[4] = MJ_MINMU
        if replacement != condim:
            model.pair_dim[pair_id] = replacement
    return tuple(changed)


def calibrate_trapezoid_sdf_contact_damping(
    model: mujoco.MjModel,
    damping: float = TRAPEZOID_SDF_DAMPING,
) -> tuple[int, ...]:
    """Calibrate direct-format damping for MJWarp's trapezoid contact manifold.

    CPU MuJoCo creates a multi-contact manifold for the source trapezoid SDF,
    while the MJWarp path uses a small stable representative manifold. The
    direct damping coefficient compensates for the remaining constraint-layout
    difference without changing stiffness or the CPU model.
    """

    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("Trapezoid SDF damping must be positive and finite.")
    changed: list[int] = []
    trapezoid_geoms: set[int] = set()
    for geom_id in range(model.ngeom):
        instance_id = int(model.geom_plugin[geom_id])
        if instance_id < 0:
            continue
        instance_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_PLUGIN,
            instance_id,
        )
        if instance_name is None or not instance_name.startswith("trapezoid"):
            continue
        if model.geom_solref[geom_id, 0] >= 0.0:
            raise ValueError(
                f"Trapezoid geom {geom_id} must use direct-format negative solref."
            )
        model.geom_solref[geom_id, 1] = -damping
        changed.append(geom_id)
        trapezoid_geoms.add(geom_id)
    if not changed:
        raise ValueError("No trapezoid SDF geoms were found for MJWarp calibration.")
    for pair_id in range(model.npair):
        if (
            int(model.pair_geom1[pair_id]) not in trapezoid_geoms
            and int(model.pair_geom2[pair_id]) not in trapezoid_geoms
        ):
            continue
        if model.pair_solref[pair_id, 0] >= 0.0:
            raise ValueError(
                f"Trapezoid contact pair {pair_id} must use direct-format negative solref."
            )
        model.pair_solref[pair_id, 1] = -damping
    return tuple(changed)


def assert_mujoco_warp_capacity(data: object, *, context: str) -> None:
    """Synchronize and fail if MJWarp truncated contacts or constraints."""

    wp.synchronize()
    overflow = np.asarray(data.overflow.numpy(), dtype=np.int64)
    nacon = int(data.nacon.numpy()[0])
    nefc = np.asarray(data.nefc.numpy(), dtype=np.int64)
    failures: list[str] = []
    if np.any(overflow):
        failures.append(f"overflow flags={overflow.tolist()}")
    if nacon > int(data.naconmax):
        failures.append(f"contacts={nacon}>{int(data.naconmax)}")
    maximum_nefc = int(np.max(nefc, initial=0))
    if maximum_nefc > int(data.njmax):
        failures.append(f"constraints={maximum_nefc}>{int(data.njmax)}")
    if failures:
        raise RuntimeError(f"MJWarp capacity failure during {context}: " + ", ".join(failures))


@wp.func
def _safe_sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


@wp.func
def _extruded_distance_and_gradient(
    planar_distance: float,
    planar_gradient: wp.vec3,
    axial_distance: float,
    axial_gradient: wp.vec3,
) -> tuple[float, wp.vec3]:
    outside_planar = wp.max(planar_distance, 0.0)
    outside_axial = wp.max(axial_distance, 0.0)
    outside = wp.length(wp.vec2(outside_planar, outside_axial))
    distance = (
        wp.min(wp.max(planar_distance, axial_distance), 0.0)
        + outside
    )
    if outside > 1.0e-12:
        return (
            distance,
            planar_gradient * (outside_planar / outside)
            + axial_gradient * (outside_axial / outside),
        )
    if planar_distance > axial_distance + 1.0e-12:
        return distance, planar_gradient
    if axial_distance > planar_distance + 1.0e-12:
        return distance, axial_gradient
    gradient = planar_gradient + axial_gradient
    gradient_length = wp.length(gradient)
    if gradient_length > 1.0e-12:
        gradient = gradient / gradient_length
    return distance, gradient


@wp.func
def _hollow_cylinder_distance_and_gradient(
    point: wp.vec3,
    attributes: vec_pluginattr,
) -> tuple[float, wp.vec3]:
    halfheight = attributes[0]
    radius = attributes[1]
    halfthickness = attributes[2]
    radial_distance = wp.length(wp.vec2(point[0], point[1]))
    side_distance = wp.abs(radial_distance - radius) - halfthickness
    cap_distance = wp.abs(point[2]) - halfheight
    side_gradient = wp.vec3(0.0)
    if radial_distance > 1.0e-12:
        scale = _safe_sign(radial_distance - radius) / radial_distance
        side_gradient = wp.vec3(
            point[0] * scale,
            point[1] * scale,
            0.0,
        )
    return _extruded_distance_and_gradient(
        side_distance,
        side_gradient,
        cap_distance,
        wp.vec3(0.0, 0.0, _safe_sign(point[2])),
    )


@wp.func
def _hollow_cylinder_distance(point: wp.vec3, attributes: vec_pluginattr) -> float:
    distance, gradient = _hollow_cylinder_distance_and_gradient(point, attributes)
    return distance


@wp.func
def _hollow_cylinder_gradient(point: wp.vec3, attributes: vec_pluginattr) -> wp.vec3:
    distance, gradient = _hollow_cylinder_distance_and_gradient(point, attributes)
    return gradient


@wp.func
def _half_annulus_distance_and_gradient(
    point: wp.vec2,
    radius: float,
    halfthickness: float,
) -> tuple[float, wp.vec2]:
    inner_radius = radius - halfthickness
    outer_radius = radius + halfthickness
    radial = wp.length(point)
    radial_direction = wp.vec2(0.0)
    if radial > 1.0e-12:
        radial_direction = point / radial

    if (
        point[0] >= 0.0
        and radial >= inner_radius
        and radial <= outer_radius
    ):
        inner_distance = radial - inner_radius
        outer_distance = outer_radius - radial
        cut_distance = point[0]
        minimum = wp.min(wp.min(inner_distance, outer_distance), cut_distance)
        gradient = wp.vec2(0.0)
        if inner_distance <= minimum + 1.0e-12:
            gradient = gradient - radial_direction
        if outer_distance <= minimum + 1.0e-12:
            gradient = gradient + radial_direction
        if cut_distance <= minimum + 1.0e-12:
            gradient = gradient + wp.vec2(-1.0, 0.0)
        gradient_length = wp.length(gradient)
        if gradient_length > 1.0e-12:
            gradient = gradient / gradient_length
        return -minimum, gradient

    if point[0] >= 0.0:
        return (
            wp.abs(radial - radius) - halfthickness,
            radial_direction * _safe_sign(radial - radius),
        )

    positive_y = wp.clamp(point[1], inner_radius, outer_radius)
    negative_y = wp.clamp(point[1], -outer_radius, -inner_radius)
    positive_delta = point - wp.vec2(0.0, positive_y)
    negative_delta = point - wp.vec2(0.0, negative_y)
    positive_distance = wp.length(positive_delta)
    negative_distance = wp.length(negative_delta)
    if positive_distance < negative_distance - 1.0e-12:
        return positive_distance, positive_delta / positive_distance
    if negative_distance < positive_distance - 1.0e-12:
        return negative_distance, negative_delta / negative_distance
    gradient = positive_delta + negative_delta
    gradient_length = wp.length(gradient)
    if gradient_length > 1.0e-12:
        gradient = gradient / gradient_length
    return positive_distance, gradient


@wp.func
def _half_hollow_cylinder_distance_and_gradient(
    point: wp.vec3,
    attributes: vec_pluginattr,
) -> tuple[float, wp.vec3]:
    planar_distance, planar_gradient = _half_annulus_distance_and_gradient(
        wp.vec2(point[0], point[1]),
        attributes[1],
        attributes[2],
    )
    return _extruded_distance_and_gradient(
        planar_distance,
        wp.vec3(planar_gradient[0], planar_gradient[1], 0.0),
        wp.abs(point[2]) - attributes[0],
        wp.vec3(0.0, 0.0, _safe_sign(point[2])),
    )


@wp.func
def _half_hollow_cylinder_distance(
    point: wp.vec3,
    attributes: vec_pluginattr,
) -> float:
    distance, gradient = _half_hollow_cylinder_distance_and_gradient(
        point,
        attributes,
    )
    return distance


@wp.func
def _half_hollow_cylinder_gradient(
    point: wp.vec3,
    attributes: vec_pluginattr,
) -> wp.vec3:
    distance, gradient = _half_hollow_cylinder_distance_and_gradient(
        point,
        attributes,
    )
    return gradient


@wp.func
def _closest_segment_point_2d(
    point: wp.vec2,
    start: wp.vec2,
    end: wp.vec2,
) -> tuple[wp.vec2, float, wp.vec2]:
    edge = end - start
    edge_length_squared = wp.dot(edge, edge)
    parameter = 0.0
    if edge_length_squared > 1.0e-12:
        parameter = wp.clamp(
            wp.dot(point - start, edge) / edge_length_squared,
            0.0,
            1.0,
        )
    closest = start + parameter * edge
    delta = point - closest
    outward = wp.vec2(edge[1], -edge[0])
    outward_length = wp.length(outward)
    if outward_length > 1.0e-12:
        outward = outward / outward_length
    return delta, wp.dot(delta, delta), outward


@wp.func
def _trapezoid_xz_distance_and_normal(
    point: wp.vec2,
    base_width: float,
    top_width: float,
    height: float,
) -> tuple[float, wp.vec2]:
    halfheight = 0.5 * height
    bottom_left = wp.vec2(-0.5 * base_width, -halfheight)
    bottom_right = wp.vec2(0.5 * base_width, -halfheight)
    top_right = wp.vec2(0.5 * top_width, halfheight)
    top_left = wp.vec2(-0.5 * top_width, halfheight)

    delta, best_distance_squared, best_outward = _closest_segment_point_2d(
        point,
        bottom_left,
        bottom_right,
    )
    best_delta = delta

    delta, distance_squared, outward = _closest_segment_point_2d(
        point,
        bottom_right,
        top_right,
    )
    if distance_squared < best_distance_squared:
        best_delta = delta
        best_distance_squared = distance_squared
        best_outward = outward

    delta, distance_squared, outward = _closest_segment_point_2d(
        point,
        top_right,
        top_left,
    )
    if distance_squared < best_distance_squared:
        best_delta = delta
        best_distance_squared = distance_squared
        best_outward = outward

    delta, distance_squared, outward = _closest_segment_point_2d(
        point,
        top_left,
        bottom_left,
    )
    if distance_squared < best_distance_squared:
        best_delta = delta
        best_distance_squared = distance_squared
        best_outward = outward

    inside = (
        (bottom_right[0] - bottom_left[0]) * (point[1] - bottom_left[1])
        - (bottom_right[1] - bottom_left[1]) * (point[0] - bottom_left[0])
        >= 0.0
        and (top_right[0] - bottom_right[0]) * (point[1] - bottom_right[1])
        - (top_right[1] - bottom_right[1]) * (point[0] - bottom_right[0])
        >= 0.0
        and (top_left[0] - top_right[0]) * (point[1] - top_right[1])
        - (top_left[1] - top_right[1]) * (point[0] - top_right[0])
        >= 0.0
        and (bottom_left[0] - top_left[0]) * (point[1] - top_left[1])
        - (bottom_left[1] - top_left[1]) * (point[0] - top_left[0])
        >= 0.0
    )
    distance = wp.sqrt(best_distance_squared)
    normal = best_outward
    if distance > 1.0e-12:
        if inside:
            normal = -best_delta / distance
        else:
            normal = best_delta / distance
    if inside:
        distance = -distance
    return distance, normal


@wp.func
def _trapezoid_distance_and_gradient(
    point: wp.vec3,
    attributes: vec_pluginattr,
) -> tuple[float, wp.vec3]:
    planar_distance, planar_gradient = _trapezoid_xz_distance_and_normal(
        wp.vec2(point[0], point[2]),
        attributes[0],
        attributes[1],
        attributes[2],
    )
    return _extruded_distance_and_gradient(
        planar_distance,
        wp.vec3(planar_gradient[0], 0.0, planar_gradient[1]),
        wp.abs(point[1]) - 0.5 * attributes[3],
        wp.vec3(0.0, _safe_sign(point[1]), 0.0),
    )


@wp.func
def _trapezoid_distance(point: wp.vec3, attributes: vec_pluginattr) -> float:
    distance, gradient = _trapezoid_distance_and_gradient(point, attributes)
    return distance


@wp.func
def _trapezoid_gradient(point: wp.vec3, attributes: vec_pluginattr) -> wp.vec3:
    distance, gradient = _trapezoid_distance_and_gradient(point, attributes)
    return gradient


@wp.func
def _trapezoid_sphere_contact(
    center: wp.vec3,
    attributes: vec_pluginattr,
    radius: float,
    contact_index: int,
) -> tuple[bool, float, wp.vec3, wp.vec3, float]:
    base_width = attributes[0]
    top_width = attributes[1]
    height = attributes[2]
    depth = attributes[3]
    if (
        base_width <= 0.0
        or top_width <= 0.0
        or height <= 0.0
        or depth <= 0.0
        or radius <= 0.0
    ):
        return False, 0.0, wp.vec3(0.0), wp.vec3(0.0), 1.0

    xz_distance, xz_normal = _trapezoid_xz_distance_and_normal(
        wp.vec2(center[0], center[2]),
        base_width,
        top_width,
        height,
    )
    y_distance = wp.abs(center[1]) - 0.5 * depth
    outside_xz = wp.max(xz_distance, 0.0)
    outside_y = wp.max(y_distance, 0.0)
    outside_distance = wp.length(wp.vec2(outside_xz, outside_y))

    normal = wp.vec3(0.0)
    prism_distance = 0.0
    if outside_distance > 1.0e-12:
        xz_scale = outside_xz / outside_distance
        y_scale = outside_y / outside_distance
        normal = wp.vec3(
            xz_normal[0] * xz_scale,
            _safe_sign(center[1]) * y_scale,
            xz_normal[1] * xz_scale,
        )
        prism_distance = outside_distance
    elif xz_distance >= y_distance:
        normal = wp.vec3(xz_normal[0], 0.0, xz_normal[1])
        prism_distance = xz_distance
    else:
        normal = wp.vec3(0.0, _safe_sign(center[1]), 0.0)
        prism_distance = y_distance

    # The center point preserves the exact sphere/prism surface midpoint. Four
    # symmetric satellites add a small, deterministic tangential footprint for
    # oblique impacts. Their reduced effective lever avoids multiplying the
    # full-radius friction torque fivefold while retaining manifold damping.
    offset = wp.vec3(0.0)
    lever_scale = 1.0
    rail_span = radius * TRAPEZOID_MANIFOLD_RAIL_SPAN_RATIO
    depth_span = radius * TRAPEZOID_MANIFOLD_DEPTH_SPAN_RATIO
    if contact_index == 1:
        offset = wp.vec3(rail_span, 0.0, 0.0)
        lever_scale = TRAPEZOID_MANIFOLD_SATELLITE_LEVER_RATIO
    elif contact_index == 2:
        offset = wp.vec3(-rail_span, 0.0, 0.0)
        lever_scale = TRAPEZOID_MANIFOLD_SATELLITE_LEVER_RATIO
    elif contact_index == 3:
        offset = wp.vec3(0.0, depth_span, 0.0)
        lever_scale = TRAPEZOID_MANIFOLD_SATELLITE_LEVER_RATIO
    elif contact_index == 4:
        offset = wp.vec3(0.0, -depth_span, 0.0)
        lever_scale = TRAPEZOID_MANIFOLD_SATELLITE_LEVER_RATIO
    elif contact_index != 0:
        return False, 0.0, wp.vec3(0.0), wp.vec3(0.0), 1.0
    return True, prism_distance - radius, normal, offset, lever_scale


def _plugin_types(model: mujoco.MjModel) -> SDFPluginTypes:
    plugin_types: dict[str, set[int]] = {
        "hollow_cylinder": set(),
        "half_hollow_cylinder": set(),
        "trapezoid": set(),
    }
    for instance_id in range(model.nplugin):
        instance_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_PLUGIN,
            instance_id,
        )
        if instance_name is None:
            continue
        if instance_name.startswith("half_hollow_cylinder"):
            key = "half_hollow_cylinder"
        elif instance_name.startswith("hollow_cylinder"):
            key = "hollow_cylinder"
        elif instance_name.startswith("trapezoid"):
            key = "trapezoid"
        else:
            continue
        plugin_types[key].add(int(model.plugin[instance_id]))

    missing = [name for name, values in plugin_types.items() if not values]
    ambiguous = {
        name: sorted(values) for name, values in plugin_types.items() if len(values) > 1
    }
    if missing or ambiguous:
        raise ValueError(
            "Could not identify unique table SDF plugin types: "
            f"missing={missing}, ambiguous={ambiguous}"
        )
    return SDFPluginTypes(
        hollow_cylinder=next(iter(plugin_types["hollow_cylinder"])),
        half_hollow_cylinder=next(iter(plugin_types["half_hollow_cylinder"])),
        trapezoid=next(iter(plugin_types["trapezoid"])),
    )


def register_mujoco_billiards_sdf(model: mujoco.MjModel) -> SDFPluginTypes:
    """Register Warp distance and analytical-gradient functions for the table SDFs.

    Registration must happen before MJWarp compiles a collision kernel in the
    current process.
    """

    if int(model.opt.sdf_initpoints) < TRAPEZOID_MANIFOLD_CONTACTS:
        raise ValueError(
            "MuJoCo sdf_initpoints must be at least "
            f"{TRAPEZOID_MANIFOLD_CONTACTS} for the trapezoid contact manifold; "
            f"got {int(model.opt.sdf_initpoints)}."
        )

    plugin_types = _plugin_types(model)
    hollow_cylinder_type = plugin_types.hollow_cylinder
    half_hollow_cylinder_type = plugin_types.half_hollow_cylinder
    trapezoid_type = plugin_types.trapezoid

    @wp.func
    def user_sdf(
        point: wp.vec3,
        attributes: vec_pluginattr,
        sdf_type: int,
    ) -> float:
        if sdf_type == wp.static(hollow_cylinder_type):
            return _hollow_cylinder_distance(point, attributes)
        if sdf_type == wp.static(half_hollow_cylinder_type):
            return _half_hollow_cylinder_distance(point, attributes)
        if sdf_type == wp.static(trapezoid_type):
            return _trapezoid_distance(point, attributes)
        return 1.0e6

    @wp.func
    def user_sdf_grad(
        point: wp.vec3,
        attributes: vec_pluginattr,
        sdf_type: int,
    ) -> wp.vec3:
        if sdf_type == wp.static(hollow_cylinder_type):
            return _hollow_cylinder_gradient(point, attributes)
        if sdf_type == wp.static(half_hollow_cylinder_type):
            return _half_hollow_cylinder_gradient(point, attributes)
        if sdf_type == wp.static(trapezoid_type):
            return _trapezoid_gradient(point, attributes)
        return wp.vec3(0.0)

    @wp.func
    def user_sdf_sphere_contact_supported(sdf_type: int) -> bool:
        return sdf_type == wp.static(trapezoid_type)

    @wp.func
    def user_sdf_sphere_contact_count(sdf_type: int) -> int:
        if sdf_type == wp.static(trapezoid_type):
            return TRAPEZOID_MANIFOLD_CONTACTS
        return 0

    @wp.func
    def user_sdf_sphere_contact(
        center: wp.vec3,
        attributes: vec_pluginattr,
        sdf_type: int,
        radius: float,
        contact_index: int,
    ) -> tuple[bool, float, wp.vec3, wp.vec3, float]:
        if sdf_type == wp.static(trapezoid_type):
            return _trapezoid_sphere_contact(
                center,
                attributes,
                radius,
                contact_index,
            )
        return False, 0.0, wp.vec3(0.0), wp.vec3(0.0), 1.0

    mjw._src.collision_sdf.user_sdf = user_sdf
    mjw._src.collision_sdf.user_sdf_grad = user_sdf_grad
    mjw._src.collision_sdf.user_sdf_sphere_contact_supported = (
        user_sdf_sphere_contact_supported
    )
    mjw._src.collision_sdf.user_sdf_sphere_contact_count = (
        user_sdf_sphere_contact_count
    )
    mjw._src.collision_sdf.user_sdf_sphere_contact = user_sdf_sphere_contact
    return plugin_types
