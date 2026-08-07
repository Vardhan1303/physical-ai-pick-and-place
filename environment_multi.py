#!/usr/bin/env python
"""
environment_multi.py — Phases 2-4: multi-object, class-agnostic,
marker-guided tabletop scene.

Adds cylinder / box / triangular prism / sphere object families to the
SAME scene, each carrying its own unique ArUco marker on a camera-facing
surface, on top of environment.py's PickPlaceEnv (arena, side-oblique
camera, destination bin, controller conventions — all reused unmodified).
The only extension point touched is PickPlaceEnv._build_movable_objects
(factored out for exactly this purpose — see its docstring), plus the
generic marker_id -> body_id / marker_id -> marker_size accessors already
added to the base class.

Marker ID convention (per project spec):
    cylinder = 0
    box      = 1
    prism    = 2
    sphere   = 3

Design notes on marker placement, since none of this is guesswork:
  - cylinder: identical mechanism to Phase 1 (radial decal tangent to the
    curved surface, azimuth computed from the camera's actual XY offset —
    see environment.py::_add_aruco_decal). Continuous azimuth alignment is
    possible because the surface is curved.
  - triangular prism: the prism's own cross-section is CONSTRUCTED (as an
    inline convex-hull mesh — 6 vertices, two triangular caps, MuJoCo's
    compiler computes the convex hull automatically when no <face> list is
    given) so that one of its three rectangular side faces exactly faces
    the shared camera azimuth by design, not by snapping to a fixed
    orientation. See _prism_vertices below for the vertex placement math.
  - box: a box's faces are axis-aligned in its own local frame, and with
    yaw=0 (the baseline/marker-visible default — see
    PickPlaceEnv.randomize_object_rotation) that local frame equals the
    world frame, so the marker must snap to whichever of the box's 4
    vertical faces (+/-X, +/-Y) most closely faces the camera azimuth,
    rather than continuously tracking it the way the cylinder/prism can.
    Documented approximation, not a bug: for this project's fixed camera
    azimuth (~155 degrees) the nearest face is within ~25 degrees of exact
    alignment, well within what a ~40-degree-elevation camera can still
    read a marker at.
  - sphere: identical radial-tangent-patch mechanism to the cylinder
    (a sphere is locally just a curved surface, same math), but as a
    single small tangent PATCH rather than a band running the object's
    full height, since a sphere has no "vertical extent" to band around.
    Documented per the project's own instruction: "a perfectly curved
    sphere cannot physically contain a perfectly planar marker without
    such a patch."

Run directly for a quick scene self-test (renders one frame per object,
detects all 4 markers):
    python environment_multi.py
"""
import xml.etree.ElementTree as ET

import numpy as np

from robosuite.models.objects import BallObject, BoxObject, CylinderObject, PrimitiveObject
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, new_element
from robosuite.utils.placement_samplers import UniformRandomSampler

from environment import (
    ARUCO_DICT_NAME,
    MARKER_PATTERN_PX,
    MARKER_QUIET_ZONE_PX,
    PickPlaceEnv,
    SIDE_CAMERA_NAME,
    ensure_marker_png,
    radial_decal_quat,
)

MARKER_ID_CYLINDER = 0
MARKER_ID_BOX = 1
MARKER_ID_PRISM = 2
MARKER_ID_SPHERE = 3
MARKER_ID_BY_SHAPE = {
    "cylinder": MARKER_ID_CYLINDER,
    "box": MARKER_ID_BOX,
    "prism": MARKER_ID_PRISM,
    "sphere": MARKER_ID_SPHERE,
}
SHAPE_BY_MARKER_ID = {v: k for k, v in MARKER_ID_BY_SHAPE.items()}

# (rgba) baseline color per shape family — randomized per-episode around
# these (see _random_rgba) so "randomize color" doesn't produce colors so
# extreme they stop looking like plausible tabletop objects.
_BASE_COLORS = {
    "cylinder": np.array([0.8, 0.15, 0.15]),
    "box": np.array([0.2, 0.55, 0.85]),
    "prism": np.array([0.9, 0.65, 0.15]),
    "sphere": np.array([0.35, 0.75, 0.35]),
}


def _prism_vertices(circumradius: float, half_height: float, front_azimuth: float) -> np.ndarray:
    """
    6 vertices (2 triangular caps) of a triangular prism whose axis is
    local Z, sized to fit in a circle of radius `circumradius`, with one
    rectangular side face's outward normal pointing EXACTLY along
    `front_azimuth` (radians, standard math convention).

    For an equilateral triangle inscribed in a circle of radius R with
    vertices at angles (az-60, az+60, az+180), the edge between the
    az-60 and az+60 vertices has its outward normal (away from the third,
    az+180 vertex) pointing exactly along `az` — elementary trig (see
    module docstring), verified by direct computation, not assumed.
    """
    verts = []
    for z in (-half_height, half_height):
        for offset_deg in (-60.0, 60.0, 180.0):
            ang = front_azimuth + np.radians(offset_deg)
            verts.append((circumradius * np.cos(ang), circumradius * np.sin(ang), z))
    return np.array(verts, dtype=float)


class PrismObject(PrimitiveObject):
    """
    A triangular-prism object — the one shape family with no native
    robosuite primitive. Built as a raw inline convex-hull mesh asset
    (MuJoCo computes the hull automatically from a vertex soup when no
    <face> list is supplied — standard, documented MuJoCo compiler
    behavior, not a hack) rather than an external .stl/.obj file, so the
    whole object stays self-contained the same way the existing
    PrimitiveObject family (Box/Cylinder/Ball) is.

    size = (circumradius, half_height, front_azimuth_rad) — 3 values, to
    fit PrimitiveObject's existing size-as-a-flat-list convention, even
    though front_azimuth isn't really a "size". Kept together so
    set_scale-style code that reads self.size stays consistent.
    """

    def __init__(self, name, circumradius, half_height, front_azimuth, rgba=None,
                 density=None, friction=None, solref=None, solimp=None,
                 material=None, joints="default", obj_type="all",
                 duplicate_collision_geoms=True):
        super().__init__(
            name=name,
            size=[circumradius, half_height, front_azimuth],
            rgba=rgba, density=density, friction=friction,
            solref=solref, solimp=solimp, material=material,
            joints=joints, obj_type=obj_type,
            duplicate_collision_geoms=duplicate_collision_geoms,
        )
        verts = _prism_vertices(circumradius, half_height, front_azimuth)
        # Manually prefixed (self.naming_prefix is already available at
        # this point — set at the very start of PrimitiveObject.__init__)
        # rather than relying on add_prefix's automatic pass, exactly the
        # same trick environment.py::_add_aruco_decal already uses for its
        # marker material, since this mesh is appended to self.asset
        # directly rather than through append_material (which only knows
        # about texture/material elements, not meshes).
        self._mesh_name = self.naming_prefix + "prism_mesh"
        mesh_el = new_element(tag="mesh", name=self._mesh_name, vertex=array_to_string(verts.flatten()))
        self.asset.append(mesh_el)

        # Geometry facts needed by the marker-decal placement code
        # (environment_multi.py's own _add_face_decal), same category as
        # CylinderObject exposing self.size[0]/[1] for its own decal code.
        self.circumradius = circumradius
        self.half_height = half_height
        self.front_azimuth = front_azimuth
        # Apothem of the front face (distance from axis to the face
        # plane) — see module docstring's equilateral-triangle trig.
        self.front_apothem = 0.5 * circumradius
        self.front_face_half_width = 0.5 * (circumradius * np.sqrt(3.0))

    def sanity_check(self):
        assert len(self.size) == 3, "prism size should be (circumradius, half_height, front_azimuth)"

    def _get_object_subtree(self):
        obj = new_element(tag="body", name="main")
        common = {"mesh": self._mesh_name, "pos": array_to_string([0, 0, 0])}
        if self.obj_type in {"collision", "all"}:
            col = dict(common)
            col.update(self.get_collision_attrib_template())
            col["name"] = "g0"
            col["type"] = "mesh"
            col["density"] = str(self.density)
            col["friction"] = array_to_string(self.friction)
            col["solref"] = array_to_string(self.solref)
            col["solimp"] = array_to_string(self.solimp)
            obj.append(new_element(tag="geom", **col))
        if self.obj_type in {"visual", "all"}:
            vis = dict(common)
            vis.update(self.get_visual_attrib_template())
            vis["name"] = "g0_vis"
            vis["type"] = "mesh"
            if self.material == "default":
                vis["rgba"] = "0.5 0.5 0.5 1"
                vis["material"] = "mat"
            elif self.material is not None:
                vis["material"] = self.material.mat_attrib["name"]
            else:
                vis["rgba"] = array_to_string(self.rgba)
            obj.append(new_element(tag="geom", **vis))
        for joint_spec in self.joint_specs:
            obj.append(new_element(tag="joint", **joint_spec))
        site_attr = self.get_site_attrib_template()
        site_attr["name"] = "default_site"
        obj.append(new_element(tag="site", **site_attr))
        return obj

    @property
    def bottom_offset(self):
        return np.array([0, 0, -self.half_height])

    @property
    def top_offset(self):
        return np.array([0, 0, self.half_height])

    @property
    def horizontal_radius(self):
        return self.circumradius

    def get_bounding_box_half_size(self):
        return np.array([self.circumradius, self.circumradius, self.half_height])


class MultiObjectPickPlaceEnv(PickPlaceEnv):
    """
    Same scene/camera/bin/controller conventions as PickPlaceEnv, but
    _build_movable_objects() is overridden to populate the table with
    however many of {cylinder, box, prism, sphere} are requested, each
    with its own unique ArUco marker (MARKER_ID_BY_SHAPE).

    Args (beyond PickPlaceEnv's):
        object_shapes (tuple of str): which shape families to include,
            from {"cylinder","box","prism","sphere"}. Default: all four
            (Phase 4's "all shapes on one table" scene). Pass a 1-tuple
            for Phase 2/3-style single-family testing.
        randomize_scale (bool): if True, each shape's size is drawn
            uniformly between a small per-shape min/max range (via the
            SAME rng-seeded mechanism robosuite's own PrimitiveObject
            family already uses for size_min/size_max — nothing
            reimplemented). If False (default), every shape uses a fixed
            baseline size, useful for repeatable single-object debugging
            (mirrors PickPlaceEnv's own target_object_size being fixed by
            default).
        randomize_color (bool): if True, each shape's rgba is jittered
            around its baseline color (_BASE_COLORS) using self.rng.
    """

    def __init__(self, object_shapes=("cylinder", "box", "prism", "sphere"),
                 randomize_scale=False, randomize_color=False, **kwargs):
        assert all(s in MARKER_ID_BY_SHAPE for s in object_shapes), \
            f"object_shapes must be a subset of {list(MARKER_ID_BY_SHAPE)}, got {object_shapes}"
        assert len(object_shapes) == len(set(object_shapes)), "object_shapes must not repeat a shape"
        self.object_shapes = tuple(object_shapes)
        self.randomize_scale = randomize_scale
        self.randomize_color = randomize_color
        super().__init__(**kwargs)

    def _random_rgba(self, shape: str) -> list:
        base = _BASE_COLORS[shape]
        if not self.randomize_color:
            return [*base, 1.0]
        jitter = self.rng.uniform(-0.2, 0.2, size=3) if self.rng is not None else np.random.uniform(-0.2, 0.2, 3)
        rgba = np.clip(base + jitter, 0.05, 0.95)
        return [*rgba.tolist(), 1.0]

    def _scale_factor(self) -> float:
        if not self.randomize_scale:
            return 1.0
        return float(self.rng.uniform(0.85, 1.15)) if self.rng is not None else float(np.random.uniform(0.85, 1.15))

    # ------------------------------------------------------------------
    def _build_movable_objects(self):
        camera_azimuth = float(np.arctan2(self._CAMERA_XY_OFFSET[1], self._CAMERA_XY_OFFSET[0]))

        self._marker_id_to_object = {}
        self._marker_id_to_marker_size = {}
        self.shape_objects = {}
        movable = []

        for shape in self.object_shapes:
            marker_id = MARKER_ID_BY_SHAPE[shape]
            scale = self._scale_factor()
            rgba = self._random_rgba(shape)
            material = CustomMaterial(
                texture="WoodRed" if shape != "cylinder" else "WoodRed",
                tex_name=f"{shape}_tex", mat_name=f"{shape}_mat",
                tex_attrib={"type": "cube"},
                mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
            )

            if shape == "cylinder":
                radius, half_length = 0.025 * scale, 0.09 * scale
                obj = CylinderObject(name="obj_cylinder", size=(radius, half_length),
                                      rgba=rgba, material=material, rng=self.rng)
                self._add_cylinder_decal(obj, marker_id, camera_azimuth)
            elif shape == "box":
                hx, hy, hz = 0.035 * scale, 0.035 * scale, 0.05 * scale
                obj = BoxObject(name="obj_box", size=(hx, hy, hz),
                                 rgba=rgba, material=material, rng=self.rng)
                self._add_box_decal(obj, marker_id, camera_azimuth)
            elif shape == "prism":
                circumradius, half_height = 0.05 * scale, 0.06 * scale
                obj = PrismObject(name="obj_prism", circumradius=circumradius, half_height=half_height,
                                   front_azimuth=camera_azimuth, rgba=rgba, material=material)
                self._add_prism_decal(obj, marker_id)
            elif shape == "sphere":
                radius = 0.045 * scale
                obj = BallObject(name="obj_sphere", size=(radius,), rgba=rgba, material=material)
                self._add_sphere_decal(obj, marker_id, camera_azimuth)
            else:
                raise ValueError(f"unknown shape {shape!r}")

            self.shape_objects[shape] = obj
            self._marker_id_to_object[marker_id] = obj
            movable.append(obj)

        # Backward-compat single-target aliases: aruco_prompt/grasp_planner
        # self-tests and flip_segmenter.py's own __main__ still reference
        # env.target_object / TARGET_MARKER_ID directly — point them at
        # whichever shape owns MARKER_ID_CYLINDER == environment.py's
        # TARGET_MARKER_ID space is disjoint (42 vs 0-3) so this is purely
        # a convenience alias, not required for multi_object_pipeline.py.
        self.target_object = movable[0]
        self.distractor_objects = []

        return movable

    # ------------------------------------------------------------------
    # Per-shape marker decal placement
    # ------------------------------------------------------------------
    def _prefixed_marker_material(self, obj, marker_id):
        marker_path = ensure_marker_png(marker_id, ARUCO_DICT_NAME)
        marker_material = CustomMaterial(
            texture=marker_path, tex_name=f"marker{marker_id}_tex", mat_name=f"marker{marker_id}_mat",
            tex_attrib={"type": "2d"}, mat_attrib={"specular": "0", "shininess": "0"},
        )
        obj.append_material(marker_material)
        # append_material() re-runs add_prefix over obj.asset immediately
        # (see environment.py::_add_aruco_decal for the identical
        # reasoning) — the name we must reference from the decal geom is
        # the PREFIXED one.
        return obj.naming_prefix + f"marker{marker_id}_mat"

    def _pattern_fraction(self):
        return MARKER_PATTERN_PX / (MARKER_PATTERN_PX + 2 * MARKER_QUIET_ZONE_PX)

    def _add_cylinder_decal(self, cyl_obj, marker_id, camera_azimuth):
        prefixed_mat = self._prefixed_marker_material(cyl_obj, marker_id)
        radius, half_length = cyl_obj.size
        decal_half = min(0.5 * radius, 0.35 * half_length)
        self._marker_id_to_marker_size_pending = 2 * decal_half * self._pattern_fraction()
        decal_quat = radial_decal_quat(camera_azimuth)
        decal_r = radius + 0.0006
        target_height_above_base = half_length
        height_above_base = float(np.clip(target_height_above_base, decal_half + 0.005, 2 * half_length - decal_half - 0.005))
        decal_local_z = height_above_base - half_length
        decal_pos = [decal_r * np.cos(camera_azimuth), decal_r * np.sin(camera_azimuth), decal_local_z]
        cyl_obj._obj.append(new_element(
            tag="geom", name="target_marker_decal", type="box",
            size=array_to_string([decal_half, decal_half, 0.0005]),
            pos=array_to_string(decal_pos), quat=array_to_string(decal_quat),
            group="1", material=prefixed_mat, contype="0", conaffinity="0",
        ))
        self._marker_id_to_marker_size[marker_id] = self._marker_id_to_marker_size_pending

    def _add_sphere_decal(self, ball_obj, marker_id, camera_azimuth):
        prefixed_mat = self._prefixed_marker_material(ball_obj, marker_id)
        radius = ball_obj.size[0]
        # Small tangent patch (not a full band — see module docstring):
        # sized conservatively relative to the sphere so it reads as a
        # local label, not something that visibly floats off the curve.
        decal_half = 0.35 * radius
        marker_size = 2 * decal_half * self._pattern_fraction()
        decal_quat = radial_decal_quat(camera_azimuth)
        decal_r = radius + 0.0006
        decal_pos = [decal_r * np.cos(camera_azimuth), decal_r * np.sin(camera_azimuth), 0.0]
        ball_obj._obj.append(new_element(
            tag="geom", name="target_marker_decal", type="box",
            size=array_to_string([decal_half, decal_half, 0.0005]),
            pos=array_to_string(decal_pos), quat=array_to_string(decal_quat),
            group="1", material=prefixed_mat, contype="0", conaffinity="0",
        ))
        self._marker_id_to_marker_size[marker_id] = marker_size

    def _add_box_decal(self, box_obj, marker_id, camera_azimuth):
        prefixed_mat = self._prefixed_marker_material(box_obj, marker_id)
        hx, hy, hz = box_obj.size
        # Snap to the nearest of the 4 vertical faces — see module
        # docstring for why a box (unlike the cylinder/prism/sphere) can't
        # continuously track the camera azimuth.
        cos_a, sin_a = np.cos(camera_azimuth), np.sin(camera_azimuth)
        if abs(cos_a) >= abs(sin_a):
            face_normal = np.array([np.sign(cos_a), 0.0, 0.0])
            half_extent_along_normal = hx
            face_width_half = hy
        else:
            face_normal = np.array([0.0, np.sign(sin_a), 0.0])
            half_extent_along_normal = hy
            face_width_half = hx
        face_azimuth = float(np.arctan2(face_normal[1], face_normal[0]))
        decal_half = min(0.6 * face_width_half, 0.6 * hz)
        marker_size = 2 * decal_half * self._pattern_fraction()
        decal_quat = radial_decal_quat(face_azimuth)
        decal_r = half_extent_along_normal + 0.0006
        decal_pos = [decal_r * np.cos(face_azimuth), decal_r * np.sin(face_azimuth), 0.0]
        box_obj._obj.append(new_element(
            tag="geom", name="target_marker_decal", type="box",
            size=array_to_string([decal_half, decal_half, 0.0005]),
            pos=array_to_string(decal_pos), quat=array_to_string(decal_quat),
            group="1", material=prefixed_mat, contype="0", conaffinity="0",
        ))
        self._marker_id_to_marker_size[marker_id] = marker_size

    def _add_prism_decal(self, prism_obj, marker_id):
        prefixed_mat = self._prefixed_marker_material(prism_obj, marker_id)
        az = prism_obj.front_azimuth
        decal_half = min(0.55 * prism_obj.front_face_half_width, 0.6 * prism_obj.half_height)
        marker_size = 2 * decal_half * self._pattern_fraction()
        decal_quat = radial_decal_quat(az)
        decal_r = prism_obj.front_apothem + 0.0006
        decal_pos = [decal_r * np.cos(az), decal_r * np.sin(az), 0.0]
        prism_obj._obj.append(new_element(
            tag="geom", name="target_marker_decal", type="box",
            size=array_to_string([decal_half, decal_half, 0.0005]),
            pos=array_to_string(decal_pos), quat=array_to_string(decal_quat),
            group="1", material=prefixed_mat, contype="0", conaffinity="0",
        ))
        self._marker_id_to_marker_size[marker_id] = marker_size

    # ------------------------------------------------------------------
    def _load_model(self):
        # Widen the placement range vs PickPlaceEnv's single-object default
        # (spec: "sufficient spacing for side grasping", "no overlaps") —
        # done by overriding _load_model just enough to substitute a wider
        # sampler; everything else is identical to the parent, achieved by
        # calling it and then re-pointing self.placement_initializer.
        super()._load_model()
        all_movable = [self.shape_objects[s] for s in self.object_shapes]
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=all_movable,
            x_range=[-0.18, 0.02],
            y_range=[-0.22, 0.22],
            rotation=None if self.randomize_object_rotation else 0,
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
            rng=self.rng,
        )


if __name__ == "__main__":
    import os
    import robosuite.macros as macros
    macros.IMAGE_CONVENTION = "opencv"
    from PIL import Image
    from aruco_prompt import get_target_prompt

    out_dir = "phase2_output"
    os.makedirs(out_dir, exist_ok=True)

    env = MultiObjectPickPlaceEnv(
        object_shapes=("cylinder", "box", "prism", "sphere"),
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=[SIDE_CAMERA_NAME], camera_heights=720, camera_widths=960,
        camera_depths=True, seed=0,
    )
    env.reset()
    for _ in range(10):
        env.sim.step()

    rgb, depth = env.get_camera_rgbd()
    Image.fromarray(rgb).save(os.path.join(out_dir, "multi_object_rgb.png"))
    print("RGB shape:", rgb.shape, "available marker ids:", env.get_available_marker_ids())

    for marker_id in env.get_available_marker_ids():
        detection, failure = get_target_prompt(rgb, expected_id=marker_id)
        if detection is None:
            print(f"marker {marker_id} ({SHAPE_BY_MARKER_ID[marker_id]}): NOT DETECTED — {failure.reason.value}")
        else:
            print(f"marker {marker_id} ({SHAPE_BY_MARKER_ID[marker_id]}): center={detection.center_px} "
                  f"side_px={detection.side_length_px:.1f}")

    env.close()
    print(f"Saved {out_dir}/multi_object_rgb.png")
