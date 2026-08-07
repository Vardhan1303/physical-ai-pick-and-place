#!/usr/bin/env python
"""
environment.py — Phase 1: the robosuite/MuJoCo tabletop scene for the
ArUco-Guided Point-Prompted Object Segmentation thesis project.

Builds a `PickPlaceEnv` (a robosuite `ManipulationEnv` subclass, modeled on
robosuite's own `Lift` environment — see
robosuite/environments/manipulation/lift.py in the installed package) with:
  - a Franka Panda arm on a table (robosuite's TableArena)
  - one target object (an upright CylinderObject — a bottle-like round
    body with an ArUco decal on its curved side, not a flat top face; the
    object's IDENTITY is never used by perception, only by this module for
    scene construction and by evaluation.py for ground truth)
  - optional distractor objects (0 by default — Phase 1 constraint: "one
    target object, no clutter")
  - a destination bin (a fixed visual BoxObject with no free joint)
  - `side_oblique_camera`: a NEW camera added directly to the arena's MJCF
    (robosuite arenas expose `.worldbody` as a live xml.etree Element —
    confirmed by inspecting `TableArena` in the installed robosuite 1.5.2;
    the existing frontview/agentview/birdview/sideview cameras are defined
    the same way, we're just adding one more). Positioned front-left of the
    table, elevated ~40 degrees, looking at the workspace center.
  - deterministic seeding via robosuite's own `seed=` env kwarg (propagates
    to `self.rng`, which `BoxObject`/`UniformRandomSampler` both accept —
    see Lift's usage of `rng=self.rng`)

Ground-truth accessors (`get_ground_truth_state`) are provided ONLY for
evaluation.py. Nothing in the perception/grasp-planning modules
(aruco_prompt.py, flip_segmenter.py, geometry.py, grasp_planner.py) may call
them — see the project's own constraint: "Do not use MuJoCo ground-truth
pose, dimensions, segmentation, or object class as input to the perception
or grasp planner."

Run directly to sanity-check the scene (Phase 1 deliverable — see the
verification steps in the project plan):
    python environment.py
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import mujoco

import robosuite.macros as macros
# robosuite defaults to OpenGL image convention (row 0 = bottom of image),
# which is why a naive render came out vertically flipped relative to what
# OpenCV/ArUco (aruco_prompt.py) and camera_utils' intrinsics (principal
# point measured from the top-left, v increasing downward) both expect.
# "opencv" is a real, documented robosuite macro (robosuite/macros.py,
# IMAGE_CONVENTION_MAPPING in mjcf_utils.py) — not a workaround of our own.
# Must be set before any camera sensor is constructed (i.e. before env
# instantiation), so it's set here at import time.
macros.IMAGE_CONVENTION = "opencv"

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, new_geom
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils import camera_utils

SIDE_CAMERA_NAME = "side_oblique_camera"

# Shared with the perception side (aruco_prompt.py) so both sides agree on
# what dictionary/ID they're looking for. DICT_6X6_250 matches what the
# existing pipeline (segment.py, pick_and_place_flip.py) already uses.
ARUCO_DICT_NAME = "DICT_6X6_250"
TARGET_MARKER_ID = 42  # arbitrary, chosen distinct from the old pipeline's marker IDs (0/1/2)

MARKERS_DIR = Path(__file__).resolve().parent / "markers"


def ensure_marker_png(marker_id: int, dict_name: str = ARUCO_DICT_NAME,
                       marker_px: int = 200, quiet_zone_px: int = 40) -> str:
    """
    Generates (idempotently) a printable ArUco marker PNG with a white
    quiet-zone border — required for reliable cv2.aruco detection, the
    marker pattern alone isn't enough. Returns the absolute file path.
    Cached under markers/ so repeated env construction doesn't regenerate it.
    """
    MARKERS_DIR.mkdir(parents=True, exist_ok=True)
    path = MARKERS_DIR / f"target_marker_id{marker_id}.png"
    if path.exists():
        return str(path)

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    pattern = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px, borderBits=1)
    canvas = np.full(
        (marker_px + 2 * quiet_zone_px, marker_px + 2 * quiet_zone_px), 255, dtype=np.uint8
    )
    canvas[quiet_zone_px:quiet_zone_px + marker_px, quiet_zone_px:quiet_zone_px + marker_px] = pattern
    # MuJoCo's texture loader needs RGB, not single-channel grayscale — a
    # grayscale PNG loaded silently as a flat gray face (no error, no
    # pattern) rather than failing loudly. Hit and fixed in-sandbox: the
    # marker rendered as a uniform gray patch until this 3-channel convert
    # was added.
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(path), canvas_rgb)
    return str(path)


def look_at_quat(cam_pos, target, world_up=(0, 0, 1)):
    """
    Returns a MuJoCo camera quaternion (w,x,y,z) that points the camera's
    view axis (local -Z, per MuJoCo convention) from cam_pos toward target.
    world_up disambiguates roll. Verified against robosuite's own
    frontview/agentview camera quats (same convention, same coordinate
    frame) before use here.
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.asarray(world_up, dtype=float)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    R = np.column_stack([right, up, -forward])  # local -Z = forward
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return quat


def radial_decal_quat(azimuth_rad):
    """
    Returns a MuJoCo quaternion (w,x,y,z) for a flat decal geom sitting
    tangent to a cylinder's curved side at the given azimuth (angle in the
    body's local XY plane, standard math convention: 0 = +X, pi/2 = +Y),
    with the cylinder's axis along local Z.

    Local Z (the decal's texture-correct face — see _add_aruco_decal's
    docstring for why it must be local Z) is pointed radially OUTWARD at
    that azimuth; local Y is pointed +Z (world/body up) so the marker
    renders upright rather than sideways; local X is the remaining tangent
    direction, completing a right-handed frame.
    """
    n = np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0])   # local Z -> outward radial
    up = np.array([0.0, 0.0, 1.0])                                   # local Y -> world up
    tangent = np.cross(up, n)                                        # local X
    tangent = tangent / np.linalg.norm(tangent)
    R = np.column_stack([tangent, up, n])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return quat


class PickPlaceEnv(ManipulationEnv):
    """
    Tabletop pick-and-place scene for point-prompted grasping. Structurally
    mirrors robosuite's built-in `Lift` environment (same base-class calls,
    same `_load_model`/`_setup_references`/`_reset_internal` pattern) but
    adds the side-oblique camera, a destination bin, and optional
    distractors.

    Args (beyond what ManipulationEnv/Lift already document):
        num_distractors (int): how many extra (non-target) objects to
            scatter on the table. 0 for Phase 1's "no clutter" constraint.
        target_object_size (2-tuple): (radius, half_length) of the target
            CylinderObject — a bottle-like upright cylinder, ArUco marker
            on its curved side rather than a flat top face
            placeholder target. Later phases will parameterize the target's
            actual mesh/shape; nothing in this class hard-codes "box" as a
            perception assumption — it's just what Phase 1 needs to exist.
    """

    # XY world-frame offset of side_oblique_camera relative to the table
    # center (Z is handled separately by _add_side_oblique_camera). Shared
    # with _add_aruco_decal so the marker decal's outward-facing azimuth is
    # always computed FROM the camera's actual position, not a hand-copied
    # guess that could silently drift out of sync with the camera.
    _CAMERA_XY_OFFSET = np.array([-0.45, -0.45])
    _CAMERA_XY_OFFSET_3D = np.array([-0.45, -0.45, 0.55])

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        num_distractors=0,
        target_object_size=(0.025, 0.05),
        randomize_object_rotation=False,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera=SIDE_CAMERA_NAME,
        camera_names=None,
        camera_heights=720,
        camera_widths=960,
        camera_depths=True,
        camera_segmentations=None,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        seed=None,
        **kwargs,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.num_distractors = num_distractors
        self.target_object_size = target_object_size
        # The target has a single ArUco decal on one face (see
        # _add_aruco_decal). A full random yaw would rotate that face away
        # from the camera unpredictably, which is fine for Phase 7's
        # occlusion/robustness sweeps (where failure to see the marker is
        # itself a measured outcome) but wrong for Phases 1-6, which need
        # the marker reliably visible while the rest of the pipeline is
        # built and debugged. Default: fixed yaw=0 (decal faces the camera).
        self.randomize_object_rotation = randomize_object_rotation
        self.placement_initializer = None

        if camera_names is None:
            camera_names = [SIDE_CAMERA_NAME]

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise="default",
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            seed=seed,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])
        self._add_side_oblique_camera(mujoco_arena)

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        target_material = CustomMaterial(
            texture="WoodRed", tex_name="target_tex", mat_name="target_mat",
            tex_attrib=tex_attrib, mat_attrib=mat_attrib,
        )
        self.target_object = CylinderObject(
            name="target_object",
            size_min=self.target_object_size,
            size_max=self.target_object_size,
            rgba=[0.8, 0.15, 0.15, 1],
            material=target_material,
            rng=self.rng,
        )
        self._add_aruco_decal(self.target_object)

        self.distractor_objects = []
        distractor_colors = [[0.2, 0.6, 0.8, 1], [0.3, 0.75, 0.35, 1], [0.85, 0.7, 0.2, 1]]
        for i in range(self.num_distractors):
            self.distractor_objects.append(
                BoxObject(
                    name=f"distractor_{i}",
                    size_min=[0.02, 0.02, 0.04],
                    size_max=[0.025, 0.025, 0.05],
                    rgba=distractor_colors[i % len(distractor_colors)],
                    rng=self.rng,
                )
            )

        all_movable = [self.target_object] + self.distractor_objects
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=all_movable,
            x_range=[-0.12, 0.12],
            y_range=[-0.15, 0.15],
            rotation=None if self.randomize_object_rotation else 0,
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
            rng=self.rng,
        )

        # Destination bin: fixed (no freejoint) visual + collision box off
        # to one side of the workspace, sitting ON the table surface.
        self.bin_object = BoxObject(
            name="destination_bin",
            size_min=[0.09, 0.09, 0.015],
            size_max=[0.09, 0.09, 0.015],
            rgba=[0.55, 0.55, 0.6, 0.5],
            joints=None,
        )
        bin_body = self.bin_object.get_obj()
        bin_half_size = np.array([0.09, 0.09, 0.015])
        bin_pos = self.table_offset + np.array([0.22, 0.0, self.table_full_size[2] / 2 + 0.016])
        bin_body.set("pos", " ".join(str(v) for v in bin_pos))
        # Stored for get_bin_top_center() — fixed scene furniture (like a
        # camera mount point), not per-object ground truth, so grasp_planner.py
        # /pipeline.py may legitimately read this instead of re-deriving the
        # bin's placement arithmetic by hand (a real bug hit during Phase 6
        # verification: a hand-copied version of this formula in a self-test
        # accidentally double-counted the table's half-thickness).
        self._bin_pos = bin_pos
        self._bin_half_size = bin_half_size

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=all_movable + [self.bin_object],
        )

    def _add_side_oblique_camera(self, mujoco_arena):
        """
        Adds `side_oblique_camera` directly to the arena's MJCF worldbody
        (an xml.etree Element — the same mechanism robosuite's own
        frontview/agentview/birdview/sideview cameras are defined with).
        Front-left of the table, elevated ~40 degrees above the table
        plane, looking at the workspace center — positioned to see both
        object side surfaces (for bottles) and part of the top surface.
        """
        table_center = self.table_offset
        cam_offset = self._CAMERA_XY_OFFSET_3D
        cam_pos = table_center + cam_offset
        look_target = table_center + np.array([0.0, 0.0, 0.05])
        quat = look_at_quat(cam_pos, look_target)

        camera = ET.SubElement(mujoco_arena.worldbody, "camera")
        camera.set("name", SIDE_CAMERA_NAME)
        camera.set("mode", "fixed")
        camera.set("pos", " ".join(str(v) for v in cam_pos))
        camera.set("quat", " ".join(str(v) for v in quat))
        camera.set("fovy", "45")

    def _add_aruco_decal(self, cyl_obj: CylinderObject):
        """
        Attaches an ArUco marker as a thin flat decal geom tangent to the
        target cylinder's curved side, facing side_oblique_camera.

        The facing azimuth is computed FROM the camera's actual XY offset
        (self._CAMERA_XY_OFFSET), not assumed to be straight along -Y. An
        earlier version hardcoded -Y and the marker rendered as a heavily
        foreshortened, darkly-shaded parallelogram and failed ArUco
        detection outright — the camera sits at an equal diagonal offset
        in -X and -Y ([-0.45, -0.45]), roughly 45 degrees away from -Y, so
        a decal facing pure -Y was significantly off-axis from the
        camera's real viewing direction. Confirmed by direct render +
        cv2.aruco detection pass after the fix (see THESIS_PLAN.md).

        Mechanism: `cyl_obj._obj` is the real ET.Element robosuite's own
        `PrimitiveObject._get_object_subtree_` builds for this object (see
        robosuite/models/objects/generated_objects.py) — appending a geom
        to it directly, before the object is merged into ManipulationTask,
        is the same thing robosuite's own per-instance naming-prefix pass
        (`add_prefix`) expects to walk over; it isn't a private workaround,
        it's the one mutation point robosuite exposes before assembly.

        Why a flat decal instead of a texture wrapped around the cylinder:
        the earlier box version found MuJoCo's 2D-texture mapping on a box
        geom only resolves correctly on the face normal to that geom's OWN
        local Z axis (side faces render flat gray, no pattern). The decal
        here is its own independent thin box geom (not the cylinder's own
        surface texture), so the same fix applies directly: give the decal
        a `quat` that points ITS local Z outward, radially away from the
        cylinder's axis, and the previously-proven "face normal to local Z
        renders correctly" case still holds — just aimed sideways instead
        of straight up. This specific radial placement hasn't been
        rendered before (the box case only ever needed straight-up), so it
        was verified by an actual render + cv2.aruco detection pass, not
        assumed from the math alone (see THESIS_PLAN.md).
        """
        marker_path = ensure_marker_png(TARGET_MARKER_ID, ARUCO_DICT_NAME)
        marker_material = CustomMaterial(
            texture=marker_path,
            tex_name="target_marker_tex",
            mat_name="target_marker_mat",
            tex_attrib={"type": "2d"},
            mat_attrib={"specular": "0", "shininess": "0"},
        )
        cyl_obj.append_material(marker_material)
        # append_material() immediately re-runs robosuite's own add_prefix()
        # over cyl_obj.asset (see objects.py::append_material), so the
        # material we just added is now named f"{naming_prefix}target_marker_mat",
        # not the bare "target_marker_mat" we passed to CustomMaterial. That
        # prefixing pass only touches .asset, not .obj — and the ONE pass
        # that does prefix .obj's geoms already ran earlier, during
        # CylinderObject.__init__ (_get_object_properties), before this
        # decal geom existed. So the decal must reference the
        # already-prefixed name directly, or MuJoCo's XML compiler fails
        # with "material ... not found" (hit and fixed on the box version).
        prefixed_mat_name = cyl_obj.naming_prefix + "target_marker_mat"

        radius, half_length = cyl_obj.size
        # Decal in-plane half-extents: width along the tangent (must stay
        # well under the diameter so the flat patch doesn't visibly poke
        # through the curve at its edges), height along the cylinder's
        # axis (kept short of the full length so it reads as a label band,
        # not the whole body).
        decal_half_w = 0.5 * radius
        decal_half_h = 0.4 * half_length

        # Azimuth pointing from the object toward the camera's actual XY
        # offset (see docstring above for why this isn't hardcoded to -Y).
        azimuth = float(np.arctan2(self._CAMERA_XY_OFFSET[1], self._CAMERA_XY_OFFSET[0]))
        decal_quat = radial_decal_quat(azimuth)
        decal_r = radius + 0.0006  # flush against the curved surface, tiny clearance
        decal_pos = [decal_r * np.cos(azimuth), decal_r * np.sin(azimuth), 0]

        decal = new_geom(
            name="target_marker_decal",
            type="box",
            size=[decal_half_w, decal_half_h, 0.0005],
            pos=decal_pos,
            quat=decal_quat,
            group=1,
            material=prefixed_mat_name,
            contype=0,
            conaffinity=0,
        )
        cyl_obj._obj.append(decal)

    def _setup_references(self):
        super()._setup_references()
        self.target_object_body_id = self.sim.model.body_name2id(self.target_object.root_body)
        self.distractor_body_ids = [
            self.sim.model.body_name2id(obj.root_body) for obj in self.distractor_objects
        ]

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )

    def reward(self, action=None):
        # Placeholder — Phase 1 is scene/camera only, no task reward yet.
        return 0.0

    # ------------------------------------------------------------------
    # Perception-facing accessors (camera math only — no object ground
    # truth). All backed directly by robosuite.utils.camera_utils, not
    # reimplemented, per the project's "do not invent APIs" instruction.
    # ------------------------------------------------------------------
    def get_camera_rgbd(self, camera_name=SIDE_CAMERA_NAME):
        """Returns (rgb uint8 HxWx3, depth_meters float32 HxW)."""
        cam_id = [i for i, n in enumerate(self.camera_names) if n == camera_name][0]
        obs = self._get_observations(force_update=True)
        rgb = obs[f"{camera_name}_image"]
        depth_norm = obs[f"{camera_name}_depth"]
        depth_m = camera_utils.get_real_depth_map(self.sim, depth_norm)
        return rgb, depth_m.squeeze(-1) if depth_m.ndim == 3 else depth_m

    def get_camera_intrinsics(self, camera_name=SIDE_CAMERA_NAME):
        idx = self.camera_names.index(camera_name)
        h, w = self.camera_heights[idx], self.camera_widths[idx]
        return camera_utils.get_camera_intrinsic_matrix(self.sim, camera_name, h, w)

    def get_camera_extrinsics(self, camera_name=SIDE_CAMERA_NAME):
        """4x4 camera-to-world pose (robot-base frame == world frame here,
        since the Panda's base is welded into the world at a known offset)."""
        return camera_utils.get_camera_extrinsic_matrix(self.sim, camera_name)

    def get_table_height(self) -> float:
        """World-frame Z of the table's top surface — fixed scene geometry
        (same category as a camera mount point), not object ground truth.
        For use by geometry.py/grasp_planner.py as a plane-fit/height hint."""
        return float(self.table_offset[2] + self.table_full_size[2] / 2)

    def get_bin_top_center(self) -> np.ndarray:
        """World-frame XYZ of the destination bin's top surface center —
        fixed scene furniture, legitimate for grasp_planner.py/pipeline.py
        to read directly instead of re-deriving the bin's placement
        arithmetic by hand (a real bug: a hand-copied version of this
        formula in an early self-test double-counted the table's
        half-thickness, producing a release height ~2cm off and causing
        the place-descend motion to fail to converge against the bin's
        actual solid geometry)."""
        return self._bin_pos + np.array([0.0, 0.0, self._bin_half_size[2]])

    # ------------------------------------------------------------------
    # EVALUATION-ONLY ground truth. Do not call from aruco_prompt.py,
    # flip_segmenter.py, geometry.py, or grasp_planner.py.
    # ------------------------------------------------------------------
    def get_ground_truth_state(self):
        target_pos = np.array(self.sim.data.body_xpos[self.target_object_body_id])
        target_quat = np.array(self.sim.data.body_xquat[self.target_object_body_id])
        return {
            "target_pos": target_pos,
            "target_quat": target_quat,
            "target_size": np.array(self.target_object_size),
        }

    def get_ground_truth_segmentation(self, camera_name=SIDE_CAMERA_NAME):
        """2-channel (geom_type, geom_id) segmentation — evaluation.py only."""
        idx = self.camera_names.index(camera_name)
        h, w = self.camera_heights[idx], self.camera_widths[idx]
        return camera_utils.get_camera_segmentation(self.sim, camera_name, h, w)


if __name__ == "__main__":
    import os
    from PIL import Image

    out_dir = "phase1_output"
    os.makedirs(out_dir, exist_ok=True)

    env = PickPlaceEnv(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[SIDE_CAMERA_NAME],
        camera_heights=720,
        camera_widths=960,
        camera_depths=True,
        num_distractors=0,
        seed=0,
    )
    env.reset()
    for _ in range(10):
        env.sim.step()

    rgb, depth = env.get_camera_rgbd()
    K = env.get_camera_intrinsics()
    Rt = env.get_camera_extrinsics()

    Image.fromarray(rgb).save(os.path.join(out_dir, "side_oblique_rgb.png"))
    np.save(os.path.join(out_dir, "side_oblique_depth.npy"), depth)

    print("RGB shape:", rgb.shape, "dtype:", rgb.dtype)
    print("Depth shape:", depth.shape, "min/max (m):", float(depth.min()), float(depth.max()))
    print("Intrinsics K:\n", K)
    print("Extrinsics (camera->world):\n", Rt)
    print(f"Saved {out_dir}/side_oblique_rgb.png and side_oblique_depth.npy")

    env.close()
