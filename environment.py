#!/usr/bin/env python
"""
environment.py — Phase 1: the robosuite/MuJoCo tabletop scene for the
ArUco-Guided Point-Prompted Object Segmentation thesis project.

Builds a `PickPlaceEnv` (a robosuite `ManipulationEnv` subclass, modeled on
robosuite's own `Lift` environment — see
robosuite/environments/manipulation/lift.py in the installed package) with:
  - a Franka Panda arm on a table (robosuite's TableArena)
  - one target object (BoxObject placeholder for Phase 1 — later phases
    swap in bottle/can/cup meshes; the object's IDENTITY is never used by
    perception, only by this module for scene construction and by
    evaluation.py for ground truth)
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
import xml.etree.ElementTree as ET

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
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils import camera_utils

SIDE_CAMERA_NAME = "side_oblique_camera"


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
        target_object_size (3-tuple): half-extents of the Phase 1 BoxObject
            placeholder target. Later phases will parameterize the target's
            actual mesh/shape; nothing in this class hard-codes "box" as a
            perception assumption — it's just what Phase 1 needs to exist.
    """

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        num_distractors=0,
        target_object_size=(0.025, 0.025, 0.05),
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera=SIDE_CAMERA_NAME,
        camera_names=None,
        camera_heights=480,
        camera_widths=640,
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
        self.target_object = BoxObject(
            name="target_object",
            size_min=self.target_object_size,
            size_max=self.target_object_size,
            rgba=[0.8, 0.15, 0.15, 1],
            material=target_material,
            rng=self.rng,
        )

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
            rotation=None,
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
        bin_pos = self.table_offset + np.array([0.22, 0.0, self.table_full_size[2] / 2 + 0.016])
        bin_body.set("pos", " ".join(str(v) for v in bin_pos))

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
        cam_offset = np.array([-0.45, -0.45, 0.55])
        cam_pos = table_center + cam_offset
        look_target = table_center + np.array([0.0, 0.0, 0.05])
        quat = look_at_quat(cam_pos, look_target)

        camera = ET.SubElement(mujoco_arena.worldbody, "camera")
        camera.set("name", SIDE_CAMERA_NAME)
        camera.set("mode", "fixed")
        camera.set("pos", " ".join(str(v) for v in cam_pos))
        camera.set("quat", " ".join(str(v) for v in quat))
        camera.set("fovy", "45")

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
        camera_heights=480,
        camera_widths=640,
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
