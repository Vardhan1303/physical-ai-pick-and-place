#!/usr/bin/env python
"""
grasp_planner.py — Phase 5: category-agnostic top-down parallel-jaw grasp
planning from a target-only point cloud (geometry.py's output).

Nothing here assumes "this is a box" or "this is a bottle" — the only
inputs are the point cloud's own geometry (a 2D footprint on the table
plane + a height range) and the gripper's physical limits. The footprint's
principal axes (via cv2.minAreaRect, same tool the old raw-MuJoCo
pipeline's shape_utils.grasp_from_mask already used for 2D masks —
generalized here to a 3D-projected footprint) decide the grasp
orientation; nothing about object identity enters the decision.

Output poses (pre-grasp, grasp, lift, place) are in ROBOT-BASE frame,
matching geometry.py's point cloud frame, ready for robot_controller.py to
execute via robosuite's OSC controller. Final gripper closing distance is
NOT fully trusted from this module's width estimate alone — robot_controller.py
closes until contact, per the project's explicit constraint.

Standalone self-test:
    python grasp_planner.py
Chains environment.py -> aruco_prompt.py -> flip_segmenter.py ->
geometry.py -> this module and prints the planned grasp.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class GraspPose:
    pos: np.ndarray     # (3,) robot-base frame
    yaw: float           # radians, rotation about world/robot Z


@dataclass
class GraspPlan:
    pregrasp: GraspPose
    grasp: GraspPose
    lift: GraspPose
    place: GraspPose
    grasp_width: float           # planned finger separation (m) — a starting point, not final
    footprint_corners: np.ndarray  # (4, 2) XY corners of the fitted rect, robot-base frame


@dataclass
class GraspFailure:
    reason: str
    detail: str = ""


def downward_grasp_rotation(yaw: float) -> np.ndarray:
    """3x3 rotation: gripper approach axis points straight down (-Z),
    rotated about Z by `yaw`. Same convention as the old pipeline's
    manipulation.make_downward_R, reimplemented locally so this module
    has no dependency on the raw-MuJoCo pipeline's code."""
    Rx180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return Rz @ Rx180


def compute_footprint(points_xy: np.ndarray):
    """cv2.minAreaRect on the table-plane projection of the point cloud.
    Returns (center(2,), width, height, angle_rad, box_corners(4,2)) where
    width <= height is NOT guaranteed — caller decides which axis to close
    the gripper along."""
    pts32 = points_xy.astype(np.float32).reshape(-1, 1, 2)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(pts32)
    box = cv2.boxPoints(((cx, cy), (w, h), angle_deg))
    return np.array([cx, cy]), float(w), float(h), np.radians(angle_deg), box


def plan_grasp(
    points_robot: np.ndarray,
    place_xy: tuple,
    gripper_max_width: float = 0.08,
    gripper_min_clearance: float = 0.005,
    table_height: Optional[float] = None,
    approach_height: float = 0.15,
    grasp_height_fraction: float = 0.5,
    width_safety_margin: float = 0.01,
    place_height: Optional[float] = None,
):
    """
    points_robot: (N, 3) target point cloud in robot-base frame
        (geometry.py's output).
    place_xy: (x, y) destination in robot-base frame (e.g. above the bin).
    table_height: robot-base-frame Z of the table surface — a scene
        constant (see geometry.py's table_height_hint reasoning), used
        only to estimate how far the object extends below its visible
        top surface (single-camera 2.5D limitation: we only ever see the
        TOP, so the object's true base height is inferred, not measured).
    place_height: robot-base-frame Z to release the object at, if it
        differs from the pick-side grasp height — e.g. the destination
        bin sits ~5.6cm above the table, so reusing the pick grasp_z for
        the place descent drives the gripper INTO the bin's solid
        geometry and the motion never converges (hit exactly this in
        Phase 6 verification). Like table_height, the bin's height is
        fixed scene furniture, not object ground truth. Defaults to the
        pick-side grasp_z if not given.

    Returns (GraspPlan, None) or (None, GraspFailure).
    """
    if len(points_robot) < 5:
        return None, GraspFailure("insufficient_points", f"only {len(points_robot)} points in target cloud")

    xy = points_robot[:, :2]
    z = points_robot[:, 2]
    top_z = float(np.percentile(z, 95))  # robust top, not raw max (avoids single-outlier skew)

    center_xy, w, h, angle_rad, corners = compute_footprint(xy)

    # Close the gripper along the SHORTER side (same rule as the old
    # pipeline's shape_utils.grasp_from_mask) — rotate the yaw 90 deg if
    # `w` (the rect's first reported side) is the LONGER one.
    if w <= h:
        grasp_width_est = w
        yaw = angle_rad
    else:
        grasp_width_est = h
        yaw = angle_rad + np.pi / 2

    # A parallel-jaw gripper's closing line is symmetric under a 180deg
    # rotation about its own approach axis — grasping "from yaw" and
    # "from yaw+180" close along the exact same physical line, just with
    # the two fingers swapped. cv2.minAreaRect's angle can return either
    # representation depending on which corner it happened to start from,
    # so without normalizing, downstream IK sometimes gets asked for a
    # wrist orientation ~180deg from one it could reach easily. Confirmed
    # in-sandbox: yaw=180deg (vs. the equivalent yaw=0deg for the same
    # box) drove the Panda's wrist into a configuration robosuite's OSC
    # controller could not converge to at all (pos error stuck ~0.3m,
    # not a slow-convergence issue — genuinely unreachable/awkward from
    # the arm's starting joint configuration), while yaw=0deg for the
    # identical footprint converged normally. Wrapping into (-90, 90]
    # degrees picks the equivalent representation closer to "no rotation,"
    # which is reachable far more often in practice.
    yaw = ((yaw + np.pi / 2) % np.pi) - np.pi / 2

    grasp_width = grasp_width_est + width_safety_margin
    if grasp_width > gripper_max_width:
        return None, GraspFailure(
            "object_too_wide",
            f"estimated grasp width {grasp_width:.3f}m (+margin) exceeds gripper max {gripper_max_width:.3f}m "
            f"along both footprint axes (w={w:.3f}, h={h:.3f}) — this simple planner only tries the two "
            f"principal axes of the footprint, not arbitrary angles",
        )
    if grasp_width < gripper_min_clearance:
        # Degenerate/too-thin footprint (e.g. FLIP undersegmented to a
        # sliver) — not physically meaningful to grasp at this width.
        return None, GraspFailure(
            "degenerate_footprint",
            f"estimated grasp width {grasp_width:.4f}m is implausibly small — likely an under-segmented mask",
        )

    if table_height is not None:
        object_height_est = max(top_z - table_height, 0.01)
        grasp_z = table_height + grasp_height_fraction * object_height_est
    else:
        # No table-height hint available: fall back to the cloud's own
        # median height (only correct if the cloud spans the object's
        # side, not just its top — documented single-camera limitation).
        grasp_z = float(np.median(z))

    release_z = grasp_z if place_height is None else place_height

    grasp_pos = np.array([center_xy[0], center_xy[1], grasp_z])
    pregrasp_pos = grasp_pos + np.array([0, 0, approach_height])
    lift_pos = pregrasp_pos.copy()
    place_pos = np.array([place_xy[0], place_xy[1], release_z])

    plan = GraspPlan(
        pregrasp=GraspPose(pregrasp_pos, yaw),
        grasp=GraspPose(grasp_pos, yaw),
        lift=GraspPose(lift_pos, yaw),
        place=GraspPose(np.array([place_xy[0], place_xy[1], release_z + approach_height]), yaw),
        grasp_width=grasp_width,
        footprint_corners=corners,
    )
    # place descent point kept separate so robot_controller can descend
    # then release, mirroring the pick side's approach/descend structure.
    plan.__dict__["place_descend"] = GraspPose(place_pos, yaw)
    return plan, None


if __name__ == "__main__":
    import robosuite.macros as macros
    macros.IMAGE_CONVENTION = "opencv"
    from environment import PickPlaceEnv, SIDE_CAMERA_NAME, TARGET_MARKER_ID
    from aruco_prompt import get_target_prompt
    from flip_segmenter import FlipTargetSegmenter
    from geometry import build_target_point_cloud, get_robot_base_transform

    env = PickPlaceEnv(
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=[SIDE_CAMERA_NAME], camera_heights=720, camera_widths=960,
        camera_depths=True, num_distractors=0, seed=0,
    )
    env.reset()
    for _ in range(10):
        env.sim.step()
    rgb, depth = env.get_camera_rgbd()
    K = env.get_camera_intrinsics()
    cam_to_world = env.get_camera_extrinsics()
    base_pos, base_mat = get_robot_base_transform(env)
    table_height_world = env.get_table_height()
    # table height in ROBOT-BASE frame (same transform geometry.py applies to points)
    table_height_robot = float(((np.array([0, 0, table_height_world]) - base_pos) @ base_mat)[2])

    detection, failure = get_target_prompt(rgb, expected_id=TARGET_MARKER_ID)
    if detection is None:
        raise SystemExit(f"[FATAL] aruco_prompt failed: {failure.reason.value}")

    segmenter = FlipTargetSegmenter(model_size="small")
    seg = segmenter.segment_from_prompt(rgb, detection.center_px, marker_side_px=detection.side_length_px)

    result = build_target_point_cloud(seg.mask_full, depth, K, cam_to_world, base_pos, base_mat,
                                       table_height_hint=table_height_world)
    print(f"point cloud size: {len(result.points_robot)}")

    # Place target: robot-base-frame XY roughly matching where
    # environment.py puts the destination bin (world [0.22, 0], offset by
    # table center) — expressed in robot-base frame the same way the point
    # cloud is, just for this standalone test.
    # Fixed scene furniture (not object ground truth) — see plan_grasp's
    # place_height doc and environment.py::get_bin_top_center's own note
    # about a hand-derived version of this arithmetic getting it wrong.
    bin_world = env.get_bin_top_center() + np.array([0, 0, 0.01])  # + small release clearance
    bin_robot = (bin_world - base_pos) @ base_mat

    plan, plan_failure = plan_grasp(result.points_robot, place_xy=(bin_robot[0], bin_robot[1]),
                                     table_height=table_height_robot, place_height=bin_robot[2])
    if plan is None:
        print(f"FAILED: {plan_failure.reason} — {plan_failure.detail}")
    else:
        print(f"grasp_width={plan.grasp_width:.4f}m yaw={np.degrees(plan.grasp.yaw):.1f}deg")
        print(f"pregrasp={plan.pregrasp.pos}")
        print(f"grasp={plan.grasp.pos}")
        print(f"lift={plan.lift.pos}")
        print(f"place_descend={plan.__dict__['place_descend'].pos}")
