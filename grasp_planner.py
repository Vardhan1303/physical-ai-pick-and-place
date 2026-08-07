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


@dataclass
class SideGraspPose:
    pos: np.ndarray   # (3,) robot-base frame
    R: np.ndarray     # (3,3) robot-base frame rotation matrix (full orientation,
                       # not just a yaw — a horizontal approach isn't expressible
                       # as a single Z-rotation of the top-down convention)


@dataclass
class SideGraspPlan:
    safe_high: SideGraspPose   # high waypoint above the table before reconfiguring
    pregrasp: SideGraspPose    # outside the object, along the marker's outward normal
    grasp: SideGraspPose        # horizontal approach target, mid-height on the visible cloud
    retreat: SideGraspPose      # horizontal retreat, same height as grasp, back along outward normal
    lift: SideGraspPose         # retreat position, lifted vertically
    transport: SideGraspPose    # above the destination, at lift height
    place_descend: SideGraspPose  # lowered to release height above the destination
    grasp_width: float            # planned finger separation (m) — a starting point, not final
    approach_direction: np.ndarray   # (3,) unit, robot-base frame, direction of travel toward the object
    closing_direction: np.ndarray    # (3,) unit, robot-base frame, finger-separation axis
    up_direction: np.ndarray          # (3,) unit, robot-base frame, table-up axis used to derive the above
    marker_pos: np.ndarray             # (3,) the selected marker's own position, robot-base frame
    marker_outward_normal: np.ndarray   # (3,) the selected marker's own outward normal, robot-base frame,
                                          # projected/normalized onto the table plane


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


def axes_to_gripper_rotation(approach_direction: np.ndarray, closing_direction: np.ndarray) -> np.ndarray:
    """
    Builds a robot-base-frame gripper rotation matrix from two physically
    meaningful axes, instead of a fixed/hard-coded Euler angle.

    Axis convention verified against the EXISTING top-down grasp math
    (downward_grasp_rotation above) rather than assumed: expanding
    `Rz(yaw) @ Rx180` by hand shows that in whatever final orientation the
    gripper is commanded to, its own local Z axis always ends up equal to
    the direction of travel/approach, and its local X axis always ends up
    equal to the direction the fingers close along (confirmed numerically
    too — see THESIS_PLAN.md's Phase-1-extension notes). Reusing that same
    convention here means side-grasp poses compose correctly with the
    EXISTING move_to_pose/OSC controller code in robot_controller.py
    without any changes there.

    local Z = approach_direction (tool axis, direction of travel toward
        the object)
    local X = closing_direction (finger-separation axis)
    local Y = Z x X (completes a right-handed frame — not necessarily
        "up"; for a horizontal approach, Y ends up horizontal too, the
        same way X and Y both end up horizontal in the top-down case when
        Z points straight down)
    """
    z = np.asarray(approach_direction, dtype=float)
    z = z / np.linalg.norm(z)
    x = np.asarray(closing_direction, dtype=float)
    x = x - np.dot(x, z) * z  # Gram-Schmidt: strip any approach-axis component, guard against
    x_norm = np.linalg.norm(x)  # slightly non-orthogonal inputs rather than assuming perfection
    if x_norm < 1e-6:
        raise ValueError("closing_direction is parallel to approach_direction — cannot build a frame")
    x = x / x_norm
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def plan_side_grasp(
    points_robot: np.ndarray,
    marker_pos_robot: np.ndarray,
    marker_outward_normal_robot: np.ndarray,
    place_xy: tuple,
    gripper_max_width: float = 0.08,
    gripper_min_clearance: float = 0.005,
    table_up: tuple = (0.0, 0.0, 1.0),
    table_height: Optional[float] = None,
    safe_height_above_table: float = 0.20,
    approach_standoff: float = 0.12,
    retreat_distance: float = 0.10,
    lift_height: float = 0.08,
    width_safety_margin: float = 0.01,
    place_height: Optional[float] = None,
    height_fraction: float = 0.7,
):
    """
    Generic, class-agnostic HORIZONTAL side-grasp planner. Unlike
    plan_grasp above (top-down, closes around a table-projected
    footprint), this approaches the object horizontally from whichever
    side its ArUco marker is visible on, and closes the gripper across
    the object's width as seen from that direction. Nothing here is
    specific to any object shape — the only shape-derived input is the
    point cloud's own extent; the approach geometry comes entirely from
    the marker's own detected pose, never from object identity or MuJoCo
    ground truth.

    points_robot: (N,3) target point cloud, robot-base frame
        (geometry.py's output).
    marker_pos_robot, marker_outward_normal_robot: the SELECTED marker's
        own position and outward-facing normal, already carried into
        robot-base frame (aruco_prompt.estimate_marker_pose +
        geometry.py's direction_camera_to_world / direction_world_to_robot_base
        / camera_to_world / world_to_robot_base) — never MuJoCo ground truth.
    place_xy: (x, y) destination in robot-base frame.
    table_height: robot-base-frame Z of the table surface — used only to
        reject a degenerate/at-the-table grasp height and to anchor the
        safe-high waypoint; same "fixed scene constant" legitimacy as
        plan_grasp's use of it.

    Returns (SideGraspPlan, None) or (None, GraspFailure).
    """
    if len(points_robot) < 5:
        return None, GraspFailure("insufficient_points", f"only {len(points_robot)} points in target cloud")

    up = np.asarray(table_up, dtype=float)
    up = up / np.linalg.norm(up)

    # Steps 1-3: resolve the marker's outward normal to a horizontal
    # approach axis. The normal is already sign-resolved toward the
    # camera by estimate_marker_pose; here we only project it onto the
    # table plane (a marker glued to a mostly-vertical side surface
    # should already be near-horizontal, but this guards against small
    # estimation tilt rather than assuming a perfectly vertical decal).
    normal = np.asarray(marker_outward_normal_robot, dtype=float)
    normal_horiz = normal - np.dot(normal, up) * up
    normal_horiz_norm = np.linalg.norm(normal_horiz)
    if normal_horiz_norm < 1e-2:
        return None, GraspFailure(
            "degenerate_marker_normal",
            f"marker outward normal is nearly parallel to table-up ({normal}) — cannot derive a horizontal approach",
        )
    outward_normal = normal_horiz / normal_horiz_norm

    # Step 4: direction of TRAVEL toward the object (opposite the outward normal).
    approach_direction = -outward_normal
    # Step 8: finger-closing axis, perpendicular to approach, in the table plane.
    closing_direction = np.cross(up, approach_direction)
    closing_norm = np.linalg.norm(closing_direction)
    if closing_norm < 1e-6:
        return None, GraspFailure("degenerate_closing_direction", "table_up parallel to approach_direction")
    closing_direction = closing_direction / closing_norm

    # Step 7: grasp height from the visible cloud's own vertical extent
    # (robust percentiles, not raw min/max — same rationale as plan_grasp's
    # top_z estimate: single stray/outlier points shouldn't skew this).
    #
    # Biased toward the UPPER part of the visible range (height_fraction
    # default 0.7, not the geometric midpoint 0.5) for a reason found
    # in-sandbox, not just tuned by feel: grasping near the table (low
    # absolute Z) with a HORIZONTAL wrist orientation is a real kinematic
    # wall for this arm, not a controller tuning issue — confirmed by
    # directly sweeping target Z at a fixed XY/orientation and finding
    # move_to_pose_interpolated converges cleanly (pos_err<6mm) around
    # ~0.06m above robot-base height here but degrades smoothly and then
    # fails as the target gets closer to the table. A real robot has the
    # same problem (the wrist and gripper body need vertical clearance
    # above the table even when approaching "horizontally"), so grasping
    # higher on a tall enough object is the physically correct fix, not a
    # workaround — this is also why the target cylinder was sized more
    # bottle-like (taller) rather than the earlier short placeholder.
    z = points_robot[:, 2]
    z_min, z_max = float(np.percentile(z, 2)), float(np.percentile(z, 98))
    grasp_z = z_min + height_fraction * (z_max - z_min)
    # Explicit reachability floor, ON TOP OF height_fraction (not instead
    # of it): the visible cloud's OWN z-range can itself be truncated low
    # (FLIP's ROI is sized around the marker's position, so if the marker
    # sits low on the object, the segmented/visible region — and therefore
    # z_max — may not extend anywhere near the object's true top either).
    # height_fraction alone can't fix that; it can only bias within
    # whatever range the perception stack actually handed it. Confirmed
    # in-sandbox that this arm needs roughly 0.14m of clearance above the
    # table to hold a horizontal wrist orientation reliably (the same
    # sweep referenced in height_fraction's comment above) — so raise
    # grasp_z to at least that floor, but never above z_max (don't grasp
    # above the visible object).
    if table_height is not None:
        min_reachable_z = table_height + 0.17
        grasp_z = min(max(grasp_z, min_reachable_z), z_max)
    if table_height is not None and grasp_z < table_height + 0.005:
        return None, GraspFailure(
            "degenerate_grasp_height",
            f"estimated grasp height {grasp_z:.3f}m is at/below the table ({table_height:.3f}m)",
        )

    # Step 9 (moved earlier): initial gripper-opening estimate — visible
    # extent along the CLOSING direction specifically (closing_direction
    # is horizontal by construction, so a plain dot product against the
    # full 3D points is correct and needs no separate XY-only special
    # case). Needed before finalizing grasp_xy below.
    closing_proj = points_robot @ closing_direction
    extent = float(np.percentile(closing_proj, 95) - np.percentile(closing_proj, 5))
    grasp_width = extent + width_safety_margin

    # Grasp XY: the point cloud's OWN centroid was found (empirically, via
    # in-sandbox execution — outcome_check kept failing even though every
    # controller stage converged) to be biased toward the near/camera-facing
    # surface, NOT the object's true central axis. A single oblique camera
    # only sees the front ~half of the object, so the raw centroid sits
    # roughly one radius short of center, along the outward normal (toward
    # the camera) — closing the gripper there grasps empty air just in
    # front of the object.
    #
    # Correction: shift the centroid along approach_direction (i.e. AWAY
    # from the camera, INTO the object) by half the visible closing-axis
    # extent. `extent` (diameter transverse to the approach axis) is used
    # as a proxy for the object's depth extent too — exact for a cylinder/
    # sphere's circular cross-section, an approximation for a box/prism
    # face, consistent with this planner's stated goal of being generic
    # rather than shape-specific.
    surface_centroid_xy = points_robot[:, :2].mean(axis=0)
    center_shift = 0.5 * extent
    grasp_xy = surface_centroid_xy + approach_direction[:2] * center_shift
    grasp_pos = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
    if grasp_width > gripper_max_width:
        return None, GraspFailure(
            "object_too_wide",
            f"estimated grasp width {grasp_width:.3f}m (+margin) exceeds gripper max {gripper_max_width:.3f}m",
        )
    if grasp_width < gripper_min_clearance:
        # Step 10 still applies regardless (close-until-contact, not this
        # estimate alone) — this rejects an implausibly small estimate
        # up front, same as plan_grasp's degenerate_footprint check.
        return None, GraspFailure(
            "degenerate_footprint",
            f"estimated grasp width {grasp_width:.4f}m is implausibly small — likely an under-segmented mask",
        )

    R = axes_to_gripper_rotation(approach_direction, closing_direction)

    # Step 11: pre-grasp OUTSIDE the object, along the marker's outward normal.
    pregrasp_pos = grasp_pos + outward_normal * approach_standoff
    # Step 6 (retreat, before lifting): horizontal, back along outward normal.
    retreat_pos = grasp_pos + outward_normal * retreat_distance
    lift_pos = retreat_pos + up * lift_height

    table_z = table_height if table_height is not None else grasp_z
    safe_high_z = table_z + safe_height_above_table
    safe_high_pos = np.array([pregrasp_pos[0], pregrasp_pos[1], safe_high_z])

    release_z = grasp_z if place_height is None else place_height
    transport_pos = np.array([place_xy[0], place_xy[1], lift_pos[2]])
    place_descend_pos = np.array([place_xy[0], place_xy[1], release_z])

    plan = SideGraspPlan(
        safe_high=SideGraspPose(safe_high_pos, R),
        pregrasp=SideGraspPose(pregrasp_pos, R),
        grasp=SideGraspPose(grasp_pos, R),
        retreat=SideGraspPose(retreat_pos, R),
        lift=SideGraspPose(lift_pos, R),
        transport=SideGraspPose(transport_pos, R),
        place_descend=SideGraspPose(place_descend_pos, R),
        grasp_width=grasp_width,
        approach_direction=approach_direction,
        closing_direction=closing_direction,
        up_direction=up,
        marker_pos=np.asarray(marker_pos_robot, dtype=float),
        marker_outward_normal=outward_normal,
    )
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
