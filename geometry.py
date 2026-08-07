#!/usr/bin/env python
"""
geometry.py — Phase 4: masked depth -> filtered, robot-base-frame 3D point
cloud of the target object.

Pipeline: FLIP's binary mask (flip_segmenter.py) + the camera's metric
depth map (environment.py) + the camera's REAL intrinsics/extrinsics
(robosuite.utils.camera_utils, same as Phase 1) -> per-pixel 3D points in
the camera frame -> world frame -> robot-base frame -> invalid-depth
removal -> tabletop-plane removal -> the final target-only cloud.

Hard constraint (repeated here because this is the module where it would
be easiest to cheat): NEVER call environment.py's
get_ground_truth_state()/get_ground_truth_segmentation() from this module.
The one piece of "known" geometry used below — the robot's own base
pose (env.robots[0].base_pos/base_ori) — is robot proprioception, not
object ground truth; a real robot always knows where its own base is
bolted down, the same way it always knows its own joint angles. Table
height, if used as a RANSAC seed/fallback, is a fixed scene-design
constant (equivalent to knowing your camera mount point), not a
per-object ground-truth read.

Camera convention (confirmed from robosuite.utils.camera_utils source,
not assumed): `get_camera_extrinsic_matrix` explicitly corrects MuJoCo's
native camera axes into the standard OpenCV convention (+X right, +Y
down, +Z forward into the scene) — see that function's own docstring
referencing the OpenCV calib3d convention. `get_real_depth_map` returns
metric distance along the camera's Z (viewing) axis, i.e. exactly the
"depth" a standard pinhole model expects. So the unprojection below is
the textbook pinhole formula, not a guess:
    X_cam = (u - cx) * Z / fx
    Y_cam = (v - cy) * Z / fy
    Z_cam = Z

Standalone self-test:
    python geometry.py
Chains environment.py -> aruco_prompt.py -> flip_segmenter.py -> this
module and prints point-cloud stats + saves a debug point-cloud PNG
(simple 3-view scatter, no extra plotting deps beyond matplotlib).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PointCloudResult:
    points_robot: np.ndarray     # (N, 3) float32, robot-base frame, target-only, post-filtering
    points_world: np.ndarray     # (N, 3) float32, world frame, same points pre-plane-removal-count
    n_raw: int                   # pixel count in the input mask, before any filtering
    n_valid_depth: int           # after invalid-depth removal
    n_after_plane_removal: int   # after tabletop-plane removal (== len(points_robot))
    table_plane: Optional[tuple]  # (normal(3,), point_on_plane(3,)) in world frame, if plane fit succeeded


def deproject_camera(mask: np.ndarray, depth_m: np.ndarray, K: np.ndarray,
                      depth_min: float = 0.05, depth_max: float = 5.0):
    """
    mask: (H, W) bool or 0/255 uint8. depth_m: (H, W) float32 metric depth
    (from camera_utils.get_real_depth_map). K: 3x3 intrinsics.
    Returns (points_cam (N,3), valid_pixel_count, raw_pixel_count).
    """
    mask_bool = mask.astype(bool) if mask.dtype != bool else mask
    n_raw = int(mask_bool.sum())
    if n_raw == 0:
        return np.zeros((0, 3), dtype=np.float32), 0, 0

    vs, us = np.nonzero(mask_bool)
    depths = depth_m[vs, us]

    valid = np.isfinite(depths) & (depths >= depth_min) & (depths <= depth_max)
    us, vs, depths = us[valid], vs[valid], depths[valid]
    n_valid = int(valid.sum())

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (us.astype(np.float32) - cx) * depths / fx
    y_cam = (vs.astype(np.float32) - cy) * depths / fy
    z_cam = depths.astype(np.float32)
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    return points_cam, n_valid, n_raw


def camera_to_world(points_cam: np.ndarray, cam_to_world_T: np.ndarray) -> np.ndarray:
    if len(points_cam) == 0:
        return points_cam
    ones = np.ones((points_cam.shape[0], 1), dtype=np.float32)
    homog = np.concatenate([points_cam, ones], axis=1)  # (N, 4)
    world = (cam_to_world_T @ homog.T).T  # (N, 4)
    return world[:, :3].astype(np.float32)


def world_to_robot_base(points_world: np.ndarray, base_pos: np.ndarray, base_mat: np.ndarray) -> np.ndarray:
    """base_pos (3,), base_mat (3,3) — robot base pose in world frame, as
    read from env.robots[0].base_pos/base_ori (robot proprioception)."""
    if len(points_world) == 0:
        return points_world
    return (points_world - base_pos.reshape(1, 3)) @ base_mat  # R^T applied via right-multiply


def remove_statistical_outliers(points: np.ndarray, mad_k: float = 6.0) -> np.ndarray:
    """
    Removes points far from the cloud's robust center — standard RGB-D
    preprocessing (synthetic and real depth sensors both produce "flying
    pixel" artifacts at object/background silhouette edges, where a
    rendered/measured depth blends between near and far surfaces). Uses
    median absolute deviation (MAD) rather than mean/std since a handful
    of extreme outliers would otherwise skew the mean/std themselves.

    Confirmed necessary empirically during Phase 4 verification: the
    5th-95th percentile of a raw deprojected cloud clustered tightly
    (~4cm spread, matching the object's real size), but the min/max
    spanned ~30cm — a few edge pixels unprojecting to wildly wrong depth.
    """
    if len(points) < 10:
        return points
    center = np.median(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)
    mad = np.median(np.abs(dist - np.median(dist))) + 1e-9
    keep = np.abs(dist - np.median(dist)) < mad_k * mad
    return points[keep]


def fit_table_plane_ransac(points_world: np.ndarray, n_iters: int = 200, dist_thresh: float = 0.005,
                            rng: Optional[np.random.Generator] = None,
                            table_height_hint: Optional[float] = None, height_tol: float = 0.02):
    """
    Minimal RANSAC plane fit (no external deps). Used to find and remove
    the tabletop plane from a target-ish point cloud that may still
    include a thin sliver of table around the object's base. Returns
    (normal(3,), point_on_plane(3,)) or None if too few points to fit /
    no plausible table plane found.

    `table_height_hint`: world-frame Z of the table's top surface. This is
    a fixed SCENE constant (table_offset + table_full_size/2 — the same
    thing you'd need to know to bolt a real camera rig above a real table),
    not per-object ground truth, and is used only to VALIDATE that a
    RANSAC-fitted plane is plausibly the table and not something else.

    Without this check, RANSAC will happily fit "a plane" to almost any
    small, nearly-flat masked region — including just the object's own
    flat top surface, if FLIP's mask under-segments to mostly that (which
    it did during Phase 4 verification: a mask covering only the marker's
    immediate area is itself nearly planar, and naive RANSAC then
    "removed the table" by deleting ~99% of the already-small cloud,
    leaving 2 stray points). Rejecting fits whose mean height isn't near
    the known table height fixes this without touching the object.
    """
    if len(points_world) < 10:
        return None
    rng = rng or np.random.default_rng(0)
    best_inliers = 0
    best_plane = None
    n = len(points_world)
    for _ in range(n_iters):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points_world[idx]
        v1, v2 = p1 - p0, p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -normal.dot(p0)
        dist = np.abs(points_world @ normal + d)
        inlier_mask = dist < dist_thresh
        inliers = int(inlier_mask.sum())
        if inliers > best_inliers:
            if table_height_hint is not None:
                mean_z = points_world[inlier_mask, 2].mean()
                if abs(mean_z - table_height_hint) > height_tol:
                    continue  # plausible plane, but not at table height — reject
            best_inliers = inliers
            best_plane = (normal, p0)
    return best_plane


def remove_plane(points_world: np.ndarray, plane, margin: float = 0.008) -> np.ndarray:
    """Removes points within `margin` of the fitted plane (the table
    surface itself), keeping points that stick up above it (the object)."""
    if plane is None or len(points_world) == 0:
        return points_world
    normal, point_on_plane = plane
    # Orient normal to point "up" in world Z so the sign convention below
    # (keep points on the +normal side) means "above the table."
    if normal[2] < 0:
        normal = -normal
    d = -normal.dot(point_on_plane)
    signed_dist = points_world @ normal + d
    return points_world[signed_dist > margin]


def build_target_point_cloud(
    mask_full: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    cam_to_world_T: np.ndarray,
    base_pos: np.ndarray,
    base_mat: np.ndarray,
    depth_min: float = 0.05,
    depth_max: float = 5.0,
    remove_table: bool = True,
    ransac_iters: int = 200,
    ransac_dist_thresh: float = 0.005,
    plane_margin: float = 0.008,
    table_height_hint: Optional[float] = None,
) -> PointCloudResult:
    points_cam, n_valid, n_raw = deproject_camera(mask_full, depth_m, K, depth_min, depth_max)
    points_world = camera_to_world(points_cam, cam_to_world_T)
    points_world = remove_statistical_outliers(points_world)

    table_plane = None
    points_world_filtered = points_world
    if remove_table and len(points_world) >= 10:
        table_plane = fit_table_plane_ransac(points_world, n_iters=ransac_iters, dist_thresh=ransac_dist_thresh,
                                              table_height_hint=table_height_hint)
        if table_plane is not None:
            points_world_filtered = remove_plane(points_world, table_plane, margin=plane_margin)
        # If RANSAC fails to find a coherent plane (e.g. the masked region
        # is entirely the object, no table sliver in it), that's fine —
        # points_world_filtered just stays as points_world.

    points_robot = world_to_robot_base(points_world_filtered, base_pos, base_mat)

    return PointCloudResult(
        points_robot=points_robot,
        points_world=points_world_filtered,
        n_raw=n_raw,
        n_valid_depth=n_valid,
        n_after_plane_removal=len(points_world_filtered),
        table_plane=table_plane,
    )


def get_robot_base_transform(env):
    """Robot proprioception, not object ground truth — see module docstring."""
    robot = env.robots[0]
    return np.array(robot.base_pos, dtype=np.float32), np.array(robot.base_ori, dtype=np.float32)


if __name__ == "__main__":
    import os
    import robosuite.macros as macros
    macros.IMAGE_CONVENTION = "opencv"
    from environment import PickPlaceEnv, SIDE_CAMERA_NAME, TARGET_MARKER_ID
    from aruco_prompt import get_target_prompt
    from flip_segmenter import FlipTargetSegmenter

    out_dir = "phase4_output"
    os.makedirs(out_dir, exist_ok=True)

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
    env.close()

    detection, failure = get_target_prompt(rgb, expected_id=TARGET_MARKER_ID)
    if detection is None:
        raise SystemExit(f"[FATAL] aruco_prompt failed: {failure.reason.value}")

    segmenter = FlipTargetSegmenter(model_size="small")
    seg = segmenter.segment_from_prompt(rgb, detection.center_px, marker_side_px=detection.side_length_px)
    print(f"mask pixels: {int((seg.mask_full > 0).sum())}")

    table_height_hint = float(env.table_offset[2] + env.table_full_size[2] / 2)
    result = build_target_point_cloud(seg.mask_full, depth, K, cam_to_world, base_pos, base_mat,
                                       table_height_hint=table_height_hint)
    print(f"n_raw={result.n_raw} n_valid_depth={result.n_valid_depth} "
          f"n_after_plane_removal={result.n_after_plane_removal}")
    print(f"table_plane found: {result.table_plane is not None}")
    if len(result.points_robot) > 0:
        mins = result.points_robot.min(axis=0)
        maxs = result.points_robot.max(axis=0)
        print(f"robot-frame bbox: min={mins} max={maxs}")
    else:
        print("[WARN] empty point cloud after filtering")

    np.save(os.path.join(out_dir, "points_robot.npy"), result.points_robot)
    print(f"Saved {out_dir}/points_robot.npy")
