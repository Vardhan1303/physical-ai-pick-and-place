#!/usr/bin/env python
"""
pick_and_place_flip.py — the full demo: real FLIP segmentation driving the
actual robot pick-and-place motion in MuJoCo.

Pipeline per detected ArUco marker (this is the whole thesis of the project:
one pipeline, no per-shape branching until the very last "which bin" step):

  1. detect marker -> its center is FLIP's point prompt (segment.py)
  2. crop a ROI around the marker, sized relative to the marker's own pixel
     size (same idea as segment.py's webcam loop, just driven by a rendered
     MuJoCo frame instead of a camera)
  3. FLIP segments the object in that ROI -> binary mask
  4. shape_utils.classify_shape(mask) -> 'square' | 'circle' | 'triangle'
     shape_utils.grasp_from_mask(mask) -> grasp point (mask centroid) +
     grasp angle (cv2.minAreaRect, closes along the shorter side) — this is
     the SAME generic function for every shape, no special-casing
  5. shape_utils.unproject_pixel_to_table(...) turns the grasp pixel into a
     3D world point via the fixed-table-height assumption
  6. shape is logged/printed but does NOT choose the destination — every
     object goes to the same shared bin (staggered into sub-slots by pick
     order only, purely to avoid objects landing on top of each other)
  7. manipulation.run_pick_and_place(...) drives the arm: gripper starts
     open -> moves above the object -> descends -> CLOSES the gripper ->
     (weld locks the object to the gripper so it survives lateral transport,
     see manipulation.py's docstring for why) -> lifts -> transports -> lowers
     -> releases -> retracts

Run from the project root on your Windows machine (needs the real
flip_position extension + ONNX weights — see segment.py):
    python pick_and_place_flip.py
"""
import os

# Must be set before importing segment.py, since it reads this at import time.
os.environ.setdefault("FLIP_WEIGHTS_DIR", r"V:\projects\Iphoreos\FLIP-main\model\weights")

import cv2
import numpy as np
import mujoco

from segment import FlipSegmenter, make_detector, marker_center_and_size
from shape_utils import classify_shape, grasp_from_mask, unproject_pixel_to_table
from manipulation import run_pick_and_place, set_home_pose, set_park_pose
from ik_utils import get_arm_qpos_addrs

SCENE_PATH = "assets/franka_emika_panda/pickplace_scene.xml"
CAM_NAME = "top_down"
SETTLE_STEPS = 500  # let free-jointed objects land on the table before capturing

ROI_MIN_HALF_PX = 45     # floor, in case a marker renders very small
ROI_SCALE = 2.3          # ROI half-size = max(ROI_MIN_HALF_PX, marker_side_px * ROI_SCALE)
SIGMA = 0.35

TABLE_Z = 0.4
GRASP_HEIGHT = 0.4275     # table_z + object half-height (0.0275 at the 1.4x size)

# One shared bin (see pickplace_scene.xml's "bin" geom) — shape no longer
# selects a destination. Objects are staggered into sub-slots within the
# bin's footprint purely so a later object doesn't land on/collide with one
# already sitting there; the slot is assigned by pick ORDER, not shape.
BIN_CENTER = (0.55, 0.0)
BIN_SLOT_OFFSETS_Y = [-0.07, 0.0, 0.07]  # > object width (0.055) so slots don't touch

OBJECT_BODY_NAMES = ["obj_square", "obj_circle", "obj_triangle"]


def nearest_object_body(model, data, world_xy):
    """
    Resolves which simulated body a grasp point belongs to, by spatial
    proximity alone — this is a simulation bookkeeping step (MuJoCo's weld
    constraint needs a specific body name to attach to), not a
    classification step. FLIP already told us WHERE and (via shape) WHICH
    BIN; this just answers "which physics object is sitting at that point."
    """
    best_name, best_dist = None, float("inf")
    for name in OBJECT_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        dist = np.linalg.norm(data.xpos[body_id][:2] - np.asarray(world_xy))
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name


def perceive(model, data, flip, detector, cam_name, img_w=960, img_h=720):
    """
    Renders the scene from cam_name, detects every ArUco marker, and runs
    FLIP + the generic grasp/classification pipeline on each. Returns a list
    of dicts: {marker_id, shape, world_xy, grasp_yaw, mask}.
    """
    renderer = mujoco.Renderer(model, height=img_h, width=img_w)
    renderer.update_scene(data, camera=cam_name)
    frame_rgb = renderer.render()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]

    corners_list, ids, _rejected = detector.detectMarkers(frame_bgr)
    if ids is None:
        return []

    ids_flat = np.asarray(ids).reshape(-1)
    results = []

    for i, marker_corners in enumerate(corners_list):
        marker_id = int(ids_flat[i])
        pts = marker_corners.reshape(4, 2)
        center, side_px = marker_center_and_size(pts)
        cx, cy = center

        half = max(ROI_MIN_HALF_PX, side_px * ROI_SCALE)
        x0 = int(np.clip(cx - half, 0, w - 1))
        x1 = int(np.clip(cx + half, 0, w))
        y0 = int(np.clip(cy - half, 0, h - 1))
        y1 = int(np.clip(cy + half, 0, h))
        if x1 - x0 < 10 or y1 - y0 < 10:
            print(f"[WARN] marker {marker_id}: ROI too small, skipping")
            continue

        roi_bgr = frame_bgr[y0:y1, x0:x1]
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

        rel_x = (cx - x0) / (x1 - x0) * 2 - 1
        rel_y = (cy - y0) / (y1 - y0) * 2 - 1

        mask = flip.segment(roi_rgb, rel_x, rel_y, SIGMA, SIGMA)
        mask_u8 = (mask > 0.5).astype(np.uint8) * 255

        shape, circularity = classify_shape(mask_u8)
        grasp = grasp_from_mask(mask_u8)
        if shape == "unknown" or grasp is None:
            print(f"[WARN] marker {marker_id}: segmentation/classification failed, skipping")
            continue
        gx_roi, gy_roi, grasp_yaw = grasp

        # ROI-local pixel -> full-frame pixel
        px_full = x0 + gx_roi
        py_full = y0 + gy_roi

        world_point = unproject_pixel_to_table(model, data, cam_name, px_full, py_full, w, h, TABLE_Z)

        print(f"marker id={marker_id} shape={shape} (circularity={circularity}) "
              f"world_xy=({world_point[0]:.4f},{world_point[1]:.4f}) yaw={np.degrees(grasp_yaw):.1f}deg")

        results.append({
            "marker_id": marker_id,
            "shape": shape,
            "world_xy": (float(world_point[0]), float(world_point[1])),
            "grasp_yaw": float(grasp_yaw),
        })

    return results


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    arm_addrs = get_arm_qpos_addrs(model)
    # Park (not home) for perception: the home/ready pose reaches directly
    # over the table center and occludes the circle marker from the
    # overhead camera — see manipulation.PARK_QPOS's comment.
    set_park_pose(model, data, arm_addrs)
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)

    detector = make_detector("DICT_6X6_250")
    flip = FlipSegmenter("small")

    detections = perceive(model, data, flip, detector, CAM_NAME)
    if not detections:
        print("[WARN] No objects detected/segmented — nothing to pick.")
        return

    # Deterministic order (by marker id) so repeated runs behave the same.
    detections.sort(key=lambda d: d["marker_id"])

    for idx, det in enumerate(detections):
        shape = det["shape"]
        obj_name = nearest_object_body(model, data, det["world_xy"])
        slot_y = BIN_SLOT_OFFSETS_Y[idx % len(BIN_SLOT_OFFSETS_Y)]
        place_xy = (BIN_CENTER[0], BIN_CENTER[1] + slot_y)

        print(f"--- picking marker {det['marker_id']} ({shape}, body={obj_name}) "
              f"-> bin slot at {place_xy} ---")
        run_pick_and_place(
            model, data, obj_name,
            pick_xy=det["world_xy"],
            place_xy=place_xy,
            grasp_yaw=det["grasp_yaw"],
            grasp_height=GRASP_HEIGHT,
        )
        for _ in range(200):
            mujoco.mj_step(model, data)

    print("Done — all detected objects processed.")


if __name__ == "__main__":
    main()
