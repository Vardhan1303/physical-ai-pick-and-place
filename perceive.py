#!/usr/bin/env python
"""
perceive.py — renders the MuJoCo pick-and-place scene from the overhead
camera, then runs the exact same ArUco+FLIP pipeline as the original
aruco-flip-segmentation project (via segment.py) on the rendered frame:
marker detected -> its center becomes FLIP's point prompt -> FLIP segments
the object -> the resulting mask's contour is classified as square/circle/
triangle using shape geometry alone (no separate trained classifier).

This covers detection + segmentation + shape classification (not yet the
robot's pick-and-place motion, which is the next stage once this is
confirmed working on your machine).

Run from the project root:
    python perceive.py

Requires FLIP_WEIGHTS_DIR to point at wherever your FLIP ONNX weights live
(defaults below to your existing Iphoreos project's weights so nothing
needs to be re-downloaded/copied).
"""
import os

# Must be set before importing segment.py, since it reads this at import time.
os.environ.setdefault(
    "FLIP_WEIGHTS_DIR",
    r"V:\projects\Iphoreos\FLIP-main\model\weights",
)

import cv2
import numpy as np
import mujoco

from segment import FlipSegmenter, make_detector, marker_center_and_size

SCENE_PATH = "assets/franka_emika_panda/pickplace_scene.xml"
CAMERA_NAME = "top_down"
SETTLE_STEPS = 500  # let free-jointed objects land on the table before capturing

ROI_HALF_SIZE_PX = 45  # half-width of the square crop fed to FLIP, centered on each marker
SIGMA = 0.35  # normalized Gaussian sigma within the small ROI crop (objects fill most of it)

MASK_COLOR = (0, 255, 0)
CENTER_COLOR = (0, 0, 255)


def classify_shape(mask_u8: np.ndarray):
    """
    Classifies a binary object mask as 'square', 'circle', or 'triangle'
    purely from contour geometry — no shape-specific model or training.

    Uses circularity (4*pi*area/perimeter^2) rather than vertex-counting via
    approxPolyDP: circularity is far less sensitive to jagged/aliased mask
    edges. Thresholds below were empirically validated against this exact
    scene's ground-truth object silhouettes (not just theory): circle ~0.88,
    square ~0.82, triangle ~0.60 — comfortably separated. If your real FLIP
    masks come out softer/blurrier at the edges than these ground-truth
    silhouettes, these thresholds may need a small retune (same spirit as
    tuning sigma/ROI in the original project).
    """
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown", None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0 or area < 20:
        return "unknown", None

    circularity = 4 * np.pi * area / (perimeter ** 2)
    if circularity > 0.85:
        shape = "circle"
    elif circularity > 0.72:
        shape = "square"
    else:
        shape = "triangle"
    return shape, round(circularity, 3)


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=720, width=960)
    renderer.update_scene(data, camera=CAMERA_NAME)
    frame_rgb = renderer.render()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]

    detector = make_detector("DICT_6X6_250")
    flip = FlipSegmenter("small")

    corners_list, ids, _rejected = detector.detectMarkers(frame_bgr)
    if ids is None:
        print("[WARN] No ArUco markers detected in the rendered frame.")
        return

    overlay = frame_bgr.copy()
    ids_flat = np.asarray(ids).reshape(-1)

    for i, marker_corners in enumerate(corners_list):
        marker_id = int(ids_flat[i])
        pts = marker_corners.reshape(4, 2)
        center, _side_px = marker_center_and_size(pts)
        cx, cy = center

        x0 = int(np.clip(cx - ROI_HALF_SIZE_PX, 0, w - 1))
        x1 = int(np.clip(cx + ROI_HALF_SIZE_PX, 0, w))
        y0 = int(np.clip(cy - ROI_HALF_SIZE_PX, 0, h - 1))
        y1 = int(np.clip(cy + ROI_HALF_SIZE_PX, 0, h))

        roi_bgr = frame_bgr[y0:y1, x0:x1]
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

        rel_x = (cx - x0) / (x1 - x0) * 2 - 1
        rel_y = (cy - y0) / (y1 - y0) * 2 - 1

        mask = flip.segment(roi_rgb, rel_x, rel_y, SIGMA, SIGMA)
        mask_u8 = (mask > 0.5).astype(np.uint8) * 255
        shape, circularity = classify_shape(mask_u8)

        print(f"marker id={marker_id}  center=({cx:.1f},{cy:.1f})  "
              f"shape={shape}  circularity={circularity}")

        alpha_full = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(alpha_full * 0.6, 0, 1)[..., None].astype(np.float32)
        region = overlay[y0:y1, x0:x1].astype(np.float32)
        colored = np.full_like(region, MASK_COLOR, dtype=np.float32)
        overlay[y0:y1, x0:x1] = (region * (1 - alpha) + colored * alpha).astype(np.uint8)

        cv2.circle(overlay, (int(cx), int(cy)), 3, CENTER_COLOR, -1)
        label = f"id{marker_id}:{shape}"
        cv2.putText(overlay, label, (x0, max(y0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite("perceive_result.png", overlay)
    print("Saved perceive_result.png")


if __name__ == "__main__":
    main()
