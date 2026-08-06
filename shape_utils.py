#!/usr/bin/env python
"""
shape_utils.py — shared geometry helpers used by both perceive.py and
pick_and_place.py: turning a FLIP mask into (a) a shape classification and
(b) a grasp point + orientation, and turning a 2D pixel into a 3D world
point via the fixed-table-height assumption.

None of this is shape-specific by design — the same functions run for
square, circle, and triangle. That's the whole point of the demo: FLIP
gives a generic segmentation, and everything downstream (classification,
grasp geometry) is generic image processing on top of it, not a per-class
trained model.
"""
import numpy as np
import cv2
import mujoco


def classify_shape(mask_u8: np.ndarray):
    """
    Classifies a binary mask as 'square', 'circle', or 'triangle' from
    contour circularity alone (4*pi*area/perimeter^2). Thresholds validated
    against this project's actual scene geometry: circle ~0.88, square
    ~0.82, triangle ~0.60.
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


def grasp_from_mask(mask_u8: np.ndarray):
    """
    Returns (cx_px, cy_px, grasp_angle_rad) from a binary mask:
      - center = the mask's own centroid (image moments), NOT the raw
        marker position — once FLIP has segmented the object, the grasp
        point should come from what was actually segmented.
      - grasp_angle = cv2.minAreaRect's rotation, adjusted so the gripper
        closes along the SHORTER of the rect's two dimensions (approaches
        perpendicular to the longer axis). This one rule handles all three
        shapes without any per-shape branching: it's near-arbitrary (and
        harmless) for the roughly-symmetric square/circle, and gives a
        real, meaningful grasp axis for the triangle.
    """
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)

    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]

    (_, _), (w, h), angle_deg = cv2.minAreaRect(c)
    # OpenCV's angle is the rotation of the rect's "w" side. If h is the
    # longer side, the gripper's closing axis should be rotated 90 deg
    # further so it still closes along the shorter dimension.
    if h > w:
        angle_deg += 90.0

    return cx, cy, np.deg2rad(angle_deg)


def unproject_pixel_to_table(model, data, cam_name: str, px: float, py: float,
                              img_width: int, img_height: int, table_z: float):
    """
    Casts a ray from the camera through pixel (px, py) and intersects it
    with the horizontal plane z = table_z, using the fixed-height
    assumption (no depth camera needed): we know objects sit on a table at
    a known height, so this alone is enough to recover 3D X/Y.

    Handles both perspective and orthographic cameras (model.cam_projection:
    0=perspective, 1=orthographic — confirmed empirically, not documented
    plainly). For an orthographic camera, cam_fovy is the FULL vertical
    extent of the view volume in world length units (not degrees, and not
    a half-extent) — also confirmed empirically by rendering known marker
    positions and solving for the pixel<->world mapping, since MuJoCo's own
    docs are easy to misread here. Every ray is parallel to the camera's
    view axis (no convergence at the camera position like perspective), so
    the pixel offset shifts the ray's ORIGIN rather than its direction.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)  # columns = camera's local x,y,z axes in world frame
    cam_right, cam_up, cam_back = cam_mat[:, 0], cam_mat[:, 1], cam_mat[:, 2]
    forward_dir = -cam_back  # camera looks down its local -z

    aspect = img_width / img_height
    ndc_x = (2 * px / img_width) - 1
    ndc_y = 1 - (2 * py / img_height)

    is_orthographic = int(model.cam_projection[cam_id]) == 1

    if is_orthographic:
        half_h = model.cam_fovy[cam_id] / 2.0
        half_w = half_h * aspect
        ray_origin = cam_pos + ndc_x * half_w * cam_right + ndc_y * half_h * cam_up
        world_dir = forward_dir
    else:
        tan_half_fovy = np.tan(np.deg2rad(model.cam_fovy[cam_id]) / 2)
        tan_half_fovx = tan_half_fovy * aspect
        local_dir = np.array([ndc_x * tan_half_fovx, ndc_y * tan_half_fovy, -1.0])
        world_dir = cam_mat @ local_dir
        world_dir /= np.linalg.norm(world_dir)
        ray_origin = cam_pos

    if abs(world_dir[2]) < 1e-8:
        return None  # ray parallel to table plane, shouldn't happen for a top-down camera

    t = (table_z - ray_origin[2]) / world_dir[2]
    world_point = ray_origin + t * world_dir
    return world_point
