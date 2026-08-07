#!/usr/bin/env python
"""
aruco_prompt.py — Phase 2: turns an RGB frame into a single (u, v) point
prompt for FLIP, by detecting an ArUco marker and taking its center pixel.

This module NEVER looks at object class, shape, or size — only the marker's
id/corners/center. The marker is a target-SELECTION mechanism ("segment
whatever object this point belongs to"), not a source of geometry. Nothing
here reads environment.py's ground-truth accessors.

Uses cv2.aruco directly (same DICT_6X6_250 dictionary + ArucoDetector
pattern as the existing pipeline's segment.py/pick_and_place_flip.py) — no
reimplementation of marker detection.

Standalone self-test (independently testable, per the project's module
requirements):
    python aruco_prompt.py
Renders one frame from environment.py's side_oblique_camera and runs
detection against it, printing the result and saving a debug overlay.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import cv2
import numpy as np


class PromptFailureReason(Enum):
    NO_MARKER_FOUND = "no_marker_found"
    MULTIPLE_MARKERS_AMBIGUOUS = "multiple_markers_ambiguous"
    MARKER_OUT_OF_FRAME = "marker_out_of_frame"          # corners clipped by image border
    MARKER_TOO_SMALL = "marker_too_small"                # decoded, but too few px to trust the center


@dataclass
class MarkerDetection:
    """A single successfully-decoded ArUco marker, usable as a FLIP prompt."""
    marker_id: int
    corners: np.ndarray          # (4, 2) float32, pixel coords, order as returned by OpenCV
    center_px: tuple              # (u, v) float
    side_length_px: float
    near_edge: bool               # True if any corner is within edge_margin_px of the frame border
    frame_shape: tuple            # (H, W) of the source frame this detection came from


@dataclass
class PromptFailure:
    reason: PromptFailureReason
    detail: str = ""
    candidate_ids: List[int] = field(default_factory=list)  # populated for MULTIPLE_MARKERS_AMBIGUOUS


@dataclass
class MarkerPose:
    """Camera-frame pose of a detected marker, from solvePnP. Everything
    here is a DIRECTION/POSITION derived purely from the marker's own
    detected corners + its known real-world size — never from MuJoCo
    ground truth (same "ArUco is a selection/geometry-free prompt, except
    for the marker's own physical size which is a scene-construction fact
    the same way a robot always knows its own printed marker's size" rule
    the rest of this module follows)."""
    rvec: np.ndarray               # (3,) Rodrigues rotation vector, camera frame
    tvec: np.ndarray               # (3,) marker center position, camera frame (meters)
    R_cam: np.ndarray              # (3,3) rotation matrix, marker-local -> camera frame
    outward_normal_cam: np.ndarray  # (3,) unit vector, camera frame, pointing from the
                                     # marker's face toward the camera (see sign-resolution
                                     # note in estimate_marker_pose)


ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def make_detector(dict_name: str = "DICT_6X6_250") -> cv2.aruco.ArucoDetector:
    """Same construction as segment.py::make_detector — kept independent
    here (not imported) so this module has no dependency on the FLIP side
    of the pipeline, only on cv2.aruco."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def _marker_center_and_size(corners: np.ndarray):
    center = corners.mean(axis=0)
    side_lengths = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    return center, float(np.mean(side_lengths))


def get_target_prompt(
    rgb: np.ndarray,
    detector: Optional[cv2.aruco.ArucoDetector] = None,
    dict_name: str = "DICT_6X6_250",
    expected_id: Optional[int] = None,
    edge_margin_px: int = 8,
    min_side_px: float = 10.0,
):
    """
    Detects ArUco markers in `rgb` and resolves them to a single target
    prompt point.

    Selection rule (this is the one place "which marker is the target" is
    decided — deliberately simple and explicit, not a heuristic buried
    downstream):
      - 0 markers decoded -> NO_MARKER_FOUND
      - >1 markers decoded:
          - if `expected_id` is given and exactly one decoded marker has
            that id -> use it (evaluation.py and single-target runs pass
            this; it's still just picking among what was actually
            detected, never inventing geometry)
          - otherwise -> MULTIPLE_MARKERS_AMBIGUOUS (caller must resolve,
            e.g. by asking the user, or by proximity to a previous frame's
            prompt in a video/live setting — out of scope for this module)
      - 1 marker decoded -> use it, unless it fails the size/edge checks
        below

    Returns: (MarkerDetection, None) on success, or (None, PromptFailure)
    on failure — never raises for "no marker" style outcomes, since those
    are expected, routine cases the pipeline must handle every frame, not
    exceptional ones.
    """
    if detector is None:
        detector = make_detector(dict_name)

    h, w = rgb.shape[:2]
    # cv2.aruco works on grayscale or BGR/RGB equally for detection (it
    # converts internally); pass through as-is, no channel-order assumption
    # needed for the DETECTOR step (only FLIP's segmenter cares about RGB
    # vs BGR order, and that's flip_segmenter.py's concern, not this one).
    corners_list, ids, _rejected = detector.detectMarkers(rgb)

    if ids is None or len(corners_list) == 0:
        return None, PromptFailure(PromptFailureReason.NO_MARKER_FOUND)

    ids_flat = np.asarray(ids).reshape(-1)

    if len(ids_flat) > 1:
        if expected_id is not None:
            matches = [i for i, mid in enumerate(ids_flat) if int(mid) == expected_id]
            if len(matches) == 1:
                idx = matches[0]
            else:
                return None, PromptFailure(
                    PromptFailureReason.MULTIPLE_MARKERS_AMBIGUOUS,
                    detail=f"expected_id={expected_id} matched {len(matches)} markers, need exactly 1",
                    candidate_ids=[int(m) for m in ids_flat],
                )
        else:
            return None, PromptFailure(
                PromptFailureReason.MULTIPLE_MARKERS_AMBIGUOUS,
                detail=f"{len(ids_flat)} markers detected and no expected_id given",
                candidate_ids=[int(m) for m in ids_flat],
            )
    else:
        idx = 0

    marker_id = int(ids_flat[idx])
    pts = corners_list[idx].reshape(4, 2).astype(np.float32)
    center, side_px = _marker_center_and_size(pts)

    if side_px < min_side_px:
        return None, PromptFailure(
            PromptFailureReason.MARKER_TOO_SMALL,
            detail=f"decoded marker side={side_px:.1f}px < min_side_px={min_side_px}",
        )

    near_edge = bool(
        np.any(pts[:, 0] < edge_margin_px) or np.any(pts[:, 0] > w - edge_margin_px)
        or np.any(pts[:, 1] < edge_margin_px) or np.any(pts[:, 1] > h - edge_margin_px)
    )
    if near_edge:
        # Still return a detection (the marker DID decode — OpenCV already
        # requires the full pattern to be visible to decode it at all) but
        # flag it: a marker flush against the frame border means the
        # object itself may be partially out of frame, which matters to
        # geometry.py's point-cloud completeness downstream.
        pass

    return MarkerDetection(
        marker_id=marker_id,
        corners=pts,
        center_px=(float(center[0]), float(center[1])),
        side_length_px=side_px,
        near_edge=near_edge,
        frame_shape=(h, w),
    ), None


def estimate_marker_pose(
    detection: MarkerDetection,
    K: np.ndarray,
    marker_size_m: float,
    dist_coeffs: Optional[np.ndarray] = None,
) -> MarkerPose:
    """
    Solves for the marker's camera-frame pose via cv2.solvePnP against its
    4 detected corners and a flat square object model of the marker's own
    known real-world size (environment.py::get_target_marker_size() — a
    scene-construction fact, not object ground truth; see MarkerPose's
    docstring).

    Object-point order matches cv2.aruco's own documented corner order
    (top-left, top-right, bottom-right, bottom-left of the marker AS
    PRINTED) — the same convention OpenCV's own estimatePoseSingleMarkers
    uses internally. With this correspondence, solvePnP's resulting
    rotation's 3rd column (local Z) already points from the marker's
    printed face toward the camera under OpenCV's right-handed convention
    (X right, Y up in-plane, Z out of the page) — confirmed by direct
    in-sandbox check (see THESIS_PLAN.md's Phase-1-extension notes):
    printing `outward_normal_cam` for the cylinder's camera-facing decal
    gave a vector with a large negative Z component in camera frame (i.e.
    pointing back toward the camera, which sits in front of the marker
    along +Z_cam), as expected.

    Sign is still explicitly re-verified (not just trusted from the
    convention) against the marker's own known camera-relative position:
    a marker facing the camera must have its outward normal pointing
    roughly opposite `tvec` (back toward the camera at the origin), so if
    `dot(normal, tvec) > 0` the normal is flipped. Cheap, and catches any
    future corner-order mismatch immediately instead of silently planning
    grasps 180 degrees wrong.
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float32)

    half = marker_size_m / 2.0
    obj_pts = np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, detection.corners.astype(np.float32), K.astype(np.float32), dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed to converge on marker pose")

    rvec = rvec.reshape(3)
    tvec = tvec.reshape(3)
    R_cam, _ = cv2.Rodrigues(rvec)
    normal_cam = R_cam[:, 2].copy()

    tvec_dir = tvec / (np.linalg.norm(tvec) + 1e-9)
    if np.dot(normal_cam, tvec_dir) > 0:
        normal_cam = -normal_cam

    return MarkerPose(rvec=rvec, tvec=tvec, R_cam=R_cam, outward_normal_cam=normal_cam)


def draw_debug_overlay(rgb: np.ndarray, detection: Optional[MarkerDetection],
                        failure: Optional[PromptFailure]) -> np.ndarray:
    """Returns a BGR debug image with the marker (if any) annotated —
    used by pipeline.py's per-stage image dump and by this module's
    self-test."""
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    if detection is not None:
        pts = detection.corners.astype(int)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
        cx, cy = detection.center_px
        cv2.circle(overlay, (int(cx), int(cy)), 6, (0, 0, 255), -1)
        label = f"id={detection.marker_id} side={detection.side_length_px:.0f}px"
        if detection.near_edge:
            label += " [NEAR EDGE]"
        cv2.putText(overlay, label, (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        msg = f"FAILED: {failure.reason.value}" if failure else "FAILED: unknown"
        cv2.putText(overlay, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return overlay


if __name__ == "__main__":
    import os
    import robosuite.macros as macros
    macros.IMAGE_CONVENTION = "opencv"
    from environment import PickPlaceEnv, SIDE_CAMERA_NAME, TARGET_MARKER_ID

    out_dir = "phase2_output"
    os.makedirs(out_dir, exist_ok=True)

    env = PickPlaceEnv(
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=[SIDE_CAMERA_NAME], camera_heights=720, camera_widths=960,
        camera_depths=True, num_distractors=0, seed=0,
    )
    env.reset()
    for _ in range(10):
        env.sim.step()
    rgb, _depth = env.get_camera_rgbd()
    env.close()

    detection, failure = get_target_prompt(rgb, expected_id=TARGET_MARKER_ID)
    if detection:
        print(f"OK: marker_id={detection.marker_id} center_px={detection.center_px} "
              f"side_px={detection.side_length_px:.1f} near_edge={detection.near_edge}")
    else:
        print(f"FAILED: {failure.reason.value} — {failure.detail}")

    overlay = draw_debug_overlay(rgb, detection, failure)
    cv2.imwrite(os.path.join(out_dir, "aruco_debug.png"), overlay)
    print(f"Saved {out_dir}/aruco_debug.png")
