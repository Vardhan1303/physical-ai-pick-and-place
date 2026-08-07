#!/usr/bin/env python
"""
side_grasp_pipeline.py — Phase 1 of the multi-object/side-grasp
extension: validates a TRUE HORIZONTAL side approach on the existing
single cylinder scene, before any multi-object/multi-shape work begins.

This module intentionally does NOT touch pipeline.py or
execute_grasp_plan (the working top-down pipeline stays exactly as it
was, still used by evaluation.py) — it is a new, separate orchestration
script over the new side-grasp additions:
  - aruco_prompt.estimate_marker_pose        (marker pose/normal, camera frame)
  - geometry.direction_camera_to_world / direction_world_to_robot_base
  - grasp_planner.plan_side_grasp / axes_to_gripper_rotation
  - robot_controller.execute_side_grasp_plan
Everything else (env capture, FLIP segmentation, point-cloud
construction) is the SAME already-verified code the top-down pipeline
uses, imported directly, not reimplemented.

Saves, to run_dir (default runs/side_grasp_<timestamp>/):
  00_rgb.png              — raw camera capture
  01_aruco.png             — detected marker corners + center (existing overlay)
  02_marker_pose.png       — marker's estimated pose, drawn as camera-frame axes
  03_flip_*                — FLIP's own debug trail (rgb/prompt/raw/cleaned/overlay)
  04_point_cloud.png       — top-down + side views of the target cloud, with the
                              planned approach/closing/up axes drawn at the grasp point
  05_execution.mp4          — video of the horizontal approach + lift + place
  log.json                  — full stage-by-stage log, including the planned pose

Standalone usage:
    python side_grasp_pipeline.py
"""
import datetime
import json
import os
import sys
import time
import traceback

import cv2
import numpy as np
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"

from environment import PickPlaceEnv, SIDE_CAMERA_NAME, TARGET_MARKER_ID
from aruco_prompt import get_target_prompt, draw_debug_overlay, estimate_marker_pose
from flip_segmenter import FlipTargetSegmenter, ROI_TALL_OBJECT_MIN_HALF_PX
from geometry import (
    build_target_point_cloud, get_robot_base_transform,
    camera_to_world, world_to_robot_base,
    direction_camera_to_world, direction_world_to_robot_base,
)
from grasp_planner import plan_side_grasp
from robot_controller import execute_side_grasp_plan
from pipeline import StageLogger, VideoRecorder  # reused as-is, not reimplemented

SETTLE_STEPS = 30


def draw_marker_pose_axes(rgb: np.ndarray, K: np.ndarray, pose, axis_length: float) -> np.ndarray:
    """BGR debug image with the marker's estimated camera-frame pose drawn
    as XYZ axes (cv2.drawFrameAxes) — a direct visualization of what
    estimate_marker_pose actually solved for, independent of any later
    processing (projection onto the table plane, cross products, etc.)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    dist_coeffs = np.zeros(5, dtype=np.float32)
    cv2.drawFrameAxes(bgr, K.astype(np.float32), dist_coeffs, pose.rvec, pose.tvec, axis_length)
    return bgr


def save_point_cloud_debug(path, points_robot: np.ndarray, plan, marker_pos: np.ndarray):
    """Top-down (XY) + side (XZ) views of the target cloud in robot-base
    frame, with the planned approach (green), closing (orange), and up
    (purple) axes drawn at the grasp point — satisfies the "debug
    visualization ... that shows the planned approach, closing, and up
    axes" requirement in robot-base frame, which is the frame the grasp
    is actually planned and executed in (more meaningful here than a
    reprojection back onto the 2D image)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scale = 0.06
    gp = plan.grasp.pos

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    ax.scatter(points_robot[:, 0], points_robot[:, 1], s=3, c="gray", alpha=0.5, label="target cloud")
    ax.scatter([gp[0]], [gp[1]], c="red", s=60, marker="x", label="grasp point")
    ax.scatter([marker_pos[0]], [marker_pos[1]], c="blue", s=60, marker="^", label="marker")
    ax.arrow(gp[0], gp[1], plan.approach_direction[0] * scale, plan.approach_direction[1] * scale,
              color="green", width=0.0015, head_width=0.008, length_includes_head=True, label="approach")
    ax.arrow(gp[0], gp[1], plan.closing_direction[0] * scale, plan.closing_direction[1] * scale,
              color="orange", width=0.0015, head_width=0.008, length_includes_head=True, label="closing")
    ax.set_xlabel("robot-base X (m)")
    ax.set_ylabel("robot-base Y (m)")
    ax.set_title("Top-down (XY): approach (green) / closing (orange)")
    ax.axis("equal")
    ax.legend(fontsize=7, loc="best")

    ax2 = axes[1]
    ax2.scatter(points_robot[:, 0], points_robot[:, 2], s=3, c="gray", alpha=0.5)
    ax2.scatter([gp[0]], [gp[2]], c="red", s=60, marker="x")
    ax2.arrow(gp[0], gp[2], plan.approach_direction[0] * scale, 0.0,
               color="green", width=0.0015, head_width=0.008, length_includes_head=True, label="approach (horiz.)")
    ax2.arrow(gp[0], gp[2], 0.0, plan.up_direction[2] * scale,
               color="purple", width=0.0015, head_width=0.008, length_includes_head=True, label="up")
    ax2.set_xlabel("robot-base X (m)")
    ax2.set_ylabel("robot-base Z (m)")
    ax2.set_title("Side view (XZ): approach (green) / up (purple)")
    ax2.axis("equal")
    ax2.legend(fontsize=7, loc="best")

    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_side_grasp_pipeline(
    env=None,
    run_dir=None,
    model_size="small",
    seed=0,
    max_steps_per_move=400,
    record_video=True,
    target_marker_id=None,
):
    """
    target_marker_id: which ArUco marker to select as the grasp target.
    Defaults to environment.TARGET_MARKER_ID (Phase 1's single-cylinder
    scene) for backward compatibility. Pass one of
    environment_multi.MARKER_ID_BY_SHAPE's values (0/1/2/3) when `env` is
    a MultiObjectPickPlaceEnv — this is the ONE thing that changes between
    single-object and multi-object mode; every other line below reads the
    target's body id / marker size through the generic
    env.get_object_body_id/get_marker_size accessors (added to
    environment.py's PickPlaceEnv base class specifically so this
    function works unmodified against either environment).
    """
    if target_marker_id is None:
        target_marker_id = TARGET_MARKER_ID
    if run_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join("runs", f"side_grasp_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    log = StageLogger()

    own_env = env is None
    if own_env:
        env = PickPlaceEnv(
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
            camera_names=[SIDE_CAMERA_NAME], camera_heights=720, camera_widths=960,
            camera_depths=True, num_distractors=0, seed=seed,
            # Side-grasp reorientation needs a much larger step budget than
            # the top-down pipeline's default horizon=1000 — confirmed
            # in-sandbox: robosuite raises "executing action in terminated
            # episode" mid-sequence at the default horizon once
            # execute_side_grasp_plan's larger interpolation budgets
            # (see its own docstring) are actually used.
            horizon=5000,
        )

    timings = {}
    t_start = time.time()

    try:
        env.reset()
        for _ in range(SETTLE_STEPS):
            env.sim.step()
        log.log("reset", f"env reset + {SETTLE_STEPS} settle steps, seed={seed if own_env else 'external'}")

        t0 = time.time()
        rgb, depth = env.get_camera_rgbd()
        K = env.get_camera_intrinsics()
        cam_to_world = env.get_camera_extrinsics()
        base_pos, base_mat = get_robot_base_transform(env)
        table_height_world = env.get_table_height()
        table_height_robot = float(((np.array([0, 0, table_height_world]) - base_pos) @ base_mat)[2])
        marker_size_m = env.get_marker_size(target_marker_id)
        cv2.imwrite(os.path.join(run_dir, "00_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        timings["capture"] = time.time() - t0
        log.log("capture", f"rgb={rgb.shape} depth_range=({depth.min():.3f},{depth.max():.3f})m "
                             f"marker_size_m={marker_size_m:.4f}")

        # --- ArUco: detect + select target_marker_id, then estimate pose ---
        t0 = time.time()
        detection, failure = get_target_prompt(rgb, expected_id=target_marker_id)
        overlay = draw_debug_overlay(rgb, detection, failure)
        cv2.imwrite(os.path.join(run_dir, "01_aruco.png"), overlay)
        if detection is None:
            log.log("aruco_prompt", f"FAILED: {failure.reason.value} — {failure.detail}", ok=False)
            log.save(os.path.join(run_dir, "log.json"))
            return {"success": False, "reason": f"aruco_prompt:{failure.reason.value}", "run_dir": run_dir}
        log.log("aruco_prompt", f"marker_id={detection.marker_id} center={detection.center_px} "
                                  f"side_px={detection.side_length_px:.1f} near_edge={detection.near_edge}")

        pose = estimate_marker_pose(detection, K, marker_size_m)
        timings["aruco_prompt"] = time.time() - t0
        pose_img = draw_marker_pose_axes(rgb, K, pose, axis_length=marker_size_m * 1.5)
        cv2.imwrite(os.path.join(run_dir, "02_marker_pose.png"), pose_img)

        marker_pos_world = camera_to_world(pose.tvec.reshape(1, 3), cam_to_world)[0]
        marker_normal_world = direction_camera_to_world(pose.outward_normal_cam, cam_to_world)
        marker_pos_robot = world_to_robot_base(marker_pos_world.reshape(1, 3), base_pos, base_mat)[0]
        marker_normal_robot = direction_world_to_robot_base(marker_normal_world, base_mat)
        log.log("marker_pose", f"pos_robot={marker_pos_robot} outward_normal_robot={marker_normal_robot}")

        # --- FLIP segmentation from the marker's center pixel ---
        # ROI sizing: start from the marker's OWN detected pixel size
        # (segment_from_prompt's default marker_side_px*ROI_SCALE
        # behavior — no forced large floor), then grow ONCE if the
        # resulting mask touches the ROI's border (a truncation signal —
        # the object likely extends past what we cropped). This replaced
        # an earlier version that always forced a large fixed floor
        # (ROI_TALL_OBJECT_MIN_HALF_PX) for every object: that worked for
        # the single tall Phase-1 cylinder but, confirmed in-sandbox on
        # the multi-object scene, was oversized for smaller/closer-spaced
        # objects (a box picked up a 0.130m-wide mask that had swallowed
        # part of a neighboring object). Growing on-demand from a
        # marker-relative base is the generic, shape-agnostic fix — no
        # per-object-type branching either way.
        t0 = time.time()
        segmenter = FlipTargetSegmenter(model_size=model_size)

        def _touches_border(seg_result):
            x0, y0, x1, y1 = seg_result.roi_bbox
            m = seg_result.mask_full
            return bool(
                m[y0, x0:x1].any() or m[y1 - 1, x0:x1].any()
                or m[y0:y1, x0].any() or m[y0:y1, x1 - 1].any()
            )

        seg = segmenter.segment_from_prompt(
            rgb, detection.center_px, marker_side_px=detection.side_length_px,
            debug_dir=run_dir, debug_tag="03_flip",
        )
        if int((seg.mask_full > 0).sum()) > 0 and _touches_border(seg):
            log.log("flip_segmenter:regrow", "initial mask touched ROI border — retrying with a larger ROI")
            seg = segmenter.segment_from_prompt(
                rgb, detection.center_px, marker_side_px=detection.side_length_px,
                debug_dir=run_dir, debug_tag="03_flip_grown",
                min_half_px=ROI_TALL_OBJECT_MIN_HALF_PX,
            )
        timings["flip_segmenter"] = time.time() - t0
        n_fg = int((seg.mask_full > 0).sum())
        log.log("flip_segmenter", f"mask_px={n_fg} confidence={seg.confidence:.3f} roi_bbox={seg.roi_bbox}",
                 ok=n_fg > 0)
        if n_fg == 0:
            log.save(os.path.join(run_dir, "log.json"))
            return {"success": False, "reason": "flip_segmenter:empty_mask", "run_dir": run_dir}

        # --- Point cloud ---
        t0 = time.time()
        pc = build_target_point_cloud(seg.mask_full, depth, K, cam_to_world, base_pos, base_mat,
                                        table_height_hint=table_height_world)
        timings["geometry"] = time.time() - t0
        log.log("geometry", f"n_raw={pc.n_raw} n_valid_depth={pc.n_valid_depth} "
                              f"n_final={pc.n_after_plane_removal} table_plane_found={pc.table_plane is not None}",
                 ok=pc.n_after_plane_removal > 0)
        if pc.n_after_plane_removal == 0:
            log.save(os.path.join(run_dir, "log.json"))
            return {"success": False, "reason": "geometry:empty_cloud", "run_dir": run_dir}

        # --- Side-grasp planning ---
        t0 = time.time()
        bin_world = env.get_bin_top_center() + np.array([0, 0, 0.01])
        bin_robot = (bin_world - base_pos) @ base_mat
        plan, plan_failure = plan_side_grasp(
            pc.points_robot, marker_pos_robot, marker_normal_robot,
            place_xy=(bin_robot[0], bin_robot[1]),
            table_height=table_height_robot, place_height=bin_robot[2],
        )
        timings["grasp_planner"] = time.time() - t0
        if plan is None:
            log.log("grasp_planner", f"FAILED: {plan_failure.reason} — {plan_failure.detail}", ok=False)
            log.save(os.path.join(run_dir, "log.json"))
            return {"success": False, "reason": f"grasp_planner:{plan_failure.reason}", "run_dir": run_dir}
        log.log("grasp_planner",
                 f"grasp_width={plan.grasp_width:.4f}m grasp_pos={plan.grasp.pos} "
                 f"approach={plan.approach_direction} closing={plan.closing_direction} up={plan.up_direction}")

        save_point_cloud_debug(os.path.join(run_dir, "04_point_cloud.png"), pc.points_robot, plan, marker_pos_robot)

        # --- Execution ---
        recorder = VideoRecorder(env, os.path.join(run_dir, "05_execution.mp4")) if record_video else None
        step_cb = recorder.maybe_capture if recorder else None

        target_body_id = env.get_object_body_id(target_marker_id)
        pos_before = np.array(env.sim.data.body_xpos[target_body_id]).copy()

        t0 = time.time()
        exec_result = execute_side_grasp_plan(env, plan, base_pos, base_mat,
                                                step_callback=step_cb, max_steps_per_move=max_steps_per_move)
        timings["robot_controller"] = time.time() - t0
        if recorder:
            recorder.close()

        for s in exec_result.stages:
            contact_note = " [stopped_by_contact]" if s.stopped_by_contact else ""
            log.log(f"robot_controller:{s.name}",
                     f"steps={s.steps_taken} pos_err={s.final_pos_error:.4f} "
                     f"rot_err={s.final_rot_error:.4f}{contact_note}", ok=s.success)

        pos_after = np.array(env.sim.data.body_xpos[target_body_id]).copy()

        bin_center = env.get_bin_top_center()
        bin_half = env._bin_half_size
        in_bin_xy = (
            abs(pos_after[0] - bin_center[0]) < bin_half[0]
            and abs(pos_after[1] - bin_center[1]) < bin_half[1]
        )
        lifted_then_placed = pos_after[2] > table_height_world - 0.02
        outcome_success = bool(exec_result.success and in_bin_xy and lifted_then_placed)
        log.log("outcome_check",
                 f"pos_before={pos_before} pos_after={pos_after} in_bin_xy={in_bin_xy} "
                 f"controller_success={exec_result.success}", ok=outcome_success)

        lift_stage = next((s for s in exec_result.stages if s.name == "lift"), None)
        lift_success = bool(lift_stage.success) if lift_stage else False

        log.save(os.path.join(run_dir, "log.json"))
        return {
            "success": outcome_success,
            "reason": exec_result.failure_reason,
            "run_dir": run_dir,
            "grasp_width": plan.grasp_width,
            "n_points": len(pc.points_robot),
            "timings": timings,
            "total_time": time.time() - t_start,
            "lift_success": lift_success,
            "plan": plan,
        }

    except Exception as e:
        log.log("exception", f"{type(e).__name__}: {e}\n{traceback.format_exc()}", ok=False)
        log.save(os.path.join(run_dir, "log.json"))
        return {"success": False, "reason": f"exception:{e}", "run_dir": run_dir}

    finally:
        if own_env:
            env.close()


if __name__ == "__main__":
    result = run_side_grasp_pipeline()
    summary_keys = ["success", "reason", "run_dir", "grasp_width", "n_points", "timings", "total_time",
                     "lift_success"]
    summary = {k: result[k] for k in summary_keys if k in result}
    print(json.dumps(summary, default=str, indent=2))
    sys.exit(0 if result["success"] else 1)
