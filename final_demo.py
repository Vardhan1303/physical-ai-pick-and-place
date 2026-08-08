#!/usr/bin/env python
"""
final_demo.py — the single, polished perception-to-pick-and-place demo.

One deterministic rollout of the existing pipeline (environment.py,
aruco_prompt.py, flip_segmenter.py, geometry.py, grasp_planner.py,
robot_controller.py — all reused unmodified in their core logic, only
extended with small, additive hooks: see demo_config.py, the three
presentation cameras in environment.py, and execute_side_grasp_plan's new
on_stage_end/home_pose_world/speed/tolerance parameters in
robot_controller.py):

  reset -> capture RGB-D (side_oblique_camera) -> detect ArUco marker
  -> marker centre = FLIP point prompt -> FLIP mask -> mask+depth =
  target point cloud (robot-base frame) -> side-grasp plan (marker
  outward normal = approach axis, point cloud = centre/height/width)
  -> home -> safe waypoint -> side pre-grasp -> horizontal approach
  -> close -> retreat -> lift -> transport -> place-descend -> release
  -> retract -> home.

Produces, under demo_output/:
  images/00_rgb.png            raw perception RGB
  images/01_aruco_prompt.png   ArUco marker + centre-point prompt
  images/02_flip_mask.png      FLIP binary mask
  images/03_flip_overlay.png   FLIP mask over RGB
  images/04_grasp_plan.png     target centre, approach/closing axes, pre-grasp + grasp pose
  images/05_pick.png           gripping/lifting the cylinder
  images/06_place.png          releasing the cylinder in the tray
  videos/01_overview.mp4               wide view (overview_camera)
  videos/02_perception_camera.mp4      side_oblique_camera + live debug overlay
  videos/03_side_grasp_closeup.mp4     close side view (side_grasp_closeup_camera)
  videos/04_place_closeup.mp4          close tray view (place_closeup_camera)
  log.json                     full run log (see build_log_entries below)

All four videos and all seven images come from ONE simulation rollout
(same seed, same object pose, same robot trajectory) — the four cameras
are captured simultaneously every step, not via separate replay passes.

Standalone usage:
    python final_demo.py
"""
import argparse
import dataclasses
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
import robosuite.utils.transform_utils as T

from environment import (
    PickPlaceEnv, SIDE_CAMERA_NAME, OVERVIEW_CAMERA_NAME,
    SIDE_CLOSEUP_CAMERA_NAME, PLACE_CLOSEUP_CAMERA_NAME, TARGET_MARKER_ID,
)
from aruco_prompt import get_target_prompt, draw_debug_overlay, estimate_marker_pose
from flip_segmenter import FlipTargetSegmenter, ROI_TALL_OBJECT_MIN_HALF_PX, ROI_MIN_HALF_PX, ROI_SCALE
from geometry import (
    build_target_point_cloud, get_robot_base_transform,
    camera_to_world, world_to_robot_base, world_to_camera, project_to_pixels,
    direction_camera_to_world, direction_world_to_robot_base,
)
from grasp_planner import plan_side_grasp
from robot_controller import execute_side_grasp_plan, GRIPPER_OPEN
from demo_config import DemoConfig
from pipeline import VideoRecorder

SETTLE_STEPS = 30
DEMO_CAMERAS = [SIDE_CAMERA_NAME, OVERVIEW_CAMERA_NAME, SIDE_CLOSEUP_CAMERA_NAME, PLACE_CLOSEUP_CAMERA_NAME]


# ----------------------------------------------------------------------
# Debug-overlay drawing (presentation only — none of this feeds back into
# perception or planning; it only visualizes what those stages already
# produced).
# ----------------------------------------------------------------------
def draw_flip_overlay(rgb: np.ndarray, mask_full: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    color = np.zeros_like(bgr)
    color[mask_full > 0] = (0, 255, 0)
    return cv2.addWeighted(bgr, 1.0, color, 0.45, 0)


def draw_grasp_plan_overlay(bgr: np.ndarray, K, cam_to_world, base_pos, base_mat, plan,
                              axis_len_m: float = 0.06) -> np.ndarray:
    """Projects the planned grasp point, pre-grasp point, and the
    approach/closing axes (robot-base frame) back into this camera's
    pixel space via world_to_camera + project_to_pixels (geometry.py),
    and draws them — approach in green, closing in orange, grasp point in
    red, pre-grasp point in blue."""
    out = bgr.copy()

    def to_px(p_robot):
        p_world = base_pos + base_mat @ np.asarray(p_robot, dtype=np.float64)
        p_cam = world_to_camera(p_world.reshape(1, 3).astype(np.float32), cam_to_world)
        uv, valid = project_to_pixels(p_cam, K)
        if not valid[0] or not np.all(np.isfinite(uv[0])):
            return None
        return (int(round(uv[0, 0])), int(round(uv[0, 1])))

    gp_px = to_px(plan.grasp.pos)
    pg_px = to_px(plan.pregrasp.pos)
    approach_end_px = to_px(plan.grasp.pos + plan.approach_direction * axis_len_m)
    closing_end_px = to_px(plan.grasp.pos + plan.closing_direction * axis_len_m)

    if gp_px and approach_end_px:
        cv2.arrowedLine(out, gp_px, approach_end_px, (0, 200, 0), 3, tipLength=0.25)
    if gp_px and closing_end_px:
        cv2.arrowedLine(out, gp_px, closing_end_px, (0, 140, 255), 3, tipLength=0.25)
    if gp_px:
        cv2.circle(out, gp_px, 7, (0, 0, 255), -1)
        cv2.putText(out, "grasp", (gp_px[0] + 8, gp_px[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    if pg_px:
        cv2.circle(out, pg_px, 7, (255, 120, 0), -1)
        cv2.putText(out, "pre-grasp", (pg_px[0] + 8, pg_px[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 0), 2)
    return out


def draw_perception_overlay(bgr: np.ndarray, detection, seg, plan, K, cam_to_world, base_pos, base_mat) -> np.ndarray:
    """Combines: ArUco polygon + id + centre dot, FLIP ROI box, FLIP mask
    contour, and the planned grasp axes — everything the spec asks
    02_perception_camera.mp4 to overlay, in one pass."""
    out = bgr.copy()

    if detection is not None:
        pts = detection.corners.astype(int)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        cx, cy = detection.center_px
        cv2.circle(out, (int(cx), int(cy)), 6, (0, 0, 255), -1)
        cv2.putText(out, f"id={detection.marker_id}", (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if seg is not None:
        x0, y0, x1, y1 = seg.roi_bbox
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 1)
        contours, _ = cv2.findContours(seg.mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 0), 2)

    if plan is not None:
        out = draw_grasp_plan_overlay(out, K, cam_to_world, base_pos, base_mat, plan)

    return out


class OverlayVideoRecorder(VideoRecorder):
    """VideoRecorder with a per-frame overlay hook — used only for
    02_perception_camera.mp4 (see module docstring). The overlay itself
    is computed ONCE (detection/mask/plan don't change after the
    perception pass), just redrawn on each new live frame so the burned-in
    debug info stays visible throughout the whole rollout, not just a
    single still."""
    def __init__(self, *args, overlay_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlay_fn = overlay_fn

    def maybe_capture(self):
        self._counter += 1
        if self._counter % self.capture_every != 0:
            return
        frame = self.env.get_camera_rgb(camera_name=self.camera_name)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
            bgr = cv2.resize(bgr, (self.width, self.height))
        if self.overlay_fn:
            bgr = self.overlay_fn(bgr)
        self.writer.write(bgr)


# ----------------------------------------------------------------------
def run_final_demo(cfg: DemoConfig = None, out_dir: str = "demo_output",
                    max_steps_per_move: int = 400, record_video: bool = True):
    cfg = cfg or DemoConfig()
    images_dir = os.path.join(out_dir, "images")
    videos_dir = os.path.join(out_dir, "videos")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    run_log = {"config": dataclasses.asdict(cfg), "stages": []}

    def log(stage, **fields):
        entry = {"stage": stage, "t": round(time.time(), 3), **fields}
        run_log["stages"].append(entry)
        print(f"[{stage}] " + " ".join(f"{k}={v}" for k, v in fields.items() if k != "stage"))

    env = PickPlaceEnv(
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=DEMO_CAMERAS, camera_heights=720, camera_widths=960, camera_depths=True,
        num_distractors=0, seed=cfg.seed, horizon=8000,
    )

    try:
        env.reset()
        for _ in range(SETTLE_STEPS):
            env.sim.step()
        log("reset", seed=cfg.seed, settle_steps=SETTLE_STEPS)

        # Reproducible home pose — captured right after reset/settle, and
        # returned to at the very end of the motion sequence (see
        # execute_side_grasp_plan's home_pose_world parameter).
        obs = env._get_observations(force_update=True)
        home_pos_world = obs["robot0_eef_pos"].copy()
        home_R_world = T.quat2mat(obs["robot0_eef_quat"].copy())
        log("home_pose", pos=home_pos_world.round(4).tolist())

        # --- Perception: capture ---
        t0 = time.time()
        rgb, depth = env.get_camera_rgbd(SIDE_CAMERA_NAME)
        K = env.get_camera_intrinsics(SIDE_CAMERA_NAME)
        cam_to_world = env.get_camera_extrinsics(SIDE_CAMERA_NAME)
        base_pos, base_mat = get_robot_base_transform(env)
        table_height_world = env.get_table_height()
        table_height_robot = float(((np.array([0, 0, table_height_world]) - base_pos) @ base_mat)[2])
        marker_size_m = env.get_target_marker_size()
        cv2.imwrite(os.path.join(images_dir, "00_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        log("capture", rgb_shape=list(rgb.shape), depth_range=[round(float(depth.min()), 3), round(float(depth.max()), 3)],
            marker_size_m=round(marker_size_m, 4), dt=round(time.time() - t0, 3))

        # --- ArUco detection + centre-point prompt ---
        t0 = time.time()
        detection, failure = get_target_prompt(rgb, expected_id=TARGET_MARKER_ID)
        overlay = draw_debug_overlay(rgb, detection, failure)
        cv2.imwrite(os.path.join(images_dir, "01_aruco_prompt.png"), overlay)
        if detection is None:
            log("aruco_prompt", success=False, reason=failure.reason.value, detail=failure.detail)
            _save_log(out_dir, run_log)
            return {"success": False, "reason": f"aruco_prompt:{failure.reason.value}", "out_dir": out_dir}
        log("aruco_prompt", success=True, marker_id=detection.marker_id, center_px=list(detection.center_px),
            side_px=round(detection.side_length_px, 1), dt=round(time.time() - t0, 3))

        pose = estimate_marker_pose(detection, K, marker_size_m)
        marker_pos_world = camera_to_world(pose.tvec.reshape(1, 3), cam_to_world)[0]
        marker_normal_world = direction_camera_to_world(pose.outward_normal_cam, cam_to_world)
        marker_pos_robot = world_to_robot_base(marker_pos_world.reshape(1, 3), base_pos, base_mat)[0]
        marker_normal_robot = direction_world_to_robot_base(marker_normal_world, base_mat)
        log("marker_pose", pos_robot=marker_pos_robot.round(4).tolist(),
            outward_normal_robot=marker_normal_robot.round(4).tolist())

        # --- FLIP segmentation from the marker centre ---
        t0 = time.time()
        segmenter = FlipTargetSegmenter(model_size=cfg.model_size)

        def touches_border(seg_result):
            x0, y0, x1, y1 = seg_result.roi_bbox
            m = seg_result.mask_full
            return bool(m[y0, x0:x1].any() or m[y1 - 1, x0:x1].any()
                        or m[y0:y1, x0].any() or m[y0:y1, x1 - 1].any())

        seg = segmenter.segment_from_prompt(rgb, detection.center_px, marker_side_px=detection.side_length_px)
        n_fg = int((seg.mask_full > 0).sum())
        marker_area_px = float(detection.side_length_px) ** 2
        regrown = False
        if n_fg > 0 and (touches_border(seg) or n_fg < 6 * marker_area_px):
            initial_half = max(ROI_MIN_HALF_PX, detection.side_length_px * ROI_SCALE)
            grown_half = min(ROI_TALL_OBJECT_MIN_HALF_PX, initial_half * 1.6)
            seg = segmenter.segment_from_prompt(rgb, detection.center_px, marker_side_px=detection.side_length_px,
                                                 min_half_px=grown_half)
            n_fg = int((seg.mask_full > 0).sum())
            regrown = True
        flip_dt = time.time() - t0

        cv2.imwrite(os.path.join(images_dir, "02_flip_mask.png"), seg.mask_full)
        cv2.imwrite(os.path.join(images_dir, "03_flip_overlay.png"), draw_flip_overlay(rgb, seg.mask_full))
        log("flip_segmenter", success=n_fg > 0, mask_area_px=n_fg, confidence=round(seg.confidence, 3),
            roi_bbox=list(seg.roi_bbox), regrown=regrown, inference_time_s=round(flip_dt, 3))
        if n_fg == 0:
            _save_log(out_dir, run_log)
            return {"success": False, "reason": "flip_segmenter:empty_mask", "out_dir": out_dir}

        # --- Target point cloud (mask + depth, camera -> world -> robot base) ---
        t0 = time.time()
        pc = build_target_point_cloud(seg.mask_full, depth, K, cam_to_world, base_pos, base_mat,
                                        table_height_hint=table_height_world)
        log("point_cloud", n_raw=pc.n_raw, n_valid_depth=pc.n_valid_depth,
            n_after_plane_removal=pc.n_after_plane_removal, table_plane_found=pc.table_plane is not None,
            dt=round(time.time() - t0, 3))
        if pc.n_after_plane_removal == 0:
            _save_log(out_dir, run_log)
            return {"success": False, "reason": "geometry:empty_cloud", "out_dir": out_dir}
        estimated_center = pc.points_robot.mean(axis=0)
        log("point_cloud_stats", estimated_center_robot=estimated_center.round(4).tolist())

        # --- Side-grasp planning (marker normal = approach axis, point
        # cloud = centre/height/width — see grasp_planner.py) ---
        t0 = time.time()
        bin_world = env.get_bin_top_center() + np.array([0, 0, 0.01])
        bin_robot = (bin_world - base_pos) @ base_mat
        place_height = cfg.place_height if cfg.place_height is not None else bin_robot[2]
        plan, plan_failure = plan_side_grasp(
            pc.points_robot, marker_pos_robot, marker_normal_robot,
            place_xy=(bin_robot[0], bin_robot[1]),
            gripper_max_width=cfg.max_gripper_opening,
            gripper_min_clearance=cfg.min_gripper_opening,
            table_height=table_height_robot,
            safe_height_above_table=cfg.safe_waypoint_height,
            approach_standoff=cfg.pregrasp_standoff,
            lift_height=cfg.lift_height,
            width_safety_margin=cfg.gripper_open_margin,
            place_height=place_height,
            height_fraction=cfg.grasp_height_ratio,
        )
        if plan is None:
            log("grasp_planner", success=False, reason=plan_failure.reason, detail=plan_failure.detail)
            _save_log(out_dir, run_log)
            return {"success": False, "reason": f"grasp_planner:{plan_failure.reason}", "out_dir": out_dir}
        gripper_opening = cfg.clamp_gripper_opening(plan.grasp_width - cfg.gripper_open_margin)
        log("grasp_planner", success=True, estimated_width=round(plan.grasp_width, 4),
            gripper_opening_target=round(gripper_opening, 4),
            grasp_pos=plan.grasp.pos.round(4).tolist(), pregrasp_pos=plan.pregrasp.pos.round(4).tolist(),
            lift_pos=plan.lift.pos.round(4).tolist(), place_pos=plan.place_descend.pos.round(4).tolist(),
            dt=round(time.time() - t0, 3))

        grasp_plan_img = draw_grasp_plan_overlay(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), K, cam_to_world,
                                                   base_pos, base_mat, plan)
        cv2.imwrite(os.path.join(images_dir, "04_grasp_plan.png"), grasp_plan_img)

        # --- Video recorders (all 4 cameras, one shared rollout) ---
        recorders = []
        if record_video:
            perception_overlay_fn = lambda bgr: draw_perception_overlay(
                bgr, detection, seg, plan, K, cam_to_world, base_pos, base_mat)
            recorders = [
                VideoRecorder(env, os.path.join(videos_dir, "01_overview.mp4"),
                               camera_name=OVERVIEW_CAMERA_NAME),
                OverlayVideoRecorder(env, os.path.join(videos_dir, "02_perception_camera.mp4"),
                                     camera_name=SIDE_CAMERA_NAME, width=960, height=720,
                                     overlay_fn=perception_overlay_fn),
                VideoRecorder(env, os.path.join(videos_dir, "03_side_grasp_closeup.mp4"),
                               camera_name=SIDE_CLOSEUP_CAMERA_NAME),
                VideoRecorder(env, os.path.join(videos_dir, "04_place_closeup.mp4"),
                               camera_name=PLACE_CLOSEUP_CAMERA_NAME),
            ]

        def step_callback():
            for r in recorders:
                r.maybe_capture()

        still_frames_captured = {"pick": False, "place": False}

        def on_stage_end(name, stage):
            log("motion_stage", name=name, success=stage.success, steps=stage.steps_taken,
                pos_err=round(stage.final_pos_error, 4), rot_err=round(stage.final_rot_error, 4),
                stopped_by_contact=stage.stopped_by_contact)
            if name == "lift" and not still_frames_captured["pick"]:
                frame = env.get_camera_rgb(camera_name=SIDE_CLOSEUP_CAMERA_NAME)
                cv2.imwrite(os.path.join(images_dir, "05_pick.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                still_frames_captured["pick"] = True
            if name == "open_gripper" and not still_frames_captured["place"]:
                frame = env.get_camera_rgb(camera_name=PLACE_CLOSEUP_CAMERA_NAME)
                cv2.imwrite(os.path.join(images_dir, "06_place.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                still_frames_captured["place"] = True

        target_body_id = env.target_object_body_id
        pos_before = np.array(env.sim.data.body_xpos[target_body_id]).copy()

        t0 = time.time()
        exec_result = execute_side_grasp_plan(
            env, plan, base_pos, base_mat,
            step_callback=step_callback, max_steps_per_move=max_steps_per_move,
            pos_tol=cfg.controller_position_tolerance, rot_tol=cfg.controller_orientation_tolerance,
            side_approach_speed=cfg.side_approach_speed, final_approach_speed=cfg.final_approach_speed,
            home_pose_world=(home_pos_world, home_R_world), on_stage_end=on_stage_end,
        )
        exec_dt = time.time() - t0

        for r in recorders:
            r.close()

        pos_after = np.array(env.sim.data.body_xpos[target_body_id]).copy()
        bin_center = env.get_bin_top_center()
        bin_half = env._bin_half_size
        in_bin_xy = (abs(pos_after[0] - bin_center[0]) < bin_half[0]
                     and abs(pos_after[1] - bin_center[1]) < bin_half[1])
        lifted_then_placed = pos_after[2] > table_height_world - 0.02
        outcome_success = bool(exec_result.success and in_bin_xy and lifted_then_placed)
        log("outcome", success=outcome_success, controller_success=exec_result.success,
            in_bin_xy=in_bin_xy, pos_before=pos_before.round(4).tolist(), pos_after=pos_after.round(4).tolist(),
            failure_reason=exec_result.failure_reason, dt=round(exec_dt, 3))

        _save_log(out_dir, run_log)
        return {
            "success": outcome_success,
            "reason": exec_result.failure_reason,
            "out_dir": out_dir,
            "grasp_width": plan.grasp_width,
            "gripper_opening": gripper_opening,
        }

    except Exception as e:
        run_log["stages"].append({"stage": "exception", "error": f"{type(e).__name__}: {e}",
                                    "traceback": traceback.format_exc()})
        _save_log(out_dir, run_log)
        return {"success": False, "reason": f"exception:{e}", "out_dir": out_dir}

    finally:
        env.close()


def _save_log(out_dir, run_log):
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump(run_log, f, indent=2, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="demo_output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--max-steps-per-move", type=int, default=400)
    args = parser.parse_args()

    cfg = DemoConfig(seed=args.seed)
    result = run_final_demo(cfg, out_dir=args.out_dir, record_video=not args.no_video,
                              max_steps_per_move=args.max_steps_per_move)
    print("\n" + json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["success"] else 1)
