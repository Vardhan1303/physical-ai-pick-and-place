#!/usr/bin/env python
"""
pipeline.py — Phase 7: the full closed loop, RGB-D -> ArUco -> FLIP ->
point cloud -> grasp -> Panda execution -> outcome check, wiring together
every module from Phases 1-6 with stage-by-stage logging and per-run
artifact saving (video + per-stage debug images), per the project's
requirement.

This module does NOT introduce any new perception/planning logic of its
own — it only sequences the real calls already verified independently in
environment.py, aruco_prompt.py, flip_segmenter.py, geometry.py,
grasp_planner.py, and robot_controller.py, and handles logging/artifacts.

Success verification here is a SIMULATOR-STATE convenience check (is the
target object's final position inside the bin's footprint) for
human-readable per-run logging — it is NOT the project's rigorous success
metric. That lives in evaluation.py, which is the only module allowed to
read MuJoCo ground truth for STATISTICS across many controlled trials.
Checking "did this one demo run look like it worked" via simulator state
here is a logging convenience, not a perception/planning input — nothing
upstream of this check (aruco_prompt/flip_segmenter/geometry/grasp_planner)
used it.

Standalone usage:
    python pipeline.py
Runs one full trial with default settings and writes everything to
runs/<timestamp>/.
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
from aruco_prompt import get_target_prompt, draw_debug_overlay
from flip_segmenter import FlipTargetSegmenter
from geometry import build_target_point_cloud, get_robot_base_transform
from grasp_planner import plan_grasp
from robot_controller import execute_grasp_plan

SETTLE_STEPS = 30


class StageLogger:
    """Prints `[STAGE] name: detail` lines (matching the old pipeline's
    logging style in pick_and_place_flip.py::perceive) and accumulates a
    JSON-serializable log for saving alongside each run's artifacts."""

    def __init__(self):
        self.entries = []

    def log(self, name: str, detail: str, ok: bool = True):
        tag = "OK" if ok else "FAIL"
        line = f"[STAGE:{tag}] {name}: {detail}"
        print(line)
        self.entries.append({"stage": name, "ok": ok, "detail": detail})

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.entries, f, indent=2, default=str)


class VideoRecorder:
    """Captures frames from side_oblique_camera during robot_controller's
    step_callback and writes an mp4. sim.render() returns OpenGL-convention
    frames (row 0 = bottom) regardless of the IMAGE_CONVENTION macro (that
    macro only affects the observation-dict layer, not this lower-level
    call — see environment.py's own note on this), so frames are flipped
    here before writing."""

    def __init__(self, env, path, width=640, height=480, fps=20, capture_every=4):
        self.env = env
        self.width, self.height = width, height
        self.capture_every = capture_every
        self.fps = fps
        self._counter = 0
        self.final_path = path
        # cv2's "mp4v" fourcc writes raw MPEG-4 part 2 in an MP4 container.
        # That combination is legal but many players (Windows Media Player
        # in particular — confirmed by the user, who saw solid static/noise
        # instead of the recording) don't ship a decoder for it and instead
        # of failing cleanly just render garbage. Write to a scratch file
        # with this codec (cv2 has no built-in H.264 encoder in most wheel
        # builds), then transcode to H.264/yuv420p — the combination every
        # mainstream player supports — via ffmpeg in close().
        self._raw_path = path + ".raw.mp4"
        self.writer = cv2.VideoWriter(self._raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    def maybe_capture(self):
        self._counter += 1
        if self._counter % self.capture_every != 0:
            return
        frame = self.env.sim.render(camera_name=SIDE_CAMERA_NAME, width=self.width, height=self.height)
        frame = frame[::-1]  # OpenGL -> top-down convention, see class docstring
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr)

    def close(self):
        self.writer.release()
        self._transcode_to_h264()

    @staticmethod
    def _resolve_ffmpeg():
        """
        Finds an ffmpeg binary WITHOUT assuming it's on PATH — confirmed
        necessary: the first version of this fix called "ffmpeg" directly
        via subprocess, which works in the (Linux, ffmpeg-preinstalled)
        sandbox but silently fails with FileNotFoundError on a fresh
        Windows conda env with no system ffmpeg, at which point the old
        code fell back to shipping the original broken raw mp4v file —
        i.e. the exact bug this was supposed to fix, just hidden behind a
        try/except. Two-tier lookup instead of one:
          1. shutil.which("ffmpeg") — a real system install, if present.
          2. imageio_ffmpeg's bundled static binary (pip-installable,
             `pip install imageio-ffmpeg`, no system PATH entry needed at
             all) — the reliable cross-platform fallback.
        Returns the resolved path, or None if neither is available (the
        caller then fails LOUDLY instead of silently shipping mp4v).
        """
        import shutil
        found = shutil.which("ffmpeg")
        if found:
            return found
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    def _transcode_to_h264(self):
        import subprocess
        ffmpeg_exe = self._resolve_ffmpeg()
        if ffmpeg_exe is None:
            print("[VideoRecorder] No ffmpeg binary found (checked PATH and imageio_ffmpeg) — "
                  "cannot produce a widely-playable video. Run `pip install imageio-ffmpeg` and "
                  "retry, or install ffmpeg and add it to PATH. Keeping the raw (likely "
                  f"unplayable-in-Windows-Media-Player) file at {self._raw_path} for now — "
                  f"it was NOT renamed to {self.final_path}, so you can tell at a glance that "
                  "the fix didn't apply.")
            return
        try:
            result = subprocess.run(
                [
                    ffmpeg_exe, "-y", "-loglevel", "error",
                    "-i", self._raw_path,
                    "-c:v", "libx264",
                    # Baseline profile / level 3.0, not libx264's default
                    # (High profile + CABAC): after switching to plain
                    # libx264 the user STILL saw static in Windows Media
                    # Player. WMP's built-in H.264 decoder is old and only
                    # reliably handles Baseline/Main profile content;
                    # High-profile B-frames/CABAC are a common cause of
                    # exactly this "plays as noise" failure mode on that
                    # specific player (VLC/modern players handle High fine,
                    # which is why the ffprobe/ffmpeg-decode self-check
                    # alone didn't catch it). Baseline is the safe
                    # lowest-common-denominator choice for a native Windows
                    # player, at the cost of file size, and 640x480 @ 20fps
                    # clips are tiny anyway.
                    "-profile:v", "baseline", "-level", "3.0",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    self.final_path,
                ],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0 or not os.path.exists(self.final_path):
                # ffmpeg unavailable or failed — fall back to the raw file
                # rather than silently losing the recording.
                print(f"[VideoRecorder] ffmpeg transcode failed ({result.returncode}): "
                      f"{result.stderr.strip()[-500:]}; keeping raw mp4v file at {self._raw_path}")
                if os.path.exists(self._raw_path) and not os.path.exists(self.final_path):
                    os.replace(self._raw_path, self.final_path)
                return
            os.remove(self._raw_path)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"[VideoRecorder] ffmpeg not available ({e}); keeping raw mp4v file at {self._raw_path}")
            if os.path.exists(self._raw_path) and not os.path.exists(self.final_path):
                os.replace(self._raw_path, self.final_path)


def run_pipeline(
    env=None,
    run_dir=None,
    model_size="small",
    num_distractors=0,
    seed=0,
    max_steps_per_move=250,
    record_video=True,
    post_capture_callback=None,
):
    """
    post_capture_callback: optional fn(env) invoked immediately after
    RGB-D capture, before ArUco/FLIP/grasp/execution touch anything —
    i.e. while the scene still matches exactly what was captured. This
    exists so evaluation.py can snapshot ground truth (segmentation,
    object pose) at the SAME instant the perception stages see the scene,
    without pipeline.py itself importing or calling any ground-truth
    accessor. A real bug this fixes: reading ground truth AFTER
    execute_grasp_plan runs compares FLIP's on-table mask against the
    object's POST-PICK position (already moved into the bin) — silently
    producing IoU=0 for otherwise-correct runs, caught during Phase 8
    verification. pipeline.py stays agnostic to what the callback does;
    it never reads ground truth itself.
    """
    if run_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join("runs", ts)
    os.makedirs(run_dir, exist_ok=True)
    log = StageLogger()

    own_env = env is None
    if own_env:
        env = PickPlaceEnv(
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
            camera_names=[SIDE_CAMERA_NAME], camera_heights=720, camera_widths=960,
            camera_depths=True, num_distractors=num_distractors, seed=seed,
        )

    timings = {}
    t_start = time.time()

    try:
        env.reset()
        for _ in range(SETTLE_STEPS):
            env.sim.step()
        seed_note = seed if own_env else "externally-constructed env"
        log.log("reset", f"env reset + {SETTLE_STEPS} settle steps, seed={seed_note}")

        t0 = time.time()
        rgb, depth = env.get_camera_rgbd()
        K = env.get_camera_intrinsics()
        cam_to_world = env.get_camera_extrinsics()
        base_pos, base_mat = get_robot_base_transform(env)
        table_height_world = env.get_table_height()
        table_height_robot = float(((np.array([0, 0, table_height_world]) - base_pos) @ base_mat)[2])
        cv2.imwrite(os.path.join(run_dir, "00_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        timings["capture"] = time.time() - t0
        log.log("capture", f"rgb={rgb.shape} depth_range=({depth.min():.3f},{depth.max():.3f})m")

        if post_capture_callback is not None:
            post_capture_callback(env)

        t0 = time.time()
        detection, failure = get_target_prompt(rgb, expected_id=TARGET_MARKER_ID)
        timings["aruco_prompt"] = time.time() - t0
        overlay = draw_debug_overlay(rgb, detection, failure)
        cv2.imwrite(os.path.join(run_dir, "01_aruco.png"), overlay)
        if detection is None:
            log.log("aruco_prompt", f"FAILED: {failure.reason.value} — {failure.detail}", ok=False)
            log.save(os.path.join(run_dir, "log.json"))
            return {"success": False, "reason": f"aruco_prompt:{failure.reason.value}", "run_dir": run_dir,
                    "timings": timings, "total_time": time.time() - t_start, "rgb": rgb, "depth": depth,
                    "detection": None, "mask_full": None, "points_robot": None}
        log.log("aruco_prompt", f"marker_id={detection.marker_id} center={detection.center_px} "
                                 f"side_px={detection.side_length_px:.1f} near_edge={detection.near_edge}")

        t0 = time.time()
        segmenter = FlipTargetSegmenter(model_size=model_size)
        seg = segmenter.segment_from_prompt(
            rgb, detection.center_px, marker_side_px=detection.side_length_px,
            debug_dir=run_dir, debug_tag="02_flip",
        )
        timings["flip_segmenter"] = time.time() - t0
        n_fg = int((seg.mask_full > 0).sum())
        log.log("flip_segmenter", f"mask_px={n_fg} confidence={seg.confidence:.3f} roi_bbox={seg.roi_bbox}",
                 ok=n_fg > 0)
        if n_fg == 0:
            log.save(os.path.join(run_dir, "log.json"))
            return {"success": False, "reason": "flip_segmenter:empty_mask", "run_dir": run_dir,
                    "timings": timings, "total_time": time.time() - t_start, "rgb": rgb, "depth": depth,
                    "detection": detection, "mask_full": seg.mask_full, "points_robot": None}

        t0 = time.time()
        pc = build_target_point_cloud(seg.mask_full, depth, K, cam_to_world, base_pos, base_mat,
                                       table_height_hint=table_height_world)
        timings["geometry"] = time.time() - t0
        log.log("geometry", f"n_raw={pc.n_raw} n_valid_depth={pc.n_valid_depth} "
                             f"n_final={pc.n_after_plane_removal} table_plane_found={pc.table_plane is not None}",
                 ok=pc.n_after_plane_removal > 0)
        if pc.n_after_plane_removal == 0:
            log.save(os.path.join(run_dir, "log.json"))
            return {"success": False, "reason": "geometry:empty_cloud", "run_dir": run_dir,
                    "timings": timings, "total_time": time.time() - t_start, "rgb": rgb, "depth": depth,
                    "detection": detection, "mask_full": seg.mask_full, "points_robot": pc.points_robot}

        t0 = time.time()
        bin_world = env.get_bin_top_center() + np.array([0, 0, 0.01])
        bin_robot = (bin_world - base_pos) @ base_mat
        plan, plan_failure = plan_grasp(pc.points_robot, place_xy=(bin_robot[0], bin_robot[1]),
                                         table_height=table_height_robot, place_height=bin_robot[2])
        timings["grasp_planner"] = time.time() - t0
        if plan is None:
            log.log("grasp_planner", f"FAILED: {plan_failure.reason} — {plan_failure.detail}", ok=False)
            log.save(os.path.join(run_dir, "log.json"))
            return {"success": False, "reason": f"grasp_planner:{plan_failure.reason}", "run_dir": run_dir,
                    "timings": timings, "total_time": time.time() - t_start, "rgb": rgb, "depth": depth,
                    "detection": detection, "mask_full": seg.mask_full, "points_robot": pc.points_robot}
        log.log("grasp_planner", f"grasp_width={plan.grasp_width:.4f}m yaw={np.degrees(plan.grasp.yaw):.1f}deg "
                                  f"grasp_pos={plan.grasp.pos}")

        recorder = VideoRecorder(env, os.path.join(run_dir, "03_execution.mp4")) if record_video else None
        step_cb = recorder.maybe_capture if recorder else None

        target_body_id = env.target_object_body_id
        pos_before = np.array(env.sim.data.body_xpos[target_body_id]).copy()

        t0 = time.time()
        exec_result = execute_grasp_plan(env, plan, base_pos, base_mat,
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

        # Simulator-state convenience check for THIS run's human-readable
        # outcome — see module docstring. NOT the project's rigorous
        # success metric (that's evaluation.py, across many trials).
        bin_center = env.get_bin_top_center()
        bin_half = env._bin_half_size
        in_bin_xy = (
            abs(pos_after[0] - bin_center[0]) < bin_half[0]
            and abs(pos_after[1] - bin_center[1]) < bin_half[1]
        )
        lifted_then_placed = pos_after[2] > table_height_world - 0.02  # didn't fall on the floor
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
            "rgb": rgb,
            "depth": depth,
            "detection": detection,
            "mask_full": seg.mask_full,
            "points_robot": pc.points_robot,
            "plan": plan,
            "exec_result": exec_result,
            "lift_success": lift_success,
        }

    except Exception as e:
        log.log("exception", f"{type(e).__name__}: {e}\n{traceback.format_exc()}", ok=False)
        log.save(os.path.join(run_dir, "log.json"))
        return {"success": False, "reason": f"exception:{e}", "run_dir": run_dir}

    finally:
        if own_env:
            env.close()


if __name__ == "__main__":
    result = run_pipeline()
    # Large arrays (rgb/depth/mask_full/points_robot) and object handles
    # (detection/plan/exec_result) are useful when calling run_pipeline()
    # programmatically (evaluation.py does exactly that), but dumping them
    # to the console makes the CLI output unreadable — print a compact
    # summary here instead.
    summary_keys = ["success", "reason", "run_dir", "grasp_width", "n_points", "timings", "total_time",
                     "lift_success"]
    summary = {k: result[k] for k in summary_keys if k in result}
    print(json.dumps(summary, default=str, indent=2))
    sys.exit(0 if result["success"] else 1)
