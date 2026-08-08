#!/usr/bin/env python
"""
run_interactive_demo.py — run this on YOUR OWN machine (with a display
attached, not the sandbox) to watch the ArUco-guided FLIP side-grasp pick
of the single bottle-like cylinder live in an on-screen MuJoCo window.

Single-object only, by design — this is the Phase 1 scene (one bottle,
one ArUco marker on its side, one destination bin), the same scene
side_grasp_pipeline.py already validates headlessly. This script does not
build or reimplement anything new: it constructs the SAME PickPlaceEnv
(environment.py) with an on-screen viewer enabled, and calls the SAME
run_side_grasp_pipeline function everything else in this repo uses.

Requirements (on your machine):
  - this repo's own conda/virtualenv (robosuite==1.5.2, mujoco==3.3.0 —
    see requirements.txt's version-pin note) with a real display
    available. Will NOT work over a headless SSH session.
  - FLIP_WEIGHTS_DIR pointing at your local FLIP-main/model/weights
    (same as every other script here)
  - `pip install imageio-ffmpeg` if you don't already have a system
    ffmpeg on PATH — needed for the saved .mp4 to be a widely-playable
    H.264 file rather than the older codec Windows Media Player can't
    decode (see pipeline.py::VideoRecorder._resolve_ffmpeg).

Usage:
    python run_interactive_demo.py
    python run_interactive_demo.py --seed 3 --no-video
"""
import argparse
import sys

import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"

from environment import PickPlaceEnv, SIDE_CAMERA_NAME
from side_grasp_pipeline import run_side_grasp_pipeline

SETTLE_STEPS = 30


def run_interactive(seed=0, model_size="small", max_steps_per_move=400,
                     record_video=True, run_dir="runs/interactive"):
    print(f"Building scene (single bottle, seed={seed})...")
    print("Opening the MuJoCo viewer window now — if nothing appears, make sure "
          "you're running this on a machine with a display attached (not over a "
          "headless SSH session).")

    env = PickPlaceEnv(
        has_renderer=True,             # on-screen window
        has_offscreen_renderer=True,   # still needed for side_oblique_camera capture
        render_camera=None,            # let the viewer use its own free camera,
                                        # independent of the perception camera
        use_camera_obs=True,
        camera_names=[SIDE_CAMERA_NAME],
        camera_heights=720, camera_widths=960, camera_depths=True,
        num_distractors=0, seed=seed, horizon=5000,
    )
    env.reset()
    for _ in range(SETTLE_STEPS):
        env.sim.step()
        env.render()

    try:
        # extra_step_callback=env.render is what makes this interactive:
        # execute_side_grasp_plan (robot_controller.py) calls the step
        # callback after every single env.step(); run_side_grasp_pipeline
        # combines this with its own video-recording callback (see its
        # extra_step_callback docstring), so the on-screen window updates
        # live every step regardless of whether --no-video was passed.
        result = run_side_grasp_pipeline(
            env=env, run_dir=run_dir, model_size=model_size,
            max_steps_per_move=max_steps_per_move, record_video=record_video,
            extra_step_callback=env.render,
        )
        print(f"\nsuccess={result['success']} reason={result.get('reason')} run_dir={result['run_dir']}")
        # Hold the window open a bit after finishing so you can see the
        # final pose instead of the window closing immediately.
        for _ in range(100):
            env.render()
    finally:
        env.close()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--max-steps-per-move", type=int, default=400)
    parser.add_argument("--no-video", action="store_true", help="Skip saving the .mp4 (viewer window still works).")
    args = parser.parse_args()

    result = run_interactive(
        seed=args.seed, model_size=args.model_size,
        max_steps_per_move=args.max_steps_per_move, record_video=not args.no_video,
    )
    sys.exit(0 if result["success"] else 1)
