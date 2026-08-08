#!/usr/bin/env python
"""
run_interactive_demo.py — run this on YOUR OWN machine (not the sandbox)
to watch the ArUco-guided FLIP pick-and-place live in an on-screen MuJoCo
viewer window.

This is a thin wrapper around the exact same pipeline used everywhere
else in this project (side_grasp_pipeline.run_side_grasp_pipeline /
multi_object_pipeline.run_multi_object_pipeline) — nothing about
perception or grasping is reimplemented here. The only difference from
the headless runs is:
  - the env is built with has_renderer=True (an on-screen MuJoCo window)
    IN ADDITION TO has_offscreen_renderer=True (still needed for the
    side_oblique_camera perception capture — robosuite supports both at
    once)
  - every simulation step also calls env.render(), so you watch the arm
    move live instead of only getting a saved .mp4 afterward

Requirements (on your machine, not the sandbox):
  - the SAME project virtualenv already set up for this repo (mujoco,
    robosuite, opencv, torch/onnxruntime for FLIP — see the project's own
    setup notes) with a real display available (a monitor / X server —
    this will NOT work over a headless SSH session without X forwarding
    or a virtual display)
  - FLIP_WEIGHTS_DIR pointing at your local FLIP-main/model/weights, same
    as every other script in this repo

Usage examples:
    # Single cylinder, Phase-1-style validation run:
    python run_interactive_demo.py --shapes cylinder

    # All four objects, pick every marker in sequence (Mode B):
    python run_interactive_demo.py --shapes cylinder box prism sphere

    # Just two objects, specific pick order:
    python run_interactive_demo.py --shapes cylinder box --marker-ids 1 0

    # Use config.yaml's settings instead of CLI flags:
    python run_interactive_demo.py --from-config
"""
import argparse
import sys

import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"

from environment import SIDE_CAMERA_NAME
from environment_multi import MultiObjectPickPlaceEnv, MARKER_ID_BY_SHAPE, SHAPE_BY_MARKER_ID
from side_grasp_pipeline import run_side_grasp_pipeline

SETTLE_STEPS = 30


def run_interactive(object_shapes, target_marker_ids=None, seed=0, model_size="small",
                     max_steps_per_move=400, record_video=True, run_dir="runs/interactive"):
    print(f"Building scene: shapes={object_shapes} seed={seed}")
    print("Opening the MuJoCo viewer window now — if nothing appears, make sure "
          "you're running this on a machine with a display attached (not over a "
          "headless SSH session).")

    env = MultiObjectPickPlaceEnv(
        object_shapes=object_shapes,
        has_renderer=True,             # <-- on-screen window
        has_offscreen_renderer=True,   # <-- still needed for side_oblique_camera capture
        render_camera=None,            # let the viewer use its own free/default camera,
                                        # independent of the perception camera
        use_camera_obs=True,
        camera_names=[SIDE_CAMERA_NAME],
        camera_heights=720, camera_widths=960, camera_depths=True,
        seed=seed, horizon=5000,
    )
    env.reset()
    for _ in range(SETTLE_STEPS):
        env.sim.step()
        env.render()

    if target_marker_ids is None:
        target_marker_ids = sorted(MARKER_ID_BY_SHAPE[s] for s in object_shapes)

    results = {}
    try:
        for marker_id in target_marker_ids:
            shape = SHAPE_BY_MARKER_ID.get(marker_id, f"id{marker_id}")
            obj_run_dir = f"{run_dir}/object_{marker_id}_{shape}"
            print(f"\n=== Picking marker_id={marker_id} ({shape}) — watch the viewer window ===")

            # Live rendering happens via the VideoRecorder.maybe_capture
            # monkeypatch installed by _install_render_hook() below (see
            # its docstring for why) — every env.step() inside
            # execute_side_grasp_plan calls that hook already, so the
            # on-screen window updates every step with no extra plumbing
            # needed here.
            result = run_side_grasp_pipeline(
                env=env, run_dir=obj_run_dir, model_size=model_size,
                max_steps_per_move=max_steps_per_move, record_video=record_video,
                target_marker_id=marker_id,
            )
            # run_side_grasp_pipeline doesn't itself take a step_callback
            # override (it always builds its own VideoRecorder-driven
            # callback when record_video=True) — patch the callback in
            # AFTER execution isn't possible post-hoc, so instead we rely
            # on VideoRecorder's own step_callback path to ALSO render:
            # see _RenderingVideoRecorder below, installed via monkeypatch
            # right before this loop starts (see __main__).
            results[marker_id] = result
            print(f"    -> success={result['success']} reason={result.get('reason')}")
    finally:
        env.close()

    n_success = sum(1 for r in results.values() if r["success"])
    print(f"\nDone: {n_success}/{len(target_marker_ids)} objects placed successfully.")
    return results


def _install_render_hook():
    """
    side_grasp_pipeline.run_side_grasp_pipeline builds its own VideoRecorder
    (imported from pipeline.py) and uses ITS maybe_capture as the
    step_callback whenever record_video=True — there's no separate hook to
    also call env.render() each step without either (a) duplicating
    run_side_grasp_pipeline here, which the project's own instructions say
    not to do ("do not rewrite working components unnecessarily"), or (b)
    making VideoRecorder itself render when the env has an on-screen
    viewer. (b) is the smaller, most localized change: VideoRecorder
    already receives `env`, so it can check env.has_renderer /
    env.viewer once and call env.render() from maybe_capture()
    alongside its existing frame-capture — a no-op for every other
    (headless) caller since has_renderer is False there.
    """
    from pipeline import VideoRecorder

    _orig_maybe_capture = VideoRecorder.maybe_capture

    def maybe_capture_and_render(self):
        if getattr(self.env, "has_renderer", False):
            self.env.render()
        _orig_maybe_capture(self)

    VideoRecorder.maybe_capture = maybe_capture_and_render


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", nargs="+", choices=list(MARKER_ID_BY_SHAPE), default=["cylinder"],
                         help="Which shape families to put on the table (default: just the cylinder, "
                              "matching Phase 1's single-object validation scene).")
    parser.add_argument("--marker-ids", nargs="+", type=int, default=None,
                         help="Which marker ids to pick, in order (default: all present, ascending).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--max-steps-per-move", type=int, default=400)
    parser.add_argument("--no-video", action="store_true", help="Skip saving .mp4 files (viewer window still works).")
    parser.add_argument("--from-config", action="store_true", help="Ignore the above flags and use config.yaml instead.")
    args = parser.parse_args()

    _install_render_hook()

    if args.from_config:
        from config import load_config, target_marker_ids_from_config
        cfg = load_config()
        shapes = tuple(cfg["scene"]["object_shapes"])
        marker_ids = target_marker_ids_from_config(cfg)
        results = run_interactive(
            object_shapes=shapes, target_marker_ids=marker_ids,
            seed=cfg["scene"]["seed"], model_size=cfg["task"]["model_size"],
            max_steps_per_move=cfg["task"]["max_steps_per_move"],
            record_video=cfg["task"]["record_video"],
        )
    else:
        results = run_interactive(
            object_shapes=tuple(args.shapes), target_marker_ids=args.marker_ids,
            seed=args.seed, model_size=args.model_size,
            max_steps_per_move=args.max_steps_per_move, record_video=not args.no_video,
        )

    n_success = sum(1 for r in results.values() if r["success"])
    sys.exit(0 if n_success == len(results) else 1)
