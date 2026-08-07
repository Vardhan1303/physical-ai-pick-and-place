#!/usr/bin/env python
"""
multi_object_pipeline.py — Phase 4: multi-object target-selection mode
("Mode B" in the project spec).

All four object families (cylinder/box/prism/sphere) are present on the
table at once. This script picks them ONE AT A TIME, by marker ID:
  - detect all visible markers
  - select only `target_marker_id`
  - run the SAME generic side-grasp pipeline (side_grasp_pipeline.py's
    run_side_grasp_pipeline, unmodified except for the target_marker_id
    parameter it already accepts) for that one marker
  - place it in the bin
  - move to the next marker id in the requested sequence

No object-type-specific branching anywhere in this file or in the
perception/grasp code it calls — grasp_planner.plan_side_grasp already IS
the generic planner (same function used for the Phase 1 cylinder), the
only thing that changes per object is WHICH marker's pose/mask feeds it.

Does not touch evaluation/metrics (explicitly out of scope for this pass
per the project owner) — this is execution only: does the arm actually
detect, segment, plan, and attempt each object.

Usage:
    python multi_object_pipeline.py
    python multi_object_pipeline.py --shapes cylinder box
    python multi_object_pipeline.py --marker-ids 1 3
"""
import argparse
import datetime
import json
import os
import sys

import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"

from environment import SIDE_CAMERA_NAME
from environment_multi import MultiObjectPickPlaceEnv, MARKER_ID_BY_SHAPE, SHAPE_BY_MARKER_ID
from side_grasp_pipeline import run_side_grasp_pipeline

SETTLE_STEPS = 30


def run_multi_object_pipeline(
    object_shapes=("cylinder", "box", "prism", "sphere"),
    target_marker_ids=None,
    run_dir=None,
    model_size="small",
    seed=0,
    max_steps_per_move=400,
    record_video=True,
    randomize_scale=False,
    randomize_color=False,
):
    """
    Builds ONE MultiObjectPickPlaceEnv (all of `object_shapes` on the
    table together) and picks `target_marker_ids` in sequence (default:
    every marker present, in ascending id order — cylinder, box, prism,
    sphere). Each pick is a full, independent run of
    side_grasp_pipeline.run_side_grasp_pipeline against the SAME env
    (env is passed in and NOT reset between picks — a picked object is
    already off the table / in the bin, exactly like a real sequential
    pick-and-place task; only the first call's env.reset() places
    everything).

    Returns a dict with one entry per attempted marker id plus an overall
    summary — no CSV/evaluation logging (explicitly out of scope for this
    pass; see evaluation.py for the separate, not-yet-extended metrics
    path).
    """
    if run_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join("runs", f"multi_object_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    if target_marker_ids is None:
        target_marker_ids = sorted(MARKER_ID_BY_SHAPE[s] for s in object_shapes)

    env = MultiObjectPickPlaceEnv(
        object_shapes=object_shapes,
        randomize_scale=randomize_scale,
        randomize_color=randomize_color,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=[SIDE_CAMERA_NAME], camera_heights=720, camera_widths=960,
        camera_depths=True, seed=seed, horizon=5000,
    )
    env.reset()
    for _ in range(SETTLE_STEPS):
        env.sim.step()

    results = {}
    try:
        for marker_id in target_marker_ids:
            shape = SHAPE_BY_MARKER_ID.get(marker_id, f"id{marker_id}")
            obj_run_dir = os.path.join(run_dir, f"object_{marker_id}_{shape}")
            print(f"\n=== Picking marker_id={marker_id} ({shape}) -> {obj_run_dir} ===", flush=True)

            # env passed in explicitly (own_env=False in
            # run_side_grasp_pipeline) so it is NOT reset or closed
            # between objects — the whole point of Mode B is sequential
            # picks off the SAME table state.
            result = run_side_grasp_pipeline(
                env=env, run_dir=obj_run_dir, model_size=model_size,
                max_steps_per_move=max_steps_per_move, record_video=record_video,
                target_marker_id=marker_id,
            )
            results[marker_id] = {"shape": shape, **{k: v for k, v in result.items() if k != "plan"}}
            print(f"    -> success={result['success']} reason={result.get('reason')}", flush=True)
    finally:
        env.close()

    n_success = sum(1 for r in results.values() if r["success"])
    summary = {
        "run_dir": run_dir,
        "object_shapes": list(object_shapes),
        "attempted": list(target_marker_ids),
        "n_success": n_success,
        "n_attempted": len(target_marker_ids),
        "per_object": results,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, default=str, indent=2)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", choices=list(MARKER_ID_BY_SHAPE), default=None,
                         help="Which shape families to put on the table (default: all 4).")
    parser.add_argument("--marker-ids", nargs="+", type=int, default=None,
                         help="Which marker ids to pick, in order (default: all present, ascending).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--max-steps-per-move", type=int, default=400)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--randomize-scale", action="store_true")
    parser.add_argument("--randomize-color", action="store_true")
    args = parser.parse_args()

    shapes = tuple(args.shapes) if args.shapes else ("cylinder", "box", "prism", "sphere")

    summary = run_multi_object_pipeline(
        object_shapes=shapes,
        target_marker_ids=args.marker_ids,
        seed=args.seed,
        model_size=args.model_size,
        max_steps_per_move=args.max_steps_per_move,
        record_video=not args.no_video,
        randomize_scale=args.randomize_scale,
        randomize_color=args.randomize_color,
    )
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "per_object"}, indent=2))
    sys.exit(0 if summary["n_success"] == summary["n_attempted"] else 1)
