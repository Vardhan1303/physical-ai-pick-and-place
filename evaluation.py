#!/usr/bin/env python
"""
evaluation.py — Phase 8: controlled experiments + the project's ONLY
ground-truth-gated metrics.

This is the one module allowed to call environment.py's
get_ground_truth_state()/get_ground_truth_segmentation(). It never feeds
that ground truth INTO pipeline.py's decision-making — it constructs the
env, hands it to pipeline.run_pipeline() (which runs the real
perception->planning->execution loop exactly as it would standalone), and
only AFTER that call returns does it read ground truth to score what
already happened. Nothing pipeline.py/aruco_prompt.py/flip_segmenter.py/
geometry.py/grasp_planner.py decided was influenced by it.

Metrics computed per trial:
  - aruco_target_selection_success: did aruco_prompt find the marker at all
  - segmentation_iou: FLIP's mask vs. the true target-object silhouette
    (from get_ground_truth_segmentation, element/geom-level)
  - point_cloud_centroid_error_m: |predicted cloud centroid (world) -
    true object center (world)| — a coarse completeness/error proxy given
    this is a single fixed camera (2.5D geometry, see project README)
  - valid_grasp_proposal: did grasp_planner produce a plan
  - lift_success: did the "lift" stage of robot_controller converge
  - pick_and_place_success: pipeline.py's own outcome check
  - total_time_s, per-stage timings

Sweeps over trial configs (seed / num_distractors / randomize_object_rotation)
so pose/clutter/rotation variation can be added by extending TRIAL_CONFIGS
below — this is the harness Phase 7's later work (distractors, unusual
shapes, occlusion) plugs into, not a finished exhaustive study.

Usage:
    python evaluation.py [n_trials]
Writes eval_results/metrics.csv and eval_results/*.png.
"""
import csv
import os
import sys

import mujoco
import numpy as np
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"

from environment import PickPlaceEnv, SIDE_CAMERA_NAME
from geometry import get_robot_base_transform
from pipeline import run_pipeline

OUT_DIR = "eval_results"


def build_ground_truth_mask(env, camera_name=SIDE_CAMERA_NAME):
    """
    Binary (H, W) mask of the TARGET OBJECT's true silhouette, built from
    robosuite's real element-level segmentation
    (camera_utils.get_camera_segmentation, wrapped by
    environment.py::get_ground_truth_segmentation — not reimplemented).
    Used ONLY here, to score FLIP's mask after the fact.
    """
    seg = env.get_ground_truth_segmentation(camera_name)  # (H, W, 2): [geom_type, geom_id]
    geom_ids = seg[:, :, 1]

    target_geom_names = ["target_object_g0_vis", "target_marker_decal"]
    target_ids = set()
    for name in target_geom_names:
        gid = mujoco.mj_name2id(env.sim.model._model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            target_ids.add(gid)

    mask = np.isin(geom_ids, list(target_ids))
    return mask


def compute_iou(pred_mask, gt_mask) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(intersection) / float(union)


def run_trial(trial_id: int, seed: int, num_distractors: int = 0,
              randomize_object_rotation: bool = False, model_size: str = "small"):
    run_dir = os.path.join(OUT_DIR, "runs", f"trial_{trial_id:03d}")
    env = PickPlaceEnv(
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=[SIDE_CAMERA_NAME], camera_heights=720, camera_widths=960,
        camera_depths=True, num_distractors=num_distractors,
        randomize_object_rotation=randomize_object_rotation, seed=seed,
    )

    row = {
        "trial_id": trial_id, "seed": seed, "num_distractors": num_distractors,
        "randomize_object_rotation": randomize_object_rotation,
        "aruco_success": False, "segmentation_iou": float("nan"),
        "point_cloud_centroid_error_m": float("nan"), "valid_grasp_proposal": False,
        "lift_success": False, "pick_and_place_success": False,
        "total_time_s": float("nan"), "failure_reason": None,
    }

    # Ground truth is captured via this callback, invoked by
    # pipeline.run_pipeline() itself immediately after RGB-D capture — the
    # exact instant the scene matches what FLIP/geometry.py will see, and
    # BEFORE any robot motion. Reading it any later (e.g. after
    # execute_grasp_plan) compares against the object's POST-PICK
    # position instead — a real bug hit during Phase 8 verification (see
    # pipeline.py's post_capture_callback docstring). pipeline.py itself
    # never calls into this; the callback is defined and owned entirely
    # here in evaluation.py.
    gt_snapshot = {}

    def snapshot_ground_truth(env_):
        gt_snapshot["mask"] = build_ground_truth_mask(env_)
        gt_snapshot["state"] = env_.get_ground_truth_state()

    try:
        result = run_pipeline(env=env, run_dir=run_dir, model_size=model_size, record_video=False,
                               post_capture_callback=snapshot_ground_truth)

        row["failure_reason"] = result.get("reason")
        row["total_time_s"] = result.get("total_time")
        for stage_name, t in (result.get("timings") or {}).items():
            row[f"time_{stage_name}_s"] = t

        row["aruco_success"] = result.get("detection") is not None
        row["valid_grasp_proposal"] = result.get("plan") is not None
        row["lift_success"] = bool(result.get("lift_success", False))
        row["pick_and_place_success"] = bool(result.get("success", False))

        mask_full = result.get("mask_full")
        if mask_full is not None and "mask" in gt_snapshot:
            row["segmentation_iou"] = compute_iou(mask_full, gt_snapshot["mask"])

        points_robot = result.get("points_robot")
        if points_robot is not None and len(points_robot) > 0 and "state" in gt_snapshot:
            base_pos, base_mat = get_robot_base_transform(env)
            points_world = points_robot @ base_mat.T + base_pos
            pred_centroid = points_world.mean(axis=0)
            row["point_cloud_centroid_error_m"] = float(
                np.linalg.norm(pred_centroid - gt_snapshot["state"]["target_pos"])
            )

    except Exception as e:
        row["failure_reason"] = f"trial_exception:{e}"
    finally:
        env.close()

    return row


def run_evaluation(n_trials: int = 10, out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # Default sweep: vary seed only (pose/settle-noise variation from
    # placement randomness within the fixed [-0.12,0.12]x[-0.15,0.15]
    # range — see environment.py's placement_initializer). Extend this
    # list with num_distractors>0 / randomize_object_rotation=True entries
    # to exercise Phase 7's clutter/occlusion sweeps once distractor
    # objects and non-box target shapes are added.
    configs = [{"seed": i, "num_distractors": 0, "randomize_object_rotation": False} for i in range(n_trials)]

    rows = []
    for i, cfg in enumerate(configs):
        print(f"=== trial {i+1}/{len(configs)} (seed={cfg['seed']}) ===")
        row = run_trial(trial_id=i, **cfg)
        rows.append(row)
        print(f"  aruco={row['aruco_success']} iou={row['segmentation_iou']} "
              f"grasp={row['valid_grasp_proposal']} lift={row['lift_success']} "
              f"pick_place={row['pick_and_place_success']} time={row['total_time_s']}")

    fieldnames = sorted({k for r in rows for k in r.keys()})
    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved {csv_path}")

    _save_plots(rows, out_dir)

    n = len(rows)
    summary = {
        "n_trials": n,
        "aruco_success_rate": sum(r["aruco_success"] for r in rows) / n,
        "valid_grasp_rate": sum(r["valid_grasp_proposal"] for r in rows) / n,
        "lift_success_rate": sum(r["lift_success"] for r in rows) / n,
        "pick_and_place_success_rate": sum(r["pick_and_place_success"] for r in rows) / n,
        "mean_iou": float(np.nanmean([r["segmentation_iou"] for r in rows])),
        "mean_total_time_s": float(np.nanmean([r["total_time_s"] for r in rows])),
    }
    print("Summary:", summary)
    return rows, summary


def _save_plots(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    stages = ["aruco_success", "valid_grasp_proposal", "lift_success", "pick_and_place_success"]
    rates = [sum(r[s] for r in rows) / n for s in stages]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(stages, rates, color="#4C72B0")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success rate")
    ax.set_title(f"Pipeline stage success rates (n={n} trials)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "stage_success_rates.png"), dpi=120)
    plt.close(fig)

    ious = [r["segmentation_iou"] for r in rows if not np.isnan(r["segmentation_iou"])]
    if ious:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(ious, bins=min(10, max(3, len(ious))), color="#55A868")
        ax.set_xlabel("Segmentation IoU vs ground truth")
        ax.set_ylabel("Count")
        ax.set_title("FLIP mask IoU distribution")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "iou_histogram.png"), dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_evaluation(n_trials=n)
