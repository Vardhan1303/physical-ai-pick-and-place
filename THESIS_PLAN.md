# ArUco-Guided Point-Prompted Object Segmentation for Class-Agnostic Robotic Grasping in MuJoCo

## Implementation status: all 8 phases done and verified

`environment.py`, `aruco_prompt.py`, `flip_segmenter.py`, `geometry.py`, `grasp_planner.py`, `robot_controller.py`, `pipeline.py`, and `evaluation.py` are all written and independently verified, including a full closed-loop run: real ArUco detection -> real FLIP ONNX segmentation (with the actual `flip_position` C extension built and running, not stubbed) -> depth-based point cloud -> grasp planning -> real robosuite OSC_POSE execution on the Panda -> the object physically placed in the destination bin. See `README_thesis.md` for how to run each phase, expected output, known limitations, and troubleshooting (several real bugs were found and fixed while building this — grayscale-PNG texture loading, MuJoCo 2D-texture face mapping, a mujoco/robosuite version incompatibility, a quaternion double-cover control bug, an unreachable-wrist grasp yaw, and a bin-height/release-point calculation error — each documented at its fix site and summarized in the README's troubleshooting section).

Replaces the earlier raw-MuJoCo-XML pick-and-place demo that used to live in this repo (`pick_and_place_flip.py`, `manipulation.py`, `ik_utils.py`, `shape_utils.py`, `assets/`) with a robosuite-based architecture built for the research question below. That old pipeline has since been removed from the repo entirely — it is fully superseded, not kept alongside this one. See `README.md` for installation and usage.

**Research question:** Can an ArUco-derived point prompt allow FLIP to segment previously unseen tabletop objects accurately enough for closed-loop RGB-D robotic grasping, compared with a category-trained YOLO segmentation baseline?

**Core constraint, stated precisely because it drives every module below:** ArUco selects *which* object to grasp (a point prompt), never *what* it is. FLIP's mask supplies the object's extent; depth supplies its visible 3D geometry. MuJoCo ground truth (pose, size, segmentation, class) may only be read by `evaluation.py`, never by `aruco_prompt.py`, `flip_segmenter.py`, `geometry.py`, or `grasp_planner.py`.

## 1. Project folder structure

```
physical-ai-pick-and-place/
├── requirements.txt              # pipeline deps (mujoco==3.3.0 pinned, see file header)
├── THESIS_PLAN.md                # this document
├── README.md                     # installation, usage, results, credits
│
├── environment.py                # Phase 1 — DONE, verified in-sandbox
├── aruco_prompt.py               # Phase 2
├── flip_segmenter.py             # Phase 3
├── geometry.py                   # Phase 4
├── grasp_planner.py              # Phase 5
├── robot_controller.py           # Phase 6
├── pipeline.py                   # Phase 7 (orchestrates 1-6)
├── evaluation.py                 # Phase 7/8 (ground truth, metrics)
│
├── runs/                         # pipeline.py output: per-run video + stage images (created at runtime)
├── eval_results/                 # evaluation.py output: CSV + plots (created at runtime)
│
├── segment.py                    # EXISTING — FlipSegmenter wrapper, reused by flip_segmenter.py
└── markers/                      # ArUco marker image(s) used by the scene
```

Each of the 8 new modules is independently importable and testable (per the constraint that every module must be independently testable) — none of them do `if __name__ == "__main__": run_everything()`; each has its own minimal self-test block instead, following `environment.py`'s pattern.

## 2. Dependency list

See `requirements.txt`. Headline finding: robosuite 1.5.2's OSC controller breaks on MuJoCo ≥ ~3.9 (`AttributeError: 'MjData' object has no attribute 'qM'` — the attribute was renamed to `.M`), reproduced directly in-sandbox. Pin `mujoco==3.3.0` for this pipeline.

## 3. Implementation plan

### Phase 1 — `environment.py` (DONE)

`PickPlaceEnv(ManipulationEnv)`, modeled directly on robosuite's own `Lift` environment (same `_load_model`/`_setup_references`/`_reset_internal` structure — not invented). Contains:

- `TableArena` + Panda robot, one `CylinderObject` target (an upright, bottle-like round body with an ArUco decal on its curved side, not a flat top face — nothing here hard-codes object shape as a perception assumption, it's only a scene-construction choice), 0 distractors by default, a fixed destination-bin `BoxObject` with no free joint.
- `side_oblique_camera`: added directly to `TableArena.worldbody` (a real `xml.etree.ElementTree.Element`, the same mechanism robosuite's own `frontview`/`agentview`/`birdview`/`sideview` cameras use). Position `[-0.45, -0.45, 1.35]` relative to world origin (table top at `z=0.8`), looking at `[0, 0, 0.85]`, computed via a `look_at_quat()` helper using `mujoco.mju_mat2Quat` — elevation **40.83°**, inside the required 35-50° band.
- RGB/depth/intrinsics/extrinsics accessors backed directly by `robosuite.utils.camera_utils` (`get_real_depth_map`, `get_camera_intrinsic_matrix`, `get_camera_extrinsic_matrix`) — no reimplementation.
- `get_ground_truth_state()` / `get_ground_truth_segmentation()` — evaluation-only, clearly separated from the perception-facing accessors.
- Deterministic seeding via robosuite's own `seed=` kwarg → `self.rng`, passed into `CylinderObject(..., rng=self.rng)` and `UniformRandomSampler(..., rng=self.rng)`.

**Real bug found and fixed while verifying this in-sandbox:** robosuite defaults to OpenGL image convention (row 0 = image bottom). A raw render came out vertically flipped relative to what OpenCV/ArUco and `camera_utils`' intrinsics (principal point from top-left, v increasing downward) both expect. Fixed with the documented `robosuite.macros.IMAGE_CONVENTION = "opencv"` macro, set at import time before any camera sensor is built — confirmed via robosuite's own source (`mjcf_utils.IMAGE_CONVENTION_MAPPING = {"opengl": 1, "opencv": -1}`), not a guess.

**Object update, 2026-08-07: box placeholder replaced with a bottle-like cylinder.** `environment.py`'s target switched from `BoxObject` to `CylinderObject` (radius/half-length size, same `PrimitiveObject` family, same procedural-geom mechanism), with the ArUco decal moved from the box's flat top face to a thin flat patch tangent to the cylinder's curved side — matching how a real bottle carries a marker/label on its body rather than its cap. Two things had to be gotten right, both confirmed by an actual render + `cv2.aruco` detection pass rather than assumed from the geometry alone:
- The decal is still its own independent thin box geom (not a texture wrapped around the cylinder's own surface), so the earlier finding — MuJoCo's 2D-texture mapping on a box only renders correctly on the face normal to that geom's OWN local `quat`-defined Z axis — still applies; it just needed a `quat` that points local Z radially outward at the chosen azimuth instead of straight up (`environment.py::radial_decal_quat`).
- That outward azimuth must actually point toward the camera. A first attempt hardcoded straight -Y and rendered as a heavily foreshortened, darkly-shaded parallelogram that ArUco couldn't decode — `side_oblique_camera` sits at an equal diagonal offset in -X and -Y (`[-0.45, -0.45]`, i.e. ~225° azimuth, not 270°/-Y), so a decal facing pure -Y was ~45° off the camera's actual viewing direction. Fixed by deriving the decal's azimuth directly from the same `_CAMERA_XY_OFFSET` the camera itself uses, so the two can't drift out of sync.

Net effect on downstream phases: `grasp_planner.py` needed no changes at all (its `cv2.minAreaRect`-on-point-cloud footprint logic is already shape-agnostic), and FLIP's segmentation quality *improved* substantially on the cylinder versus the box placeholder (mean IoU against ground truth rose from roughly 0.2 to roughly 0.77 in evaluation sweeps) — see `README.md`'s Results section.

### Phase 2 — `aruco_prompt.py`

`cv2.aruco` (the same `DICT_6X6_250` dictionary and detector pattern already used by `segment.py`/`pick_and_place_flip.py` in the old pipeline). Input: RGB frame from `env.get_camera_rgbd()`. Output: `{marker_id, corners, center_px}` or a typed failure (`NoMarkerFound`, `MultipleMarkersFound`, `MarkerPartiallyVisible`, `MarkerOutOfFrame`) — validated via corner-count and in-frame bounding checks, not just "did detectMarkers return something."

### Phase 3 — `flip_segmenter.py`

Thin wrapper around the EXISTING `segment.py::FlipSegmenter` (already proven working against the real ONNX weights on the user's Windows machine in the old pipeline) — not reimplemented. Input: RGB + one positive `(x, y)` focus point in FLIP's normalized coordinate convention (same `rel_x`/`rel_y` mapping `pick_and_place_flip.py::perceive` already uses). Output: binary mask + FLIP's own confidence/diagnostics if the underlying model exposes them. Optional post-FLIP cleanup (largest connected component containing the prompt point, hole filling, tiny-component removal) applied only after inference, never influencing FLIP's own inference call. Saves per-call debug visualizations (RGB, prompt point, raw mask, cleaned mask, overlay) to `runs/<run_id>/stageN_flip/`.

### Phase 4 — `geometry.py`

Masked-depth-pixels → 3D point cloud via `env.get_camera_intrinsics()` (pixel → camera frame), then camera→robot-base via `env.get_camera_extrinsics()`. Invalid-depth removal (0 / inf / out-of-clip-range) and RANSAC or height-threshold tabletop-plane removal (table height is a robot-frame constant computed from the *robot base offset*, not from any per-object ground truth). Returns a filtered, target-only point cloud. Never touches `get_ground_truth_state()`.

### Phase 5 — `grasp_planner.py`

Category-agnostic top-down parallel-jaw planner. Footprint = masked point cloud projected onto the table plane (convex hull or occupancy grid). Orientation = PCA on the 2D footprint or `cv2.minAreaRect`-equivalent on the projected hull (same principle `shape_utils.grasp_from_mask` already uses in the old pipeline for 2D masks, generalized to a 3D-projected footprint here). Grasp width checked against the Panda's gripper limits (`0.0` to `0.08m` — from `panda.xml`'s finger joint range, already vendored in this repo). Outputs pre-grasp/grasp/lift/place poses in robot-base frame. Final closing distance is contact-determined at execution time (Phase 6), not purely geometric.

### Phase 6 — `robot_controller.py`

Uses robosuite's composite `OSC_POSE` controller (`suite.load_part_controller_config` + `refactor_composite_controller_config`, the same real loading path found in `robosuite/controllers/config/robots/default_panda.json` — not invented). Action space: 6-dim delta pose + 1-dim gripper, per `OSC_POSE`'s documented convention. Sequence: pre-grasp → descend → close-until-contact (via `env.robots[0].composite_controller`'s gripper action, monitored against a force/contact signal or a closing-distance timeout) → lift → transport → open. Includes safety heights, per-stage timeouts, and a simple recovery path (abort + retract) on IK/contact failure. Deterministic in this initial version — no retries with randomized perturbation yet.

### Phase 7 — `pipeline.py`

Orchestrates Phases 1-6 end to end: RGB-D → `aruco_prompt` → `flip_segmenter` → `geometry` → `grasp_planner` → `robot_controller` → success check (object height above table + inside destination bin footprint, read from the *executed* robot/object state, not from ground truth object identity). Stage-by-stage logs (`[STAGE] aruco: marker_id=3 center=(412,207)`, etc., matching the existing pipeline's logging style in `pick_and_place_flip.py::perceive`). Saves a video (reusing the `frame_callback` pattern already built for `manipulation.py::run_pick_and_place`) plus per-stage images to `runs/<run_id>/`.

### Phase 8 — `evaluation.py`

Ground-truth-gated experiment runner — the ONLY module allowed to call `env.get_ground_truth_state()` / `get_ground_truth_segmentation()`. Controlled sweeps over pose/scale/viewpoint/lighting/occlusion/clutter/unseen-shape (driven by `environment.py`'s existing `num_distractors` and `UniformRandomSampler` randomization, extended with scale/lighting parameters as needed). Metrics: ArUco target-selection success rate, FLIP segmentation IoU vs. ground-truth mask, point-cloud completeness/error vs. ground-truth mesh, valid-grasp-proposal rate, successful-lift rate, successful-pick-and-place rate, per-stage and total latency. Saves CSV + matplotlib plots to `eval_results/`. Phase 8 also adds the YOLOv8-seg baseline (`ultralytics`) sharing the same camera/point-cloud/grasp-planner/evaluation code path as FLIP, differing only in the segmentation module.

## 4. Phase 1 implementation

Written and verified: `environment.py` (see repo root). Self-test block at the bottom (`python environment.py`) instantiates the env, steps 10 physics steps, captures RGB+depth from `side_oblique_camera`, prints intrinsics/extrinsics, and saves `phase1_output/side_oblique_rgb.png` + `side_oblique_depth.npy`.

## 5. Verification instructions (do this before starting Phase 2)

1. **Fresh environment.** Create a virtualenv and `pip install -r requirements.txt` (note the `mujoco==3.3.0` pin, see above).
2. **Run the self-test:** `python environment.py` from the repo root. Expect it to print RGB shape `(480, 640, 3)`, a depth min/max in a physically sane range (roughly 0.3-5m for this scene), a 3x3 intrinsics matrix, and a 4x4 extrinsics matrix — then `Saved phase1_output/side_oblique_rgb.png and side_oblique_depth.npy`.
3. **Visually inspect `phase1_output/side_oblique_rgb.png`.** You should see: the table surface, the red target cylinder standing on it, the gray destination bin, and the Panda gripper entering frame from the upper-left — all right-side-up. (This caught a real bug during development: without `robosuite.macros.IMAGE_CONVENTION = "opencv"`, the same scene renders vertically flipped — table underside "up", objects appearing to float against a bright void. If your image looks like that, the macro isn't taking effect before env construction.)
4. **Elevation sanity check:** the camera should be looking down at roughly 35-50° above the table plane — enough to see both the target cylinder's curved side (where the ArUco decal sits) and part of its top in the same frame (confirm both are visible in the saved image).
5. **Determinism check:** run `python environment.py` twice; since Phase 1 uses `num_distractors=0` and a single fixed-size target, the object's rendered position should be visually identical between runs (seed=0 both times). This confirms seeding is wired correctly before Phase 2 relies on it for reproducible marker placement.
6. Once 1-5 pass, proceed to Phase 2 (`aruco_prompt.py`): add an ArUco texture to `target_object`'s camera-facing surface and confirm `cv2.aruco` detects it in the Phase 1 camera's RGB output before writing any FLIP-facing code.
