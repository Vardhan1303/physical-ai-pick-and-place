# ArUco-Guided Point-Prompted Object Segmentation for Class-Agnostic Robotic Grasping in MuJoCo

Full pipeline status: **all 8 modules implemented and verified end-to-end**, including a real Panda pick-and-place run in simulation (ArUco detected -> real FLIP ONNX segmentation -> point cloud -> grasp plan -> robosuite OSC execution -> object placed in the bin). See `THESIS_PLAN.md` for the architecture and the per-module design rationale; this file covers installation and how to run each phase.

## Installation

This pipeline needs a **separate virtual environment** from this repo's older raw-MuJoCo demo (`pick_and_place_flip.py` etc.) — the two pin incompatible MuJoCo versions on purpose (see `requirements_thesis.txt`'s header comment for the exact `AttributeError` this avoids).

```
python -m venv .venv_thesis
.venv_thesis\Scripts\activate        (Windows)
pip install -r requirements_thesis.txt
```

### FLIP weights + the `flip_position` C extension

`flip_segmenter.py` wraps the existing `segment.py::FlipSegmenter`, which needs:
1. The FLIP ONNX weights (`flip-encoder-<size>.onnx`, `flip-predictor-<size>.onnx`). By default this repo's scripts point `FLIP_WEIGHTS_DIR` at `V:\projects\Iphoreos\FLIP-main\model\weights` (the Iphoreos project's copy) — override the `FLIP_WEIGHTS_DIR` environment variable if yours live elsewhere.
2. The compiled `flip_position` C extension (samples FLIP's multi-resolution input patches — no pure-Python fallback exists). Build it once:
   ```
   cd <path-to-FLIP-main>/ext
   python setup.py build_ext --inplace
   ```
   then make sure the resulting `flip_position*.so`/`.pyd` is importable (either run scripts from that directory, add it to `PYTHONPATH`, or copy the built extension into your venv's `site-packages`). Despite the extension's own comment suggesting Linux/WSL2 for building, it built and ran cleanly with plain GCC in this project's Linux verification sandbox — the friction the comment describes is specifically about MSVC on native Windows, not Linux in general.

### Rendering backend

MuJoCo needs a working OpenGL context for offscreen rendering. On Windows this is normally automatic. If you ever run this headless on Linux and hit `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` or similar, that means EGL/OSMesa aren't available — use `MUJOCO_GL=glx` under `Xvfb` instead (what this project's own sandbox verification used).

## Running each phase

All commands run from the repo root, inside the thesis venv.

```
python environment.py       # Phase 1: scene + camera sanity check -> phase1_output/
python aruco_prompt.py      # Phase 2: ArUco detection on a live render -> phase2_output/
python flip_segmenter.py    # Phase 3: real FLIP segmentation from the ArUco prompt -> phase3_output/
python geometry.py          # Phase 4: masked depth -> robot-frame point cloud -> phase4_output/
python grasp_planner.py     # Phase 5: grasp pose planning (prints the plan)
python robot_controller.py  # Phase 6: executes ONE grasp plan on the Panda (prints per-stage results)
python pipeline.py          # Phase 7: full closed loop -> runs/<timestamp>/ (rgb, aruco/flip debug images, video, log.json)
python evaluation.py N      # Phase 8: N controlled trials -> eval_results/metrics.csv + plots
```

Each module is independently runnable and only depends on the modules before it in this list (`geometry.py` needs Phases 1-3's outputs, etc.) — none of them require the others to have been run first as separate processes; each self-test re-derives what it needs by calling the earlier modules directly.

## Expected output (Phase 1, sanity check)

`python environment.py` should print an RGB shape, a depth range roughly 0.3-5m, a 3x3 intrinsics matrix, and a 4x4 extrinsics matrix, then save `phase1_output/side_oblique_rgb.png` — open it and confirm: table surface, target object, destination bin, and the Panda gripper are all visible, right-side-up, from an oblique elevated angle (not upside down — see Troubleshooting).

## Expected output (Phase 7, full pipeline)

`python pipeline.py` prints one `[STAGE:OK]`/`[STAGE:FAIL]` line per stage (reset, capture, aruco_prompt, flip_segmenter, geometry, grasp_planner, then one line per robot_controller sub-stage, then outcome_check), then a JSON summary with `"success": true/false`. A successful run's `runs/<timestamp>/` directory contains:
- `00_rgb.png`, `01_aruco.png` — capture + detected marker
- `02_flip_00_rgb_prompt.png` / `_01_raw_mask_roi.png` / `_02_cleaned_mask_roi.png` / `_03_overlay_full.png` — FLIP's segmentation debug trail
- `03_execution.mp4` — the arm executing the plan
- `log.json` — the full stage-by-stage log

## Known limitations (don't oversell these)

- **Single camera = partial 2.5D geometry.** The point cloud only ever contains what `side_oblique_camera` can see in one frame — for the current box placeholder, that's mostly the top face. Grasp height is therefore an *estimate* (table height + half the visible top's height above it), not a measurement, and the system relies on `robot_controller.py`'s close-until-contact / stop-on-stall behavior to make up the difference, exactly as specified. It does not, and cannot from one camera, reconstruct the object's full hidden geometry.
- **FLIP under-segmentation on this synthetic placeholder.** In verification runs, FLIP's mask covered mostly the marker's immediate neighborhood rather than the object's full visible extent (confirmed via the ground-truth IoU in `evaluation.py`, typically ~0.2 on the current placeholder box). This is a sigma/ROI-scale calibration question (`flip_segmenter.py`'s `DEFAULT_SIGMA`, `ROI_SCALE`), not a wiring bug — worth tuning once real bottle/can/cup meshes replace the placeholder box, since FLIP was trained on real-object imagery, not a flat-colored synthetic box.
- **ArUco detection is placement-sensitive.** Across a small evaluation sweep (3 seeds), the marker was undetected in 2/3 trials purely from object-placement variation within the current sampling range — a real, honest finding from `evaluation.py`, not a bug being hidden. Expanding `evaluation.py`'s trial count and correlating failures against object XY position would be the natural next diagnostic step before Phase 7's clutter/occlusion sweeps.
- **Fixed target rotation (yaw=0) through Phase 6.** The target's ArUco decal sits on its top face and only renders correctly when un-rotated (`environment.py`'s `randomize_object_rotation=False` default) — a real, documented MuJoCo texture-mapping limitation found during Phase 2 verification (see `environment.py::_add_aruco_decal`'s docstring), not an arbitrary restriction.

## Troubleshooting

- **Rendered image looks upside down** (table underside "up", objects floating against a bright void): `robosuite.macros.IMAGE_CONVENTION` wasn't set to `"opencv"` before the environment was constructed. It must be set at import time, before `PickPlaceEnv(...)` — see the top of `environment.py`.
- **`AttributeError: 'MjData' object has no attribute 'qM'`**: wrong MuJoCo version. This pipeline needs `mujoco==3.3.0` (or another version that still exposes `MjData.qM`), not whatever `pip install mujoco` resolves to by default. See `requirements_thesis.txt`.
- **ArUco marker renders as a flat gray square with no visible pattern**: either (a) the marker PNG was saved as grayscale instead of RGB (MuJoCo's texture loader silently fails on grayscale-only PNGs), or (b) the decal geom is on a side face instead of the top face (this MuJoCo build's `type="2d"` texture mapping only resolves correctly on the face normal to the geom's local Z axis). Both are explained and fixed in `environment.py::_add_aruco_decal`'s docstring.
- **A robot motion stage stalls with a large, non-shrinking position error**: check whether it's a *descend* stage (grasp approach or place release) — those are expected to sometimes stop early due to real contact (the object landing on the bin's surface, for instance), which `robot_controller.py::move_to_pose`'s `stop_on_stall` handles. If it's a *large lateral/reconfiguration* move (pregrasp, transport) stalling far from the target instead, check the planned grasp yaw — this pipeline normalizes yaw into `(-90, 90]` degrees in `grasp_planner.py` specifically because un-normalized values near 180 degrees drove the Panda into an unreachable wrist configuration during verification.
