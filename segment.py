#!/usr/bin/env python
"""
segment.py — ArUco-guided FLIP segmentation demo (Iphoreos)
=============================================================

Pipeline: webcam frame -> detect ArUco marker(s) -> marker center becomes the
object prompt -> crop a region-of-interest (ROI) around the marker -> feed the
ROI + a 2D Gaussian prompt (center + rough size) into a FLIP-Tiny/Small ONNX
model -> get a segmentation mask back -> overlay + record.

WHY A ROI CROP INSTEAD OF THE FULL FRAME
-----------------------------------------
FLIP predicts a mask value per requested pixel coordinate (`mask_coordinates`).
Asking for every pixel of a full webcam frame (e.g. 640x480 = 307,200 points)
on every frame is unnecessarily expensive and also makes the Gaussian sigma
harder to reason about. Instead we crop a square ROI around each marker
(sized relative to the marker's own pixel size), run FLIP only on that crop
at a fixed resolution, and paste the resulting mask back into the full frame.
This keeps runtime roughly constant regardless of camera resolution or how
many bottles are in frame.

HARD DEPENDENCY: the `flip_position` C extension
--------------------------------------------------
FLIP's multi-resolution patch sampling (picking which image patches around
the Gaussian prompt get fed to the encoder) is implemented in C, not ONNX.
You must build it once (see ext/setup.py in the FLIP repo) before this
script will run — see the README notes in this project folder, or the
message this script prints if the import fails.

USAGE
-----
    python segment.py --model small --dict DICT_6X6_250

Keys while the window is focused:
    q / ESC  quit
    r        start/stop recording an .mp4 to ./captures
    s        save a snapshot .png to ./captures
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Locate the FLIP repo + weights relative to this script.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FLIP_DIR = BASE_DIR / "FLIP-main"
WEIGHTS_DIR = FLIP_DIR / "model" / "weights"

try:
    import flip_position  # compiled C extension (built via FLIP-main/ext/setup.py)
except ImportError:
    sys.exit(
        "\n[FATAL] Could not import `flip_position`.\n"
        "This is the compiled C extension that samples FLIP's input patches;\n"
        "there is no pure-Python fallback. Build it once with:\n\n"
        f"    cd {FLIP_DIR / 'ext'}\n"
        "    python setup.py build install\n\n"
        "Note: the extension's build flags (-march=native, -fopenmp, -flto) are\n"
        "GCC-style and will NOT work with MSVC on native Windows. Easiest path:\n"
        "build and run this whole script inside WSL2 (Ubuntu) with build-essential\n"
        "installed, or a Linux/macOS machine. Webcam access from WSL2 needs\n"
        "usbipd-win USB passthrough for your laptop camera.\n"
    )

PATCH_SIZES = [1.0, 2.0, 4.0, 8.0, 16.0]  # matches min/max_patch_size in configs/flip-*.json


# ---------------------------------------------------------------------------
# FLIP ONNX wrapper (encoder + predictor pair, no torch needed at runtime)
# ---------------------------------------------------------------------------
class FlipSegmenter:
    def __init__(self, model_size: str, num_tokens: int = 512):
        enc_path = WEIGHTS_DIR / f"flip-encoder-{model_size}.onnx"
        pred_path = WEIGHTS_DIR / f"flip-predictor-{model_size}.onnx"
        if not enc_path.exists() or not pred_path.exists():
            sys.exit(f"[FATAL] Missing ONNX weights, expected:\n  {enc_path}\n  {pred_path}")

        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(enc_path), providers=providers)
        self.predictor = ort.InferenceSession(str(pred_path), providers=providers)
        self.num_tokens = num_tokens

    @staticmethod
    def _grid_coordinates(h: int, w: int) -> np.ndarray:
        """Reimplementation of FLIP's generate_grid_coordinates (torch -> numpy)."""
        xs = np.linspace(-1, 1, w, dtype=np.float32) * (w / 256.0)
        ys = np.linspace(-1, 1, h, dtype=np.float32) * (h / 256.0)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")  # both (h, w)
        return np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)

    def segment(self, roi_rgb: np.ndarray, mu_x: float, mu_y: float,
                sigma_x: float, sigma_y: float, rot_a: float = 1.0, rot_b: float = 0.0) -> np.ndarray:
        """
        roi_rgb: HxWx3 uint8 crop, channel order matching training (try RGB first;
                 if masks look nonsensical, flip to BGR — see note in build_position_prompt).
        mu_x, mu_y, sigma_x, sigma_y: normalized to [-1, 1] within this crop.
        Returns: float32 mask in [0, 1], shape (H, W) matching roi_rgb.
        """
        h, w = roi_rgb.shape[:2]
        position = np.array([mu_x, mu_y, sigma_x, sigma_y, rot_a, rot_b], dtype=np.float32)

        patches, coords, _indices, _lengths = flip_position.sample_continuous_patches(
            np.ascontiguousarray(roi_rgb, dtype=np.uint8), position, self.num_tokens, PATCH_SIZES
        )

        scaled_position = position.copy()
        scaled_position[0] *= w / 256.0
        scaled_position[1] *= h / 256.0
        scaled_position[2] *= w / 256.0
        scaled_position[3] *= h / 256.0
        scaled_position = scaled_position.reshape(1, 6)

        enc_inputs = {"position": scaled_position}
        for i, p in enumerate(PATCH_SIZES):
            key = f"p{int(p)}"
            patch_arr = patches[i].astype(np.float32).transpose(0, 3, 1, 2) / 255.0  # N,3,p,p
            enc_inputs[f"patches_{key}"] = patch_arr
            enc_inputs[f"coords_{key}"] = coords[i].astype(np.float32)

        k_cached, v_cached = self.encoder.run(["k_cached", "v_cached"], enc_inputs)

        mask_coords = self._grid_coordinates(h, w)
        pred_inputs = {
            "position": scaled_position,
            "mask_coordinates": mask_coords,
            "k_cached": k_cached,
            "v_cached": v_cached,
        }
        (mask_logits,) = self.predictor.run(["mask_logits"], pred_inputs)
        mask = 1.0 / (1.0 + np.exp(-mask_logits.reshape(-1)))  # sigmoid
        return mask.reshape(h, w).astype(np.float32)


# ---------------------------------------------------------------------------
# ArUco helpers
# ---------------------------------------------------------------------------
ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,  # default: matches chev.me/arucogen's default sheet
    "DICT_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def make_detector(dict_name: str):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def marker_center_and_size(corners: np.ndarray):
    """corners: (4, 2) array for one marker, in the order OpenCV returns."""
    center = corners.mean(axis=0)
    side_lengths = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    return center, float(np.mean(side_lengths))


def to_portrait(frame: np.ndarray, target_w: int, target_h: int, mode: str = "crop") -> np.ndarray:
    """
    Fit an arbitrary-aspect-ratio frame into a fixed target_w x target_h portrait
    canvas, regardless of the camera's actual capture resolution.

    mode="crop": center-crop the frame to the target aspect ratio first, then
                 resize to exact target size. No black bars, but the sides of
                 a landscape frame get cut off (fine here since the bottle
                 should already be roughly centered in frame).
    mode="pad":  resize to fit entirely within the target box, then letterbox
                 the remaining space with black bars. Nothing gets cropped,
                 but you lose screen space to bars.
    """
    h, w = frame.shape[:2]
    target_aspect = target_w / target_h
    src_aspect = w / h

    if mode == "crop":
        if src_aspect > target_aspect:
            # source is relatively wider than target -> crop left/right
            new_w = int(h * target_aspect)
            x0 = (w - new_w) // 2
            cropped = frame[:, x0:x0 + new_w]
        else:
            # source is relatively taller than target -> crop top/bottom
            new_h = int(w / target_aspect)
            y0 = (h - new_h) // 2
            cropped = frame[y0:y0 + new_h, :]
        return cv2.resize(cropped, (target_w, target_h))

    elif mode == "pad":
        if src_aspect > target_aspect:
            new_w = target_w
            new_h = int(target_w / src_aspect)
        else:
            new_h = target_h
            new_w = int(target_h * src_aspect)
        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y0 = (target_h - new_h) // 2
        x0 = (target_w - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    raise ValueError(f"Unknown fit mode: {mode}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tiny", "small", "middle"], default="small")
    ap.add_argument("--cam-index", type=int, default=0)
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--dict", choices=list(ARUCO_DICTS.keys()), default="DICT_6X6_250")
    ap.add_argument("--roi-scale-x", type=float, default=6.0,
                     help="ROI width = marker side length * roi-scale-x")
    ap.add_argument("--roi-scale-y", type=float, default=11.0,
                     help="ROI height = marker side length * roi-scale-y (bigger than roi-scale-x since "
                          "bottles are tall and narrow — a square crop wastes width and shortchanges height)")
    ap.add_argument("--roi-offset-y", type=float, default=0.35,
                     help="shift the ROI center below the marker, as a fraction of ROI half-height. "
                          "Markers usually aren't at the true vertical center of the object (more bottle "
                          "below the marker than above it), so a positive offset reclaims that space.")
    ap.add_argument("--full-frame", action="store_true",
                     help="skip ROI cropping entirely and run FLIP on the whole camera frame instead. "
                          "Costs the same compute (mask resolution is fixed by --roi-size either way), "
                          "but the object occupies a smaller fraction of that fixed budget, so edges are "
                          "usually coarser than a well-fitted ROI. --sigma-x/-y need to be much smaller "
                          "in this mode since the object is now tiny relative to the whole frame.")
    ap.add_argument("--roi-size", type=int, default=256, help="ROI is resized to this square before FLIP")
    ap.add_argument("--sigma-x", type=float, default=0.28,
                     help="normalized Gaussian sigma (horizontal) within the ROI")
    ap.add_argument("--sigma-y", type=float, default=0.42,
                     help="normalized Gaussian sigma (vertical) within the ROI — bottles are tall and "
                          "narrow, so this defaults larger than sigma-x")
    ap.add_argument("--num-tokens", type=int, default=512)
    ap.add_argument("--mask-thresh", type=float, default=0.5,
                     help="only used with --hard-mask; ignored otherwise")
    ap.add_argument("--hard-mask", action="store_true",
                     help="use a thresholded binary mask (old behavior, blocky edges) instead of the "
                          "default smooth/anti-aliased soft mask")
    ap.add_argument("--mask-alpha", type=float, default=0.5)
    ap.add_argument("--print-every", type=int, default=15,
                     help="print detected marker id/center to the console every N frames (0 disables)")
    ap.add_argument("--portrait", action="store_true",
                     help="display/record in a fixed mobile-portrait window, independent of camera resolution")
    ap.add_argument("--portrait-size", default="405x720",
                     help="WxH of the portrait canvas actually recorded/saved, e.g. 405x720 (9:16, low-res)")
    ap.add_argument("--fit-mode", choices=["crop", "pad"], default="crop",
                     help="crop: fill the portrait frame, cutting off the sides (default). "
                          "pad: show the full frame with black letterbox bars.")
    ap.add_argument("--preview-max-height", type=int, default=800,
                     help="cap the live preview WINDOW height (pixels) so it fits your screen; "
                          "doesn't affect the resolution actually recorded/saved")
    ap.add_argument("--preview-max-width", type=int, default=1200,
                     help="cap the live preview WINDOW width (pixels)")
    ap.add_argument("--out-dir", default=str(BASE_DIR / "captures"))
    args = ap.parse_args()

    portrait_w, portrait_h = (int(v) for v in args.portrait_size.lower().split("x"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = make_detector(args.dict)
    flip = FlipSegmenter(args.model, num_tokens=args.num_tokens)

    window_name = "Iphoreos - ArUco + FLIP segmentation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_sized = False  # set once we know the real output frame's aspect ratio

    # CAP_DSHOW avoids Windows Media Foundation's hardware-accelerated capture
    # path, which has known driver bugs with some external webcams/capture
    # cards (can hard-hang the GPU driver). DirectShow is the safer default here.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.cam_index, backend)
    if not cap.isOpened():
        sys.exit(f"[FATAL] Could not open camera index {args.cam_index}")

    # Explicitly cap the requested resolution — some webcams/capture devices
    # default to a very high native resolution (e.g. 4K) that can overload
    # capture + display pipelines. 1280x720 is plenty for this pipeline
    # since we crop small ROIs around each marker anyway.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)

    writer = None
    recording = False
    MASK_COLOR = (0, 255, 0)   # BGR green, fixed for every marker
    CENTER_COLOR = (0, 0, 255)  # BGR red, fixed for every marker
    frame_idx = 0
    fps = 0.0
    fps_smoothing = 0.9  # exponential moving average, higher = smoother/slower to react
    last_t = time.time()

    print("Keys: q/ESC quit | r start/stop recording | s snapshot")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] frame grab failed")
            break
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            inst_fps = 1.0 / dt
            fps = fps_smoothing * fps + (1 - fps_smoothing) * inst_fps if fps > 0 else inst_fps
        h, w = frame.shape[:2]
        overlay = frame.copy()
        frame_idx += 1
        do_print = args.print_every > 0 and frame_idx % args.print_every == 0

        corners_list, ids, _rejected = detector.detectMarkers(frame)

        if ids is not None:
            ids_flat = np.asarray(ids).reshape(-1)  # some cv2 builds return (N,) instead of (N,1)
            for i, marker_corners in enumerate(corners_list):
                marker_id = int(ids_flat[i])
                try:
                    pts = marker_corners.reshape(4, 2)
                    center, side_px = marker_center_and_size(pts)
                    cx, cy = center

                    if do_print:
                        print(f"marker id={marker_id} center=({cx:.1f}, {cy:.1f})")

                    if args.full_frame:
                        x0, y0, x1, y1 = 0, 0, w, h
                    else:
                        half_w = max(side_px * args.roi_scale_x / 2.0, 20)
                        half_h = max(side_px * args.roi_scale_y / 2.0, 20)
                        # bias the crop center downward so more of it covers the
                        # base of the bottle, since the marker sits well above
                        # the object's true vertical midpoint
                        cy_shifted = cy + args.roi_offset_y * half_h
                        x0 = int(np.clip(cx - half_w, 0, w - 1))
                        x1 = int(np.clip(cx + half_w, 0, w))
                        y0 = int(np.clip(cy_shifted - half_h, 0, h - 1))
                        y1 = int(np.clip(cy_shifted + half_h, 0, h))
                        if x1 - x0 < 10 or y1 - y0 < 10:
                            continue

                    roi_bgr = frame[y0:y1, x0:x1]
                    # Resize preserving the crop's real aspect ratio (FLIP doesn't
                    # require a square input — forcing one would squish a tall
                    # narrow bottle sideways). roi_size caps the longer side.
                    crop_h, crop_w = roi_bgr.shape[:2]
                    if crop_h >= crop_w:
                        target_h = args.roi_size
                        target_w = max(8, round(crop_w * (args.roi_size / crop_h)))
                    else:
                        target_w = args.roi_size
                        target_h = max(8, round(crop_h * (args.roi_size / crop_w)))
                    roi_resized = cv2.resize(roi_bgr, (target_w, target_h))
                    # NOTE: FLIP's training pipeline naming suggests RGB; if masks look
                    # inverted/nonsensical, swap this to roi_resized directly (BGR).
                    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)

                    # marker center within the crop, normalized to [-1, 1]
                    rel_x = (cx - x0) / (x1 - x0) * 2 - 1
                    rel_y = (cy - y0) / (y1 - y0) * 2 - 1

                    mask = flip.segment(roi_rgb, rel_x, rel_y, args.sigma_x, args.sigma_y)

                    if args.hard_mask:
                        mask_bin = (mask > args.mask_thresh).astype(np.float32)
                        # NEAREST here would blow the binary blocks up into visible
                        # jaggies; still resize the (now 0/1) values with linear so
                        # the upsample at least interpolates between them cleanly.
                        alpha_full = cv2.resize(mask_bin, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                    else:
                        # Default: resize the *continuous* probability mask (not a
                        # thresholded one) with linear interpolation, then use it
                        # directly as a soft per-pixel blend weight. This is what
                        # actually fixes the blocky/jagged edges — smoothing a
                        # binary mask after the fact can't recover detail that
                        # thresholding already destroyed.
                        alpha_full = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)

                    alpha = np.clip(alpha_full * args.mask_alpha, 0, 1)[..., None].astype(np.float32)
                    region = overlay[y0:y1, x0:x1].astype(np.float32)
                    colored = np.full_like(region, MASK_COLOR, dtype=np.float32)
                    blended = region * (1 - alpha) + colored * alpha
                    overlay[y0:y1, x0:x1] = blended.astype(np.uint8)

                    cv2.circle(overlay, (int(cx), int(cy)), 5, CENTER_COLOR, -1)
                except Exception as e:
                    # Don't let one bad marker/frame kill the whole webcam session —
                    # print the error once and keep going.
                    print(f"[WARN] segmentation failed for marker {marker_id}: {e}")

        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if recording:
            cv2.putText(overlay, "REC", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # All ArUco detection + FLIP inference above ran on the native camera
        # frame (so nothing gets cut off before the model sees it). The
        # portrait reshape only happens here, right before display/record,
        # so the output window/file is a fixed mobile size regardless of
        # what resolution the camera actually captures at.
        if args.portrait:
            display_frame = to_portrait(overlay, portrait_w, portrait_h, args.fit_mode)
        else:
            display_frame = overlay

        if not window_sized:
            # Size the window to display_frame's ACTUAL aspect ratio, scaled down
            # to fit the screen. WINDOW_NORMAL stretches content non-uniformly to
            # fill whatever shape the window happens to be — sizing it correctly
            # up front (once) is what avoids the vertical-squeeze distortion.
            dh, dw = display_frame.shape[:2]
            scale = min(args.preview_max_width / dw, args.preview_max_height / dh, 1.0)
            cv2.resizeWindow(window_name, int(dw * scale), int(dh * scale))
            window_sized = True

        cv2.imshow(window_name, display_frame)

        if recording and writer is not None:
            writer.write(display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            if not recording:
                ts = time.strftime("%Y%m%d-%H%M%S")
                path = str(out_dir / f"capture-{ts}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                rec_size = (portrait_w, portrait_h) if args.portrait else (w, h)
                writer = cv2.VideoWriter(path, fourcc, 20.0, rec_size)
                recording = True
                print(f"[REC] recording to {path}")
            else:
                recording = False
                if writer is not None:
                    writer.release()
                    writer = None
                print("[REC] stopped")
        elif key == ord("s"):
            ts = time.strftime("%Y%m%d-%H%M%S")
            path = str(out_dir / f"snapshot-{ts}.png")
            cv2.imwrite(path, display_frame)
            print(f"[SNAP] saved {path}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
