#!/usr/bin/env python
"""
flip_segmenter.py — Phase 3: point-prompted FLIP segmentation.

Thin wrapper around the EXISTING `segment.py::FlipSegmenter` (the real
ONNX-based encoder/predictor pair + `flip_position` C extension, already
proven against real weights — see segment.py's own docstring and the
pipeline's prior use in pick_and_place_flip.py). This module does not
reimplement FLIP inference; it only adds:
  1. ROI cropping around the ArUco prompt point (same idea as
     pick_and_place_flip.py::perceive — crop scaled to the marker's own
     pixel size, run FLIP on the crop, paste the mask back)
  2. optional post-inference mask cleanup (largest connected component
     containing the prompt, hole filling, tiny-component removal) —
     applied strictly AFTER FLIP's own inference call, never influencing it
  3. debug visualization saving (RGB / prompt point / raw mask / cleaned
     mask / overlay)

Input contract: one RGB frame + one positive (u, v) focus point (in FULL
FRAME pixel coordinates — as produced by aruco_prompt.get_target_prompt).
Output: a full-frame-sized binary mask (H, W) plus the raw ROI-local soft
mask for diagnostics. This module never reads MuJoCo ground truth.

Standalone self-test:
    python flip_segmenter.py
Chains environment.py -> aruco_prompt.py -> this module on one rendered
frame and saves debug images to phase3_output/.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Same default-weights-location convention pick_and_place_flip.py already
# uses: point at the Iphoreos project's FLIP-main (where the real ONNX
# weights + the `ext/` C-extension source actually live), overridable via
# the FLIP_WEIGHTS_DIR env var. Must be set before importing segment.py,
# since it reads this at import time.
os.environ.setdefault("FLIP_WEIGHTS_DIR", r"V:\projects\Iphoreos\FLIP-main\model\weights")

from segment import FlipSegmenter as _RawFlipSegmenter  # noqa: E402  (env var must be set first)

ROI_MIN_HALF_PX = 45
ROI_SCALE = 2.3          # ROI half-size = max(ROI_MIN_HALF_PX, marker_side_px * ROI_SCALE)
ROI_RESIZE = 256          # ROI is resized to this square before FLIP, matching segment.py's default
DEFAULT_SIGMA = 0.35


@dataclass
class SegmentationResult:
    mask_full: np.ndarray        # (H, W) uint8, 0/255 — full-frame-sized binary mask
    mask_roi_soft: np.ndarray    # (h, w) float32 in [0,1] — FLIP's raw ROI-local output, for diagnostics
    roi_bbox: tuple               # (x0, y0, x1, y1) in full-frame pixel coords
    confidence: float             # mean soft-mask value inside the final binary mask (simple proxy;
                                   # FLIP's ONNX heads here don't expose a separate calibrated confidence head)


class FlipTargetSegmenter:
    def __init__(self, model_size: str = "small", num_tokens: int = 512):
        self._flip = _RawFlipSegmenter(model_size, num_tokens=num_tokens)
        self.model_size = model_size

    def segment_from_prompt(
        self,
        rgb_full: np.ndarray,
        prompt_px: tuple,
        marker_side_px: Optional[float] = None,
        sigma_x: float = DEFAULT_SIGMA,
        sigma_y: float = DEFAULT_SIGMA,
        cleanup: bool = True,
        debug_dir: Optional[str] = None,
        debug_tag: str = "flip",
    ) -> SegmentationResult:
        h, w = rgb_full.shape[:2]
        cx, cy = prompt_px

        half = ROI_MIN_HALF_PX if marker_side_px is None else max(ROI_MIN_HALF_PX, marker_side_px * ROI_SCALE)
        x0 = int(np.clip(cx - half, 0, w - 1))
        x1 = int(np.clip(cx + half, 0, w))
        y0 = int(np.clip(cy - half, 0, h - 1))
        y1 = int(np.clip(cy + half, 0, h))
        if x1 - x0 < 10 or y1 - y0 < 10:
            raise ValueError(f"ROI too small around prompt {prompt_px}: ({x0},{y0})-({x1},{y1})")

        roi_rgb = rgb_full[y0:y1, x0:x1]
        roi_h, roi_w = roi_rgb.shape[:2]
        # Preserve aspect ratio (same reasoning as segment.py: forcing a
        # square would squish a non-square crop).
        if roi_h >= roi_w:
            target_h = ROI_RESIZE
            target_w = max(8, round(roi_w * (ROI_RESIZE / roi_h)))
        else:
            target_w = ROI_RESIZE
            target_h = max(8, round(roi_h * (ROI_RESIZE / roi_w)))
        roi_resized = cv2.resize(roi_rgb, (target_w, target_h))

        rel_x = (cx - x0) / (x1 - x0) * 2 - 1
        rel_y = (cy - y0) / (y1 - y0) * 2 - 1

        # --- real FLIP inference call — unmodified, unwrapped-by-cleanup ---
        mask_soft = self._flip.segment(roi_resized, rel_x, rel_y, sigma_x, sigma_y)
        # ---------------------------------------------------------------

        mask_roi_full = cv2.resize(mask_soft, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
        mask_bin_roi = (mask_roi_full > 0.5).astype(np.uint8) * 255

        prompt_roi_xy = (int(round(cx - x0)), int(round(cy - y0)))
        if cleanup:
            mask_bin_roi = self._cleanup_mask(mask_bin_roi, prompt_roi_xy)

        mask_full = np.zeros((h, w), dtype=np.uint8)
        mask_full[y0:y1, x0:x1] = mask_bin_roi

        confidence = float(mask_roi_full[mask_bin_roi > 0].mean()) if np.any(mask_bin_roi) else 0.0

        if debug_dir:
            self._save_debug(debug_dir, debug_tag, rgb_full, prompt_px, (x0, y0, x1, y1),
                              mask_roi_full, mask_bin_roi, mask_full)

        return SegmentationResult(
            mask_full=mask_full,
            mask_roi_soft=mask_roi_full,
            roi_bbox=(x0, y0, x1, y1),
            confidence=confidence,
        )

    @staticmethod
    def _cleanup_mask(mask_bin: np.ndarray, prompt_xy: tuple) -> np.ndarray:
        """
        Post-FLIP-inference cleanup ONLY (never feeds back into FLIP):
          1. keep the connected component containing the prompt point (if
             the prompt itself lands on background, fall back to the
             largest component — FLIP's soft mask is usually still peaked
             near the prompt even if thresholding clips the exact pixel)
          2. fill internal holes
          3. (implicit) tiny stray components elsewhere in the ROI are
             dropped by virtue of step 1 only keeping ONE component
        """
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        if num_labels <= 1:
            return mask_bin  # nothing found at all — return as-is (caller sees an empty mask)

        px, py = prompt_xy
        h, w = mask_bin.shape
        px = int(np.clip(px, 0, w - 1))
        py = int(np.clip(py, 0, h - 1))
        target_label = labels[py, px]
        if target_label == 0:
            # Prompt pixel itself is background post-threshold; fall back
            # to the largest non-background component.
            areas = stats[1:, cv2.CC_STAT_AREA]
            target_label = 1 + int(np.argmax(areas))

        component_mask = (labels == target_label).astype(np.uint8) * 255

        # Fill holes: flood-fill from a corner on the INVERSE mask, then
        # invert back — standard "fill enclosed background" trick.
        h2, w2 = component_mask.shape
        flood = component_mask.copy()
        ff_mask = np.zeros((h2 + 2, w2 + 2), dtype=np.uint8)
        cv2.floodFill(flood, ff_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        filled = cv2.bitwise_or(component_mask, holes)
        return filled

    @staticmethod
    def _save_debug(debug_dir, tag, rgb_full, prompt_px, roi_bbox, mask_roi_soft, mask_bin_roi, mask_full):
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        x0, y0, x1, y1 = roi_bbox
        bgr_full = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR)

        annotated = bgr_full.copy()
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.circle(annotated, (int(prompt_px[0]), int(prompt_px[1])), 5, (0, 0, 255), -1)
        cv2.imwrite(str(Path(debug_dir) / f"{tag}_00_rgb_prompt.png"), annotated)

        raw_vis = (np.clip(mask_roi_soft, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(Path(debug_dir) / f"{tag}_01_raw_mask_roi.png"), raw_vis)

        cv2.imwrite(str(Path(debug_dir) / f"{tag}_02_cleaned_mask_roi.png"), mask_bin_roi)

        overlay = bgr_full.copy()
        color = np.zeros_like(overlay)
        color[mask_full > 0] = (0, 255, 0)
        blended = cv2.addWeighted(overlay, 1.0, color, 0.45, 0)
        cv2.imwrite(str(Path(debug_dir) / f"{tag}_03_overlay_full.png"), blended)


if __name__ == "__main__":
    import robosuite.macros as macros
    macros.IMAGE_CONVENTION = "opencv"
    from environment import PickPlaceEnv, SIDE_CAMERA_NAME, TARGET_MARKER_ID
    from aruco_prompt import get_target_prompt

    out_dir = "phase3_output"

    env = PickPlaceEnv(
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=[SIDE_CAMERA_NAME], camera_heights=720, camera_widths=960,
        camera_depths=True, num_distractors=0, seed=0,
    )
    env.reset()
    for _ in range(10):
        env.sim.step()
    rgb, _depth = env.get_camera_rgbd()
    env.close()

    detection, failure = get_target_prompt(rgb, expected_id=TARGET_MARKER_ID)
    if detection is None:
        raise SystemExit(f"[FATAL] aruco_prompt failed: {failure.reason.value} — cannot test flip_segmenter")

    print(f"prompt: marker_id={detection.marker_id} center_px={detection.center_px} "
          f"side_px={detection.side_length_px:.1f}")

    segmenter = FlipTargetSegmenter(model_size="small")
    result = segmenter.segment_from_prompt(
        rgb, detection.center_px, marker_side_px=detection.side_length_px,
        debug_dir=out_dir, debug_tag="phase3",
    )
    n_fg = int((result.mask_full > 0).sum())
    print(f"mask pixels: {n_fg} | roi_bbox={result.roi_bbox} | confidence={result.confidence:.3f}")
    print(f"Saved debug images to {out_dir}/")
