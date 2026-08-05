#!/usr/bin/env python
"""
generate_markers.py — creates one ArUco marker PNG per object (square/circle/
triangle), saved into assets/markers/. These get applied as a small flat
"sticker" texture on top of each simulated object, standing in for the
printed markers you stuck on your real bottles.

Usage:
    python generate_markers.py
"""
from pathlib import Path

import cv2

OUT_DIR = Path(__file__).resolve().parent / "assets" / "markers"
DICT = cv2.aruco.DICT_6X6_250  # matches segment.py's default
MARKER_PX = 512  # high-res source; MuJoCo will downsample as needed

# One marker ID per object — arbitrary IDs, just need to be distinct.
# The shape itself is deliberately NOT derivable from which ID this is;
# shape gets classified later from the FLIP mask's contour, not from this.
OBJECTS = {
    "square": 0,
    "circle": 1,
    "triangle": 2,
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)

    for name, marker_id in OBJECTS.items():
        img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_PX)
        # MuJoCo textures want RGB (or RGBA); OpenCV's marker image is single-channel.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out_path = OUT_DIR / f"marker_{name}.png"
        cv2.imwrite(str(out_path), img_rgb)
        print(f"Saved {out_path} (id={marker_id})")


if __name__ == "__main__":
    main()
