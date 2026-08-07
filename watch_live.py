#!/usr/bin/env python
"""
watch_live.py — same pipeline as pick_and_place_flip.py (real FLIP
perception -> generic grasp -> arm motion), but opens MuJoCo's interactive
viewer and drives it live instead of running headless. You get the same
window as `python -m mujoco.viewer --mjcf=...`, except the arm actually
moves through the real pick-and-place sequence in front of you, synced to
roughly real time, rather than you having to drag things around by hand or
wait for a rendered .mp4 afterward.

Run from the project root on your Windows machine (same requirements as
pick_and_place_flip.py — real flip_position extension + ONNX weights):
    python watch_live.py

Close the viewer window (or Ctrl+C in the terminal) to exit.
"""
import time

import mujoco
import mujoco.viewer

import pick_and_place_flip as pf
from manipulation import set_park_pose
from ik_utils import get_arm_qpos_addrs


def main():
    model = mujoco.MjModel.from_xml_path(pf.SCENE_PATH)
    data = mujoco.MjData(model)
    dt = model.opt.timestep  # pace step_callback to roughly real time

    with mujoco.viewer.launch_passive(model, data) as viewer:

        def synced_step():
            """Called after every physics step during arm motion — sync the
            GUI and sleep the physics timestep so the arm's motion plays at
            roughly real speed instead of flashing through in a fraction of
            a second (headless stepping is much faster than real time)."""
            viewer.sync()
            time.sleep(dt)

        arm_addrs = get_arm_qpos_addrs(model)
        # Park (not home) for perception — see manipulation.PARK_QPOS's
        # comment: the home/ready pose occludes the circle marker.
        set_park_pose(model, data, arm_addrs)
        for _ in range(pf.SETTLE_STEPS):
            mujoco.mj_step(model, data)
            synced_step()

        detector = pf.make_detector("DICT_6X6_250")
        flip = pf.FlipSegmenter("small")

        detections = pf.perceive(model, data, flip, detector, pf.CAM_NAME)
        if not detections:
            print("[WARN] No objects detected/segmented — nothing to pick.")
        else:
            detections.sort(key=lambda d: d["marker_id"])

            for idx, det in enumerate(detections):
                if not viewer.is_running():
                    break
                shape = det["shape"]
                obj_name = pf.nearest_object_body(model, data, det["world_xy"])
                slot_y = pf.BIN_SLOT_OFFSETS_Y[idx % len(pf.BIN_SLOT_OFFSETS_Y)]
                place_xy = (pf.BIN_CENTER[0], pf.BIN_CENTER[1] + slot_y)

                print(f"--- picking marker {det['marker_id']} ({shape}, body={obj_name}) "
                      f"-> bin slot at {place_xy} ---")
                pf.run_pick_and_place(
                    model, data, obj_name,
                    pick_xy=det["world_xy"],
                    place_xy=place_xy,
                    grasp_yaw=det["grasp_yaw"],
                    grasp_height=pf.GRASP_HEIGHT,
                    step_callback=synced_step,
                )
                for _ in range(200):
                    mujoco.mj_step(model, data)
                    synced_step()

            print("Done — all detected objects processed. Close the viewer window to exit.")

        # Keep the window open (and physics gently settling) after the
        # sequence finishes, until you close it yourself.
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    main()
