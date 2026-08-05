#!/usr/bin/env python
"""
manipulation.py — executes a full pick-and-place motion on the Panda arm
given a 3D pick point, a 3D place point, and a grasp yaw angle. Shared by
the real perception-driven script (pick_and_place.py) and by tests that
substitute ground-truth segmentation for FLIP.

The arm's joint actuators are PD position servos (built into the Menagerie
Panda model), so "moving" a joint means setting data.ctrl to the target
angle and stepping physics long enough for the PD controller to converge —
no separate trajectory/motion-planning library needed for this demo.
"""
import numpy as np
import mujoco

from ik_utils import solve_ik, get_arm_qpos_addrs

GRIPPER_OPEN_CTRL = 255.0
GRIPPER_CLOSED_CTRL = 0.0

APPROACH_HEIGHT = 0.15  # meters above the table for pre-grasp/pre-place poses
SETTLE_STEPS = 300       # physics steps per motion segment (PD converges + settles)
GRASP_STEPS = 200        # extra steps to let the gripper actually close/grip


def make_downward_R(yaw_rad: float) -> np.ndarray:
    """World-frame rotation: gripper's approach axis points straight down
    (-world Z), rotated about world Z by yaw_rad — this is the "grasp
    angle" computed from the mask's minAreaRect."""
    Rx180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return Rz @ Rx180


def _goto(model, data, hand_id, arm_addrs, arm_actuator_ids, target_pos, target_R, steps=SETTLE_STEPS):
    ok = solve_ik(model, data, hand_id, arm_addrs, target_pos, target_R)
    target_qpos = [data.qpos[a] for a in arm_addrs]
    for _ in range(steps):
        for act_id, q in zip(arm_actuator_ids, target_qpos):
            data.ctrl[act_id] = q
        mujoco.mj_step(model, data)
    return ok


def _set_gripper(model, data, gripper_actuator_id, ctrl_value, steps=GRASP_STEPS):
    for _ in range(steps):
        data.ctrl[gripper_actuator_id] = ctrl_value
        mujoco.mj_step(model, data)


def run_pick_and_place(model, data, pick_pos, place_pos, grasp_yaw,
                        grasp_height, frame_callback=None):
    """
    pick_pos, place_pos: (x, y) tuples — Z is derived from grasp_height /
    APPROACH_HEIGHT internally.
    grasp_height: world Z of the object's grasp point (its vertical
    center — where the gripper should close around it).
    frame_callback: optional fn() called after every physics step, used to
    record video frames during the motion (keeps this module independent
    of any video-writing code).
    """
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    arm_addrs = get_arm_qpos_addrs(model)
    arm_actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
                        for i in range(1, 8)]
    gripper_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")

    R = make_downward_R(grasp_yaw)
    px, py = pick_pos
    lx, ly = place_pos

    def step_with_callback(steps):
        if frame_callback is None:
            return
        for _ in range(steps):
            frame_callback()

    # 1. Pre-grasp: above the object, gripper open.
    _set_gripper(model, data, gripper_actuator_id, GRIPPER_OPEN_CTRL, steps=1)
    _goto(model, data, hand_id, arm_addrs, arm_actuator_ids,
          np.array([px, py, grasp_height + APPROACH_HEIGHT]), R)

    # 2. Descend to grasp height.
    _goto(model, data, hand_id, arm_addrs, arm_actuator_ids,
          np.array([px, py, grasp_height]), R)

    # 3. Close gripper.
    _set_gripper(model, data, gripper_actuator_id, GRIPPER_CLOSED_CTRL)

    # 4. Lift.
    _goto(model, data, hand_id, arm_addrs, arm_actuator_ids,
          np.array([px, py, grasp_height + APPROACH_HEIGHT]), R)

    # 5. Transport (stay high) to above the target bin.
    _goto(model, data, hand_id, arm_addrs, arm_actuator_ids,
          np.array([lx, ly, grasp_height + APPROACH_HEIGHT]), R)

    # 6. Descend to release height.
    _goto(model, data, hand_id, arm_addrs, arm_actuator_ids,
          np.array([lx, ly, grasp_height]), R)

    # 7. Open gripper to release.
    _set_gripper(model, data, gripper_actuator_id, GRIPPER_OPEN_CTRL)

    # 8. Retract.
    _goto(model, data, hand_id, arm_addrs, arm_actuator_ids,
          np.array([lx, ly, grasp_height + APPROACH_HEIGHT]), R)
