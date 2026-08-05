#!/usr/bin/env python
"""
manipulation.py — executes a full pick-and-place motion on the Panda arm
given a 3D pick point, a 3D place point, a grasp yaw angle, and the object
body to grasp.

The arm's joint actuators are PD position servos (built into the Menagerie
Panda model), so "moving" a joint means setting data.ctrl to the target
angle and stepping physics long enough for the PD controller to converge.
Motion between waypoints is broken into fine sub-steps (goto_linear) to
keep the Cartesian path roughly straight rather than letting joint-space
PD control bow through an arbitrary curve between two poses.

GRASP HOLD: pure friction grasping of small rigid objects in MuJoCo proved
unreliable under lateral motion after extensive tuning (friction, contact
solver softness, grip stiffness, motion smoothness) — verified by direct
testing, not assumed. Since this project's purpose is a working perception
-> action demo rather than a grasp-physics research contribution, a weld
equality constraint is activated between the gripper and object the moment
it's grasped, and released at the target bin. The fingers still physically
close around the object for a real contact at pick/place; only the
"carrying" phase is assisted.
"""
import numpy as np
import mujoco

from ik_utils import solve_ik, get_tcp_pose, get_arm_qpos_addrs

GRIPPER_OPEN_CTRL = 255.0
GRIPPER_CLOSED_CTRL = 0.0

APPROACH_HEIGHT = 0.15   # meters above the grasp height for pre-grasp/pre-place/transport
N_WAYPOINTS = 12         # sub-waypoints per motion segment, keeps the Cartesian path near-linear
STEPS_PER_WAYPOINT = 60  # physics steps per sub-waypoint (PD convergence + settle)
GRASP_STEPS = 300        # steps to let the gripper actually close/open


def make_downward_R(yaw_rad: float) -> np.ndarray:
    """World-frame rotation: gripper's approach axis points straight down
    (-world Z), rotated about world Z by yaw_rad — the grasp angle from the
    mask's minAreaRect."""
    Rx180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return Rz @ Rx180


def set_home_pose(model, data, arm_addrs):
    """Initializes the arm+gripper to a sane starting configuration above
    the table. Sets qpos directly rather than using mj_resetDataKeyframe,
    since the Menagerie keyframe only defines the arm's 9 DOFs and would
    zero out every object's free-joint qpos too (a real bug hit and fixed
    during development of this project)."""
    home_arm_qpos = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
    for addr, val in zip(arm_addrs, home_arm_qpos):
        data.qpos[addr] = val
    for jname in ("finger_joint1", "finger_joint2"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        data.qpos[model.jnt_qposadr[jid]] = 0.04
    mujoco.mj_forward(model, data)
    return home_arm_qpos


def _goto_linear(model, data, hand_id, arm_addrs, arm_actuator_ids, target_pos, target_R,
                  n_wp=N_WAYPOINTS, steps_per_wp=STEPS_PER_WAYPOINT, frame_callback=None):
    tcp, _ = get_tcp_pose(model, data, hand_id)
    for a in np.linspace(0, 1, n_wp + 1)[1:]:
        wp = tcp * (1 - a) + np.asarray(target_pos) * a
        solve_ik(model, data, hand_id, arm_addrs, wp, target_R, max_iters=150)
        target_qpos = [data.qpos[k] for k in arm_addrs]
        for _ in range(steps_per_wp):
            for act_id, q in zip(arm_actuator_ids, target_qpos):
                data.ctrl[act_id] = q
            mujoco.mj_step(model, data)
        # One frame per waypoint (not per whole segment) so recorded video
        # shows smooth motion rather than a jump-cut between segment ends.
        if frame_callback:
            frame_callback()


def _set_gripper(model, data, gripper_actuator_id, ctrl_value, steps=GRASP_STEPS, frame_callback=None):
    capture_every = max(steps // 4, 1)
    for i in range(steps):
        data.ctrl[gripper_actuator_id] = ctrl_value
        mujoco.mj_step(model, data)
        if frame_callback and i % capture_every == 0:
            frame_callback()


def _activate_weld(model, data, eq_id, hand_id, obj_body_id):
    """Locks the weld to the object's CURRENT relative pose w.r.t. the hand
    (not the compile-time default, which would snap the object back to
    wherever it started relative to the gripper)."""
    hand_pos = data.xpos[hand_id].copy()
    hand_quat = data.xquat[hand_id].copy()
    obj_pos = data.xpos[obj_body_id].copy()
    obj_quat = data.xquat[obj_body_id].copy()

    hand_mat = np.zeros(9)
    mujoco.mju_quat2Mat(hand_mat, hand_quat)
    hand_mat = hand_mat.reshape(3, 3)
    rel_pos = hand_mat.T @ (obj_pos - hand_pos)

    hand_quat_neg = np.zeros(4)
    mujoco.mju_negQuat(hand_quat_neg, hand_quat)
    rel_quat = np.zeros(4)
    mujoco.mju_mulQuat(rel_quat, hand_quat_neg, obj_quat)

    model.eq_data[eq_id, 3:6] = rel_pos
    model.eq_data[eq_id, 6:10] = rel_quat
    data.eq_active[eq_id] = 1


def _deactivate_weld(data, eq_id):
    data.eq_active[eq_id] = 0


def run_pick_and_place(model, data, object_name: str, pick_xy, place_xy, grasp_yaw,
                        grasp_height, frame_callback=None):
    """
    object_name: one of 'obj_square', 'obj_circle', 'obj_triangle' — used to
    look up the matching weld constraint ('weld_square' etc.) by name.
    pick_xy, place_xy: (x, y) world tuples. grasp_height: world Z of the
    object's vertical center (where the gripper closes around it).
    frame_callback: optional fn() called after each physics step, for
    recording video frames without coupling this module to any video code.
    """
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    weld_name = "weld_" + object_name.split("_", 1)[1]
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)

    arm_addrs = get_arm_qpos_addrs(model)
    arm_actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
                        for i in range(1, 8)]
    gripper_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")

    R = make_downward_R(grasp_yaw)
    px, py = pick_xy
    lx, ly = place_xy

    def goto(target_pos, **kwargs):
        _goto_linear(model, data, hand_id, arm_addrs, arm_actuator_ids, target_pos, R,
                     frame_callback=frame_callback, **kwargs)

    # 1. Pre-grasp: above the object, gripper open.
    _set_gripper(model, data, gripper_actuator_id, GRIPPER_OPEN_CTRL, steps=1)
    goto([px, py, grasp_height + APPROACH_HEIGHT])

    # 2. Descend to grasp height.
    goto([px, py, grasp_height])

    # 3. Close gripper, then lock the weld at the pose it actually grasped at.
    _set_gripper(model, data, gripper_actuator_id, GRIPPER_CLOSED_CTRL, frame_callback=frame_callback)
    _activate_weld(model, data, eq_id, hand_id, obj_body_id)

    # 4. Lift.
    goto([px, py, grasp_height + APPROACH_HEIGHT])

    # 5. Transport (stay high) to above the target bin.
    goto([lx, ly, grasp_height + APPROACH_HEIGHT])

    # 6. Descend to release height.
    goto([lx, ly, grasp_height])

    # 7. Release: drop the weld, then open the gripper.
    _deactivate_weld(data, eq_id)
    _set_gripper(model, data, gripper_actuator_id, GRIPPER_OPEN_CTRL, frame_callback=frame_callback)

    # 8. Retract.
    goto([lx, ly, grasp_height + APPROACH_HEIGHT])
