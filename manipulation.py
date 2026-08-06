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

from ik_utils import solve_ik, get_tcp_pose, get_arm_qpos_addrs, mat2quat, quat2mat, slerp

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
    during development of this project).

    Also sets data.ctrl to match: the arm's actuators are PD position
    servos, so if a caller steps physics afterward (e.g. to let free-jointed
    objects settle onto the table before perception) without first setting
    ctrl, the controllers see ctrl=0 and drag every joint toward zero,
    silently undoing this whole function before the first real motion even
    starts. This was a real bug — the arm drifted from x=0.55 to x=0.13
    during a 500-step settle loop, and every reported "IK failure"
    downstream of it was actually IK correctly targeting the goal from that
    already-wrong starting configuration."""
    home_arm_qpos = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
    for addr, val in zip(arm_addrs, home_arm_qpos):
        data.qpos[addr] = val
    for jname in ("finger_joint1", "finger_joint2"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        data.qpos[model.jnt_qposadr[jid]] = 0.04
    mujoco.mj_forward(model, data)

    for i, val in enumerate(home_arm_qpos):
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i + 1}")
        data.ctrl[act_id] = val
    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    data.ctrl[gripper_act_id] = GRIPPER_OPEN_CTRL

    return home_arm_qpos


# The Panda's home/ready pose reaches directly over the table center, which
# happens to sit right above the circle object's marker — occluding it from
# the overhead camera used for perception (found by direct testing: with
# set_home_pose's ctrl fix applied, the arm actually holds that position
# instead of drifting, and the circle marker vanished from detection). Park
# the arm off to the side of the table before running perception, then let
# the first pick-and-place motion move it from there.
PARK_QPOS = [-2.6, -1.2, 0, -2.8, 0, 1.6, 0.8]


def set_park_pose(model, data, arm_addrs):
    """Same idea as set_home_pose (sets qpos AND holds it via ctrl so a
    settle loop doesn't drag the arm away from it), but swung off to the
    side so it doesn't block the overhead camera's view of the table."""
    for addr, val in zip(arm_addrs, PARK_QPOS):
        data.qpos[addr] = val
    for jname in ("finger_joint1", "finger_joint2"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        data.qpos[model.jnt_qposadr[jid]] = 0.04
    mujoco.mj_forward(model, data)

    for i, val in enumerate(PARK_QPOS):
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i + 1}")
        data.ctrl[act_id] = val
    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    data.ctrl[gripper_act_id] = GRIPPER_OPEN_CTRL

    return PARK_QPOS


def _goto_linear(model, data, hand_id, arm_addrs, arm_actuator_ids, target_pos, target_R,
                  n_wp=N_WAYPOINTS, steps_per_wp=STEPS_PER_WAYPOINT, frame_callback=None):
    tcp, start_R = get_tcp_pose(model, data, hand_id)
    start_quat = mat2quat(start_R)
    target_quat = mat2quat(target_R)
    for a in np.linspace(0, 1, n_wp + 1)[1:]:
        wp = tcp * (1 - a) + np.asarray(target_pos) * a
        # Orientation is slerped too (not jumped straight to target_R) — a
        # large single-shot orientation change can send the damped IK solver
        # into a bad configuration on the very first step; see ik_utils.slerp.
        wp_R = quat2mat(slerp(start_quat, target_quat, a))
        solve_ik(model, data, hand_id, arm_addrs, wp, wp_R, max_iters=150)
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
