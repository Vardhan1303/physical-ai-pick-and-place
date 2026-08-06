#!/usr/bin/env python
"""
ik_utils.py — small damped-least-squares inverse kinematics solver for the
Panda arm, using MuJoCo's own mj_jac (no external IK library / dependency).

Solves for the 7 arm joint angles that bring a fixed offset point on the
"hand" body (the TCP — tool center point, roughly where the fingertips meet)
to a target world position + orientation.
"""
import numpy as np
import mujoco

# Standard Franka Panda flange-to-fingertip distance along the hand's local
# +Z (the gripper's approach/closing axis).
TCP_LOCAL_OFFSET = np.array([0.0, 0.0, 0.1034])

ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


def get_tcp_pose(model, data, hand_body_id):
    """Returns (world_pos, world_rotmat) of the TCP point rigidly attached
    to the hand body."""
    hand_pos = data.xpos[hand_body_id]
    hand_mat = data.xmat[hand_body_id].reshape(3, 3)
    tcp_pos = hand_pos + hand_mat @ TCP_LOCAL_OFFSET
    return tcp_pos, hand_mat


def orientation_error(r_target: np.ndarray, r_current: np.ndarray) -> np.ndarray:
    """Standard axis-angle-ish orientation error: sum of cross products of
    corresponding columns of the two rotation matrices. Zero when aligned,
    small-angle-accurate near convergence — enough for damped IK."""
    err = np.zeros(3)
    for i in range(3):
        err += np.cross(r_current[:, i], r_target[:, i])
    return 0.5 * err


def solve_ik(model, data, hand_body_id, arm_qpos_addrs, target_pos, target_rotmat,
             max_iters=200, pos_tol=0.002, rot_tol=0.02, damping=0.05, step_scale=1.0):
    """
    Iteratively updates data.qpos in-place (only the 7 arm joint DOFs) so the
    TCP reaches target_pos/target_rotmat. Returns True if converged.

    arm_qpos_addrs: list of 7 qpos indices for joint1..joint7 (order matters).
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    dof_cols = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in ARM_JOINT_NAMES]

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)  # refresh xpos/xmat from current qpos
        tcp_pos, tcp_mat = get_tcp_pose(model, data, hand_body_id)

        pos_err = target_pos - tcp_pos
        rot_err = orientation_error(target_rotmat, tcp_mat)

        if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
            return True

        mujoco.mj_jac(model, data, jacp, jacr, tcp_pos, hand_body_id)
        J = np.vstack([jacp[:, dof_cols], jacr[:, dof_cols]])  # 6x7
        err = np.concatenate([pos_err, rot_err])  # 6

        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 err
        JJt = J @ J.T
        lam2 = damping ** 2
        dq = J.T @ np.linalg.solve(JJt + lam2 * np.eye(6), err)
        dq *= step_scale

        for i, addr in enumerate(arm_qpos_addrs):
            data.qpos[addr] += dq[i]
            lo, hi = model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINT_NAMES[i])]
            data.qpos[addr] = np.clip(data.qpos[addr], lo, hi)

    return False


def get_arm_qpos_addrs(model):
    return [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ARM_JOINT_NAMES]


def mat2quat(mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


def quat2mat(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Quaternion spherical linear interpolation. Used to smoothly interpolate
    the gripper's target ORIENTATION across waypoints, not just its position
    — without this, a large single-shot orientation change (e.g. a grasp yaw
    far from the arm's current wrist angle) can send the damped-least-squares
    IK solver into a bad/degenerate configuration on the very first step
    (observed directly: yaw=-90deg from the home pose converged to the hand
    dropping to floor level instead of the intended pregrasp height)."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)
