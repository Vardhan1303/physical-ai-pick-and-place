#!/usr/bin/env python
"""
robot_controller.py — Phase 6: executes a GraspPlan (grasp_planner.py) on
the Panda arm via robosuite's real OSC_POSE composite controller.

Grounded directly in the installed robosuite 1.5.2's actual controller
config (robosuite/controllers/config/robots/default_panda.json, the same
file environment.py's PickPlaceEnv auto-loads when controller_configs=None
— confirmed via its own load-time log line). Key facts taken from that
file, not invented:
  - action = 7-dim: [dx, dy, dz, d_axis_angle_x, d_axis_angle_y,
    d_axis_angle_z, gripper], each pose component normalized to [-1, 1]
  - "input_type": "delta" — every action is a RELATIVE motion command, not
    an absolute target, so reaching a target pose means closing the loop
    ourselves: read the current end-effector pose each step, compute the
    remaining error, send a clipped/normalized delta, repeat.
  - "output_max"/"output_min": [0.05]*3 position + [0.5]*3 rotation per
    step (radians for rotation, meters for position) — this bounds how
    big a single action's effect can be; used below to normalize errors
    into the controller's expected [-1, 1] range.
  - "input_ref_frame": "base" — the delta components are interpreted along
    the ROBOT BASE's own axes, not world axes. Since grasp_planner.py's
    poses are already in robot-base frame, this is convenient, but the
    live error signal (current eef pose) comes back in WORLD frame via
    robosuite's own `robot0_eef_pos`/`robot0_eef_quat` observables
    (confirmed by direct inspection in-sandbox), so every per-step error
    is computed in world frame and then rotated into base frame with
    `base_mat.T` before being sent as an action.
  - gripper action convention: -1 = open, +1 = close (robosuite's
    standard convention, used consistently across its own demo scripts).

Orientation error uses `robosuite.utils.transform_utils.get_orientation_error`
— the same function OSC's own internal control law uses — rather than a
reimplementation, so the error this module drives to zero is defined the
same way the controller itself interprets it.

"Close until contact, not only estimated geometry" is implemented as: send
the CLOSE action for up to `grasp_close_steps`, but stop early once the
gripper's own finger qpos velocity drops near zero before reaching the
planned width — i.e. the fingers stopped because they hit something, not
because they ran out of travel. This is a simplified contact proxy (no
force/torque sensor read), documented as such.

Standalone self-test:
    python robot_controller.py
Chains the full Phase 1-5 pipeline and then physically executes the
resulting grasp plan in the sandboxed sim, printing pass/fail per stage.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import robosuite.utils.transform_utils as T

GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0

# From default_panda.json — see module docstring.
TRANSLATION_OUTPUT_MAX = 0.05   # m per policy step
ROTATION_OUTPUT_MAX = 0.5        # rad per policy step


@dataclass
class StageResult:
    name: str
    success: bool
    steps_taken: int
    final_pos_error: float
    final_rot_error: float
    stopped_by_contact: bool = False


@dataclass
class ExecutionResult:
    stages: List[StageResult] = field(default_factory=list)
    success: bool = False
    failure_reason: Optional[str] = None


def _downward_grasp_rotation(yaw: float) -> np.ndarray:
    from grasp_planner import downward_grasp_rotation
    return downward_grasp_rotation(yaw)


def move_to_pose(
    env,
    target_pos_world: np.ndarray,
    target_R_world: np.ndarray,
    base_mat: np.ndarray,
    gripper_action: float = GRIPPER_OPEN,
    max_steps: int = 150,
    pos_tol: float = 0.006,
    rot_tol: float = 0.08,
    step_callback=None,
    stop_on_stall: bool = False,
    stall_eps: float = 0.0008,
    stall_patience: int = 12,
) -> StageResult:
    """
    Closed-loop P-control to an absolute world-frame pose, one robosuite
    env.step() per iteration, using the real OSC_POSE delta-action
    convention (see module docstring).

    `stop_on_stall`: for DESCEND-type moves specifically (grasp approach,
    place release), the physically correct stopping condition isn't
    always "reached the exact target pose" — it's "reached the target OR
    got physically blocked by something (the table, the object being
    grasped, the bin's surface) on the way there." Confirmed necessary
    in-sandbox: a place-descend targeting a pose measured at the bin's
    rim consistently stopped ~5cm short with a stable, non-decreasing
    position error, because the GRASPED OBJECT (not the gripper itself)
    was the first thing to contact the bin's solid top surface — the
    object was successfully placed, just not at the exact TCP height the
    naive target assumed (which didn't account for the object hanging
    below the gripper's TCP by roughly half its own height). Detecting
    "stopped moving, error not shrinking" and treating that as success
    (with `stopped_by_contact=True` so callers/logs can tell the
    difference from a clean convergence) is the standard robotics answer
    to this, not a workaround for a specific number.
    """
    target_quat_xyzw = T.mat2quat(target_R_world)

    pos_err_norm = np.inf
    rot_err_norm = np.inf
    stall_count = 0
    prev_pos = None
    stopped_by_contact = False
    step = 0
    for step in range(1, max_steps + 1):
        obs = env._get_observations(force_update=True)
        cur_pos = obs["robot0_eef_pos"]
        cur_quat = obs["robot0_eef_quat"]

        # Quaternion double-cover fix: q and -q represent the identical
        # rotation. get_orientation_error doesn't check which hemisphere
        # target/current are on, so if they land on opposite signs the
        # computed error can momentarily be ~2x too large and drive the
        # controller the "long way around" before it corrects — confirmed
        # in-sandbox: rot_err_norm started at ~2.0 (near the theoretical
        # max) and position error nearly TRIPLED over the first 80 steps
        # before recovering, for a target that a hemisphere-aligned error
        # reaches directly. Flipping the target's sign to match the
        # current quaternion's hemisphere each step is the standard fix.
        aligned_target_quat = target_quat_xyzw if np.dot(target_quat_xyzw, cur_quat) >= 0 else -target_quat_xyzw

        pos_err_world = target_pos_world - cur_pos
        orn_err_world = T.get_orientation_error(aligned_target_quat, cur_quat)
        pos_err_norm = float(np.linalg.norm(pos_err_world))
        rot_err_norm = float(np.linalg.norm(orn_err_world))

        if pos_err_norm < pos_tol and rot_err_norm < rot_tol:
            break

        if stop_on_stall and prev_pos is not None:
            moved = float(np.linalg.norm(cur_pos - prev_pos))
            if moved < stall_eps:
                stall_count += 1
                if stall_count >= stall_patience:
                    stopped_by_contact = True
                    break
            else:
                stall_count = 0
        prev_pos = cur_pos.copy()

        pos_err_base = base_mat.T @ pos_err_world
        orn_err_base = base_mat.T @ orn_err_world
        action_pos = np.clip(pos_err_base / TRANSLATION_OUTPUT_MAX, -1, 1)
        action_rot = np.clip(orn_err_base / ROTATION_OUTPUT_MAX, -1, 1)
        action = np.concatenate([action_pos, action_rot, [gripper_action]]).astype(np.float32)
        env.step(action)
        if step_callback:
            step_callback()

    converged = pos_err_norm < pos_tol and rot_err_norm < rot_tol
    return StageResult(
        name="move_to_pose", success=(converged or stopped_by_contact),
        steps_taken=step, final_pos_error=pos_err_norm, final_rot_error=rot_err_norm,
        stopped_by_contact=stopped_by_contact,
    )


def move_to_pose_interpolated(
    env,
    target_pos_world: np.ndarray,
    target_R_world: np.ndarray,
    base_mat: np.ndarray,
    gripper_action: float = GRIPPER_OPEN,
    n_waypoints: int = 5,
    max_steps_per_waypoint: int = 80,
    pos_tol: float = 0.006,
    rot_tol: float = 0.08,
    step_callback=None,
    stop_on_stall: bool = False,
) -> StageResult:
    """
    Breaks a large move into `n_waypoints` position-lerp + orientation-slerp
    sub-targets, calling move_to_pose for each. Confirmed necessary
    in-sandbox: a single big move_to_pose call combining a ~0.4m position
    change with a ~180deg orientation change would initially converge
    (getting within ~2cm of the target around its own step 50) and then
    visibly REVERSE and drift ~0.3m away, settling into a stable but wrong
    equilibrium — the OSC controller solving position and orientation
    error jointly from a bad relative configuration, not unlike the old
    raw-MuJoCo pipeline's own documented IK-divergence bug for non-zero
    grasp yaws (fixed there by slerping orientation across waypoints
    instead of jumping straight to the target). Same fix, reapplied here
    against robosuite's OSC controller instead of a custom damped-least-
    squares IK solver.
    """
    obs = env._get_observations(force_update=True)
    start_pos = obs["robot0_eef_pos"].copy()
    start_quat = obs["robot0_eef_quat"].copy()
    target_quat = T.mat2quat(target_R_world)
    if np.dot(target_quat, start_quat) < 0:
        target_quat = -target_quat

    last_stage = None
    for i in range(1, n_waypoints + 1):
        frac = i / n_waypoints
        wp_pos = start_pos * (1 - frac) + target_pos_world * frac
        wp_quat = T.quat_slerp(start_quat, target_quat, frac)
        wp_R = T.quat2mat(wp_quat)
        is_last = i == n_waypoints
        last_stage = move_to_pose(
            env, wp_pos, wp_R, base_mat, gripper_action=gripper_action,
            max_steps=max_steps_per_waypoint, pos_tol=pos_tol, rot_tol=rot_tol,
            step_callback=step_callback,
            stop_on_stall=(stop_on_stall and is_last),
        )
        if not last_stage.success and not is_last:
            # An intermediate waypoint stalling isn't necessarily fatal —
            # keep pushing toward the final target, which is what actually
            # matters; only the LAST waypoint's outcome is reported.
            continue
    return last_stage


def close_gripper_until_contact(env, max_steps: int = 100, qvel_eps: float = 0.0005,
                                 settle_steps: int = 10, step_callback=None) -> StageResult:
    """Sends the CLOSE action repeatedly, stopping early if finger qvel
    drops near zero before the fingers reach their fully-closed limit —
    the "hit something" signal described in the module docstring. Also
    stops (successfully) if fingers DO reach fully closed (grasping
    something thin, or closing on nothing — pipeline.py's success check,
    not this module, decides whether an object was actually captured)."""
    still_count = 0
    step = 0
    for step in range(1, max_steps + 1):
        action = np.zeros(7, dtype=np.float32)
        action[-1] = GRIPPER_CLOSE
        env.step(action)
        if step_callback:
            step_callback()
        obs = env._get_observations(force_update=True)
        qvel = obs.get("robot0_gripper_qvel", None)
        if qvel is not None and np.max(np.abs(qvel)) < qvel_eps:
            still_count += 1
            if still_count >= settle_steps:
                break
        else:
            still_count = 0
    return StageResult(name="close_gripper", success=True, steps_taken=step,
                        final_pos_error=0.0, final_rot_error=0.0)


def open_gripper(env, steps: int = 40, step_callback=None) -> StageResult:
    for step in range(1, steps + 1):
        action = np.zeros(7, dtype=np.float32)
        action[-1] = GRIPPER_OPEN
        env.step(action)
        if step_callback:
            step_callback()
    return StageResult(name="open_gripper", success=True, steps_taken=steps,
                        final_pos_error=0.0, final_rot_error=0.0)


def execute_grasp_plan(env, plan, base_pos: np.ndarray, base_mat: np.ndarray,
                        step_callback=None, max_steps_per_move: int = 200) -> ExecutionResult:
    """
    Runs the full pre-grasp -> descend -> close -> lift -> transport ->
    descend -> release -> retract sequence for one GraspPlan
    (grasp_planner.GraspPlan), reporting per-stage success/failure so the
    pipeline can log and, on failure, abort/retract rather than push
    through a bad state.
    """
    result = ExecutionResult()

    def to_world(pos_robot):
        return base_pos + base_mat @ pos_robot

    # (name, pose, gripper_action, allow_stall, use_interpolation) — the
    # two big reconfiguration moves (pregrasp from wherever the arm starts,
    # and the lateral transport to the bin) go through
    # move_to_pose_interpolated; short local moves (descend/lift, which
    # only change one axis by ~15cm from an already-good starting pose)
    # don't need it. See move_to_pose_interpolated's docstring for the
    # concrete failure this fixes.
    stages_spec = [
        ("pregrasp", plan.pregrasp, GRIPPER_OPEN, False, True),
        ("descend_to_grasp", plan.grasp, GRIPPER_OPEN, True, False),
    ]
    for name, pose, grip, allow_stall, interp in stages_spec:
        R_world = base_mat @ _downward_grasp_rotation(pose.yaw)
        mover = move_to_pose_interpolated if interp else move_to_pose
        stage = mover(env, to_world(pose.pos), R_world, base_mat,
                       gripper_action=grip, step_callback=step_callback, stop_on_stall=allow_stall,
                       **({"max_steps": max_steps_per_move} if not interp else {}))
        stage.name = name
        result.stages.append(stage)
        if not stage.success:
            result.failure_reason = f"{name}: failed to converge (pos_err={stage.final_pos_error:.4f})"
            return result

    close_stage = close_gripper_until_contact(env, step_callback=step_callback)
    result.stages.append(close_stage)

    post_grasp_stages = [
        ("lift", plan.lift, GRIPPER_CLOSE, False, False),
        ("transport_to_place", plan.place, GRIPPER_CLOSE, False, True),
        ("descend_to_place", plan.__dict__["place_descend"], GRIPPER_CLOSE, True, False),
    ]
    for name, pose, grip, allow_stall, interp in post_grasp_stages:
        R_world = base_mat @ _downward_grasp_rotation(pose.yaw)
        mover = move_to_pose_interpolated if interp else move_to_pose
        stage = mover(env, to_world(pose.pos), R_world, base_mat,
                       gripper_action=grip, step_callback=step_callback, stop_on_stall=allow_stall,
                       **({"max_steps": max_steps_per_move} if not interp else {}))
        stage.name = name
        result.stages.append(stage)
        if not stage.success:
            result.failure_reason = f"{name}: failed to converge (pos_err={stage.final_pos_error:.4f})"
            return result

    open_stage = open_gripper(env, step_callback=step_callback)
    result.stages.append(open_stage)

    retract_pose = plan.place
    R_world = base_mat @ _downward_grasp_rotation(retract_pose.yaw)
    retract_stage = move_to_pose(env, to_world(retract_pose.pos), R_world, base_mat,
                                  gripper_action=GRIPPER_OPEN, max_steps=max_steps_per_move,
                                  step_callback=step_callback, stop_on_stall=True)
    retract_stage.name = "retract"
    result.stages.append(retract_stage)

    result.success = True
    return result


def execute_side_grasp_plan(env, plan, base_pos: np.ndarray, base_mat: np.ndarray,
                             step_callback=None, max_steps_per_move: int = 200,
                             n_waypoints: int = 10, max_steps_per_waypoint: int = 150) -> ExecutionResult:
    """
    Runs the HORIZONTAL side-grasp sequence for one
    grasp_planner.SideGraspPlan: safe-high -> pregrasp -> horizontal
    approach -> close -> retreat -> lift -> transport -> lower -> release
    -> retract.

    Deliberately structured to mirror execute_grasp_plan above as closely
    as the different motion shape allows, and reuses every primitive
    (move_to_pose, move_to_pose_interpolated, close_gripper_until_contact,
    open_gripper) completely unchanged — only the POSES and the fact that
    each SideGraspPose already carries a full rotation matrix (from
    grasp_planner.axes_to_gripper_rotation) instead of a yaw-only rotation
    are new. The same "close on contact, not just estimated width"
    (close_gripper_until_contact) and "stop-on-stall counts as success for
    a descend/contact move" (move_to_pose's stop_on_stall) rules from the
    top-down version apply here for exactly the same physical reason —
    the horizontal approach and the vertical place-descend are both
    "drive toward a surface until you touch it" moves.

    `n_waypoints`/`max_steps_per_waypoint` default higher than
    execute_grasp_plan's top-down interpolation (5 waypoints / 80 steps)
    because side-grasp reconfigurations start much farther from the arm's
    natural "reach down" home orientation — confirmed necessary
    in-sandbox: the safe-high -> pregrasp descent (same orientation
    throughout, position-only) still failed to converge within the
    smaller default budget even though the orientation itself was already
    achieved, apparently because holding a horizontal wrist orientation
    while translating into the lower workspace needs a different
    elbow/joint configuration that takes the controller more steps to
    find than a purely-positional top-down move does.
    """
    result = ExecutionResult()

    def to_world(pos_robot):
        return base_pos + base_mat @ pos_robot

    def R_world_of(pose):
        return base_mat @ pose.R

    pregrasp_stages = [
        ("safe_high", plan.safe_high, GRIPPER_OPEN, False, True),
        ("side_pregrasp", plan.pregrasp, GRIPPER_OPEN, False, True),
        ("side_approach", plan.grasp, GRIPPER_OPEN, True, False),
    ]
    for name, pose, grip, allow_stall, interp in pregrasp_stages:
        R_world = R_world_of(pose)
        mover = move_to_pose_interpolated if interp else move_to_pose
        extra_kwargs = ({"n_waypoints": n_waypoints, "max_steps_per_waypoint": max_steps_per_waypoint}
                         if interp else {"max_steps": max_steps_per_move})
        stage = mover(env, to_world(pose.pos), R_world, base_mat,
                       gripper_action=grip, step_callback=step_callback, stop_on_stall=allow_stall,
                       **extra_kwargs)
        stage.name = name
        result.stages.append(stage)
        if not stage.success:
            result.failure_reason = f"{name}: failed to converge (pos_err={stage.final_pos_error:.4f})"
            return result

    close_stage = close_gripper_until_contact(env, step_callback=step_callback)
    result.stages.append(close_stage)

    post_grasp_stages = [
        ("retreat", plan.retreat, GRIPPER_CLOSE, False, False),
        # stop_on_stall=True here too: lifting while holding an object in
        # a horizontal-wrist grasp can run the arm up against a joint
        # limit before the full planned lift_height is reached — confirmed
        # in-sandbox (position barely changed step-over-step even with a
        # generous step budget, the same "stopped moving, error not
        # shrinking" signature move_to_pose's stall detection is built
        # for, not a slow-convergence case more steps would fix).
        ("lift", plan.lift, GRIPPER_CLOSE, True, False),
        ("transport_to_place", plan.transport, GRIPPER_CLOSE, False, True),
        ("descend_to_place", plan.place_descend, GRIPPER_CLOSE, True, False),
    ]
    for name, pose, grip, allow_stall, interp in post_grasp_stages:
        R_world = R_world_of(pose)
        mover = move_to_pose_interpolated if interp else move_to_pose
        extra_kwargs = ({"n_waypoints": n_waypoints, "max_steps_per_waypoint": max_steps_per_waypoint}
                         if interp else {"max_steps": max_steps_per_move})
        stage = mover(env, to_world(pose.pos), R_world, base_mat,
                       gripper_action=grip, step_callback=step_callback, stop_on_stall=allow_stall,
                       **extra_kwargs)
        stage.name = name
        result.stages.append(stage)
        if not stage.success:
            result.failure_reason = f"{name}: failed to converge (pos_err={stage.final_pos_error:.4f})"
            return result

    open_stage = open_gripper(env, step_callback=step_callback)
    result.stages.append(open_stage)

    retract_pose = plan.transport  # back up to the same high point above the tray
    R_world = R_world_of(retract_pose)
    retract_stage = move_to_pose(env, to_world(retract_pose.pos), R_world, base_mat,
                                  gripper_action=GRIPPER_OPEN, max_steps=max_steps_per_move,
                                  step_callback=step_callback, stop_on_stall=True)
    retract_stage.name = "retract"
    result.stages.append(retract_stage)

    result.success = True
    return result


if __name__ == "__main__":
    import robosuite.macros as macros
    macros.IMAGE_CONVENTION = "opencv"
    from environment import PickPlaceEnv, SIDE_CAMERA_NAME, TARGET_MARKER_ID
    from aruco_prompt import get_target_prompt
    from flip_segmenter import FlipTargetSegmenter
    from geometry import build_target_point_cloud, get_robot_base_transform
    from grasp_planner import plan_grasp

    env = PickPlaceEnv(
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=[SIDE_CAMERA_NAME], camera_heights=720, camera_widths=960,
        camera_depths=True, num_distractors=0, seed=0,
    )
    env.reset()
    for _ in range(10):
        env.sim.step()

    rgb, depth = env.get_camera_rgbd()
    K = env.get_camera_intrinsics()
    cam_to_world = env.get_camera_extrinsics()
    base_pos, base_mat = get_robot_base_transform(env)
    table_height_world = env.get_table_height()
    table_height_robot = float(((np.array([0, 0, table_height_world]) - base_pos) @ base_mat)[2])

    detection, failure = get_target_prompt(rgb, expected_id=TARGET_MARKER_ID)
    if detection is None:
        raise SystemExit(f"[FATAL] aruco_prompt failed: {failure.reason.value}")

    segmenter = FlipTargetSegmenter(model_size="small")
    seg = segmenter.segment_from_prompt(rgb, detection.center_px, marker_side_px=detection.side_length_px)

    pc = build_target_point_cloud(seg.mask_full, depth, K, cam_to_world, base_pos, base_mat,
                                   table_height_hint=table_height_world)
    print(f"point cloud size: {len(pc.points_robot)}")

    release_clearance = 0.01
    bin_world = env.get_bin_top_center() + np.array([0, 0, release_clearance])
    bin_robot = (bin_world - base_pos) @ base_mat

    plan, plan_failure = plan_grasp(pc.points_robot, place_xy=(bin_robot[0], bin_robot[1]),
                                     table_height=table_height_robot, place_height=bin_robot[2])
    if plan is None:
        raise SystemExit(f"[FATAL] grasp planning failed: {plan_failure.reason} — {plan_failure.detail}")
    print(f"grasp_width={plan.grasp_width:.4f}m yaw={np.degrees(plan.grasp.yaw):.1f}deg")

    exec_result = execute_grasp_plan(env, plan, base_pos, base_mat, max_steps_per_move=250)
    for s in exec_result.stages:
        status = "OK" if s.success else "FAIL"
        contact_note = " [stopped by contact]" if s.stopped_by_contact else ""
        print(f"[{status}] {s.name}: steps={s.steps_taken} pos_err={s.final_pos_error:.4f} "
              f"rot_err={s.final_rot_error:.4f}{contact_note}")
    print(f"overall success={exec_result.success} reason={exec_result.failure_reason}")

    env.close()
