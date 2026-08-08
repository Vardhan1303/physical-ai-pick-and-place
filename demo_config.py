#!/usr/bin/env python
"""
demo_config.py — single, clear configuration surface for the polished
single-object demo (final_demo.py). Every tunable the project spec calls
out by name lives here, as one dataclass, rather than scattered across
call sites.

This is deliberately separate from config.py/config.yaml (which configure
the earlier multi-object experiment scripts, out of scope for this demo
per the project's own instruction to not extend multi-object work at this
stage) — final_demo.py does not import those.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DemoConfig:
    # --- Grasp geometry ---
    gripper_open_margin: float = 0.015        # m, added on top of the FLIP/depth-estimated object width
    min_gripper_opening: float = 0.01         # m, reject an estimate implausibly smaller than this
    max_gripper_opening: float = 0.08         # m, Panda's physical max finger opening
    grasp_height_ratio: float = 0.7           # 0-1, where up the visible object's vertical extent to grasp
                                                # (0=bottom of visible cloud, 1=top) — biased high by default,
                                                # see grasp_planner.py's own note on why (wrist clearance above
                                                # the table is a real kinematic constraint for a horizontal grasp)

    # --- Motion waypoints ---
    pregrasp_standoff: float = 0.12           # m, pre-grasp distance outside the object along the approach axis
    lift_height: float = 0.08                 # m, vertical lift after the grasp closes
    place_height: Optional[float] = None      # m, world-frame Z to release at (None = derived from the tray's own top surface)
    safe_waypoint_height: float = 0.20        # m above the table, used for the safe/transit poses

    # --- Motion speed (0-1 scale factor on the controller's per-step
    # position/orientation gain — 1.0 = full speed, smaller = slower,
    # more controlled motion. See robot_controller.py::move_to_pose's
    # speed_scale parameter.) ---
    side_approach_speed: float = 1.0          # pre-grasp reconfiguration moves (safe waypoint <-> side pre-grasp)
    final_approach_speed: float = 0.4         # the final horizontal approach into contact — deliberately slower

    # --- Controller convergence tolerances ---
    # 0.008, not the tighter 0.006 the underlying primitives default to:
    # confirmed in-sandbox that 0.006 leaves several reconfiguration
    # stages (side_pregrasp, safe_high) converging to within
    # 0.0002-0.0006m of the tolerance and then timing out on step budget
    # right at the edge — not a systematic problem, just not enough
    # margin for ordinary run-to-run convergence variance. 8mm is still a
    # precise stop for a demo of this scale.
    controller_position_tolerance: float = 0.008   # m
    controller_orientation_tolerance: float = 0.08  # rad

    # --- Perception ---
    model_size: str = "small"                 # FLIP model size: tiny | small
    seed: int = 0                              # deterministic scene/episode seed

    def clamp_gripper_opening(self, estimated_width: float) -> float:
        target = estimated_width + self.gripper_open_margin
        return float(min(max(target, self.min_gripper_opening), self.max_gripper_opening))


if __name__ == "__main__":
    import dataclasses
    import json
    cfg = DemoConfig()
    print(json.dumps(dataclasses.asdict(cfg), indent=2))
