from environment import PickPlaceEnv, SIDE_CAMERA_NAME
from PIL import Image
import numpy as np

env = PickPlaceEnv(
    has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
    camera_names=["frontview", SIDE_CAMERA_NAME], camera_heights=480, camera_widths=640,
    camera_depths=False, num_distractors=0, seed=0,
)
env.reset()
for _ in range(10):
    env.sim.step()
obs = env._get_observations(force_update=True)
Image.fromarray(obs["frontview_image"]).save("/tmp/frontview.png")
Image.fromarray(obs[SIDE_CAMERA_NAME + "_image"]).save("/tmp/side.png")

cam_id = env.sim.model.camera_name2id(SIDE_CAMERA_NAME)
print("cam pos", env.sim.model.cam_pos[cam_id])
print("cam quat", env.sim.model.cam_quat[cam_id])
print("table_offset", env.table_offset)
print("robot base pos", env.robots[0].robot_model.base_xpos_offset["table"](env.table_full_size[0]))

# also print target object pos to confirm it's on the table
print("target pos", env.sim.data.body_xpos[env.target_object_body_id])
env.close()
