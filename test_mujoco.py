import mujoco

model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/panda.xml")
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

renderer = mujoco.Renderer(model)
renderer.update_scene(data)
img = renderer.render()

import cv2
cv2.imwrite("panda_test.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("Saved panda_test.png — open it and confirm the arm is visible.")