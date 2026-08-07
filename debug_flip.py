from PIL import Image
import numpy as np
img = np.array(Image.open("/tmp/frontview.png"))
Image.fromarray(img[::-1]).save("/tmp/frontview_flipped.png")
