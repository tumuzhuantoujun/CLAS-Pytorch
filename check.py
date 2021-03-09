from utils.read_data import resize_images
import imageio
import numpy as np

a2c = resize_images(np.stack(imageio.mimread('./misc/Pat44_A2C.gif'), axis = 0), (256, 256)).astype(np.uint8)
a4c = resize_images(np.stack(imageio.mimread('./misc/Pat44_A4C.gif'), axis = 0), (256, 256)).astype(np.uint8)
imageio.mimsave('a2c.gif', a2c)
imageio.mimsave('a4c.gif', a4c)