from utils.read_data import resize_images
import imageio
import numpy as np

a2c = resize_images(np.stack(imageio.mimread('./misc/Pat44_A2C_images.gif'), axis = 0), (256, 256)).astype(np.uint8)
a4c = resize_images(np.stack(imageio.mimread('./misc/Pat44_A4C_images.gif'), axis = 0), (256, 256)).astype(np.uint8)
imageio.mimsave('Pat44_A2C_images.gif', a2c, fps=10)
imageio.mimsave('Pat44_A4C_images.gif', a4c, fps=10)

a2c = np.stack(imageio.mimread('./misc/Pat44_A2C_segmentation.gif'), axis = 0).astype(np.uint8)
a4c = np.stack(imageio.mimread('./misc/Pat44_A4C_segmentation.gif'), axis = 0).astype(np.uint8)
imageio.mimsave('Pat44_A2C_segmentation.gif', a2c, fps=10)
imageio.mimsave('Pat44_A4C_segmentation.gif', a4c, fps=10)