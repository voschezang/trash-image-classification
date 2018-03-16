import numpy as np
import sklearn, os, sys
from sklearn import svm
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL
from PIL import ImageEnhance

print('------')
print(os.getcwd())
os.chdir('src/')

print('------')
print(os.getcwd())

import data, config, image

dataset = data.init_dataset()

# Import train+test data
img_dir = config.dataset_dir + 'train/'
img_name = img_dir + dataset.train[0]

img = PIL.Image.open(img_name)  # fill in directory
img2 = image.transform_image(img, contrast=2, brightness=2)
# end_img.save('') # fill in directory
#     end_img.show()
img2

img = image.transform_random(img)
img
