""" Functions that transform images (np matrices)
"""
import numpy as np
import sklearn, os, sys
from sklearn import svm
# from skimage import filters
import skimage.io, cv2, PIL
from PIL import ImageEnhance
import matplotlib.pyplot as plt

import config
from utils import utils

# (important)
#   cv2.imread() and skimage.io.imread() produce different results

### --------------------------------------------------------------------
### Full img transformations
# ## --------------------------------------------------------------------


def normalize(img=np.array()):
    return img / 255


def denormalize(img=np.array()):
    return img * 255


def to_np_array(PIL_img, mode='RGB'):
    # convert PIL.Image to np.ndarray
    return np.array(PIL_img.convert(mode))


def to_PIL_Image(img):
    # convert np.ndarray to PIL.Image
    return PIL.Image.fromarray(img)


def encode_super_img(img, mutate=False):
    mutation = transform_random(img)
    rgb = to_np_array(mutation, mode='RGB')
    img_ = show_edges(mutation.copy())
    edges = to_np_array(img_, mode='P')
    # reshape 'edges' to fit 'rgb'
    edges = edges[:, :, np.newaxis]
    encoded = np.append(rgb, edges, axis=2)
    return normalize(encoded)


def decode_super_img(img):
    return denormalize(img[:, :, :3])


def transform_image(img, sharpness=1.5, bw=1.0, contrast=0.8, brightness=1):
    """
    sharpness (factor)
    - 0.0 gives a blurred image,
      1.0 gives the original image,
      2.0 gives a sharpened image.
    bw = black & white color factor
    - 0.0 gives a black and white image.
      1.0 gives the original image.
    conf = contrast factor
    - 0.0 gives a solid grey image.
     1.0 gives the original image.
    brightness factor
    - 0.0 gives a black image.
      1.0 gives the original image.
    """
    trans_img = img.copy()

    trans_img = ImageEnhance.Contrast(trans_img)
    trans_img = trans_img.enhance(contrast)

    trans_img = ImageEnhance.Sharpness(trans_img)
    trans_img = trans_img.enhance(sharpness)

    trans_img = ImageEnhance.Color(trans_img)
    trans_img = trans_img.enhance(bw)

    trans_img = ImageEnhance.Brightness(trans_img)
    result_img = trans_img.enhance(brightness)

    return result_img


def show_edges(img):
    return img.filter(PIL.ImageFilter.FIND_EDGES)


def rotate_img(img, amt):
    return img.rotate(45)


def transform_random(img, scale=[1, 1, 1]):
    # params are from a skewed distribution
    # mutation_vector = np.random.random(3) * scale
    lowest = 0.2
    highest = 4
    mutation_vector = [
        utils.random_skewed(lowest, highest, skew=2) for _ in range(3)
    ]
    # mutation_vector = [1, 1, 1] + np.random.random(3) * scale

    s = mutation_vector[0]
    c = mutation_vector[1]
    b = mutation_vector[2]
    return transform_image(img, sharpness=s, contrast=c, brightness=b)


### --------------------------------------------------------------------
### Indiviual pixel transformations
### --------------------------------------------------------------------


def replace(v1, v2):
    return v2


def add(v1, v2):
    # mean of r,g,b
    return np.mean([v1, v2], axis=0)
    # return v1 + np.minimum(v1, v2) / 2


def overlay(v1, v2):
    # add smallest element values
    intermediate = np.minimum(v1, v2)
    return v1 + intermediate  # np.minimum(v1,v2)


### --------------------------------------------------------------------
### Sampling & generative functions
### --------------------------------------------------------------------

# def sample_from_matrix(m):
#     # m :: 2d matrix
#     x = np.random.randint(0,m.shape[1] - 1)
#     y = np.random.randint(0,m.shape[0] - 1)
#     print('sampl m',x,y)
#     return m[y,x]


def sample_from_img(img):
    # return a random pixel (either rgb or bw)
    x = np.random.randint(0, img.shape[1] - 1)
    y = np.random.randint(0, img.shape[0] - 1)
    #     r = img[:,:,0]
    #     g = img[:,:,1]
    #     b = img[:,:,2]
    return img[y, x]


def random_img(dims=(10, 10, 3)):
    return np.random.randint(0, 255, size=dims, dtype=np.uint8)


def add_img(img, img2, x, y, f=replace):
    # f = function that mutates pixel-vectors
    # np index: [y][x]
    max_y, max_x = img.shape[0:2]
    h, w = img2.shape[0:2]
    xs = np.clip(np.arange(x, x + w), 0, max_x - 1)
    ys = np.clip(np.arange(y, y + h), 0, max_y - 1)
    print('sha', img.shape)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            # y is the first index
            img[y, x] = f(img[y, x], img2[j, i])

    return img


def add_noise(img, x, y, w, h, f=sample_from_img):
    # f = function that mutates pixel-vectors
    # np index: [y][x]
    max_y, max_x = img.shape[0:2]
    xs = np.clip(np.arange(x, x + w), 0, max_x - 1)
    ys = np.clip(np.arange(y, y + h), 0, max_y - 1)
    for x in xs:
        for y in ys:
            # y is the first index
            img[y, x] = f(img)

    return img


### --------------------------------------------------------------------
### Vector manipulation
### --------------------------------------------------------------------


def scale_vector(x, w, scale=1., max_range=1):
    # scale & center a 1-dimensional vector (not cartesian-like),
    # vector: [x, x + dx]
    delta = (w * scale) - w
    x -= int(round(delta / 2.))
    w += delta  # w + delta/2 + delta/2
    (x, w) = np.clip([x, w], 0, max_range)
    return int(round(x)), int(round(w))


def extend_coordinates(x, y, w, h, img=[[]], scale=1.0, lower=1.0):
    # selection is a rectangle that starts at x,y with dims w,h
    # increase selection while preserving aspect ratio
    # increase d-width, d-height
    (x, w) = scale_vector(x, w, scale, img.shape[1])
    (y, h) = scale_vector(y, h, scale, img.shape[0])

    # increase the bottom of the selection
    # this can be used to remove human-bodies from imgs
    h = int(round(h * lower))

    return x, y, w, h
