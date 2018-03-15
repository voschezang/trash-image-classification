""" Functions that transform images (np matrices)
"""
import numpy as np
import sklearn, os, sys
from sklearn import svm
from skimage import data, io, filters
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import cv2

import config

# (important)
#   cv2.imread() and skimage.io.imread() produce different results

### --------------------------------------------------------------------
### Pixel transformations
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
