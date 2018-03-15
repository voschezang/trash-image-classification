""" Functions that transform images (np matrices)
Including:
- img transformation
- face detection
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


def detect_faces(img, scale=1.3, nn=5):
    # https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
    # minSize and maxSize are not specified
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        config.cascades_dir + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale, minNeighbors=nn)
    return faces, gray


def detect_eyes(gray_img, faces=[], min_eyes=2, scale=1.3, nn=1):
    # search for eyes within each found face
    # save faces which contain min_eyes eyes
    result = {}  # :: {face: eyes}
    eye_cascade = cv2.CascadeClassifier(
        config.cascades_dir + 'haarcascade_eye.xml')
    for face in faces:
        (x, y, w, h) = face
        roi_gray = gray_img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=scale, minNeighbors=nn)
        result[faces] = eyes

        # save the faces that contain at least n eyes
        if len(eyes) > min_eyes:
            result[face] = eyes

    return result


# def draw_faces()
# def draw_eyes()
