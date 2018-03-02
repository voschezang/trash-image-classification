""" Functions specific to our dataset
"""
import pandas
import os, sklearn, skimage, pandas, numpy as np
# from sklearn import svm
# from skimage import data, io, filters

import config
from utils import utils

# from src.utils import utils # custom functions, in local environment


def init_dataset():
    # alt: use utils.Dataset
    labels = pandas.read_csv(config.dataset_dir + 'labels.csv')
    train = os.listdir(config.dataset_dir + 'train/')
    test = os.listdir(config.dataset_dir + 'test/')
    return train, test, labels


def dict_index_to_label(labels):
    # labels :: list or set()
    # return { int: label }
    unique_labels3 = set(labels)
    return {k: v for k, v in enumerate(unique_labels3)}


def dict_label_to_index(labels):
    # labels :: list or set()
    # return { label: int }
    unique_labels = set(labels)
    return {k: v for v, k in enumerate(unique_labels)}


def filename_to_class(labels, filename='aed285c5eae61e3e7ddb5f78e6a7a977.jpg'):
    # labels :: pandas.df :: { id: breed }
    # index_dict :: { value: index } :: { breed: int }

    label = labels.loc[labels['id'] == utils.stem(filename)]
    return label.breed.item()


def read_img(folder='train', img_name='aed285c5eae61e3e7ddb5f78e6a7a977.jpg'):
    if not folder[-1] == '/':
        folder += '/'
    filename = config.dataset_dir + folder + img_name
    return skimage.io.imread(filename)


def gen_random_img(example_img):
    # generate a random image with the shape and datatype of the original image
    return np.empty_like(example_img)
