""" Functions that are specific to our dataset
"""
import pandas
import os, sklearn, skimage, pandas, numpy as np
# from sklearn import svm
# from skimage import data, io, filters
from collections import namedtuple

import config
from utils import utils

# from utils import utils # custom functions, in local environment

Dataset = namedtuple(
    'Dataset',
    ['train', 'test', 'labels', 'dict_index_to_label', 'dict_label_to_index'])

print(""" Dataset :: namedtuple(
    ['train' = ['img_name']
    , 'test' = ['img_name']
    , 'labels' = pandas.df('img_name','breed')
    , 'dict_index_to_label' = dict to convert label_index -> label_name
    , 'dict_label_to_index'= dict to convert label_name -> label_index
    """)


def init_dataset():
    # alt: use utils.Dataset
    labels = pandas.read_csv(config.dataset_dir + 'labels.csv')
    train = os.listdir(config.dataset_dir + 'train/')
    test = os.listdir(config.dataset_dir + 'test/')

    # create a label dicts to convert labels to numerical data and vice versa
    # the order is arbitrary, as long as we can convert them back to the original classnames
    unique_labels = set(labels['breed'])
    dict_index_to_label_ = dict_index_to_label(unique_labels)
    dict_label_to_index_ = dict_label_to_index(unique_labels)
    # return data as a namedtuple
    return Dataset(train, test, labels, dict_index_to_label_,
                   dict_label_to_index_)


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


def read_img(folder='train',
             img_name='aed285c5eae61e3e7ddb5f78e6a7a977.jpg',
             verbose=False):
    if not folder[-1] == '/':
        folder += '/'
    filename = config.dataset_dir + folder + img_name
    if verbose: print('reading file: ' + filename)
    return skimage.io.imread(filename)


# Collect test data (+labels)
# ignore images that have different dimensions
# extract_data :: Dataset -> Bool -> ([np.array (flattened)], [label])
def extract_data(dataset, img_list, dimensions, verbose=False):
    print('extract data:', len(img_list))
    labels = dataset.labels
    dict_label_to_index = dataset.dict_label_to_index
    img_data = []
    img_labels = []
    for img_name in img_list:
        img = read_img('train/', img_name, verbose)
        if img.shape == dimensions:
            img_data.append(img.flatten())
            breed = filename_to_class(labels, img_name)
            breed_index = dict_label_to_index[breed]
            img_labels.append(breed_index)
            # else: print('dims')
    return (img_data, img_labels)


def gen_random_img(example_img):
    # generate a random image with the shape and datatype of the original image
    return np.empty_like(example_img)
