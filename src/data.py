""" Functions that are specific to our dataset
"""
import pandas
import os, sklearn, skimage, skimage.io, pandas, numpy as np
import keras.utils
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


def labels_to_vectors(dataset, train_labels, test_labels):
    # dataset contains dicts to convert
    train = textlabels_to_numerical(dataset, train_labels)
    test = textlabels_to_numerical(dataset, test_labels)
    y_train = keras.utils.to_categorical(train)
    y_test = keras.utils.to_categorical(test)
    return y_train, y_test


def textlabels_to_numerical(dataset, labels):
    # transform ['label'] => [index]
    # (list of text => list of indices)
    return [dataset.dict_label_to_index[label] for label in labels]


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


def get_label(img_name='aed285c5eae61e3e7ddb5f78e6a7a977.jpg', labels=[]):
    # labels :: pandas.df :: { id: breed }
    # index_dict :: { value: index } :: { breed: int }
    label = labels.loc[labels['id'] == utils.stem(img_name)]
    return label.breed.item()


# TODO rmv this function in svm-notebooks
def filename_to_class(labels, filename='aed285c5eae61e3e7ddb5f78e6a7a977.jpg'):
    # labels :: pandas.df :: { id: breed }
    # index_dict :: { value: index } :: { breed: int }
    return get_label(filename, labels)


def read_img(folder='train',
             img_name='aed285c5eae61e3e7ddb5f78e6a7a977.jpg',
             verbose=False):
    if not folder[-1] == '/':
        folder += '/'
    filename = config.dataset_dir + folder + img_name
    if verbose: print('reading file: ' + filename)
    return skimage.io.imread(filename)


def crop(img, size=250, verbose=False):
    # I think the smallest img has shape 160x160
    a, b, c = img.shape
    if a < size or b < size:
        if verbose: print('WARNING, img too small', a, b, 'but size is', size)
        return (False, img)
    return (True, img[0:size, 0:size])


# def flatten(img):
#     return np.array(crop(img)).flatten()


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


def extract_all(dataset, img_list, reshaper=crop, verbose=False):
    print('extract all data:', len(img_list))
    x_train = []
    y_train = []
    for img_name in img_list:
        img = read_img('train/', img_name, verbose)
        success, img = reshaper(img, verbose=verbose)
        if success:
            x_train.append(img)
            y_train.append(get_label(img_name, dataset.labels))
    x_train = np.stack(x_train)
    amt = x_train.shape[0]
    return (x_train, y_train, amt)


def show_info(data):
    print('__ info: __')
    print('length: ', len(data))
    print('type: ', type(data))
    if type(data) is np.ndarray:
        print('shape: ', data.shape)


def gen_random_img(example_img):
    # generate a random image with the shape and datatype of the original image
    return np.empty_like(example_img)
