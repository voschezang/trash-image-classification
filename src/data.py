""" Functions that are specific to our dataset
"""
import pandas, collections
import os, sklearn, skimage, skimage.io, pandas, numpy as np
import keras.utils
# from sklearn import svm
# from skimage import data, io, filters
from collections import namedtuple
from scipy import misc, ndimage
from numpy import array
import config
from utils import utils
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from utils import utils # custom functions, in local environment

Dataset = namedtuple(
    'Dataset',
    ['train', 'test', 'validation', 'labels', 'dict_index_to_label', 'dict_label_to_index'])

print(""" Dataset :: namedtuple(
    ['train' = ['img_name']
    , 'test' = ['img_name']
    , 'validation' = ['img_name']
    , 'labels' = pandas.df('img_name','classification')
    , 'dict_index_to_label' = dict to convert label_index -> label_name
    , 'dict_label_to_index'= dict to convert label_name -> label_index
    """)


def init_dataset():
    # alt: use utils.Dataset
    labels = pandas.read_csv(config.dataset_dir + 'labels.csv')
    print(labels['classification'])
    train = os.listdir(config.dataset_dir + 'train/')
    test = os.listdir(config.dataset_dir + 'test/')
    validation = os.listdir(config.dataset_dir + 'validation/')
#     test_final = os.listdir(config.dataset_dir + 'test_final/')
#     train_final = os.listdir(config.dataset_dir + 'train_final/')

    # create a label dicts to convert labels to numerical data and vice versa
    # the order is arbitrary, as long as we can convert them back to the original classnames
    unique_labels = set(labels['classification'])
    dict_index_to_label_ = dict_index_to_label(unique_labels)
    dict_label_to_index_ = dict_label_to_index(unique_labels)
    # return data as a namedtuple
    return Dataset(train, test, validation, labels, dict_index_to_label_,
                   dict_label_to_index_)

def one_hot(values):
    values = array(values)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def labels_to_vectors(dataset, train_labels, test_labels, validation_labels):
    # dataset contains dicts to convert
    # TODO make sure that every label is present in both y_test and y_test
    train = textlabels_to_numerical(dataset, train_labels)
    test = textlabels_to_numerical(dataset, test_labels)
    validation = textlabels_to_numerical(dataset, validation_labels)
    y_train = keras.utils.to_categorical(train)
    y_test = keras.utils.to_categorical(test)
    y_validation = keras.utils.to_categorical(validation)

    return y_train, y_test, y_validation


def y_to_label_dict(dataset, vector=[]):
    n = vector.shape[0]
    result = {}  # :: {label: score}
    for i in range(n):
        label = dataset.dict_index_to_label[i]
        result[label] = vector[i]
    return result


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


def get_label(img_name='aed285c5eae61e3e7ddb5f78e6a7a977.jpg',
              labels=pandas.DataFrame()):
    # labels :: pandas.df :: { id: classification }
    # index_dict :: { value: index } :: { classification: int }
    label = labels.loc[labels['id'] == img_name]
    return label.classification.item()


# TODO rmv this function in svm-notebooks
def filename_to_class(labels, filename='aed285c5eae61e3e7ddb5f78e6a7a977.jpg'):
    # labels :: pandas.df :: { id: classification }
    # index_dict :: { value: index } :: { classification: int }
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
            classification = filename_to_class(labels, img_name)
            classification_index = dict_label_to_index[classification]
            img_labels.append(classification_index)
            # else: print('dims')
  
    return (img_data, img_labels)

def resize(image, dim1, dim2): 
    return (True, misc.imresize(image, (dim1, dim2)))

            
def extract_all(dataset, img_list, reshaper=crop, verbose=False):
    # labels :: df['id','class']
    print('extract all data:', len(img_list))
    x_train = []
    y_train = []
    for img_name in img_list:
        if not img_name[-4:] == '.jpg':
            img_name += '.jpg'
        img = read_img('train/', img_name, verbose)
        
        x_train.append(img)
        y_train.append(get_label(img_name, dataset.labels))
    x_train = np.stack(x_train)
    amt = x_train.shape[0]
    return (x_train, y_train, amt)

def extract_all_test(dataset, img_list, reshaper=crop, verbose=False):
    # labels :: df['id','class']
    print('extract all data:', len(img_list))
    x_test = []
    y_test = []
    for img_name in img_list:
        if not img_name[-4:] == '.jpg':
            img_name += '.jpg'
        img = read_img('test/', img_name, verbose)
        x_test.append(img)
        y_test.append(get_label(img_name, dataset.labels))
    x_test = np.stack(x_test)
    amt = x_test.shape[0]
    return (x_test, y_test, amt)

def extract_all_validation(dataset, img_list, reshaper=crop, verbose=False):
    # labels :: df['id','class']
    print('extract all data:', len(img_list))
    x_validation = []
    y_validation = []
    for img_name in img_list:
        if not img_name[-4:] == '.jpg':
            img_name += '.jpg'
        img = read_img('validation/', img_name, verbose)
        x_validation.append(img)
        y_validation.append(get_label(img_name, dataset.labels))
    x_validation = np.stack(x_validation)
    amt = x_validation.shape[0]
    return (x_validation, y_validation, amt)

def items_with_label(labels, label='scottish_deerhound'):
    # return all items with label x
    #:labels :: pandas.DataFrame[item,'label']
    id_col, label_col = labels.keys()[0:2]
    return labels.loc[labels[label_col] == label][id_col]


def top_classes(labels, amt=3):
    # return classes that have the most instances
    #:labels :: pandas.DataFrame['id','classification']
    counter = collections.Counter(labels['classification'])
    ls = list(counter.items())
    sorted_list = sorted(ls, key=lambda x: x[1], reverse=True)[:amt]
    return [label for label, _ in sorted_list]


def show_info(data):
    print('__ info: __')
    print('length: ', len(data))
    print('type: ', type(data))
    if type(data) is np.ndarray:
        print('shape: ', data.shape)


def gen_random_img(example_img):
    # generate a random image with the shape and datatype of the original image
    return np.empty_like(example_img)
