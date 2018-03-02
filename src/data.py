""" Functions specific to our dataset
"""
import pandas
import os, sklearn, pandas, numpy as np
from sklearn import svm
from skimage import data, io, filters

# from src.utils import utils # custom functions, in local environment


def stem(filename='aed285c5eae61e3e7ddb5f78e6a7a977.jpg'):
    # rmv file extension (.jpg)
    return filename.split('.')[0]


def filename_to_class_index(filename='aed285c5eae61e3e7ddb5f78e6a7a977.jpg',
                            labels=[],
                            index_dict={}):
    # labels :: pandas.df :: { id: breed }
    # index_dict :: { value: index } :: { breed: int }
    label = labels.loc[labels['id'] == stem(filename)]
    return label_value_to_index[label.breed.item()]
