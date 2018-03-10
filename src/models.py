""" NN models
"""

# import nn libs
import keras
from keras import backend as K
from sklearn.decomposition import PCA
from keras.utils import to_categorical
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam


def sequential(input_shape, output_length):
    model = keras.Sequential()
    model.add(Dense(128, input_shape=input_shape))
    model.add(Activation('relu'))  # because relu is awesome
    # ... more layers

    model.add(Dense(output_length))
    model.add(Activation('softmax'))
    # in addition, return a function that displays information about the model
    return model, model.summary
