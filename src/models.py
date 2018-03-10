""" NN models
"""

# import nn libs
import keras
from keras import backend as K
from sklearn.decomposition import PCA
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten


def sequential(input_shape, output_length):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape))
    model.add(Activation('relu'))  # because relu is awesome
    # ... more layers

    model.add(Dense(output_length))
    model.add(Activation('softmax'))
    # in addition, return a function that displays information about the model
    return model, model.summary


def sequential_conv(input_shape, output_length):
    model = Sequential()
    model.add(
        Conv2D(
            16, kernel_size=(3, 3), activation='relu',
            input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(
        Dropout(0.25)
    )  # Dropout 25% of the nodes of the previous layer during training
    model.add(Flatten())  # Flatten, and add a fully connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(
        output_length,
        activation='softmax'))  # Last layer: 10 class nodes, with dropout
    return model, model.summary
