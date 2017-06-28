import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import ZeroPadding2D
import matplotlib.pylab as plt
import numpy as np
import theano.tensor.nnet.abstract_conv as absconv
import cv2
import h5py
import os


def vggcam(nb_classes, input_shape=(3, None, None), num_input_channels=1024):
    '''
    :param nb_classes: # classes (IMAGENET = 1000)
    :param input_shape: image shape
    :param num_input_channels: channels CAM layer
    :param bounding_box:  Query processing (Oxford/Paris)
    :return: instance of the model VGG-16 CAM
    '''

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='relu5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='relu5_3'))

    # Add another conv layer with ReLU + GAP
    model.add(Convolution2D(num_input_channels, 3, 3, activation='relu', border_mode="same", name='CAM_relu'))

    # Global Average Pooling
    model.add(GlobalAveragePooling2D(name='CAM_pool'))

    # Add the W layer
    model.add(Dense(nb_classes, activation='softmax'))

    model.name = "vgg_cam"

    return model

