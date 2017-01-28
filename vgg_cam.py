import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.convolutional import ZeroPadding2D
import matplotlib.pylab as plt
import numpy as np
import theano.tensor.nnet.abstract_conv as absconv
import cv2
import h5py
import os


def VGGCAM(nb_classes, input_shape, num_input_channels=1024, bounding_box=None):
    """
    Build Convolution Neural Network
    args : nb_classes (int) number of classes
    returns : model (keras NN) the Neural Net model
    """

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

    if bounding_box is None:
        # Add another conv layer with ReLU + GAP
        model.add(Convolution2D(num_input_channels, 3, 3, activation='relu', border_mode="same", name='CAM_relu'))

        model.add(AveragePooling2D(strides=[input_shape[1] / 16, input_shape[2] / 16],
                                   pool_size=[input_shape[1] / 16, input_shape[2] / 16], name='CAM_pool'))
        model.add(Flatten())
        # Add the W layer
        model.add(Dense(nb_classes, activation='softmax'))
    else:
        print 'bounding box: ', bounding_box
        x, y, dx, dy = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]

        f_x, f_y, f_dx, f_dy = int((x - (x % 16)) / 16), int((y - (y % 16)) / 16), \
                               int((dx - (dx % 16)) / 16), int((dy - (dy % 16)) / 16)

        if f_dy >= input_shape[1]/16:
            f_dy = input_shape[1]/16

        if f_dx >= input_shape[2] / 16:
            f_dx = input_shape[2] / 16

        coord_y2 = input_shape[1] / 16 - f_dy
        coord_x2 = input_shape[2] / 16 - f_dx

        print f_x, f_y, f_dx, f_dy
        print coord_y2
        print coord_x2
        print '1'
        model.add(Cropping2D(cropping=((f_y, coord_y2), (f_x, coord_x2))))
        print '2'
        # Add another conv layer with ReLU + GAP
        model.add(Convolution2D(num_input_channels, 3, 3, activation='relu', border_mode="same", name='CAM_relu'))
        print '3'
        model.add(AveragePooling2D(strides=[f_dy-f_y, f_dx-f_x],
                                   pool_size=[f_dy-f_y, f_dx-f_x], name='CAM_pool'))
        model.add(Flatten())
        print '4'
        # Add the W layer
        model.add(Dense(nb_classes, activation='softmax'))
        print'5'

    model.name = "VGGCAM"
    print'6'
    return model

