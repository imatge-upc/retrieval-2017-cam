from scipy.misc import imread, imresize, imsave
import numpy as np
import os
import math
import h5py
import matplotlib.pyplot as plt
import sys


def create_folders(path):
    if not os.path.exists(path):
        print 'Creating path: ', path
        os.makedirs(path)
    else:
        print 'Path already exists'


def load_data(filepath):
    with h5py.File(filepath, 'r') as hf:
        data = np.array(hf.get('data'))
        #print 'Shape of the array features: ', data.shape
        return data


def save_data(data, path, name):
    with h5py.File(path + name, 'w') as hf:
        hf.create_dataset('data', data=data)


def preprocess_images(images, img_width, img_height, mean_value):
    print ("Preprocessing Images... ")
    num_images = len(images)
    x = np.zeros((num_images, 3, img_height, img_width), dtype=np.float32)
    for i in range(0, num_images):
        # print str(i + 1) + "/" + str(num_images)
        images[i] = imresize(images[i], [img_height, img_width]).astype(dtype=np.float32)
        # print images[i].shape
        # RGB -> BGR
        R = np.copy(images[i][:, :, 0])
        B = np.copy(images[i][:, :, 2])
        images[i][:, :, 0] = B
        images[i][:, :, 2] = R

        # Subtract mean
        images[i][:, :, 0] -= mean_value[0]
        images[i][:, :, 1] -= mean_value[1]
        images[i][:, :, 2] -= mean_value[2]

        x[i, :, :, :] = np.transpose(images[i], (2, 0, 1))
        #print x[i].shape
        #print 'Preprocessed ' + str(num_images)
    print x.shape
    return x


def preprocess_query(image, img_width, img_height, mean_value):
    print ("Preprocessing query... ")
    image = imresize(image, [img_height, img_width]).astype(dtype=np.float32)
    # RGB -> BGR
    R = np.copy(image[:, :, 0])
    B = np.copy(image[:, :, 2])
    image[:, :, 0] = B
    image[:, :, 2] = R

    # Subtract mean
    image[:, :, 0] -= mean_value[0]
    image[:, :, 1] -= mean_value[1]
    image[:, :, 2] -= mean_value[2]

    image = np.transpose(image, (2, 0, 1))
    print image.shape
    return image