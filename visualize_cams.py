from keras.models import *
from keras.callbacks import *
import keras.backend as K
import cv2
import os
import sys
import h5py
import utils_datasets as ud
import numpy as np
from scipy.misc import imread, imresize, imsave
from cam_utils import extract_feat_and_cam_masks, visualize_cam
from models_utils import choose_model

# Define Paths
path_images = '/imatge/ajimenez/work/ITR/oxford/datasets_hdf5/places/1024x720/'

output_path_heatmaps = '/imatge/ajimenez/workspace/ITR/cam_heatmaps/'

cams_name = '/imatge/ajimenez/work/ITR/results_ITR/cam_masks/googlenet_places/oxford_480x480'

dataset = 'Oxford'

if dataset == 'Oxfnkjnord':
    f_dataset = open(path_images+'oxford_h_info.txt')
    name_h = f_dataset.readline()
    n_chunks_h = int(f_dataset.readline())
    f_dataset.close()
    f_dataset = open(path_images + 'oxford_v_info.txt')
    name_v = f_dataset.readline()
    n_chunks_v = int(f_dataset.readline())
    f_dataset.close()

top_classes = 5
batchsize = 50

model_name = 'googlenet'

# Load Data
#print 'Loading data...'
#images, image_names = uf.read_data_oxford(path_images)

#print images[0,0,0:10,0:10]
#print images[0,0,100:120,0:10]

#print images.shape
#print image_names.shape



# Extract CAMs

images, image_names = ud.read_dataset(path_images + 'oxford_queries_h.h5')
#images, image_names = uf.read_data_oxford('../datasets_hdf5/oxford_test_224/oxford.h5')
#print images.shape
x = np.zeros((1, 3, 720, 1024))

x[0, :, :, :] = images[0]


model = choose_model(model_name, 'h')

imsave('asdf.jpg', np.transpose(images[0], (1, 2, 0)))
#print x.shape
visualize_cam(model, 1, x, top_classes, output_path_heatmaps, image_names)




#extract_cam_masks(googlenet, batchsize, images, top_classes, cams_name)
#images [num_images, 3, width, height]
#x = np.zeros((1, 3, 224, 224))
#x[0, :, :, :] = images[1170]

#scores = googlenet.predict(x, batch_size=1)
#print scores

#print googlenet.predict(x, batch_size=1).sort()[::-1]


#x[1, :, :, :] = images[10]
#visualize_cam(googlenet, 1, x, top_classes, output_path_heatmaps, image_names)



