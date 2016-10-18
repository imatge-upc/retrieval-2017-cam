from keras.models import *
from keras.callbacks import *
import keras.backend as K
import cv2
import os
import sys
import h5py
import utils_oxford as uf
import numpy as np
from scipy.misc import imread, imresize, imsave
from cam_utils import extract_feat_and_cam_masks, visualize_cam

# Define Paths
path_images = '/imatge/ajimenez/workspace/ITR/datasets_hfd5/oxford_test_224/oxford_bb.h5'
output_path_heatmaps = '/imatge/ajimenez/work/results_ITR/cam_heatmaps/'

cams_name = '/imatge/ajimenez/work/results_ITR/cam_masks/googlenet_places/oxford_bb_c_f_s.h5'



# Load Data
print 'Loading data...'
images, image_names = uf.read_data_oxford(path_images)

print images[0,0,0:10,0:10]
print images[0,0,100:120,0:10]

print images.shape
print image_names.shape

# Load Model
print 'Loading model...'
googlenet = model_from_json(open('../models/inception_cam_model_structure.json').read())

googlenet.load_weights('../models/inception_cam_weights.h5')

googlenet.summary()

# Extract CAMs
print 'Extracting CAMs...'
batchsize = 100
top_classes = 205


# extract_cam_masks(googlenet, batchsize, images, top_classes)
#images [num_images, 3, width, height]
#x = np.zeros((1, 3, 224, 224))
#x[0, :, :, :] = images[1170]

#scores = googlenet.predict(x, batch_size=1)
#print scores

#print googlenet.predict(x, batch_size=1).sort()[::-1]


#x[1, :, :, :] = images[10]
#visualize_cam(googlenet, 1, x, top_classes, output_path_heatmaps, image_names)

extract_feat_and_cam_masks(googlenet, batchsize, images, top_classes, cams_name)

