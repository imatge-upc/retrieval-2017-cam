from keras.models import *
from keras.callbacks import *
import keras.backend as K
import os
import sys
import h5py
import utils_oxford as uf
import numpy as np
from cam_utils import extract_feat_and_cam_masks
from models_utils import choose_model, extract_features
from utils_datasets import read_dataset_properties, read_dataset, create_folders

# Define Paths

dataset = 'Oxford'
#dataset = 'Paris'

local_search = True

batchsize = 15

model_name = 'vgg_16_imagenet'

layer = 'conv5_3'

dim = '1024x720'

if dataset == 'Oxford':
    # Path Dataset
    dataset_path = '/imatge/ajimenez/work/ITR/oxford/datasets_hdf5/imagenet/' + dim + '/'
    name_h, n_chunks_h, batchsize_h, total_imgs_h = read_dataset_properties(dataset_path+'oxford_h_info.txt')
    name_v, n_chunks_v, batchsize_v, total_imgs_v = read_dataset_properties(dataset_path + 'oxford_v_info.txt')
    #feat_name = '/imatge/ajimenez/work/ITR/oxford/features/vgg_imagenet/oxford_1024x720/'
    feat_name = '/imatge/ajimenez/work/ITR/oxford/features/'+model_name+'/'+layer+'/'+dim+'/'
    create_folders(feat_name)
    feat_name += 'oxford'

if dataset == 'Paris':
    dataset_path = '/imatge/ajimenez/work/ITR/paris/datasets_hdf5/imagenet/'+dim+'/'
    name_h, n_chunks_h, batchsize_h, total_imgs_h = read_dataset_properties(dataset_path + 'paris_h_info.txt')
    name_v, n_chunks_v, batchsize_v, total_imgs_v = read_dataset_properties(dataset_path + 'paris_v_info.txt')
    feat_name = '/imatge/ajimenez/work/ITR/paris/features/'+model_name+'/'+layer+'/'+dim+'/'
    create_folders(feat_name)
    feat_name += 'paris'

# For 1024x720 15 images at a time, 55 GB mem aprox

print 'Dataset: ', dataset
print 'Batch size: ', batchsize
print 'Local search: ', local_search
print 'Model: ', model_name
print 'Layer: ', layer

t = time.time()
# Horizontal

model = choose_model(model_name, 'h')


# for i in range(0, n_chunks_h):
#     print'Extracting CAMs for chunk number ', i
#     images, image_names = read_dataset(name_h + '_' + str(i) + '.h5')
#     features_name_chunk = feat_name + '_h_' + str(i)
#     extract_features(model, layer, batchsize, images, features_name_chunk)


if local_search:
    if dataset == 'Oxford':
        images, image_names = read_dataset(dataset_path+'oxford_queries_h_ls.h5')
    elif dataset == 'Paris':
        images, image_names = read_dataset(dataset_path + 'paris_queries_h_ls.h5')
    features_name_chunk = feat_name + '_queries_h_ls'
else:
    if dataset == 'Oxford':
        images, image_names = read_dataset(dataset_path+'oxford_queries_h.h5')
    elif dataset == 'Paris':
        images, image_names = read_dataset(dataset_path + 'paris_queries_h.h5')
    features_name_chunk = feat_name + '_queries_h'

extract_features(model, layer, batchsize, images, features_name_chunk)

# Vertical

model = choose_model(model_name, 'v')

# for i in range(0, n_chunks_v):
#     print'Extracting CAMs for chunk number ', i
#     images, image_names = uf.read_data_oxford(name_v + '_' + str(i) + '.h5')
#     features_name_chunk = feat_name + '_v_' + str(i)
#     extract_features(model, layer, batchsize, images, features_name_chunk)

if local_search:
    if dataset == 'Oxford':
        images, image_names = read_dataset(dataset_path+'oxford_queries_v_ls.h5')
    elif dataset == 'Paris':
        images, image_names = read_dataset(dataset_path + 'paris_queries_v_ls.h5')
    features_name_chunk = feat_name + '_queries_v_ls'
else:
    if dataset == 'Oxford':
        images, image_names = read_dataset(dataset_path+'oxford_queries_v.h5')
    elif dataset == 'Paris':
        images, image_names = read_dataset(dataset_path + 'paris_queries_v.h5')
    features_name_chunk = feat_name + '_queries_v' 

extract_features(model, layer, batchsize, images, features_name_chunk)

print 'Total time elapsed: ', time.time()-t
