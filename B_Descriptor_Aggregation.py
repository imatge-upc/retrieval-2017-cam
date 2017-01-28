import numpy as np
import time
from sklearn.decomposition import PCA
import sys
from pooling_functions import descriptor_aggregation, retrieve_n_descriptors
from utils_datasets import create_folders, save_data, load_data

# Parameters
dataset = 'Paris'
num_classes = 50

num_classes_pca = 1

model_name = 'Vgg_16_CAM'
layer = 'relu5_1'
dim = '1024x720'

local_search = False

# PCA
apply_pca = True
pca_dim = 512

t_0 = time.time()

if dataset == 'Oxford':
    path_descriptors = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/' + \
                       model_name + '/' + layer + '/' + dim + '/crow/'
    if local_search:
        path_descriptors += 'ls/'
    create_folders(path_descriptors)
    cam_descriptors_path = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'oxford_all_64_wp.h5'

    pca_descriptors_path = '/imatge/ajimenez/work/ITR/paris/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'paris_all_64_wp.h5'
    num_images = 5063
    num_img_pca = 6392
    name_descriptors = 'oxford_' + str(num_classes)
    pca_name = '_pca_' + str(pca_dim) + '_paris_'+str(num_classes_pca)

if dataset == 'Oxford105k':
    path_descriptors = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/' + \
                       model_name + '/' + layer + '/' + dim + '/crow/'
    if local_search:
        path_descriptors += 'ls/'
    create_folders(path_descriptors)
    cam_descriptors_path = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'oxford_all_64_wp.h5'

    pca_descriptors_path = '/imatge/ajimenez/work/ITR/paris/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'paris_all_64_wp.h5'
    num_images = 5063
    num_img_pca = 6392
    name_descriptors = 'oxford_' + str(num_classes)
    pca_name = '_pca_' + str(pca_dim) + '_paris_'+str(num_classes_pca)

if dataset == 'Paris':
    path_descriptors = '/imatge/ajimenez/work/ITR/paris/descriptors_new/' + \
                       model_name + '/' + layer+'/' + dim + '/crow/'
    if local_search:
        path_descriptors += 'ls'
    create_folders(path_descriptors)
    cam_descriptors_path = '/imatge/ajimenez/work/ITR/paris/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'paris_all_64_wp.h5'
    pca_descriptors_path = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'oxford_all_64_wp.h5'
    num_images = 6392
    num_img_pca = 5063
    name_descriptors = 'paris_' + str(num_classes)
    pca_name = '_pca_' + str(pca_dim) + '_oxford_'+str(num_classes_pca)


########################################################################################################################

cam_descriptors = load_data(cam_descriptors_path)

# Compute PCA
if apply_pca:
    tpca = time.time()
    pca_desc = retrieve_n_descriptors(num_classes_pca, num_img_pca, load_data(pca_descriptors_path))
    print 'Computing PCA...'
    print pca_desc.shape
    name_descriptors += pca_name
    pca_matrix = PCA(n_components=pca_dim, whiten=True)
    pca_matrix.fit(pca_desc)
    print 'PCA matrix shape:', pca_matrix.components_.shape
    print 'Time elapsed PCA: ', time.time() - tpca
else:
    pca_matrix = ''

name_descriptors += '.h5'
descriptors = descriptor_aggregation(cam_descriptors, num_images, num_classes, pca_matrix)

print 'Saving Data... ', name_descriptors
save_data(descriptors, path_descriptors, name_descriptors)
print 'Data Saved'
print 'Total time elapsed: ', time.time() - t_0