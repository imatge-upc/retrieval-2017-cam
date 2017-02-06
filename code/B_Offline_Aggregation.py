import numpy as np
import time
import sys
from pooling_functions import descriptor_aggregation, retrieve_n_descriptors, compute_pca
from utils import create_folders, save_data, load_data


# Parameters
dataset = 'Oxford'

# Num classes stored in the precomputed
num_prec_classes = 64

num_classes = 32

model_name = 'Vgg_16_CAM'
layer = 'relu5_1'
dim = '1024x720'

# PCA
apply_pca = True
pca_dim = 512
dim_descriptor = 512
num_classes_pca = 1

print 'Dataset ', dataset
print 'Num Classes ', num_classes
print 'Dimension Descriptor ', dim_descriptor
if apply_pca:
    print 'Num Classes PCA ', num_classes_pca
    print 'Dimension PCA ', pca_dim

t_0 = time.time()

# DATASET PATHS AND NAMES
if dataset == 'Oxford':
    path_descriptors = '/imatge/ajimenez/work/ITR/oxford/descriptors_new2/' + \
                       model_name + '/' + layer + '/' + dim + '/'

    create_folders(path_descriptors)
    cam_descriptors_path = '/imatge/ajimenez/work/ITR/oxford/descriptors_new2/Vgg_16_CAM/relu5_1/1024x720/' \
                           'oxford_all_32_wp.h5'

    pca_descriptors_path = '/imatge/ajimenez/work/ITR/paris/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'paris_all_64_wp.h5'
    num_images = 5063
    num_img_pca = 6392
    name_descriptors = 'oxford_' + str(num_classes)
    pca_name = '_pca_' + str(pca_dim) + '_paris_'+str(num_classes_pca)

if dataset == 'Oxford105k':
    path_descriptors = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/' + \
                       model_name + '/' + layer + '/' + dim + '/'

    create_folders(path_descriptors)
    cam_descriptors_path = '/imatge/ajimenez/work/ITR/descriptors100k/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/' \
                           'distractor_all_64_wp_'

    pca_descriptors_path = '/imatge/ajimenez/work/ITR/paris/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'paris_all_64_wp.h5'
    num_images = 100070
    num_img_pca = 6392
    name_descriptors = 'oxford_105k_' + str(num_classes)
    pca_name = '_pca_' + str(pca_dim) + '_paris_'+str(num_classes_pca)

if dataset == 'Paris':
    path_descriptors = '/imatge/ajimenez/work/ITR/paris/descriptors_new/' + \
                       model_name + '/' + layer+'/' + dim + '/crow/'

    create_folders(path_descriptors)
    cam_descriptors_path = '/imatge/ajimenez/work/ITR/paris/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'paris_all_64_wp.h5'
    pca_descriptors_path = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'oxford_all_64_wp.h5'
    num_images = 6392
    num_img_pca = 5063
    name_descriptors = 'paris_' + str(num_classes)
    pca_name = '_pca_' + str(pca_dim) + '_oxford_'+str(num_classes_pca)

if dataset == 'Paris106k':
    path_descriptors = '/imatge/ajimenez/work/ITR/paris/descriptors_new/' + \
                       model_name + '/' + layer + '/' + dim + '/crow/'

    create_folders(path_descriptors)
    cam_descriptors_path = '/imatge/ajimenez/work/ITR/descriptors100k/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/' \
                           'distractor_all_64_wp_'

    pca_descriptors_path = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'oxford_all_64_wp.h5'
    num_images = 100070
    num_img_pca = 5063
    name_descriptors = 'paris_106k_' + str(num_classes)
    pca_name = '_pca_' + str(pca_dim) + '_oxford_'+str(num_classes_pca)

########################################################################################################################

# Compute PCA
if apply_pca:
    tpca = time.time()
    pca_desc = retrieve_n_descriptors(num_classes_pca, num_img_pca, load_data(pca_descriptors_path))
    pca_matrix = compute_pca(pca_desc, pca_dim=pca_dim)
    name_descriptors += pca_name
    print 'PCA matrix shape:', pca_matrix.components_.shape
    print 'Time elapsed PCA: ', time.time() - tpca
else:
    pca_matrix = None

if dataset == 'Oxford105k' or dataset == 'Paris106k':
    n_chunks = 10
    descriptors = np.zeros((0, 512), dtype=np.float32)
    for n in range(0, n_chunks+1):
        desc = load_data(cam_descriptors_path + str(n) + '.h5')
        print desc.shape
        descriptors = np.concatenate((descriptors, descriptor_aggregation(desc, desc.shape[0]/num_prec_classes,
                                                                          num_classes, pca_matrix)))
        print descriptors.shape

elif dataset == 'Oxford' or dataset == 'Paris':
    t = time.time()
    cam_descriptors = load_data(cam_descriptors_path)
    print 'Time elapsed loading: ', time.time() - t
    descriptors = descriptor_aggregation(cam_descriptors, num_images, num_classes, pca_matrix)

name_descriptors += '.h5'

print descriptors.shape

print 'Saving Data... ', name_descriptors
save_data(descriptors, path_descriptors, name_descriptors)
print 'Data Saved'
print 'Total time elapsed: ', time.time() - t_0