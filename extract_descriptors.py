import cam_utils as cu
import numpy as np
import time
from sklearn.decomposition import PCA
import sys
from utils_datasets import read_dataset_properties, create_folders, load_data, save_data
from pooling_functions import weighted_pooling

#PARAMETERS

dataset = 'Paris'

dim = '1024x720'

layer = 'conv5_3'

local_search = True

num_features = 512

# THRESHOLD CAMS
thresh_cam = 0

# PCA
pca_on = False
if pca_on:
    pca_load_path = '/imatge/ajimenez/work/ITR/oxford/descriptors/regions/vgg_conv5_3/1024x720/oxford_fusion_32_wp.h5'
    pca_dim = 512
    pca_name = '_pca_oxford_32_wp'

# WEIGHT
weight = False
# Perform also max pooling
max_pooling = True
# Load features and masks
num_classes = 8

FUSION = True

if FUSION:
    print '--FUSION--'

save_regions_for_pca = True

if dataset == 'Oxford':
    # Path Dataset
    dataset_path = '/imatge/ajimenez/work/ITR/oxford/datasets_hdf5/places/' + dim + '/'
    name_h, n_chunks_h, batchsize_h, total_imgs_h = read_dataset_properties(dataset_path + 'oxford_h_info.txt')
    name_v, n_chunks_v, batchsize_v, total_imgs_v = read_dataset_properties(dataset_path + 'oxford_v_info.txt')
    path_descriptors = '/imatge/ajimenez/work/ITR/oxford/descriptors/' + 'vgg_' + layer+'/' + dim + '/'
    if local_search:
        path_descriptors += 'ls/'
    create_folders(path_descriptors)
    feat_path = '/imatge/ajimenez/work/ITR/oxford/features/' + 'vgg_16_imagenet' + '/' + layer + '/' + dim + '/'
    cams_path = '/imatge/ajimenez/work/ITR/oxford/cam_masks/' + 'googlenet' + '/' + dim + '/'
    if local_search:
        h_query_feat = feat_path + 'oxford_queries_h_ls.h5'
        v_query_feat = feat_path + 'oxford_queries_v_ls.h5'
        h_query_cams = cams_path + 'oxford_queries_h_ls.h5'
        v_query_cams = cams_path + 'oxford_queries_v_ls.h5'
    else:
        h_query_feat = feat_path + 'oxford_queries_h.h5'
        v_query_feat = feat_path + 'oxford_queries_v.h5'
        h_query_cams = cams_path + 'oxford_queries_h.h5'
        v_query_cams = cams_path + 'oxford_queries_v.h5'
    feat_path += 'oxford'
    cams_path += 'oxford'
    num_images = 5063
    name_descriptors = 'oxford_' + 'fusion_' + str(num_classes) + '_th_' + str(thresh_cam)

    if save_regions_for_pca:
        regions_for_pca_save_path = '/imatge/ajimenez/work/ITR/oxford/descriptors/' + 'regions/' + 'vgg_' + layer + '/'+ dim +'/'
        if local_search:
            regions_for_pca_save_path += 'ls/'
        regions_name_wp = 'oxford_fusion_' + str(num_classes)+'_wp.h5'
        regions_name_mp = 'oxford_fusion_' + str(num_classes) + '_mp.h5'
        create_folders(regions_for_pca_save_path)

if dataset == 'Paris':
    dataset_path = '/imatge/ajimenez/work/ITR/paris/datasets_hdf5/places/' + dim + '/'
    name_h, n_chunks_h, batchsize_h, total_imgs_h = read_dataset_properties(dataset_path + 'paris_h_info.txt')
    name_v, n_chunks_v, batchsize_v, total_imgs_v = read_dataset_properties(dataset_path + 'paris_v_info.txt')
    path_descriptors = '/imatge/ajimenez/work/ITR/paris/descriptors/' + 'vgg_' + layer+'/' + dim + '/'
    if local_search:
        path_descriptors += 'ls/'
    create_folders(path_descriptors)
    feat_path = '/imatge/ajimenez/work/ITR/paris/features/' + 'vgg_16_imagenet' + '/' + layer + '/' + dim + '/'
    cams_path = '/imatge/ajimenez/work/ITR/paris/cam_masks/' + 'googlenet' + '/' + dim + '/'
    if local_search:
        h_query_feat = feat_path + 'paris_queries_h_ls.h5'
        v_query_feat = feat_path + 'paris_queries_v_ls.h5'
        h_query_cams = cams_path + 'paris_queries_h_ls.h5'
        v_query_cams = cams_path + 'paris_queries_v_ls.h5'
    else:
        h_query_feat = feat_path + 'paris_queries_h.h5'
        v_query_feat = feat_path + 'paris_queries_v.h5'
        h_query_cams = cams_path + 'paris_queries_h.h5'
        v_query_cams = cams_path + 'paris_queries_v.h5'

    feat_path += 'paris'
    cams_path += 'paris'
    name_descriptors = 'paris_' + 'fusion_' + str(num_classes) + '_th_' + str(thresh_cam)
    num_images = 6392

    if save_regions_for_pca:
        regions_for_pca_save_path = '/imatge/ajimenez/work/ITR/paris/descriptors/' + 'regions/' + 'vgg_' + layer + '/' +dim+'/'
        if local_search:
            regions_for_pca_save_path += 'ls/'
        regions_name_wp = 'paris_fusion_' + str(num_classes) + '_wp.h5'
        regions_name_mp = 'paris_fusion_' + str(num_classes) + '_mp.h5'
        create_folders(regions_for_pca_save_path)


if save_regions_for_pca:
    print 'Saving regions for PCA...'
    all_regions_images_wp = np.zeros((num_images * num_classes, num_features), dtype=np.float32)
    all_regions_images_mp = np.zeros((num_images * num_classes, num_features), dtype=np.float32)

if weight:
    name_descriptors += '_weigth'

if pca_on:
    print 'Computing PCA...'
    name_descriptors += pca_name
    pca_dim = 512
    descriptors = load_data(pca_load_path)
    pca_matrix = PCA(n_components=pca_dim, whiten=True)
    pca_matrix.fit(descriptors)
else:
    pca_matrix = ''

if max_pooling:
    mp_name_result = name_descriptors + '_mp.h5'

wp_name_result = name_descriptors + '_wp.h5'


print 'Dataset: ', dataset
print 'Image Dimensions: ', dim
print 'Num images = ', num_images
print 'Num Class Activation Maps = ', num_classes
print 'Local Search: ', local_search
sys.stdout.flush()

print 'Beginning Descriptor Extraction: '
print 'Weighted sum pooling: ' + wp_name_result
if max_pooling:
    print 'Max pooling: ' + mp_name_result
sys.stdout.flush()

wsp_descriptors = np.zeros((num_images, num_features), dtype=np.float32)
wmp_descriptors = np.zeros((num_images, num_features), dtype=np.float32)

tt = time.time()

batch_size = batchsize_h

# Horizontal Images
print 'Horizontal Images...'
for n in range(0, n_chunks_h):
    print 'Chunk number ', n
    sys.stdout.flush()

    if FUSION:
        cams = cu.load_cams(cams_path + '_h_' + str(n)+'.h5', num_classes, 'cams')
        features = load_data(feat_path + '_h_' + str(n)+'.h5')
    else:
        cams, features, scores = cu.load_cams(cams_path + '_h_' + str(n)+'.h5', num_classes)
    b_s = features.shape[0]

    if thresh_cam > 0:
        for i in range(0, b_s):
            for k in range(0, num_classes):
                  cams[i][k][np.where(cams[i][k] < thresh_cam)] = 0

    ind_1 = n * batch_size
    ind_2 = batch_size * (n + 1)

    if n == n_chunks_h - 1:
        if save_regions_for_pca and max_pooling:
            wsp_descriptors[ind_1:ind_1+b_s], wmp_descriptors[ind_1:ind_1+b_s], wpr, mpr = \
                weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

            all_regions_images_wp[ind_1 * num_classes:num_classes * (ind_1+b_s)] = wpr
            all_regions_images_mp[ind_1 * num_classes:num_classes * (ind_1+b_s)] = mpr

        elif max_pooling:
            wsp_descriptors[ind_1:ind_1 + b_s], wmp_descriptors[ind_1:ind_1 + b_s] = \
                weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

        else:
            wsp_descriptors[ind_1:ind_1 + b_s] = \
                 weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

    else:
        if save_regions_for_pca and max_pooling:
            wsp_descriptors[ind_1:ind_2], wmp_descriptors[ind_1:ind_2], wpr, mpr = \
                weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

            all_regions_images_wp[ind_1 * num_classes:num_classes * ind_2] = wpr
            all_regions_images_mp[ind_1 * num_classes:num_classes * ind_2] = mpr

        elif max_pooling:
            wsp_descriptors[ind_1:ind_2], wmp_descriptors[ind_1:ind_2] = \
                weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

        else:
            wsp_descriptors[ind_1:ind_2] = \
                weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

# Position of the vector images
last_h = ind_1+b_s

# Vertical Images

print 'Vertical Images...'
for n in range(0, n_chunks_v):
    print 'Chunk number ', n
    sys.stdout.flush()
    if FUSION:
        cams = cu.load_cams(cams_path + '_v_' + str(n) + '.h5', num_classes, 'cams')
        features = load_data(feat_path + '_v_' + str(n) + '.h5')
    else:
        cams, features, scores = cu.load_cams(cams_path + '_v_' + str(n) + '.h5', num_classes)
    b_s = features.shape[0]

    if thresh_cam > 0:
        for i in range(0, b_s):
            for k in range(0, num_classes):
                  cams[i][k][np.where(cams[i][k] < thresh_cam)] = 0

    ind_1 = last_h + n * batch_size
    ind_2 = last_h + batch_size * (n + 1)

    if n == n_chunks_v - 1:
        if save_regions_for_pca and max_pooling:
            wsp_descriptors[ind_1:ind_1 + b_s], wmp_descriptors[ind_1:ind_1 + b_s], wpr, mpr = \
                weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

            all_regions_images_wp[ind_1 * num_classes:num_classes * (ind_1 + b_s)] = wpr
            all_regions_images_mp[ind_1 * num_classes:num_classes * (ind_1 + b_s)] = mpr

        elif max_pooling:
            wsp_descriptors[ind_1:ind_1 + b_s], wmp_descriptors[ind_1:ind_1 + b_s] = \
                weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

        else:
            wsp_descriptors[ind_1:ind_1 + b_s] = \
                weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)
    else:
        if save_regions_for_pca and max_pooling:
            wsp_descriptors[ind_1:ind_2], wmp_descriptors[ind_1:ind_2], wpr, mpr = \
                weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

            all_regions_images_wp[ind_1 * num_classes:num_classes * ind_2] = wpr
            all_regions_images_mp[ind_1 * num_classes:num_classes * ind_2] = mpr

        elif max_pooling:
            wsp_descriptors[ind_1:ind_2], wmp_descriptors[ind_1:ind_2] = \
                weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

        else:
            wsp_descriptors[ind_1:ind_2] = \
                weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

# Position of the vector images
last_v = ind_1 + b_s

print last_v
print 'Queries horizontal...'

# Queries Horizontal
if FUSION:
    cams = cu.load_cams(h_query_cams, num_classes, 'cams')
    features = load_data(h_query_feat)
else:
    cams, features, scores = cu.load_cams(h_query_cams, num_classes)

b_s = features.shape[0]
print b_s

if save_regions_for_pca and max_pooling:
    wsp_descriptors[last_v:last_v + b_s], wmp_descriptors[last_v:last_v + b_s], wpr, mpr = \
        weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

    all_regions_images_wp[last_v * num_classes:num_classes * (last_v + b_s)] = wpr
    all_regions_images_mp[last_v * num_classes:num_classes * (last_v + b_s)] = mpr

elif max_pooling:
    wsp_descriptors[last_v:last_v + b_s], wmp_descriptors[last_v:last_v + b_s] = \
        weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

else:
    wsp_descriptors[last_v:last_v + b_s] = \
        weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

last = last_v + b_s

# Queries Vertical
print 'Queries vertical...'

if FUSION:
    cams = cu.load_cams(v_query_cams, num_classes, 'cams')
    features = load_data(v_query_feat)
else:
    cams, features, scores = cu.load_cams(v_query_cams, num_classes)

b_s = features.shape[0]

if save_regions_for_pca and max_pooling:
    wsp_descriptors[last:last + b_s], wmp_descriptors[last:last + b_s], wpr, mpr = \
        weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

    all_regions_images_wp[last * num_classes:num_classes * (last + b_s)] = wpr
    all_regions_images_mp[last * num_classes:num_classes * (last + b_s)] = mpr

elif max_pooling:
    wsp_descriptors[last:last + b_s], wmp_descriptors[last:last + b_s] = \
        weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

else:
    wsp_descriptors[last:last + b_s] = \
        weighted_pooling(features, cams, max_pooling, save_regions_for_pca, pca_matrix)

print 'Total time elapsed: ', time.time() - tt

save_data(wsp_descriptors, path_descriptors, wp_name_result)

if save_regions_for_pca:
    save_data(all_regions_images_wp, regions_for_pca_save_path, regions_name_wp)
    save_data(all_regions_images_mp, regions_for_pca_save_path, regions_name_mp)

if max_pooling:
    save_data(wmp_descriptors, path_descriptors, mp_name_result)