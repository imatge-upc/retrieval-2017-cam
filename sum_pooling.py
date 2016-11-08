import cam_utils as cu
import utils_datasets as ud
import numpy as np
import time
from sklearn.decomposition import PCA
import sys



path_images = '/imatge/ajimenez/work/ITR/datasets_hdf5/oxford/places/oxford_1024x720/'
results_path = '/imatge/ajimenez/work/ITR/results_ITR/descriptors/vgg_imagenet/'

#Oxford


layer = 'conv5_3'
dim = '1024x720'

dataset = 'Paris'

local_search = True

if dataset == 'Oxford':
    # Path Dataset
    dataset_path = '/imatge/ajimenez/work/ITR/oxford/datasets_hdf5/places/1024x720/'
    name_h, n_chunks_h, batchsize_h, total_imgs_h = ud.read_dataset_properties(dataset_path+'oxford_h_info.txt')
    name_v, n_chunks_v, batchsize_v, total_imgs_v = ud.read_dataset_properties(dataset_path + 'oxford_v_info.txt')
    feat_path = '/imatge/ajimenez/work/ITR/oxford/features/' + 'vgg_16_imagenet' + '/' + layer + '/' + dim + '/'
    if local_search:
        h_query_feat = feat_path + 'oxford_queries_h_ls.h5'
        v_query_feat = feat_path + 'oxford_queries_v_ls.h5'
    else:
        h_query_feat = feat_path + 'oxford_queries_h.h5'
        v_query_feat = feat_path + 'oxford_queries_v.h5'
    feat_path += 'oxford'
    name_result = 'oxford_1024_720_max_poolingBLABLABLA.h5'
    num_images = 5063

if dataset == 'Paris':
    dataset_path = '/imatge/ajimenez/work/ITR/paris/datasets_hdf5/places/' + dim + '/'
    name_h, n_chunks_h, batchsize_h, total_imgs_h = ud.read_dataset_properties(dataset_path + 'paris_h_info.txt')
    name_v, n_chunks_v, batchsize_v, total_imgs_v = ud.read_dataset_properties(dataset_path + 'paris_v_info.txt')
    results_path = '/imatge/ajimenez/work/ITR/paris/descriptors/' + 'vgg_' + layer+'/' + dim + '/'
    ud.create_folders(results_path)
    feat_path = '/imatge/ajimenez/work/ITR/paris/features/' + 'vgg_16_imagenet' + '/' + layer + '/' + dim + '/'
    if local_search:
        h_query_feat = feat_path + 'paris_queries_h_ls.h5'
        v_query_feat = feat_path + 'paris_queries_v_ls.h5'
    else:
        h_query_feat = feat_path + 'paris_queries_h.h5'
        v_query_feat = feat_path + 'paris_queries_v.h5'
    feat_path += 'paris'
    name_result = 'paris_1024_720_sum_pooling.h5'
    num_images = 6392


num_features = 512

num_classes = 5

image_representations = np.zeros((num_images, num_features), dtype=np.float32)

max_pooling = False

load_features = 'features'
batch_size = 100


def sum_pooling(features):
    num_samples = features.shape[0]
    sys.stdout.flush()
    descriptors = np.zeros((num_samples, num_features), dtype=np.float32)
    for i in range(0, num_samples):
        #print 'Image: ', i
        #sys.stdout.flush()
        for f in range(0, num_features):
            if max_pooling:
                descriptors[i, f] = np.amax(features[i, f])
            else:
                descriptors[i, f] = features[i, f].sum()
    descriptors /= np.linalg.norm(descriptors, axis=1)[:, None]
    return descriptors


t = time.time()
for n in range(0, n_chunks_h):
    print 'Chunk number ', n
    sys.stdout.flush()
    if load_features == 'features':
        features = ud.load_data(feat_path +'_h_'+ str(n)+'.h5')
    else:
        cams, features, scores = cu.load_cams(cam_path + '_h_' + str(n)+'.h5', num_classes)

    ind_1 = n * batch_size
    ind_2 = batch_size * (n + 1)
    if n == n_chunks_h - 1:
        last = features.shape[0]
        image_representations[ind_1:ind_1+last] = sum_pooling(features)
    else:
        image_representations[ind_1:ind_2] = sum_pooling(features)

last_h = ind_1 + last

for n in range(0, n_chunks_v):
    print 'Chunk number ', n
    sys.stdout.flush()
    if load_features == 'features':
        features = ud.load_data(feat_path + '_v_' + str(n)+'.h5')
    else:
        cams, features, scores = cu.load_cams(cam_path + '_v_' + str(n)+'.h5', num_classes)

    ind_1 = last_h + n * batch_size
    ind_2 = last_h + batch_size * (n + 1)
    if n == n_chunks_v - 1:
        last = features.shape[0]
        image_representations[ind_1:ind_1+last] = sum_pooling(features)
    else:
        image_representations[ind_1:ind_2] = sum_pooling(features)

last_v = ind_1 + last

# Queries Horizontal

features = ud.load_data(h_query_feat)

b_s = features.shape[0]

image_representations[last_v:last_v+b_s] = sum_pooling(features)

last_t = last_v + b_s

# Queries Vertical
print 'Queries vertical...'

features = ud.load_data(v_query_feat)

b_s = features.shape[0]

image_representations[last_t:last_t+b_s] = sum_pooling(features)

print 'Time elapsed computing sum_pooling: ', time.time() - t

ud.save_data(image_representations, results_path, name_result)

print 'Total time elapsed: ', time.time()-t




