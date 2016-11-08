import numpy as np
import os
import h5py
import sys
import evaluate_oxford_paris as eval
import utils_datasets as utils
import time
from sklearn.decomposition import PCA

dataset = 'Paris'

dim = '1024x720'

if dataset == 'Oxford':
    ranking_path = '/imatge/ajimenez/work/ITR/oxford/results/vgg_conv5_3/' + dim + '/'
    ranking_image_names_list = '/imatge/ajimenez/work/ITR/oxford/lists/list_oxford_rank.txt'
    utils.create_folders(ranking_path)
    descriptors_path = '/imatge/ajimenez/work/ITR/oxford/descriptors/vgg_conv5_3/' + dim + '/'
    descriptors_name = 'oxford_fusion_32_th_0_pca_paris_8_wp_mp.h5'
    data = utils.load_data(descriptors_path+descriptors_name)
    print data[5062]
    print 'Data shape: ', data.shape
    sys.stdout.flush()

    t = time.time()
    distances, indices = eval.compute_distances(data, 5063)
    print 'Time elapsed computing distances: ', time.time()-t

    image_names = list()

    with open(ranking_image_names_list, "r") as f:
        for line in f:
            image_names.append(line)

    image_names = np.array(image_names)

    print image_names
    eval.save_rankings(indices, image_names, ranking_path, dataset)

    print len(image_names)
    sys.stdout.flush()



    eval.evaluate_oxford(ranking_path, descriptors_name)

    #eval.show_images_top(5,dataset)

    print 'Evaluated:  ' + descriptors_name

elif dataset == 'Paris':
    ranking_path = '/imatge/ajimenez/work/ITR/paris/results/vgg_conv5_3/' + dim + '/'
    ranking_image_names_list = '/imatge/ajimenez/work/ITR/paris/lists/list_paris_rank.txt'
    utils.create_folders(ranking_path)
    descriptors_path = '/imatge/ajimenez/work/ITR/paris/descriptors/vgg_conv5_3/' + dim + '/'
    descriptors_name = 'paris_1024_720_max_pooling.h5'
    data = utils.load_data(descriptors_path + descriptors_name)
    print data[0]
    print 'Data shape: ', data.shape
    sys.stdout.flush()

    t = time.time()
    distances, indices = eval.compute_distances(data, 6392)
    print 'Time elapsed computing distances: ', time.time() - t

    image_names = list()

    with open(ranking_image_names_list, "r") as f:
        for line in f:
            image_names.append(line)

    image_names = np.array(image_names)

    print image_names

    eval.save_rankings(indices, image_names, ranking_path, dataset)

    eval.evaluate_paris(ranking_path)

    print 'Evaluated: ' + descriptors_name


