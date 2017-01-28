import numpy as np
import os
import h5py
import sys
import evaluate_oxford_paris as eval
import utils_datasets as utils
import time
from sklearn.decomposition import PCA
from vgg_cam import VGGCAM
from utils_datasets import create_folders, save_data, preprocess_images, preprocess_query, load_data
from pooling_functions import weighted_cam_pooling, descriptor_aggregation, retrieve_n_descriptors, descriptor_aggregation_cl
from cam_utils import extract_feat_cam
from scipy.misc import imread
import math
from crow import compute_crow_channel_weight
import pickle
from reranking import re_ranking

imagenet_dictionary = pickle.load(open("/imatge/ajimenez/work/ITR/imagenet1000_clsid_to_human.pkl", "rb"))


def print_classes(dictionary_labels, vector_classes):
    for vc in vector_classes:
        print dictionary_labels[vc]


def load_specific_descriptors(list_classes, img_names, path_descriptors):
    name_file = path_descriptors + img_names+'.h5'
    with h5py.File(name_file, 'r') as hf:
        descriptors = np.array(hf.get('data'))
    return descriptors[list_classes]

# Parameters to set

# Dataset
dataset = 'Oxford'
n_images_oxford = 5063
n_images_paris = 6392
n_queries = 55

# Network Parameters
nb_classes = 1000
VGGCAM_weight_path = '/imatge/ajimenez/work/ITR/models/vgg_cam_weights.h5'
model_name = 'vgg_16_CAM_imagenet'
layer = 'relu5_1'
batch_size = 10

# Search Mode 6654 7831
local_search = True

# Query Expansion
query_expansion = False
n_expand = 10

# Re-ranking
do_re_ranking = False

# Descriptors for Re-ranking / Local Search
dim = '1024x720'
size_v = [720, 1024]
size_h = [1024, 720]
mean_value = [123.68, 116.779, 103.939]

apply_pca = True
num_classes_pca = 1
num_cams = 6


top_n_ranking = 100


print 'Dataset: ', dataset
print 'Num_cams ', num_cams
print 'PCA with ', num_classes_pca
print 'Q expansion ', n_expand

LOAD = False

if dataset == 'Oxford':
    image_path_oxford = '/imatge/ajimenez/work/datasets_retrieval/Oxford/1_images/'
    ranking_path = '/imatge/ajimenez/work/ITR/oxford/results/' + model_name + '/' + layer + '/' + dim + '/assdad'+\
                   str(top_n_ranking)+str(n_expand)+'/'
    ranking_image_names_list = '/imatge/ajimenez/work/ITR/oxford/lists/list_oxford_rank.txt'
    utils.create_folders(ranking_path)

    descriptors_path = '/imatge/ajimenez/work/ITR/oxford/all_descriptors/Vgg_16_CAM/relu5_1/1024x720/'
    descriptors_name = 'oxford_8_pca_512_paris_8.h5'

    pca_descriptors_path = '/imatge/ajimenez/work/ITR/paris/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'paris_all_32_wp.h5'

    ls_desc_path = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/ls/' + \
                   str(num_cams)+'_' + 'pca_'+str(num_classes_pca)+'/'

    utils.create_folders(ls_desc_path)

    image_names = list()
    with open(ranking_image_names_list, "r") as f:
        for line in f:
            image_names.append(line)
    image_names = np.array(image_names)


    sys.stdout.flush()

    t = time.time()

    sys.stdout.flush()

    path_gt = "/imatge/ajimenez/work/datasets_retrieval/Oxford/2_groundtruth/"
    query_names = ["all_souls", "ashmolean", "balliol", "bodleian", "christ_church", "cornmarket", "hertford", "keble",
                   "magdalen", "pitt_rivers", "radcliffe_camera"]

    model_v = VGGCAM(nb_classes, (3, 1024, 720))
    model_v.load_weights(VGGCAM_weight_path)

    model_h = VGGCAM(nb_classes, (3, 720, 1024))
    model_h.load_weights(VGGCAM_weight_path)
    count = 0

    if apply_pca:
        pca_dim = 512
        pca_desc = retrieve_n_descriptors(num_classes_pca, n_images_paris, load_data(pca_descriptors_path))
        print 'Computing PCA...'
        print pca_desc.shape
        pca_matrix = PCA(n_components=pca_dim, whiten=True)
        pca_matrix.fit(pca_desc)
        print 'PCA matrix shape:', pca_matrix.components_.shape
        data = np.zeros((n_images_oxford, pca_dim))
        print 'Data shape: ', data.shape
    else:
        pca_matrix = ''

    descriptors = np.zeros((nb_classes*n_images_oxford, 512), dtype=np.float32)
    for index, img_n in enumerate(image_names):
        descriptors[index*nb_classes:(index+1)*nb_classes] = load_data(descriptors_path+img_n.replace('\n','')+'.h5')

    print descriptors.shape

    tcomp = time.time()
    for query_name in query_names:
        print count
        for i in range(1, 6):
            f = open(path_gt + query_name + '_' + str(i) + '_query.txt').readline()
            f = f.replace("oxc1_", "")
            f_list = f.split(" ")
            for k in range(1, 5):
                f_list[k] = (int(math.floor(float(f_list[k]))))

            img = imread(image_path_oxford + f_list[0] + '.jpg')
            print 'Image Shape: ' + str(img.shape[0]) + 'x' + str(img.shape[1])

            x, y, dx, dy = f_list[1], f_list[2], f_list[3], f_list[4]

            f_x, f_y, f_dx, f_dy = int((x - (x % 16)) / 16), int((y - (y % 16)) / 16), \
                                   int((dx - (dx % 16)) / 16), int((dy - (dy % 16)) / 16)

            img_cropped = img[y:dy, x:dx]
            query_img_name = f_list[0]

            print 'Name of the query: ', f_list[0]
            print 'Crop Height: ', img_cropped.shape[0]
            print 'Crop Width: ', img_cropped.shape[1]
            print 'Resized into...'

            if local_search:
                h = img_cropped.shape[0] - (img_cropped.shape[0] % 16)
                w = img_cropped.shape[1] - (img_cropped.shape[1] % 16)
                img_cropped = preprocess_query(img_cropped, w, h, mean_value)

                # if img.shape[0] > img.shape[1]:
                #     size = size_v
                # else:
                #     size = size_h
                #
                # img_prep = preprocess_query(img_cropped, size[0], size[1], mean_value)
                x_feat = np.zeros((1, img_cropped.shape[0], img_cropped.shape[1], img_cropped.shape[2]), dtype=np.float32)

                #x_feat = np.zeros((1, img_prep.shape[0], img_prep.shape[1], img_prep.shape[2]), dtype=np.float32)

                #model = VGGCAM(nb_classes, (img_prep.shape[0], img_prep.shape[1], img_prep.shape[2]),
                #               bounding_box=[x, y, dx, dy])


                model = VGGCAM(nb_classes, (img_cropped.shape[0], img_cropped.shape[1], img_cropped.shape[2]))
                #print 'Model created'
                model.load_weights(VGGCAM_weight_path)
                #model.summary()
                #print 'Weights loaded'
                x_feat[0, :, :, :] = img_cropped
                features, cams, class_list = extract_feat_cam(model, layer, 1, x_feat, num_cams)
                print class_list[0]
                #print_classes(imagenet_dictionary, class_list[0])

            if LOAD:
                desc = load_data(ls_desc_path + f_list[0] + '.h5')
            else:
                if img.shape[0] > img.shape[1]:
                    size = size_v
                    img_p = preprocess_query(img, size[0], size[1], mean_value)
                    x_features = np.zeros((1, img_p.shape[0], img_p.shape[1], img_p.shape[2]), dtype=np.float32)
                    x_features[0, :, :, :] = img_p
                    if local_search:
                        features, cams, roi = extract_feat_cam(model_v, layer, 1, x_features,
                                                            num_cams, class_list[0, 0:num_cams], roi=True)
                    else:
                        features, cams, class_list = extract_feat_cam(model_v, layer, 1, x_features, num_cams)

                else:
                    size = size_h
                    img_p = preprocess_query(img, size[0], size[1], mean_value)
                    x_features = np.zeros((1, img_p.shape[0], img_p.shape[1], img_p.shape[2]), dtype=np.float32)
                    x_features[0, :, :, :] = img_p
                    if local_search:
                        features, cams, roi = extract_feat_cam(model_h, layer, 1, x_features,
                                                               num_cams, class_list[0, 0:num_cams], roi=True)
                    else:
                        features, cams, class_list = extract_feat_cam(model_h, layer, 1, x_features, num_cams)

                if local_search:
                    d_wp = weighted_cam_pooling(features[:, :, f_y:f_dy, f_x:f_dx],
                                                cams[:, :, f_y:f_dy, f_x:f_dx], max_pool=False)
                else:
                    d_wp = weighted_cam_pooling(features, cams, max_pool=False)

                #d_wp = weighted_cam_pooling(features[:, :, f_y:f_dy, f_x:f_dx],
                #                            cams[:, :, :, :], max_pool=False)

                print_classes(imagenet_dictionary, class_list[0])
                desc = descriptor_aggregation(d_wp, 1, num_cams, pca_matrix)

            tagregation=time.time()
            data = descriptor_aggregation_cl(descriptors, n_images_oxford, pca_matrix, class_list[0])
            print 'Time elapsed agregation: ', time.time() - tagregation

            #save_data(desc, ls_desc_path, f_list[0]+'.h5')

            indices_local, data_local = eval.save_ranking_one_query(data, desc, image_names, ranking_path, query_img_name)

            if do_re_ranking:
                t_rerank = time.time()
                indices_re_ranking, data_re_ranking = re_ranking(desc, class_list[0], image_names,
                                                indices_local, dataset, top_n_ranking, pca_matrix, model_h, model_v)
                print 'Time reranking: ', time.time() - t_rerank
                eval.save_ranking_indices(indices_re_ranking, image_names, query_img_name, ranking_path)

            if query_expansion:
                if do_re_ranking:
                    data_local[indices_re_ranking[0:top_n_ranking]] = data_re_ranking
                    desc_expanded = eval.expand_query(n_expand, data_local, indices_re_ranking)
                else:
                    desc_expanded = eval.expand_query(n_expand, data_local, indices_local)
                eval.save_ranking_one_query(data, desc_expanded, image_names, ranking_path, query_img_name)

    time_ranking = time.time() - tcomp


    eval.evaluate_oxford(ranking_path, descriptors_name)
    print 'Time elapsed computing distances: ', time.time() - time_ranking
    print 'Time elapsed total: ', time.time() - t
    #eval.show_images_top(5,dataset)

    print 'Evaluated:  ' + descriptors_name

elif dataset == 'Paris':
    ranking_path = '/imatge/ajimenez/work/ITR/paris/results/' + model_name + '/' + layer + '/' + dim + '/prova_/'
    ranking_image_names_list = '/imatge/ajimenez/work/ITR/paris/lists/list_paris_rank.txt'
    utils.create_folders(ranking_path)


    descriptors_path = '/imatge/ajimenez/work/ITR/paris/all_descriptors/Vgg_16_CAM/relu5_1/1024x720/'
    descriptors_name = 'paris_32_pca_512_oxford_16.h5'

    pca_descriptors_path = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'oxford_all_32_wp.h5'
    image_path_paris = '/imatge/ajimenez/work/datasets_retrieval/Paris/imatges_paris/'

    ls_desc_path = '/imatge/ajimenez/work/ITR/paris/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/ls/'+ \
                   str(num_cams)+'_' + 'pca_'+str(num_classes_pca)+'/'
    utils.create_folders(ls_desc_path)

    if local_search:
        ranking_image_names_list = '/imatge/ajimenez/work/ITR/paris/lists/list_paris_rank.txt'

    data = np.zeros((n_images_paris, 512))
    print 'Data shape: ', data.shape
    sys.stdout.flush()

    t = time.time()
    print 'Time elapsed computing distances: ', time.time() - t

    image_names = list()
    with open(ranking_image_names_list, "r") as f:
        for line in f:
            image_names.append(line)

    image_names = np.array(image_names)

    descriptors = np.zeros((nb_classes * n_images_paris, 512))
    for index, img_n in enumerate(image_names):
        descriptors[index * nb_classes:(index + 1) * nb_classes] = load_data(
            descriptors_path + img_n.replace('\n', '') + '.h5')

    path_gt = "/imatge/ajimenez/work/datasets_retrieval/Paris/imatges_paris_gt/"
    query_names = ["defense", "eiffel", "invalides", "louvre", "moulinrouge", "museedorsay", "notredame", "pantheon",
                   "pompidou", "sacrecoeur", "triomphe"]

    model_v = VGGCAM(nb_classes, (3, 1024, 720))
    model_v.load_weights(VGGCAM_weight_path)

    model_h = VGGCAM(nb_classes, (3, 720, 1024))
    model_h.load_weights(VGGCAM_weight_path)
    count = 0

    if apply_pca:
        pca_dim = 512
        pca_desc = retrieve_n_descriptors(num_classes_pca, n_images_oxford, load_data(pca_descriptors_path))
        print 'Computing PCA...'
        print pca_desc.shape
        pca_matrix = PCA(n_components=pca_dim, whiten=True)
        pca_matrix.fit(pca_desc)
        print 'PCA matrix shape:', pca_matrix.components_.shape
    else:
        pca_matrix = ''

    tcomp = time.time()
    for query_name in query_names:
        for i in range(1, 6):
            f = open(path_gt + query_name + '_' + str(i) + '_query.txt').readline()
            f_list = f.split(" ")
            for k in range(1, 5):
                f_list[k] = (int(math.floor(float(f_list[k]))))

            print f_list[0]
            query_img_name = f_list[0]
            img = imread(image_path_paris + f_list[0] + '.jpg')
            print 'Image Shape: ' + str(img.shape[0]) + 'x' + str(img.shape[1])

            x, y, dx, dy = f_list[1], f_list[2], f_list[3], f_list[4]

            f_x, f_y, f_dx, f_dy = int((x - (x % 16)) / 16), int((y - (y % 16)) / 16),\
                                   int((dx - (dx % 16)) / 16), int((dy - (dy % 16)) / 16)

            img_cropped = img[y:dy, x:dx]

            print 'Name of the query: ', f_list[0]
            print 'Crop Height: ', img_cropped.shape[0]
            print 'Crop Width: ', img_cropped.shape[1]
            print 'Resized into...'

            if local_search:
                h = img_cropped.shape[0] - (img_cropped.shape[0] % 16)
                w = img_cropped.shape[1] - (img_cropped.shape[1] % 16)
                img_cropped = preprocess_query(img_cropped, w, h, mean_value)
                x_feat = np.zeros((1, img_cropped.shape[0], img_cropped.shape[1], img_cropped.shape[2]),
                                  dtype=np.float32)
                model = VGGCAM(nb_classes, (img_cropped.shape[0], img_cropped.shape[1], img_cropped.shape[2]))
                model.load_weights(VGGCAM_weight_path)
                x_feat[0, :, :, :] = img_cropped
                features_c, cams_c, class_list = extract_feat_cam(model, layer, 1, x_feat, num_cams)
                print class_list[0]
                #print_classes(imagenet_dictionary, class_list[0])

            if LOAD:
                desc = load_data(ls_desc_path + f_list[0] + '.h5')
            else:
                if img.shape[0] > img.shape[1]:
                    size = size_v
                    img_p = preprocess_query(img, size[0], size[1], mean_value)
                    x_features = np.zeros((1, img_p.shape[0], img_p.shape[1], img_p.shape[2]), dtype=np.float32)
                    x_features[0, :, :, :] = img_p
                    if local_search:
                        features, cams, roi = extract_feat_cam(model_v, layer, 1, x_features,
                                                               num_cams, class_list[0, 0:num_cams], roi=True)
                    else:
                        features, cams, class_list = extract_feat_cam(model_v, layer, 1, x_features, num_cams)

                else:
                    size = size_h
                    img_p = preprocess_query(img, size[0], size[1], mean_value)
                    x_features = np.zeros((1, img_p.shape[0], img_p.shape[1], img_p.shape[2]), dtype=np.float32)
                    x_features[0, :, :, :] = img_p
                    if local_search:
                        features, cams, roi = extract_feat_cam(model_h, layer, 1, x_features,
                                                               num_cams, class_list[0, 0:num_cams], roi=True)
                    else:
                        features, cams, class_list = extract_feat_cam(model_h, layer, 1, x_features, num_cams)

                if local_search:
                    d_wp = weighted_cam_pooling(features[:, :, f_y:f_dy, f_x:f_dx],
                                                cams[:, :, f_y:f_dy, f_x:f_dx], max_pool=False)
                else:
                    d_wp = weighted_cam_pooling(features, cams, max_pool=False)

                print_classes(imagenet_dictionary, class_list[0])
                desc = descriptor_aggregation(d_wp, 1, num_cams, pca_matrix)

            tagregation = time.time()
            data = descriptor_aggregation_cl(descriptors, n_images_paris, pca_matrix, class_list[0])
            print 'Time elapsed agregation: ', time.time() - tagregation

            indices_local, data_local = eval.save_ranking_one_query(data, desc, image_names, ranking_path, query_img_name)

            if do_re_ranking:
                t_rerank = time.time()
                indices_re_ranking, data_re_ranking = re_ranking(desc, class_list[0], image_names,
                                                indices_local, dataset, top_n_ranking, pca_matrix, model_h, model_v)
                print 'Time reranking: ', time.time() - t_rerank
                eval.save_ranking_indices(indices_re_ranking, image_names, query_img_name, ranking_path)

            if query_expansion:
                if do_re_ranking:
                    data_local[indices_re_ranking[0:top_n_ranking]] = data_re_ranking
                    desc_expanded = eval.expand_query(n_expand, data_local, indices_re_ranking)
                else:
                    desc_expanded = eval.expand_query(n_expand, data_local, indices_local)
                eval.save_ranking_one_query(data, desc_expanded, image_names, ranking_path, query_img_name)

    time_ranking = time.time()-tcomp

    eval.evaluate_paris(ranking_path)


    #eval.show_stats(dataset, ranking_path)

    print 'Time elapsed ranking: ',  time_ranking
    print 'Time elapsed total: ', time.time() - t
    print 'Evaluated: ' + descriptors_name






