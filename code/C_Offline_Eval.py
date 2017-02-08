import numpy as np
import os
import h5py
import sys
import evaluate_oxford_paris as eval
import utils as utils
import time
from vgg_cam import VGGCAM
from cam_utils import extract_feat_cam
from utils import create_folders, save_data, preprocess_images, preprocess_query, load_data, print_classes
from pooling_functions import weighted_cam_pooling, descriptor_aggregation, retrieve_n_descriptors, compute_pca
from scipy.misc import imread
import math
from reranking import re_ranking
import pickle


imagenet_dictionary = pickle.load(open("/imatge/ajimenez/work/ITR/imagenet1000_clsid_to_human.pkl", "rb"))

# Parameters to set

# Dataset
dataset = 'Oxford'
n_images_distractors = 100070
n_images_oxford = 5063
n_images_paris = 6392
n_queries = 55

# Network Parameters
nb_classes = 1000
VGGCAM_weight_path = '/imatge/ajimenez/work/ITR/models/vgg_cam_weights.h5'
model_name = 'vgg_16_CAM_imagenet'
layer = 'relu5_1'
batch_size = 10

# Query Expansion
query_expansion = True
n_expand = 5

# Re-ranking
do_re_ranking = False

# Descriptors for Re-ranking  (Size W x H)
dim = '1024x720'
size_v = [720, 1024]
size_h = [1024, 720]
mean_value = [123.68, 116.779, 103.939]

apply_pca = True
num_classes_pca = 1
pca_dim = 512
num_cams = 32

# Num_cams2 --> Used to compute the descriptors when re-ranking
num_cams2 = 6

top_n_ranking = 1000


print 'Dataset: ', dataset
print 'Num_cams ', num_cams
print 'PCA with ', num_classes_pca

if dataset == 'Oxford':
    image_path_oxford = '/imatge/ajimenez/work/datasets_retrieval/Oxford/1_images/'
    ranking_path = '/imatge/ajimenez/work/ITR/oxford/results/' + model_name + '/' + layer + '/' + dim + '/example/'
    ranking_image_names_list = '/imatge/ajimenez/work/ITR/oxford/lists/list_oxford_rank.txt'
    utils.create_folders(ranking_path)
    descriptors_path = '/imatge/ajimenez/work/ITR/oxford/descriptors_new2/Vgg_16_CAM/relu5_1/1024x720/'
    descriptors_name = 'oxford_32_pca_512_paris_1.h5'

    pca_descriptors_path = '/imatge/ajimenez/work/ITR/paris/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'paris_all_32_wp.h5'

    ls_desc_path = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/ls/' + \
                   str(num_cams)+'_' + 'pca_'+str(num_classes_pca)+'/'
    utils.create_folders(ls_desc_path)

    data = utils.load_data(descriptors_path+descriptors_name)

    print 'Data shape: ', data.shape
    sys.stdout.flush()

    t = time.time()

    image_names = list()

    with open(ranking_image_names_list, "r") as f:
        for line in f:
            image_names.append(line)

    image_names = np.array(image_names)

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
        pca_desc = retrieve_n_descriptors(num_classes_pca, n_images_paris, load_data(pca_descriptors_path))
        pca_matrix = compute_pca(pca_desc, pca_dim=pca_dim)
        print 'PCA matrix shape:', pca_matrix.components_.shape
    else:
        pca_matrix = None

    for query_name in query_names:
        print count
        for i in range(1, 6):
            f = open(path_gt + query_name + '_' + str(i) + '_query.txt').readline()
            f = f.replace("oxc1_", "")
            f_list = f.split(" ")
            for k in range(1, 5):
                f_list[k] = (int(math.floor(float(f_list[k]))))

            query_img_name = f_list[0]
            img = imread(image_path_oxford + query_img_name + '.jpg')
            print 'Image Shape: ' + str(img.shape[0]) + 'x' + str(img.shape[1])

            # Query bounding box
            x, y, dx, dy = f_list[1], f_list[2], f_list[3], f_list[4]

            # Feature map query bounding box
            f_x, f_y, f_dx, f_dy = int((x - (x % 16)) / 16), int((y - (y % 16)) / 16), \
                                   int((dx - (dx % 16)) / 16), int((dy - (dy % 16)) / 16)

            img_cropped = img[y:dy, x:dx]

            print 'Name of the query: ', query_img_name
            print 'Crop Height: ', img_cropped.shape[0]
            print 'Crop Width: ', img_cropped.shape[1]
            print 'Resized into...'

            h = img_cropped.shape[0] - (img_cropped.shape[0] % 16)
            w = img_cropped.shape[1] - (img_cropped.shape[1] % 16)
            img_cropped = preprocess_query(img_cropped, w, h, mean_value)
            x_feat = np.zeros((1, img_cropped.shape[0], img_cropped.shape[1], img_cropped.shape[2]), dtype=np.float32)
            x_feat[0] = img_cropped
            model = VGGCAM(nb_classes, (img_cropped.shape[0], img_cropped.shape[1], img_cropped.shape[2]))
            model.load_weights(VGGCAM_weight_path)
            features_c, cams_c, class_list = extract_feat_cam(model, layer, 1, x_feat, num_cams)

            if img.shape[0] > img.shape[1]:
                size = size_v
                img_p = preprocess_query(img, size[0], size[1], mean_value)
                x_features = np.zeros((1, img_p.shape[0], img_p.shape[1], img_p.shape[2]), dtype=np.float32)
                x_features[0, :, :, :] = img_p
                features, cams, roi = extract_feat_cam(model_v, layer, 1, x_features,
                                                       num_cams, class_list[0, 0:num_cams], roi=True)

            else:
                size = size_h
                img_p = preprocess_query(img, size[0], size[1], mean_value)
                x_features = np.zeros((1, img_p.shape[0], img_p.shape[1], img_p.shape[2]), dtype=np.float32)
                x_features[0, :, :, :] = img_p
                features, cams, roi = extract_feat_cam(model_h, layer, 1, x_features,
                                                       num_cams, class_list[0, 0:num_cams], roi=True)

            d_wp = weighted_cam_pooling(features[:, :, f_y:f_dy, f_x:f_dx],
                                        cams[:, :, f_y:f_dy, f_x:f_dx], max_pool=False)

            # Compute Query Descriptor
            desc = descriptor_aggregation(d_wp, 1, num_cams, pca_matrix)

            indices_local, data_local = eval.save_ranking_one_query(data, desc, image_names, ranking_path, query_img_name)

            # When re-ranking descriptor for the query computed with less CAMs, as we know the relevant objects
            desc = descriptor_aggregation(d_wp, 1, num_cams2, pca_matrix)

            if do_re_ranking:
                t_rerank = time.time()
                indices_re_ranking, data_re_ranking = re_ranking(desc, class_list[0, 0:num_cams2], image_names,
                                                                 indices_local, dataset, top_n_ranking, pca_matrix,
                                                                 model_h, model_v)
                print 'Time reranking: ', time.time() - t_rerank
                eval.save_ranking_indices(indices_re_ranking, image_names, query_img_name, ranking_path)

            if query_expansion:
                if do_re_ranking:
                    data_local[indices_re_ranking[0:top_n_ranking]] = data_re_ranking
                    desc_expanded = eval.expand_query(n_expand, data_local, indices_re_ranking)
                else:
                    desc_expanded = eval.expand_query(n_expand, data_local, indices_local)
                eval.save_ranking_one_query(data, desc_expanded, image_names, ranking_path, query_img_name)

    print 'Time elapsed computing distances: ', time.time()-t

    eval.evaluate_oxford(ranking_path)

    print 'Evaluated:  ' + descriptors_name

elif dataset == 'Paris':
    ranking_path = '/imatge/ajimenez/work/ITR/paris/results/' + model_name + '/' + layer + '/' + dim + '/sfqwe/'
    ranking_image_names_list = '/imatge/ajimenez/work/ITR/paris/lists/list_paris_rank.txt'
    utils.create_folders(ranking_path)

    descriptors_path = '/imatge/ajimenez/work/ITR/paris/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/'
    descriptors_name = 'paris_32_pca_512_oxford_1.h5'

    pca_descriptors_path = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'oxford_all_32_wp.h5'

    image_path_paris = '/imatge/ajimenez/work/datasets_retrieval/Paris/imatges_paris/'

    ls_desc_path = '/imatge/ajimenez/work/ITR/paris/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/ls/'+ \
                   str(num_cams)+'_' + 'pca_'+str(num_classes_pca)+'/'
    utils.create_folders(ls_desc_path)

    data = utils.load_data(descriptors_path + descriptors_name)
    print 'Data shape: ', data.shape
    sys.stdout.flush()

    t = time.time()

    image_names = list()
    with open(ranking_image_names_list, "r") as f:
        for line in f:
            image_names.append(line)

    image_names = np.array(image_names)

    path_gt = "/imatge/ajimenez/work/datasets_retrieval/Paris/imatges_paris_gt/"
    query_names = ["defense", "eiffel", "invalides", "louvre", "moulinrouge", "museedorsay", "notredame", "pantheon",
                   "pompidou", "sacrecoeur", "triomphe"]

    model_v = VGGCAM(nb_classes, (3, 1024, 720))
    model_v.load_weights(VGGCAM_weight_path)

    model_h = VGGCAM(nb_classes, (3, 720, 1024))
    model_h.load_weights(VGGCAM_weight_path)
    count = 0

    if apply_pca:
        pca_desc = retrieve_n_descriptors(num_classes_pca, n_images_oxford, load_data(pca_descriptors_path))
        pca_matrix = compute_pca(pca_desc, pca_dim= pca_dim)
        print 'PCA matrix shape:', pca_matrix.components_.shape
    else:
        pca_matrix = None

    for query_name in query_names:
        for i in range(1, 6):
            f = open(path_gt + query_name + '_' + str(i) + '_query.txt').readline()
            f_list = f.split(" ")
            for k in range(1, 5):
                f_list[k] = (int(math.floor(float(f_list[k]))))

            print f_list[0]
            img = imread(image_path_paris + f_list[0] + '.jpg')
            print 'Image Shape: ' + str(img.shape[0]) + 'x' + str(img.shape[1])

            x, y, dx, dy = f_list[1], f_list[2], f_list[3], f_list[4]

            f_x, f_y, f_dx, f_dy = int((x - (x % 16)) / 16), int((y - (y % 16)) / 16),\
                                   int((dx - (dx % 16)) / 16), int((dy - (dy % 16)) / 16)

            img_cropped = img[y:dy, x:dx]
            query_img_name = f_list[0]
            print 'Name of the query: ', query_img_name
            print 'Crop Height: ', img_cropped.shape[0]
            print 'Crop Width: ', img_cropped.shape[1]
            print 'Resized into...'

            h = img_cropped.shape[0] - (img_cropped.shape[0] % 16)
            w = img_cropped.shape[1] - (img_cropped.shape[1] % 16)

            img_cropped = preprocess_query(img_cropped, w, h, mean_value)
            x_feat = np.zeros((1, img_cropped.shape[0], img_cropped.shape[1], img_cropped.shape[2]),
                              dtype=np.float32)
            x_feat[0] = img_cropped
            model = VGGCAM(nb_classes, (img_cropped.shape[0], img_cropped.shape[1], img_cropped.shape[2]))
            model.load_weights(VGGCAM_weight_path)
            features_c, cams_c, class_list = extract_feat_cam(model, layer, 1, x_feat, num_cams)

            if img.shape[0] > img.shape[1]:
                size = size_v
                img_p = preprocess_query(img, size[0], size[1], mean_value)
                x_features = np.zeros((1, img_p.shape[0], img_p.shape[1], img_p.shape[2]), dtype=np.float32)
                x_features[0, :, :, :] = img_p
                features, cams, roi = extract_feat_cam(model_v, layer, 1, x_features,
                                                       num_cams, class_list[0, 0:num_cams], roi=True)

            else:
                size = size_h
                img_p = preprocess_query(img, size[0], size[1], mean_value)
                x_features = np.zeros((1, img_p.shape[0], img_p.shape[1], img_p.shape[2]), dtype=np.float32)
                x_features[0, :, :, :] = img_p
                features, cams, roi = extract_feat_cam(model_h, layer, 1, x_features,
                                                       num_cams, class_list[0, 0:num_cams], roi=True)

            d_wp = weighted_cam_pooling(features[:, :, f_y:f_dy, f_x:f_dx],
                                       cams[:, :, f_y:f_dy, f_x:f_dx], max_pool=False)

            desc = descriptor_aggregation(d_wp, 1, num_cams, pca_matrix)

            indices_local, data_local = eval.save_ranking_one_query(data, desc, image_names, ranking_path,
                                                                    query_img_name)
            desc = descriptor_aggregation(d_wp, 1, num_cams2, pca_matrix)

            if do_re_ranking:
                t_rerank = time.time()
                indices_re_ranking, data_re_ranking = re_ranking(desc, class_list[0, 0:num_cams2], image_names,
                                                                 indices_local, dataset, top_n_ranking, pca_matrix,
                                                                 model_h, model_v)
                print 'Time reranking: ', time.time() - t_rerank
                eval.save_ranking_indices(indices_re_ranking, image_names, query_img_name, ranking_path)

            if query_expansion:
                if do_re_ranking:
                    data_local[indices_re_ranking[0:top_n_ranking]] = data_re_ranking
                    desc_expanded = eval.expand_query(n_expand, data_local, indices_re_ranking)
                else:
                    desc_expanded = eval.expand_query(n_expand, data_local, indices_local)
                eval.save_ranking_one_query(data, desc_expanded, image_names, ranking_path, query_img_name)

    eval.evaluate_paris(ranking_path)

    print 'Evaluated: ' + descriptors_name

elif dataset == 'Oxford105k':
    image_path_oxford = '/imatge/ajimenez/work/datasets_retrieval/Oxford/1_images/'
    ranking_path = '/imatge/ajimenez/work/ITR/oxford/results/' + model_name + '/' + layer + '/' + dim + '/105k__/'
    ranking_image_names_list = '/imatge/ajimenez/work/ITR/oxford/lists/list_oxford_rank.txt'
    ranking_distractors_list = '/imatge/ajimenez/workspace/ITR/lists/list_oxford_105k_rank.txt'
    utils.create_folders(ranking_path)

    descriptors_path = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/'
    distractors_path = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                       'oxford_105k_h_32_pca_512_paris_1.h5'
    if local_search:
        descriptors_path += ''
        ranking_image_names_list = '/imatge/ajimenez/work/ITR/oxford/lists/list_oxford_rank.txt'
    # descriptors_name = 'oxford_fusion_8_th_0_pca_paris_8_wp_wp.h5'
    descriptors_name = 'oxford_32_pca_512_paris_1.h5'
    pca_descriptors_path = '/imatge/ajimenez/work/ITR/paris/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'paris_all_32_wp.h5'

    ls_desc_path = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/ls/' + \
                   str(num_cams) + '_' + 'pca_' + str(num_classes_pca) + '/'
    utils.create_folders(ls_desc_path)

    data = utils.load_data(descriptors_path + descriptors_name)
    data = np.concatenate((data, utils.load_data(distractors_path)))

    # print data[5000]
    print 'Data shape: ', data.shape
    sys.stdout.flush()

    t = time.time()

    image_names = list()

    with open(ranking_image_names_list, "r") as f:
        for line in f:
            image_names.append(line)

    with open(ranking_distractors_list, "r") as f:
        for line in f:
            image_names.append(line)

    image_names = np.array(image_names)

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
        pca_desc = retrieve_n_descriptors(num_classes_pca, n_images_paris, load_data(pca_descriptors_path))
        pca_matrix = compute_pca(pca_desc, pca_dim=pca_dim)
        print 'PCA matrix shape:', pca_matrix.components_.shape
    else:
        pca_matrix = ''

    for query_name in query_names:
        print count
        for i in range(1, 6):
            f = open(path_gt + query_name + '_' + str(i) + '_query.txt').readline()
            f = f.replace("oxc1_", "")
            f_list = f.split(" ")
            for k in range(1, 5):
                f_list[k] = (int(math.floor(float(f_list[k]))))

            query_img_name = f_list[0]
            img = imread(image_path_oxford + query_img_name + '.jpg')
            print 'Image Shape: ' + str(img.shape[0]) + 'x' + str(img.shape[1])

            x, y, dx, dy = f_list[1], f_list[2], f_list[3], f_list[4]

            f_x, f_y, f_dx, f_dy = int((x - (x % 16)) / 16), int((y - (y % 16)) / 16), \
                                   int((dx - (dx % 16)) / 16), int((dy - (dy % 16)) / 16)

            img_cropped = img[y:dy, x:dx]

            print 'Name of the query: ', query_img_name
            print 'Crop Height: ', img_cropped.shape[0]
            print 'Crop Width: ', img_cropped.shape[1]
            print 'Resized into...'

            h = img_cropped.shape[0] - (img_cropped.shape[0] % 16)
            w = img_cropped.shape[1] - (img_cropped.shape[1] % 16)
            img_cropped = preprocess_query(img_cropped, w, h, mean_value)
            x_feat = np.zeros((1, img_cropped.shape[0], img_cropped.shape[1], img_cropped.shape[2]),
                              dtype=np.float32)
            x_feat[0] = img_cropped
            model = VGGCAM(nb_classes, (img_cropped.shape[0], img_cropped.shape[1], img_cropped.shape[2]))
            model.load_weights(VGGCAM_weight_path)
            features_c, cams_c, class_list = extract_feat_cam(model, layer, 1, x_feat, num_cams)
            if LOAD:
                desc = load_data(ls_desc_path + f_list[0] + '.h5')
            else:
                if img.shape[0] > img.shape[1]:
                    size = size_v
                    img_p = preprocess_query(img, size[0], size[1], mean_value)
                    x_features = np.zeros((1, img_p.shape[0], img_p.shape[1], img_p.shape[2]), dtype=np.float32)
                    x_features[0, :, :, :] = img_p
                    features, cams, roi = extract_feat_cam(model_v, layer, 1, x_features,
                                                           num_cams, class_list[0, 0:num_cams], roi=True)
                else:
                    size = size_h
                    img_p = preprocess_query(img, size[0], size[1], mean_value)
                    x_features = np.zeros((1, img_p.shape[0], img_p.shape[1], img_p.shape[2]), dtype=np.float32)
                    x_features[0, :, :, :] = img_p
                    features, cams, roi = extract_feat_cam(model_h, layer, 1, x_features,
                                                           num_cams, class_list[0, 0:num_cams], roi=True)

                d_wp = weighted_cam_pooling(features[:, :, f_y:f_dy, f_x:f_dx],
                                            cams[:, :, f_y:f_dy, f_x:f_dx], max_pool=False)

                desc = descriptor_aggregation(d_wp, 1, num_cams, pca_matrix)

            indices_local, data_local = eval.save_ranking_one_query(data, desc, image_names, ranking_path,
                                                                    query_img_name)

            desc = descriptor_aggregation(d_wp, 1, num_cams2, pca_matrix)

            if do_re_ranking:
                t_rerank = time.time()
                indices_re_ranking, data_re_ranking = re_ranking(desc, class_list[0, 0:num_cams2], image_names,
                                                                 indices_local, dataset, top_n_ranking, pca_matrix,
                                                                 model_h, model_v)
                print 'Time reranking: ', time.time() - t_rerank
                eval.save_ranking_indices(indices_re_ranking, image_names, query_img_name, ranking_path)

            if query_expansion:
                if do_re_ranking:
                    data_local[indices_re_ranking[0:top_n_ranking]] = data_re_ranking
                    desc_expanded = eval.expand_query(n_expand, data_local, indices_re_ranking)
                else:
                    desc_expanded = eval.expand_query(n_expand, data_local, indices_local)
                eval.save_ranking_one_query(data, desc_expanded, image_names, ranking_path, query_img_name)

    print 'Time elapsed computing distances: ', time.time() - t

    eval.evaluate_oxford(ranking_path)

    print 'Evaluated:  ' + descriptors_name

elif dataset == 'Paris106k':
    ranking_path = '/imatge/ajimenez/work/ITR/paris/results/' + model_name + '/' + layer + '/' + dim + '/Paris106k/'
    ranking_image_names_list = '/imatge/ajimenez/work/ITR/paris/lists/list_paris_rank.txt'
    ranking_distractors_list = '/imatge/ajimenez/workspace/ITR/lists/list_oxford_105k_rank.txt'
    utils.create_folders(ranking_path)

    descriptors_path = '/imatge/ajimenez/work/ITR/paris/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/'
    descriptors_name = 'paris_32_pca_512_oxford_1.h5'
    distractors_path = '/imatge/ajimenez/work/ITR/paris/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                       'paris_106k_h_32_pca_512_oxford_1.h5'

    pca_descriptors_path = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/' \
                           'oxford_all_32_wp.h5'
    image_path_paris = '/imatge/ajimenez/work/datasets_retrieval/Paris/imatges_paris/'

    ls_desc_path = '/imatge/ajimenez/work/ITR/paris/descriptors_new/Vgg_16_CAM/relu5_1/1024x720/crow/ls/'+ \
                   str(num_cams)+'_' + 'pca_'+str(num_classes_pca)+'/'
    utils.create_folders(ls_desc_path)

    if local_search:
        ranking_image_names_list = '/imatge/ajimenez/work/ITR/paris/lists/list_paris_rank.txt'

    data = utils.load_data(descriptors_path + descriptors_name)
    data = np.concatenate((data, utils.load_data(distractors_path)))
    print 'Data shape: ', data.shape
    sys.stdout.flush()

    t = time.time()
    print 'Time elapsed computing distances: ', time.time() - t

    image_names = list()
    with open(ranking_image_names_list, "r") as f:
        for line in f:
            image_names.append(line)

    with open(ranking_distractors_list, "r") as f:
        for line in f:
            image_names.append(line)

    image_names = np.array(image_names)

    path_gt = "/imatge/ajimenez/work/datasets_retrieval/Paris/imatges_paris_gt/"
    query_names = ["defense", "eiffel", "invalides", "louvre", "moulinrouge", "museedorsay", "notredame", "pantheon",
                   "pompidou", "sacrecoeur", "triomphe"]

    model_v = VGGCAM(nb_classes, (3, 1024, 720))
    model_v.load_weights(VGGCAM_weight_path)

    model_h = VGGCAM(nb_classes, (3, 720, 1024))
    model_h.load_weights(VGGCAM_weight_path)
    count = 0

    if apply_pca:
        pca_desc = retrieve_n_descriptors(num_classes_pca, n_images_oxford, load_data(pca_descriptors_path))
        pca_matrix = compute_pca(pca_desc, pca_dim=pca_dim)
        print 'PCA matrix shape:', pca_matrix.components_.shape
    else:
        pca_matrix = None

    for query_name in query_names:
        for i in range(1, 6):
            f = open(path_gt + query_name + '_' + str(i) + '_query.txt').readline()
            f_list = f.split(" ")
            for k in range(1, 5):
                f_list[k] = (int(math.floor(float(f_list[k]))))

            print f_list[0]
            img = imread(image_path_paris + f_list[0] + '.jpg')
            print 'Image Shape: ' + str(img.shape[0]) + 'x' + str(img.shape[1])

            x, y, dx, dy = f_list[1], f_list[2], f_list[3], f_list[4]

            f_x, f_y, f_dx, f_dy = int((x - (x % 16)) / 16), int((y - (y % 16)) / 16),\
                                   int((dx - (dx % 16)) / 16), int((dy - (dy % 16)) / 16)

            img_cropped = img[y:dy, x:dx]
            query_img_name = f_list[0]
            print 'Name of the query: ', query_img_name
            print 'Crop Height: ', img_cropped.shape[0]
            print 'Crop Width: ', img_cropped.shape[1]
            print 'Resized into...'

            h = img_cropped.shape[0] - (img_cropped.shape[0] % 16)
            w = img_cropped.shape[1] - (img_cropped.shape[1] % 16)

            img_cropped = preprocess_query(img_cropped, w, h, mean_value)
            x_feat = np.zeros((1, img_cropped.shape[0], img_cropped.shape[1], img_cropped.shape[2]),
                              dtype=np.float32)
            x_feat[0] = img_cropped
            model = VGGCAM(nb_classes, (img_cropped.shape[0], img_cropped.shape[1], img_cropped.shape[2]))
            model.load_weights(VGGCAM_weight_path)
            features_c, cams_c, class_list = extract_feat_cam(model, layer, 1, x_feat, num_cams)

            if img.shape[0] > img.shape[1]:
                size = size_v
                img_p = preprocess_query(img, size[0], size[1], mean_value)
                x_features = np.zeros((1, img_p.shape[0], img_p.shape[1], img_p.shape[2]), dtype=np.float32)
                x_features[0, :, :, :] = img_p
                features, cams, roi = extract_feat_cam(model_v, layer, 1, x_features,
                                                       num_cams, class_list[0, 0:num_cams], roi=True)

            else:
                size = size_h
                img_p = preprocess_query(img, size[0], size[1], mean_value)
                x_features = np.zeros((1, img_p.shape[0], img_p.shape[1], img_p.shape[2]), dtype=np.float32)
                x_features[0, :, :, :] = img_p
                features, cams, roi = extract_feat_cam(model_h, layer, 1, x_features,
                                                       num_cams, class_list[0, 0:num_cams], roi=True)

            d_wp = weighted_cam_pooling(features[:, :, f_y:f_dy, f_x:f_dx],
                                       cams[:, :, f_y:f_dy, f_x:f_dx], max_pool=False)

            desc = descriptor_aggregation(d_wp, 1, num_cams, pca_matrix)

            indices_local, data_local = eval.save_ranking_one_query(data, desc, image_names, ranking_path,
                                                                    query_img_name)

            desc = descriptor_aggregation(d_wp, 1, num_cams2, pca_matrix)

            if do_re_ranking:
                t_rerank = time.time()
                indices_re_ranking, data_re_ranking = re_ranking(desc, class_list[0, 0:num_cams2], image_names,
                                                                 indices_local, dataset, top_n_ranking, pca_matrix,
                                                                 model_h, model_v)
                print 'Time reranking: ', time.time() - t_rerank
                eval.save_ranking_indices(indices_re_ranking, image_names, query_img_name, ranking_path)

            if query_expansion:
                if do_re_ranking:
                    data_local[indices_re_ranking[0:top_n_ranking]] = data_re_ranking
                    desc_expanded = eval.expand_query(n_expand, data_local, indices_re_ranking)
                else:
                    desc_expanded = eval.expand_query(n_expand, data_local, indices_local)
                eval.save_ranking_one_query(data, desc_expanded, image_names, ranking_path, query_img_name)

    eval.evaluate_paris(ranking_path)

    print 'Evaluated: ' + descriptors_name