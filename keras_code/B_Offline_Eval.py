import numpy as np
import os
import h5py
import sys
import getopt
import evaluate_oxford_paris as eval
import utils as utils
import time
from vgg_cam import vggcam
from cam_utils import extract_feat_cam_fast, get_output_layer
from utils import create_folders, save_data, preprocess_images, load_data, print_classes
from pooling_functions import descriptor_aggregation, retrieve_n_descriptors, compute_pca, sum_pooling, weighted_cam_pooling
from scipy.misc import imread
import math
from reranking import re_ranking
import pickle
import cv2
from keras.models import *
from keras.callbacks import *
import keras.backend as K


imagenet_dictionary = pickle.load(open("../imagenet1000_clsid_to_human.pkl", "rb"))

# Instructions Arguments: python script.py -d 'Oxford/Paris' --nc_q 32 --pca 1 --qe 10 --re 100 --nc_re 6

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:', ['nc_q=', 'pca=', 'qe=', 're=', 'nc_re='])
    flag_nc_q = False
    flag_pca = False
    flag_d = False
    flag_nc_re = False
    flag_qe = False
    flag_re = False
except getopt.GetoptError:
    print 'script.py -d <dataset> --nc_q <nclasses_query> --pca <n_classes_pca> --qe <n_query_exp> --re <n_re_ranking> ' \
          '--nc_re <n_classes_re_ranking>'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-d':
        if arg == 'Oxford' or arg == 'Paris' or arg == 'Oxford105k' or arg == 'Paris106k':
            dataset = arg
            flag_d = True

    elif opt == '--nc_q':
            num_cams = int(arg)
            flag_nc_q = True

    elif opt == '--pca':
            num_classes_pca = int(arg)
            flag_pca = True

    elif opt == '--qe':
            n_expand = int(arg)
            query_expansion = True
            flag_qe = True

    elif opt == '--re':
            do_re_ranking = True
            top_n_ranking = int(arg)
            flag_re = True

    elif opt == '--nc_re':
            num_cams2 = int(arg)
            flag_nc_re = True

# SET FOR RE-RANKING
batch_size_re = 6   # More large = Faster, depends on GPU RAM

# Global params
n_images_distractors = 100070
n_images_oxford = 5063
n_images_paris = 6392
n_queries = 55

# Descriptors for Re-ranking  (Size W x H)
dim = '1024x720'
size_v = [720, 1024]
size_h = [1024, 720]
mean_value = [123.68, 116.779, 103.939]

# Parameters to set

# Dataset
if not flag_d:
    dataset = 'Oxford'
    print 'Default dataset: ', dataset

# Network Parameters
nb_classes = 1000
VGGCAM_weight_path = '../models/vgg_cam_weights.h5'
model_name = 'vgg_16_CAM'
layer = 'relu5_1'

model = vggcam(nb_classes)
model.load_weights(VGGCAM_weight_path)

# For faster processing of individual queries:
# Set convolutional layer to extract the CAMs (CAM_relu layer)
final_conv_layer = get_output_layer(model, "CAM_relu")

# Set layer to extract the features
conv_layer_features = get_output_layer(model, layer)

# Function to get scores, conv_maps --> Could be implemented outside, bottleneck
get_output = K.function([model.layers[0].input, K.learning_phase()],
                        [final_conv_layer.output, model.layers[-1].output, conv_layer_features.output])

count = 0

# PCA Parameters
apply_pca = True
pca_dim = 512

if not flag_pca:
    num_classes_pca = 1
    print 'Default pca_classes: ', num_classes_pca

# N Class Activation Maps
if not flag_nc_q:
    num_cams = 64
    print 'Default classes: ', num_cams

# Num_cams2 --> Used to compute the descriptors when re-ranking
if not flag_nc_re:
    num_cams_re = 6
    print 'Default classes for re-ranking: ', num_cams_re

# Re-ranking
if not flag_re:
    do_re_ranking = False
    top_n_ranking = 0
    print 'Not doing Re-ranking'

# Query Expansion
if not flag_qe:
    # Re-ranking
    query_expansion = False
    n_expand = 0
    print 'Not doing Query Expansion'

num_prec_classes = 64

print 'Dataset: ', dataset
print 'Num_cams ', num_cams
print 'PCA with ', num_classes_pca
if do_re_ranking:
    print 'Re-ranking with first ', top_n_ranking
if query_expansion:
    print 'Applying query expansion using the first ', n_expand


if dataset == 'Oxford':
    image_path = '/data/jim011/datasets_retrieval/Oxford5k/images/'
    ranking_path = '../results/oxford/' + model_name + '/' + layer + '/' + dim \
                   + '/R' + str(top_n_ranking) + 'QE' + str(n_expand)+'/off/'
    ranking_image_names_list = '../lists/list_oxford_rank.txt'
    utils.create_folders(ranking_path)

    pca_descriptors_path = '/data/jim011/paris/descriptors/Vgg_16_CAM/relu5_1/1024x720/' \
                           'paris_all_64_wp.h5'

    cam_descriptors_path = '/data/jim011/oxford/descriptors/Vgg_16_CAM/relu5_1/1024x720/oxford_all_64_wp.h5'

    n_images_pca = n_images_paris

    num_images = n_images_oxford

    t = time.time()

    image_names = list()

    with open(ranking_image_names_list, "r") as f:
        for line in f:
            image_names.append(line)

    image_names = np.array(image_names)

    print image_names

    sys.stdout.flush()

    path_gt = "/data/jim011/datasets_retrieval/Oxford5k/ground_truth/"
    query_names = ["all_souls", "ashmolean", "balliol", "bodleian", "christ_church", "cornmarket", "hertford", "keble",
                   "magdalen", "pitt_rivers", "radcliffe_camera"]


elif dataset == 'Paris':
    image_path = '/data/jim011/datasets_retrieval/Paris6k/images/'
    ranking_path = '../results/paris/' + model_name + '/' + layer + '/' + dim \
                   + '/R' + str(top_n_ranking) + 'QE' + str(n_expand) + '/off/'
    ranking_image_names_list = '../lists/list_paris_rank.txt'
    utils.create_folders(ranking_path)

    descriptors_path = '/data/jim011/paris/descriptors/Vgg_16_CAM/relu5_1/1024x720/'

    pca_descriptors_path = '/data/jim011/oxford/descriptors/Vgg_16_CAM/relu5_1/1024x720/' \
                           'oxford_all_64_wp.h5'

    cam_descriptors_path = '/data/jim011/paris/descriptors/Vgg_16_CAM/relu5_1/1024x720/paris_all_64_wp.h5'
    num_images = n_images_paris
    n_images_pca = n_images_oxford

    t = time.time()

    image_names = list()

    with open(ranking_image_names_list, "r") as f:
        for line in f:
            image_names.append(line)

    image_names = np.array(image_names)

    path_gt = "/data/jim011/datasets_retrieval/Paris6k/ground_truth/"
    query_names = ["defense", "eiffel", "invalides", "louvre", "moulinrouge", "museedorsay", "notredame", "pantheon",
                   "pompidou", "sacrecoeur", "triomphe"]


elif dataset == 'Oxford105k':
    image_path = '/data/jim011/datasets_retrieval/Oxford5k/images/'
    ranking_path = '../results/oxford105k/' + model_name + '/' + layer + '/' \
                   + dim + '/' + '/R' + str(top_n_ranking) + 'QE' + str(n_expand)+'/off/'
    ranking_image_names_list = '../lists/list_oxford_rank.txt'
    ranking_distractors_list = '../lists/list_oxford_105k_rank.txt'
    utils.create_folders(ranking_path)

    cam_distractors_path = '/data/jim011/descriptors100k/descriptors/Vgg_16_CAM/relu5_1/1024x720/distractor_all_64_wp_'

    pca_descriptors_path = '/data/jim011/paris/descriptors/Vgg_16_CAM/relu5_1/1024x720/' \
                           'paris_all_64_wp.h5'

    cam_descriptors_path = '/data/jim011/oxford/descriptors/Vgg_16_CAM/relu5_1/1024x720/oxford_all_64_wp.h5'

    n_images_pca = n_images_paris

    num_images = n_images_oxford
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

    path_gt = "/data/jim011/datasets_retrieval/Oxford5k/ground_truth/"
    query_names = ["all_souls", "ashmolean", "balliol", "bodleian", "christ_church", "cornmarket", "hertford", "keble",
                   "magdalen", "pitt_rivers", "radcliffe_camera"]


elif dataset == 'Paris106k':
    image_path = '/data/jim011/datasets_retrieval/Paris6k/images/'
    ranking_path = '../results/paris106k/' + model_name + '/' + layer + '/' \
                   + dim + '/' + '/R' + str(top_n_ranking) + 'QE' + str(n_expand)+'/off/'
    ranking_image_names_list = '../lists/list_paris_rank.txt'
    ranking_distractors_list = '../lists/list_oxford_105k_rank.txt'
    utils.create_folders(ranking_path)

    cam_distractors_path = '/data/jim011/descriptors100k/descriptors/Vgg_16_CAM/relu5_1/1024x720/distractor_all_64_wp_'

    pca_descriptors_path = '/data/jim011/oxford/descriptors/Vgg_16_CAM/relu5_1/1024x720/' \
                           'oxford_all_64_wp.h5'

    cam_descriptors_path = '/data/jim011/paris/descriptors/Vgg_16_CAM/relu5_1/1024x720/paris_all_64_wp.h5'

    n_images_pca = n_images_oxford

    sys.stdout.flush()
    num_images = n_images_paris

    image_names = list()
    with open(ranking_image_names_list, "r") as f:
        for line in f:
            image_names.append(line)

    with open(ranking_distractors_list, "r") as f:
        for line in f:
            image_names.append(line)

    image_names = np.array(image_names)

    path_gt = "/data/jim011/datasets_retrieval/Paris6k/ground_truth/"
    query_names = ["defense", "eiffel", "invalides", "louvre", "moulinrouge", "museedorsay", "notredame", "pantheon",
                   "pompidou", "sacrecoeur", "triomphe"]


maps = list()

# PCA and Aggregation --> This could be precomputed, for ablation purposes it has been included here

if apply_pca:
    pca_desc = retrieve_n_descriptors(num_classes_pca, n_images_pca, load_data(pca_descriptors_path))
    pca_matrix = compute_pca(pca_desc, pca_dim=pca_dim, whiten=True)
    print 'PCA matrix shape:', pca_matrix.components_.shape
else:
    pca_matrix = None

if dataset == 'Oxford105k' or dataset == 'Paris106k':
    n_chunks = 10
    distractors = np.zeros((0, 512), dtype=np.float32)
    for n_in in range(0, n_chunks + 1):
        desc = load_data(cam_distractors_path + str(n_in) + '.h5')
        print desc.shape
        distractors = np.concatenate((distractors, descriptor_aggregation(desc, desc.shape[0] / num_prec_classes,
                                                                          num_cams, pca_matrix)))
        print distractors.shape
        t = time.time()
        cam_descriptors = load_data(cam_descriptors_path)
        print 'Time elapsed loading: ', time.time() - t
        sys.stdout.flush()
    data = descriptor_aggregation(cam_descriptors, num_images, num_cams, pca_matrix)
    data = np.concatenate((data, distractors))

elif dataset == 'Oxford' or dataset == 'Paris':
    t = time.time()
    cam_descriptors = load_data(cam_descriptors_path)
    print 'Time elapsed loading: ', time.time() - t
    data = descriptor_aggregation(cam_descriptors, num_images, num_cams, pca_matrix)
    #data = pca_matrix.transform(cam_descriptors)
    #data /= np.linalg.norm(data)
    #data = sum_pooling(cam_descriptors, CroW = True)
    #data = cam_descriptors

    sys.stdout.flush()
t_vector = list()

# TESTING

for query_name in query_names:
    print count
    for i in range(1, 6):
        f = open(path_gt + query_name + '_' + str(i) + '_query.txt').readline()
        if dataset == 'Oxford' or dataset == 'Oxford105k':
            f = f.replace("oxc1_", "")
        f_list = f.split(" ")
        for k in range(1, 5):
            f_list[k] = (int(math.floor(float(f_list[k]))))

        query_img_name = f_list[0]
        img = imread(image_path + query_img_name + '.jpg')

        # Query bounding box
        x, y, dx, dy = f_list[1], f_list[2], f_list[3], f_list[4]

        # Feature map of the query bounding box (the dimensions of the last layer are 16 times smaller)

        f_x, f_y, f_dx, f_dy = int((x - (x % 16)) / 16), int((y - (y % 16)) / 16), \
                               int((dx - (dx % 16)) / 16), int((dy - (dy % 16)) / 16)

        img_cropped = img[y:dy, x:dx]

        print 'Name of the query: ', query_img_name

        # Make multiple of 16

        h = img_cropped.shape[0] - (img_cropped.shape[0] % 16)
        w = img_cropped.shape[1] - (img_cropped.shape[1] % 16)
        img_cropped = preprocess_images(img_cropped, w, h, mean_value)

        # Obtain the classes from the cropped query (To-do implement it directly on model --> Now we do 2 forward)
        features_c, cams_c, class_list = extract_feat_cam_fast(model, get_output, conv_layer_features,
                                                               1, img_cropped, num_cams)

        if img.shape[0] > img.shape[1]:
            size = size_v
        else:
            size = size_h

        # Query resized to be 1024x720 or 720x1024
        img_p = preprocess_images(img, size[0], size[1], mean_value)

        # Obtain features for all the image
        time_search = time.time()
        t_extract = time.time()
        features, cams, _ = extract_feat_cam_fast(model, get_output, conv_layer_features, 1, img_p,
                                               num_cams, class_list[0, 0:num_cams])

        print 'Time extraction features ', time.time() - t_extract

        # Build the descriptor with the query features
        cam_pool_t = time.time()

        # Cropped Features (Query Bounding Box)
        d_wp = weighted_cam_pooling(features[:, :, f_y:f_dy, f_x:f_dx], cams[:, :, f_y:f_dy, f_x:f_dx])

        # Cropped Query (Bounding Box)
        #d_wp = weighted_cam_pooling(features_c, cams_c)

        print 'time campool ', time.time() - cam_pool_t
        # Compute Query Descriptor
        time_agg = time.time()

        desc = descriptor_aggregation(d_wp, 1, num_cams, pca_matrix)

        # For Sum-Pooling (Raw)

        #desc = pca_matrix.transform(sum_pooling(features[:, :, f_y:f_dy, f_x:f_dx], CroW=False))
        #desc = pca_matrix.transform(sum_pooling(features[:, :, f_y:f_dy, f_x:f_dx], CroW=True))
        #desc = sum_pooling(features_c)
        #desc /= np.linalg.norm(desc)

        print 'time agg ', time.time() - time_agg

        time_ind = time.time()
        indices_local, data_local = eval.save_ranking_one_query(data, desc, image_names, ranking_path, query_img_name)
        print 'Time ind = ', time.time() - time_ind
        print 'Time search = ', time.time() - time_search
        t_vector.append(time.time() - time_search)

        if do_re_ranking:
            # When re-ranking descriptor for the query computed with less CAMs, as we know the relevant objects
            desc = descriptor_aggregation(d_wp, 1, num_cams_re, pca_matrix)
            t_rerank = time.time()
            indices_re_ranking, data_re_ranking = re_ranking(desc, class_list[0, 0:num_cams_re], batch_size_re, image_names,
                                                             indices_local, dataset, top_n_ranking, pca_matrix,
                                                             model, conv_layer_features, get_output)
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
print np.array(t_vector).sum()/55
if dataset == 'Oxford' or dataset == 'Oxford105k':
    maps.append(eval.evaluate_oxford(ranking_path))
elif dataset == 'Paris' or dataset == 'Paris106k':
    maps.append(eval.evaluate_paris(ranking_path))

maps_file = open(ranking_path + 'maps' + dataset + '_Npca_' + str(num_classes_pca) + '_Nc_' + str(num_cams) + '.txt', 'w')

print maps

for res in maps:
    maps_file.write(str(res) + '\n')

maps_file.close()
