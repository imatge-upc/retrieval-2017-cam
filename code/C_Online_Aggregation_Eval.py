import numpy as np
import os
import h5py
import sys
import evaluate_oxford_paris as eval
import utils as utils
import time
import getopt
from vgg_cam import VGGCAM
from utils import create_folders, save_data, preprocess_images, preprocess_query, load_data, print_classes
from pooling_functions import weighted_cam_pooling, descriptor_aggregation, retrieve_n_descriptors, \
    descriptor_aggregation_cl, compute_pca
from cam_utils import extract_feat_cam
from scipy.misc import imread
import math
import pickle
from reranking import re_ranking

imagenet_dictionary = pickle.load(open("/home/jim011/workspace/retrieval-2017-icmr/imagenet1000_clsid_to_human.pkl", "rb"))

# Instructions Arguments: python script.py -d 'Oxford/Paris' -nc_q 32 -pca 1 -qe 10 -re 100
try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:', ['nc_q=', 'pca=', 'qe=', 're='])
    flag_nc_q = False
    flag_pca = False
    flag_d = False
    flag_qe = False
    flag_re = False

except getopt.GetoptError:
    print 'script.py -d <dataset> --nc_q <nclasses_query> --pca <n_classes_pca> --qe <n_query_exp> --re <n_re_ranking>'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-d':
        if arg == 'Oxford' or arg == 'Paris':
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


n_images_oxford = 5063
n_images_paris = 6392
n_queries = 55

# Descriptors for Re-ranking / Local Search

# Image Pre-Processing
dim = '1024x720'
size_v = [720, 1024]
size_h = [1024, 720]
mean_value = [123.68, 116.779, 103.939]

descriptors_dim = 512

# Parameters to set

# Dataset
if not flag_d:
    dataset = 'Oxford'
    print 'Default dataset: ', dataset

# Network Parameters
model_name = 'vgg_16_CAM_imagenet'
nb_classes = 1000
VGGCAM_weight_path = '/home/jim011/workspace/retrieval-2017-icmr/models/vgg_cam_weights.h5'
layer = 'relu5_1'
batch_size_re = 8

# Search Mode
local_search = True

# PCA Application
apply_pca = True
pca_dim = 512
if not flag_pca:
    num_classes_pca = 1
    print 'Default pca_classes: ', num_classes_pca

# N Class Activation Maps
if not flag_nc_q:
    num_cams = 6
    print 'Default classes: ', num_cams

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


model_v = VGGCAM(nb_classes, (3, 1024, 720))
model_v.load_weights(VGGCAM_weight_path)

model_h = VGGCAM(nb_classes, (3, 720, 1024))
model_h.load_weights(VGGCAM_weight_path)
count = 0

print 'Dataset: ', dataset
print 'Num_cams ', num_cams
print 'PCA with ', num_classes_pca
if do_re_ranking:
    print 'Re-Ranking the top  ', top_n_ranking
if query_expansion:
    print 'Query Expansion = ', n_expand


if dataset == 'Oxford':
    images_path = '/data/jim011/datasets_retrieval/Oxford5k/images/'
    ranking_path = '/home/jim011/workspace/retrieval-2017-icmr/results/oxford/' + model_name + '/' + layer + '/' + dim + '/R' +\
                   str(top_n_ranking) + 'QE' + str(n_expand)+'/'
    ranking_image_names_list = '/home/jim011/workspace/retrieval-2017-icmr/lists/list_oxford_rank.txt'
    utils.create_folders(ranking_path)

    descriptors_path = '/data/jim011/oxford/descriptors/Vgg_16_CAM/relu5_1/1024x720/online/'

    pca_descriptors_path = '/data/jim011/paris/descriptors/Vgg_16_CAM/relu5_1/1024x720/' \
                           'paris_all_64_wp.h5'
    n_images = n_images_oxford
    image_names = list()
    with open(ranking_image_names_list, "r") as f:
        for line in f:
            image_names.append(line)
    image_names = np.array(image_names)

    sys.stdout.flush()

    t = time.time()

    sys.stdout.flush()

    path_gt = '/data/jim011/datasets_retrieval/Oxford5k/ground_truth/'
    query_names = ["all_souls", "ashmolean", "balliol", "bodleian", "christ_church", "cornmarket", "hertford", "keble",
                   "magdalen", "pitt_rivers", "radcliffe_camera"]

    if apply_pca:
        pca_dim = 512
        pca_desc = retrieve_n_descriptors(num_classes_pca, n_images_paris, load_data(pca_descriptors_path))
        pca_matrix = compute_pca(pca_desc, pca_dim=pca_dim)
        data = np.zeros((n_images_oxford, pca_dim))
        print 'Data shape: ', data.shape
    else:
        pca_matrix = None

    print 'Loading descriptors...'
    sys.stdout.flush()
    descriptors = np.zeros((nb_classes*n_images_oxford, 512), dtype=np.float32)
    for index, img_n in enumerate(image_names):
        descriptors[index*nb_classes:(index+1)*nb_classes] = load_data(descriptors_path+img_n.replace('\n', '')+'.h5')


elif dataset == 'Paris':
    ranking_path = './results/paris/' + model_name + '/' + layer + '/' + dim + '/R' +\
                   str(top_n_ranking) + 'QE' + str(n_expand)+'/'
    ranking_image_names_list = '.r/lists/list_paris_rank.txt'
    utils.create_folders(ranking_path)

    descriptors_path = '/data/jim011/paris/descriptors/Vgg_16_CAM/relu5_1/1024x720/online/'

    pca_descriptors_path = '/data/jim011/oxford/descriptors/Vgg_16_CAM/relu5_1/1024x720/' \
                           'oxford_all_64_wp.h5'
    images_path = '/data/jim011/datasets_retrieval/Paris6k/images/'

    n_images = n_images_paris

    if local_search:
        ranking_image_names_list = '/home/jim011/workspace/retrieval-2017-icmr/lists/list_paris_rank.txt'

    data = np.zeros((n_images_paris, descriptors_dim))
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

    path_gt = "/data/jim011/datasets_retrieval/Paris6k/ground_truth/"
    query_names = ["defense", "eiffel", "invalides", "louvre", "moulinrouge", "museedorsay", "notredame", "pantheon",
                   "pompidou", "sacrecoeur", "triomphe"]

    if apply_pca:
        pca_dim = 512
        pca_desc = retrieve_n_descriptors(num_classes_pca, n_images_oxford, load_data(pca_descriptors_path))
        print pca_desc.shape
        pca_matrix = compute_pca(pca_desc, pca_dim=pca_dim)
        data = np.zeros((n_images_oxford, pca_dim))
        print 'Data shape: ', data.shape
    else:
        pca_matrix = None

tcomp = time.time()

for query_name in query_names:
    print count
    for i in range(1, 6):
        f = open(path_gt + query_name + '_' + str(i) + '_query.txt').readline()
        if dataset == 'Oxford':
            f = f.replace("oxc1_", "")
        f_list = f.split(" ")
        for k in range(1, 5):
            f_list[k] = (int(math.floor(float(f_list[k]))))

        img = imread(images_path + f_list[0] + '.jpg')
        print 'Image Shape: ' + str(img.shape[0]) + 'x' + str(img.shape[1])

        # Queries bounding box
        x, y, dx, dy = f_list[1], f_list[2], f_list[3], f_list[4]

        # Feature Maps bounding box
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

            x_feat = np.zeros((1, img_cropped.shape[0], img_cropped.shape[1], img_cropped.shape[2]), dtype=np.float32)

            model = VGGCAM(nb_classes, (img_cropped.shape[0], img_cropped.shape[1], img_cropped.shape[2]))
            model.load_weights(VGGCAM_weight_path)
            x_feat[0, :, :, :] = img_cropped
            features, cams, class_list = extract_feat_cam(model, layer, 1, x_feat, num_cams)

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
        data = descriptor_aggregation_cl(descriptors, n_images, pca_matrix, class_list[0])
        print 'Time elapsed agregation: ', time.time() - tagregation

        indices_local, data_local = eval.save_ranking_one_query(data, desc, image_names, ranking_path, query_img_name)

        if do_re_ranking:
            t_rerank = time.time()
            indices_re_ranking, data_re_ranking = re_ranking(desc, class_list[0], batch_size_re, image_names,
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

if dataset == 'Oxford':
    eval.evaluate_oxford(ranking_path)
elif dataset == 'Paris':
    eval.evaluate_paris(ranking_path)

print 'Time elapsed computing distances: ', time_ranking
print 'Time elapsed total: ', time.time() - t
