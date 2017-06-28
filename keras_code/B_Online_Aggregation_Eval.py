import numpy as np
import os
import h5py
import sys
import evaluate_oxford_paris as eval
import utils as utils
import time
import getopt
from vgg_cam import vggcam
from utils import create_folders, save_data, preprocess_images, load_data, print_classes
from pooling_functions import weighted_cam_pooling, descriptor_aggregation, retrieve_n_descriptors, \
    descriptor_aggregation_cl, compute_pca
from cam_utils import extract_feat_cam_fast, get_output_layer
from scipy.misc import imread
import math
import pickle
from reranking import re_ranking
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
    flag_qe = False
    flag_re = False
    flag_nc_re = False

except getopt.GetoptError:
    print 'script.py -d <dataset> --nc_q <nclasses_query> --pca <n_classes_pca> --qe <n_query_exp> --re <n_re_ranking>' \
          '--nc_re <n_classes_re_ranking>'
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

    elif opt == '--nc_re':
        num_cams2 = int(arg)
        flag_nc_re = True


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

# Re-Ranking CAMs
batch_size_re = 6  # More large = Faster, depends on GPU RAM
num_cams_re = 6

# PCA Application
apply_pca = True
pca_dim = 512
if not flag_pca:
    num_classes_pca = 1
    print 'Default pca_classes: ', num_classes_pca

# N Class Activation Maps
if not flag_nc_q:
    num_cams = 64
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


model = vggcam(nb_classes)
model.load_weights(VGGCAM_weight_path)

count = 0

print 'Dataset: ', dataset
print 'Num_cams ', num_cams
print 'PCA with ', num_classes_pca
if do_re_ranking:
    print 'Re-Ranking the top  ', top_n_ranking
if query_expansion:
    print 'Query Expansion = ', n_expand


if dataset == 'Oxford':
    # Set Paths
    images_path = '/data/jim011/datasets_retrieval/Oxford5k/images/'
    ranking_path = '../results/oxford/' + model_name + '/' + layer + '/' + dim + '/R' +\
                   str(top_n_ranking) + 'QE' + str(n_expand)+'/online/'
    ranking_image_names_list = '../lists/list_oxford_rank.txt'
    utils.create_folders(ranking_path)
    n_images_pca = n_images_paris

    descriptors_path = '/data/jim011/oxford/descriptors/Vgg_16_CAM/relu5_1/1024x720/online/'

    pca_descriptors_path = '/data/jim011/paris/descriptors/Vgg_16_CAM/relu5_1/1024x720/' \
                           'paris_all_64_wp.h5'
    n_images = n_images_oxford

    path_gt = '/data/jim011/datasets_retrieval/Oxford5k/ground_truth/'
    query_names = ["all_souls", "ashmolean", "balliol", "bodleian", "christ_church", "cornmarket", "hertford", "keble",
                   "magdalen", "pitt_rivers", "radcliffe_camera"]

    # Load Image names
    image_names = list()
    with open(ranking_image_names_list, "r") as f:
        for line in f:
            image_names.append(line)
    image_names = np.array(image_names)

    # Load Class Vectors
    print 'Loading descriptors...'

    descriptors = np.zeros((nb_classes*n_images_oxford, 512), dtype=np.float32)
    for index, img_n in enumerate(image_names):
        descriptors[index*nb_classes:(index+1)*nb_classes] = load_data(descriptors_path+img_n.replace('\n', '')+'.h5')

    print 'Descriptors loaded'
    sys.stdout.flush()


elif dataset == 'Paris':
    # Set Paths
    images_path = '/data/jim011/datasets_retrieval/Paris6k/images/'

    ranking_path = '../results/paris/' + model_name + '/' + layer + '/' + dim + '/R' +\
                   str(top_n_ranking) + 'QE' + str(n_expand)+'/online/'
    ranking_image_names_list = '../lists/list_paris_rank.txt'
    utils.create_folders(ranking_path)

    descriptors_path = '/data/jim011/paris/descriptors/Vgg_16_CAM/relu5_1/1024x720/online/'

    pca_descriptors_path = '/data/jim011/oxford/descriptors/Vgg_16_CAM/relu5_1/1024x720/' \
                           'oxford_all_64_wp.h5'
    n_images = n_images_paris
    n_images_pca = n_images_oxford

    path_gt = "/data/jim011/datasets_retrieval/Paris6k/ground_truth/"
    query_names = ["defense", "eiffel", "invalides", "louvre", "moulinrouge", "museedorsay", "notredame", "pantheon",
                   "pompidou", "sacrecoeur", "triomphe"]

    # Load Image Names
    image_names = list()
    with open(ranking_image_names_list, "r") as f:
        for line in f:
            image_names.append(line)

    image_names = np.array(image_names)

    # Load Class Vectors
    print 'Loading descriptors...'

    descriptors = np.zeros((nb_classes * n_images_paris, 512))
    for index, img_n in enumerate(image_names):
        descriptors[index * nb_classes:(index + 1) * nb_classes] = load_data(
            descriptors_path + img_n.replace('\n', '') + '.h5')

    print 'Descriptors loaded'
    sys.stdout.flush()


maps = list()

if apply_pca:
    pca_desc = retrieve_n_descriptors(num_classes_pca, n_images_pca, load_data(pca_descriptors_path))
    print pca_desc.shape
    pca_matrix = compute_pca(pca_desc, pca_dim=pca_dim)
else:
    pca_matrix = None

for query_name in query_names:
    for i in range(1, 6):
        f = open(path_gt + query_name + '_' + str(i) + '_query.txt').readline()
        if dataset == 'Oxford':
            f = f.replace("oxc1_", "")
        f_list = f.split(" ")
        for k in range(1, 5):
            f_list[k] = (int(math.floor(float(f_list[k]))))

        img = imread(images_path + f_list[0] + '.jpg')
        #print 'Image Shape: ' + str(img.shape[0]) + 'x' + str(img.shape[1])

        # Queries bounding box
        x, y, dx, dy = f_list[1], f_list[2], f_list[3], f_list[4]

        # Feature Maps bounding box
        f_x, f_y, f_dx, f_dy = int((x - (x % 16)) / 16), int((y - (y % 16)) / 16), \
                               int((dx - (dx % 16)) / 16), int((dy - (dy % 16)) / 16)

        img_cropped = img[y:dy, x:dx]
        query_img_name = f_list[0]

        print 'Name of the query: ', f_list[0]

        h = img_cropped.shape[0] - (img_cropped.shape[0] % 16)
        w = img_cropped.shape[1] - (img_cropped.shape[1] % 16)
        img_cropped = preprocess_images(img_cropped, w, h, mean_value)

        features_c, cams_c, class_list = extract_feat_cam_fast(model, get_output, conv_layer_features, 1, img_cropped, num_cams)

        # Show Classes of Query (uncomment to see)
        # print_classes(imagenet_dictionary, class_list[0])

        if img.shape[0] > img.shape[1]:
            size = size_v
        else:
            size = size_h

        img_p = preprocess_images(img, size[0], size[1], mean_value)

        features, cams, roi = extract_feat_cam_fast(model, get_output, conv_layer_features, 1, img_p,
                                    num_cams, class_list[0, 0:num_cams])

        d_wp = weighted_cam_pooling(features[:, :, f_y:f_dy, f_x:f_dx],
                                    cams[:, :, f_y:f_dy, f_x:f_dx])

        desc = descriptor_aggregation(d_wp, 1, num_cams, pca_matrix)

        tagregation = time.time()
        data = descriptor_aggregation_cl(descriptors, n_images, pca_matrix, class_list[0])
        print 'Time elapsed agregation: ', time.time() - tagregation

        indices_local, data_local = eval.save_ranking_one_query(data, desc, image_names, ranking_path, query_img_name)

        if do_re_ranking:
            desc = descriptor_aggregation(d_wp, 1, num_cams_re, pca_matrix)
            t_rerank = time.time()
            indices_re_ranking, data_re_ranking = re_ranking(desc, class_list[0,0:num_cams_re], batch_size_re, image_names,
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

if dataset == 'Oxford':
    maps.append(eval.evaluate_oxford(ranking_path))
elif dataset == 'Paris':
    maps.append(eval.evaluate_paris(ranking_path))

maps_file = open(ranking_path + 'maps' + dataset + '_pca_' + str(num_classes_pca) + '_Nc_' + str(num_cams) + '.txt', 'w')

print maps

for res in maps:
    maps_file.write(str(res) + '\n')

maps_file.close()
