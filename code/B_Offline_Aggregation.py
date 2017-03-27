import numpy as np
import time
import sys
from pooling_functions import descriptor_aggregation, retrieve_n_descriptors, compute_pca
from utils import create_folders, save_data, load_data
import getopt


# Instructions Arguments: python script.py -d 'Oxford/Paris' --nc 32 --pca 1
try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:', ['nc=', 'pca='])
    flag_nc = False
    flag_pca = False
    flag_d = False
except getopt.GetoptError:
    print 'script.py -d <dataset> -a <aggregation>'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-d':
        if arg == 'Oxford' or arg == 'Paris' or arg == 'Oxford105k' or arg == 'Paris106k':
            dataset = arg
            flag_d = True

    elif opt == '--nc':
            num_classes = int(arg)
            flag_nc = True

    elif opt == '--pca':
            num_classes_pca = int(arg)
            flag_pca = True


# Parameters
if not flag_d:
    dataset = 'Oxford'
    print 'Default dataset: ', dataset

if not flag_nc:
    num_classes = 32
    print 'Default classes: ', num_classes


n_images_distractors = 100070
n_images_oxford = 5063
n_images_paris = 6392

# Num classes stored in the precomputed --> Have to be set up
num_prec_classes = 64

model_name = 'Vgg_16_CAM'
layer = 'relu5_1'
dim = '1024x720'

# PCA
apply_pca = True
pca_dim = 512
dim_descriptor = 512

if not flag_pca:
    num_classes_pca = 1
    print 'Default pca_classes: ', num_classes_pca


print 'Dataset ', dataset
print 'Num Classes ', num_classes
print 'Dimension Descriptor ', dim_descriptor

if apply_pca:
    print 'Num Classes PCA ', num_classes_pca
    print 'Dimension PCA ', pca_dim

t_0 = time.time()

# DATASET PATHS AND NAMES
if dataset == 'Oxford':
    path_descriptors = '/data/jim011/oxford/descriptors/' + \
                       model_name + '/' + layer + '/' + dim + '/'

    create_folders(path_descriptors)
    cam_descriptors_path = '/data/jim011/oxford/descriptors/' + model_name + '/' + layer + '/' + dim + '/' \
                           'oxford_all_64_wp.h5'

    pca_descriptors_path = '/data/jim011/paris/descriptors/' + model_name + '/' + layer + '/' + dim + '/' \
                           'paris_all_64_wp.h5'
    num_images = n_images_oxford
    num_img_pca = n_images_paris
    name_descriptors = 'oxford_' + str(num_classes)
    pca_name = '_pca_' + str(pca_dim) + '_paris_'+str(num_classes_pca)

if dataset == 'Oxford105k':
    path_descriptors = '/data/jim011/oxford/descriptors/' + \
                       model_name + '/' + layer + '/' + dim + '/'

    create_folders(path_descriptors)
    cam_descriptors_path = '/data/jim011/descriptors100k/descriptors/' + model_name + '/' + layer + '/' + dim + '/' \
                           'distractor_all_64_wp_'

    pca_descriptors_path = '/data/jim011/paris/descriptors/' + model_name + '/' + layer + '/' + dim + '/' \
                           'paris_all_64_wp.h5'
    num_images = n_images_distractors
    num_img_pca = n_images_paris
    name_descriptors = 'oxford_105k_' + str(num_classes)
    pca_name = '_pca_' + str(pca_dim) + '_paris_'+str(num_classes_pca)

if dataset == 'Paris':
    path_descriptors = '/data/jim011/paris/descriptors/' + \
                       model_name + '/' + layer+'/' + dim + '/'

    create_folders(path_descriptors)
    cam_descriptors_path = '/data/jim011/paris/descriptors/' + model_name + '/' + layer + '/' + dim + '/' \
                           'paris_all_64_wp.h5'
    pca_descriptors_path = '/data/jim011/oxford/descriptors/' + model_name + '/' + layer + '/' + dim + '/' \
                           'oxford_all_64_wp.h5'
    num_images = n_images_paris
    num_img_pca = n_images_oxford
    name_descriptors = 'paris_' + str(num_classes)
    pca_name = '_pca_' + str(pca_dim) + '_oxford_'+str(num_classes_pca)

if dataset == 'Paris106k':
    path_descriptors = '/data/jim011/paris/descriptors/' + \
                       model_name + '/' + layer + '/' + dim + '/'

    create_folders(path_descriptors)
    cam_descriptors_path = '/data/jim011/descriptors100k/descriptors/' + model_name + '/' + layer + '/' + dim + '/' \
                           'distractor_all_64_wp_'

    pca_descriptors_path = '/data/jim011/oxford/descriptors/' + model_name + '/' + layer + '/' + dim + '/' \
                           'oxford_all_64_wp.h5'
    num_images = n_images_distractors
    num_img_pca = n_images_oxford
    name_descriptors = 'paris_106k_' + str(num_classes)
    pca_name = '_pca_' + str(pca_dim) + '_oxford_'+str(num_classes_pca)

########################################################################################################################

# Compute PCA
if apply_pca:
    tpca = time.time()
    pca_desc = retrieve_n_descriptors(num_classes_pca, num_img_pca, load_data(pca_descriptors_path))
    pca_matrix = compute_pca(pca_desc, pca_dim=pca_dim, whiten=True)
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