from keras.models import *
from keras.callbacks import *
import keras.backend as K
import os
import sys
from scipy.misc import imread
import time
import h5py
import numpy as np
from cam_utils import extract_feat_cam, extract_feat_cam_all
from vgg_cam import VGGCAM
from models_utils import extract_features
from utils_datasets import create_folders, save_data, preprocess_images, preprocess_query
from pooling_functions import weighted_cam_pooling, weighted_pooling


# Dataset Selection
#dataset = 'Oxford'
#dataset = 'Paris'
dataset = 'distractors100k'

# Image Pre-processing

# Horizontal Images
size_h = [1024, 720]
# Vertical Images
size_v = [720, 1024]

dim = '1024x720'

# Mean to substract
mean_data = 'Imagenet'

# Model Selection
model_name = 'Vgg_16_CAM'

if mean_data == 'Places':
    mean_value = [104, 166.66, 122.67]
    folder = 'places/'
elif mean_data == 'Imagenet':
    mean_value = [123.68, 116.779, 103.939]
    folder = 'imagenet/'
else:
    mean_value = [0, 0, 0]

# Model Selection: VGG_CAM
if model_name == 'Vgg_16_CAM':
    nb_classes = 1000
    VGGCAM_weight_path = '/imatge/ajimenez/work/ITR/models/vgg_cam_weights.h5'


layer = 'relu5_1'
dim_descriptor = 512

# CAM Extraction
batch_size = 12
num_classes = 64


if dataset == 'Oxford':
    n_img_dataset = 5063
    train_list_path_h = "/imatge/ajimenez/work/ITR/oxford/lists/list_oxford_horizontal_no_queries.txt"
    train_list_path_v = "/imatge/ajimenez/work/ITR/oxford/lists/list_oxford_vertical_no_queries.txt"
    path_descriptors = '/imatge/ajimenez/work/ITR/oxford/descriptors_new/' + model_name + '/' + layer + '/' + dim + '/'
    descriptors_cams_path_wp = path_descriptors + 'oxford_all_' + str(num_classes) + '_wp.h5'
    descriptors_cams_path_mp = path_descriptors + 'oxford_all_' + str(num_classes) + '_mp.h5'
    create_folders(path_descriptors)

if dataset == 'Paris':
    n_img_dataset = 6392
    train_list_path_h = "/imatge/ajimenez/work/ITR/paris/lists/list_paris_horizontal_no_queries.txt"
    train_list_path_v = "/imatge/ajimenez/work/ITR/paris/lists/list_paris_vertical_no_queries.txt"
    path_descriptors = '/imatge/ajimenez/work/ITR/paris/descriptors_new/' + model_name + '/' + layer + '/' + dim + '/'
    descriptors_cams_path_wp = path_descriptors + 'paris_all_' + str(num_classes) + '_wp.h5'
    descriptors_cams_path_mp = path_descriptors + 'paris_all_' + str(num_classes) + '_mp.h5'
    create_folders(path_descriptors)

if dataset == 'distractors100k':
    n_img_dataset = 100070
    train_list_path_h = "/imatge/ajimenez/workspace/ITR/lists/list_oxford105k_horizontal.txt"
    train_list_path_v = "/imatge/ajimenez/workspace/ITR/lists/list_oxford105k_vertical.txt"
    path_descriptors = '/imatge/ajimenez/work/ITR/descriptors100k/descriptors_new/' + model_name + '/' + layer + '/' + dim + '/'
    descriptors_cams_path_wp = path_descriptors + 'distractor_all_' + str(num_classes) + '_wp.h5'
    descriptors_cams_path_mp = path_descriptors + 'distractor_all_' + str(num_classes) + '_mp.h5'
    create_folders(path_descriptors)


def extract_cam_descriptors(model_name, batch_size, num_classes, size, mean_value, image_train_list_path, desc_wp, desc_mp):
    images = [0] * batch_size
    image_names = [0] * batch_size
    counter = 0
    num_images = 0
    ind = 0
    t0 = time.time()
    print 'Horizontal size: ', size[0]
    print 'Vertical size: ', size[1]

    if model_name == 'Vgg_16_CAM':
        model = VGGCAM(nb_classes, (3, size[1], size[0]))
        model.load_weights(VGGCAM_weight_path)
    print 'Model loaded'
    model.summary()
    for line in open(image_train_list_path):
        if counter >= batch_size:
            print 'Processing image batch: ', ind
            t1 = time.time()
            data = preprocess_images(images, size[0], size[1], mean_value)
            features, cams, cl = extract_feat_cam(model, layer, batch_size, data, num_classes)
            #features, cams = extract_feat_cam_all(model, layer, batch_size, data)
            d_wp, d_mp = weighted_cam_pooling(features, cams)
            #for img_ind in range(0, batch_size):
            #    print 'Saved ' + image_names[img_ind] + '.h5'
            #    save_data(d_wp[img_ind*nb_classes:(img_ind+1)*nb_classes], path_descriptors, image_names[img_ind]+'.h5')
            desc_wp = np.concatenate((desc_wp, d_wp))
            desc_mp = np.concatenate((desc_mp, d_mp))
            print 'Batch processed, CAMs descriptors obtained!'
            print 'Time elapsed: ', time.time()-t1
            #print desc_wp.shape
            sys.stdout.flush()
            counter = 0
            ind += 1

        line = line.rstrip('\n')
        images[counter] = imread(line)
        if dataset == 'Oxford':
            line = line.replace('/imatge/ajimenez/work/datasets_retrieval/Oxford/1_images/', '')
        elif dataset == 'Paris':
            line = line.replace('/imatge/ajimenez/work/datasets_retrieval/Paris/imatges_paris/', '')
        image_names[counter] = (line.replace('.jpg', ''))
        counter += 1
        num_images += 1


    #Last batch
    print 'Last Batch'
    data = np.zeros((counter, 3, size[1], size[0]), dtype=np.float32)
    data[0:] = preprocess_images(images[0:counter], size[0], size[1], mean_value)
    features, cams, cl = extract_feat_cam(model, layer, batch_size, data, num_classes)
    #features, cams = extract_feat_cam_all(model, layer, batch_size, data)
    d_wp, d_mp = weighted_cam_pooling(features, cams)
    #for img_ind in range(0, counter):
    #    save_data(d_wp[img_ind * nb_classes:(img_ind + 1) * nb_classes], path_descriptors, image_names[img_ind] + '.h5')
    desc_wp = np.concatenate((desc_wp, d_wp))
    desc_mp = np.concatenate((desc_mp, d_mp))
    #print desc_wp.shape
    print 'Batch processed, CAMs descriptors obtained!'
    print 'Total time elapsed: ', time.time() - t0
    sys.stdout.flush()

    return desc_wp, desc_mp


########################################################################################################################
# Main Script

print 'Num classes: ', num_classes
print 'Mean: ', mean_value

t_0 = time.time()
desc_wp = np.zeros((0, dim_descriptor), dtype=np.float32)
desc_mp = np.zeros((0, dim_descriptor), dtype=np.float32)

# Horizontal Images
#desc_wp, desc_mp = \
#    extract_cam_descriptors(model_name, batch_size, num_classes, size_h, mean_value, train_list_path_h, desc_wp, desc_mp)
#save_data(desc_wp, descriptors_cams_path_wp, '_h')
#save_data(desc_mp, descriptors_cams_path_mp, '_h')
print desc_wp.shape
# Vertical Images
desc_wp, desc_mp = \
    extract_cam_descriptors(model_name, batch_size, num_classes, size_v, mean_value, train_list_path_v, desc_wp, desc_mp)

print desc_wp.shape

# Queries
if dataset == 'Oxford':
    i = 0
    with open('/imatge/ajimenez/work/ITR/oxford/lists/listcropped.txt', "r") as f:
        for line in f:
            print i
            line = line.replace('\n', '')
            img = np.array(
                (imread('/imatge/ajimenez/work/datasets_retrieval/Oxford/1_images/' + line + '.jpg')),
                dtype=np.float32)

            #img = np.transpose(img, (2, 0, 1))
            print img.shape
            if img.shape[0] > img.shape[1]:
                size = size_v
            else:
                size = size_h

            img_p = preprocess_query(img, size[0], size[1], mean_value)

            x = np.zeros((1, img_p.shape[0], img_p.shape[1], img_p.shape[2]), dtype=np.float32)
            model = VGGCAM(nb_classes, (img_p.shape[0], img_p.shape[1], img_p.shape[2]))
            model.load_weights(VGGCAM_weight_path)
            print 'Model loaded --> Extracting'
            x[0, :, :, :] = img_p
            features, cams, cl = extract_feat_cam(model, layer, batch_size, x, num_classes)
            #features, cams = extract_feat_cam_all(model, layer, batch_size, x)
            d_wp, d_mp = weighted_cam_pooling(features, cams)
            #print 'Saved '+ line +'.h5'
            #save_data(d_wp, path_descriptors, line+'.h5')
            desc_wp = np.concatenate((desc_wp, d_wp))
            desc_mp = np.concatenate((desc_mp, d_mp))
            #print desc_wp.shape
            i += 1

elif dataset == 'Paris':
    i = 0
    with open('/imatge/ajimenez/work/ITR/paris/lists/listcropped.txt', "r") as f:
        for line in f:
            print i
            line = line.replace('\n', '')
            img = np.array(
                (imread('/imatge/ajimenez/work/datasets_retrieval/Paris/imatges_paris/' + line + '.jpg')),
                dtype=np.float32)

            # img = np.transpose(img, (2, 0, 1))
            print img.shape
            if img.shape[0] > img.shape[1]:
                size = size_v
            else:
                size = size_h

            img_p = preprocess_query(img, size[0], size[1], mean_value)

            x = np.zeros((1, img_p.shape[0], img_p.shape[1], img_p.shape[2]), dtype=np.float32)
            model = VGGCAM(nb_classes, (img_p.shape[0], img_p.shape[1], img_p.shape[2]))
            model.load_weights(VGGCAM_weight_path)
            print 'Model loaded --> Extracting'
            x[0, :, :, :] = img_p
            features, cams, cl = extract_feat_cam(model, layer, batch_size, x, num_classes)
            #features, cams = extract_feat_cam_all(model, layer, batch_size, x)
            d_wp, d_mp = weighted_cam_pooling(features, cams)
            #print 'Saved ' + line + '.h5'
            #save_data(d_wp, path_descriptors, line + '.h5')
            desc_wp = np.concatenate((desc_wp, d_wp))
            desc_mp = np.concatenate((desc_mp, d_mp))
            #print desc_wp.shape
            i += 1

print 'Saving Data...'
#print desc_wp.shape
save_data(desc_wp, descriptors_cams_path_wp, '')
save_data(desc_mp, descriptors_cams_path_mp, '')
print 'Data Saved'
print 'Total time elapsed: ', time.time() - t_0