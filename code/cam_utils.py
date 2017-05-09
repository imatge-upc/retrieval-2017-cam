from keras.models import *
from keras.callbacks import *
import keras.backend as K
import cv2
import os
import sys
import h5py
import numpy as np
import time
import math
from scipy.misc import imread, imresize, imsave

classes_places = 205
classes_imagenet = 1000


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


# Extract region of interest from CAMs
def extract_ROI(heatmap, threshold):
    th = threshold * np.max(heatmap)
    heatmap = heatmap > th
    # Find the largest connected component

    contours, hierarchy = cv2.findContours(heatmap.astype('uint8'), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(ctr) for ctr in contours]

    max_contour = contours[areas.index(max(areas))]

    x, y, w, h = cv2.boundingRect(max_contour)
    if w == 0:
        w = heatmap.shape[1]
    if h == 0:
        h = heatmap.shape[0]
    return x, y, w, h


# Visualization Purposes, Draw bounding box around object of interest
def draw_bounding_box(img, full_heatmap, label, color=(0, 0, 255), threshold=0.3):
    # Apply the thresholding
    full_heatmap = cv2.resize(full_heatmap, (img.shape[1], img.shape[0]))  # , interpolation=cv2.INTER_NEAREST)
    th = threshold * np.max(full_heatmap)
    full_heatmap = full_heatmap > th
    # Find the largest connected component
    ima2, contours, hierarchy = cv2.findContours(full_heatmap.astype('uint8'), mode=cv2.RETR_EXTERNAL,
                                                  method=cv2.CHAIN_APPROX_SIMPLE)

    cv2.imwrite('contours_' + str(threshold) + '.jpg', ima2)
    areas = [cv2.contourArea(ctr) for ctr in contours]
    max_contour = contours[areas.index(max(areas))]

    x, y, w, h = cv2.boundingRect(max_contour)
    # Draw bounding box and label
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    #cv2.putText(img, label[:], (x + 3, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.01, color, 2)
    cv2.imwrite('bounded_' + str(threshold) + '.jpg', img)
    return x, y, w, h


# Method for Online aggregation
def extract_feat_cam_all(model, layer, batch_size, images, top_nclass=1000):
    '''
    Extract CAM masks for all classes, for each image in the dataset. Also extract  features
    from layer
    :param model: The network
    :param batch_size: batch_size
    :param images: images in format [num_total,3,height, width]
    :return:
    '''

    num_samples = images.shape[0]

    print 'Num of total samples: ', num_samples
    print 'Batch size: ', batch_size
    sys.stdout.flush()

    num_it = int(math.floor(num_samples / batch_size))
    last_batch = num_samples % batch_size
    batch_size_loop = batch_size

    # Set convolutional layer to extract the CAMs (CAM_relu layer)
    final_conv_layer = get_output_layer(model, "CAM_relu")

    # Set layer to extract the features
    conv_layer_features = get_output_layer(model, layer)
    f_shape = conv_layer_features.output_shape

    # Initialize Arrays
    features_conv = np.zeros((num_samples, f_shape[1], images.shape[2]%16, images.shape[3]%16))
    cams = np.zeros((images.shape[0], top_nclass, images.shape[2]%16, images.shape[3]%16), dtype=np.float32)

    # Function to get conv_maps
    get_output = K.function([model.layers[0].input, K.learning_phase()],
                            [final_conv_layer.output, conv_layer_features.output])
    # Extract weights from Dense
    weights_fc = model.layers[-1].get_weights()[0]

    for i in range(0, num_it+1):
        t0 = time.time()
        if i == num_it:
            if last_batch != 0:
                x = images[i*batch_size:batch_size*i+last_batch, :, :, :]
                batch_size_loop = last_batch
            else:
                break
        else:
            x = images[i*batch_size:batch_size*(i+1), :, :, :]

        print 'Batch number: ', i

        [conv_outputs, features] = get_output([x, 0])
        features_conv[i*batch_size:i*batch_size+features.shape[0], :, :, :] = features

        print ('Time elapsed to forward the batch: ', time.time()-t0)

        for ii in range(0, batch_size_loop):
            for k in range(0, top_nclass):
                w_class = weights_fc[:, k]
                cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[2:4])
                for ind, w in enumerate(w_class):
                    cam += w * conv_outputs[ii, ind, :, :]
                cam /= np.max(cam)
                cam[np.where(cam < 0)] = 0

                cams[i*batch_size+ii, k, :, :] = cam

        print 'Time elapsed to compute CAMs: ', time.time()-t0

    return features_conv, cams


def extract_feat_cam(model, layer, batch_size, images, top_nclass, specify_class=None, roi=False):
    '''
    :param model: Network
    :param layer: Layer to extract features
    :param batch_size: Batch size
    :param images: data [n_samples,3,H,W]
    :param top_nclass: number of CAMs to extract (Top predicted N classes)
    :param specify_class: (If we know the classes) --> Class Array
    :param roi: Region of Interest given list of classes
    :return: features, cams, class_list , roi
    '''

    # width, height of conv5_1 layer
    # 14x14 for 224x224 input image
    # H/16 x W/16 for H x W input image with VGG-16

    num_samples = images.shape[0]
    print images.shape

    class_list = np.zeros((num_samples, top_nclass), dtype=np.int32)
    print 'Num of total samples: ', num_samples
    print 'Batch size: ', batch_size
    sys.stdout.flush()

    num_it = int(math.floor(num_samples / batch_size))
    last_batch = num_samples % batch_size
    batch_size_loop = batch_size

    # Set convolutional layer to extract the CAMs (CAM_relu layer)
    final_conv_layer = get_output_layer(model, "CAM_relu")

    # Set layer to extract the features
    conv_layer_features = get_output_layer(model, layer)
    f_shape = conv_layer_features.output_shape

    # Initialize Arrays
    features_conv = np.zeros((num_samples, f_shape[1], images.shape[2] // 16, images.shape[3] // 16))
    cams = np.zeros((images.shape[0], top_nclass, images.shape[2] // 16, images.shape[3] // 16), dtype=np.float32)
    all_scores = np.zeros((num_samples, classes_imagenet))

    # Function to get scores, conv_maps
    get_output = K.function([model.layers[0].input, K.learning_phase()],
                            [final_conv_layer.output, model.layers[-1].output, conv_layer_features.output])
    # Extract weights from Dense
    weights_fc = model.layers[-1].get_weights()[0]

    # Region of interest for re-ranking (bounding box coordinates --> (num samples, num_thresholds, x,y,dx,dy)
    if roi:
        bbox_coord = np.zeros((num_samples, 5, 4), dtype=np.int16)

    for i in range(0, num_it+1):
        t0 = time.time()
        if i == num_it:
            if last_batch != 0:
                x = images[i*batch_size:batch_size*i+last_batch, :, :, :]
                batch_size_loop = last_batch
            else:
                break
        else:
            x = images[i*batch_size:batch_size*(i+1), :, :, :]

        #print 'Batch number: ', i

        [conv_outputs, scores, features] = get_output([x, 0])
        features_conv[i*batch_size:i*batch_size+features.shape[0], :, :, :] = features

        #print ('Time elapsed to forward the batch: ', time.time()-t0)

        if specify_class is None:

            for ii in range(0, batch_size_loop):
                #print 'Image number: ', ii
                indexed_scores = scores[ii].argsort()[::-1]
                for k in range(0, top_nclass):
                    w_class = weights_fc[:, indexed_scores[k]]
                    all_scores[i * batch_size + ii, k] = scores[ii, indexed_scores[k]]
                    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[2:4])
                    for ind, w in enumerate(w_class):
                        cam += w * conv_outputs[ii, ind, :, :]
                    cam /= np.max(cam)
                    cam[np.where(cam < 0)] = 0

                    cams[i*batch_size+ii, k, :, :] = cam

                    class_list[i*batch_size+ii, k] = indexed_scores[k]

        else:
            for ii in range(0, batch_size_loop):
                # print 'Image number: ', ii
                for k in range(0, top_nclass):
                    w_class = weights_fc[:, specify_class[k]]
                    all_scores[i * batch_size + ii, k] = scores[ii, specify_class[k]]
                    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[2:4])
                    for ind, w in enumerate(w_class):
                        cam += w * conv_outputs[ii, ind, :, :]
                    cam /= np.max(cam)
                    cam[np.where(cam < 0)] = 0

                    cams[i * batch_size + ii, k, :, :] = cam

                # How to compute the ROI of the image, in the paper results we average 2 most probable classes
                if roi:
                    average = True

                    if average:
                        average_cam = np.zeros((cams.shape[2], cams.shape[3]))
                        for cam in cams[i*batch_size+ii, 0:2]:
                            average_cam += cam
                        heatmap = average_cam / 2
                    else:
                        heatmap = cams[i*batch_size+ii, 0]

                    bbox_coord[i*batch_size+ii, 0, :] = extract_ROI(heatmap=heatmap, threshold=0.01)# Full Image
                    bbox_coord[i*batch_size+ii, 1, :] = extract_ROI(heatmap=heatmap, threshold=0.1)
                    bbox_coord[i*batch_size+ii, 2, :] = extract_ROI(heatmap=heatmap, threshold=0.2)
                    bbox_coord[i*batch_size+ii, 3, :] = extract_ROI(heatmap=heatmap, threshold=0.3)
                    bbox_coord[i*batch_size+ii, 4, :] = extract_ROI(heatmap=heatmap, threshold=0.4)

        print 'Time elapsed to compute CAMs & Features: ', time.time()-t0
        sys.stdout.flush()
    if specify_class is None:
        return features_conv, cams, class_list

    else:
        return features_conv, cams, bbox_coord



