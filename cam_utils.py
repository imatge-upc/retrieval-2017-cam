from keras.models import *
from keras.callbacks import *
import keras.backend as K
import cv2
import os
import sys
import h5py
import utils_oxford as uf
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


def filter_CAMs(cams_prediction, good_cams, n_final_cams=24):
    order = cams_prediction[good_cams][0:n_final_cams].argsort()[::-1]
    print order
    selected_cams = good_cams[order]
    print selected_cams
    return selected_cams[0:n_final_cams]


# Extract region of interest from CAMs
def extract_ROI(heatmap, threshold):
    th = threshold * np.max(heatmap)
    heatmap = heatmap > th
    # Find the largest connected component
    ima2, contours, hierarchy = cv2.findContours(heatmap.astype('uint8'), mode=cv2.RETR_EXTERNAL,
                                                 method=cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(ctr) for ctr in contours]
    max_contour = contours[areas.index(max(areas))]
    x, y, w, h = cv2.boundingRect(max_contour)
    if w == 0:
        w = heatmap.shape[1]
    if h == 0:
        h = heatmap.shape[0]
    return x, y, w, h


def draw_bounding_box(img, heatmap, label, k, color=(0, 0, 255), threshold=0.8):
    # Resize heatmap to the minimum dimension and fill the rest with zeros
    short_edge = min(img.shape[:2])  # shape: (height, width) for both numpy and cv2  || img.shape[1::-1]
    resized_heatmap = cv2.resize(heatmap, (short_edge, short_edge))  # , interpolation=cv2.INTER_NEAREST)
    full_heatmap = np.zeros(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    full_heatmap[yy:yy + short_edge, xx:xx + short_edge] = resized_heatmap
    # Apply the thresholding
    full_heatmap = full_heatmap > threshold
    # Find the largest connected component
    ima2, contours, hierarchy = cv2.findContours(heatmap.astype('uint8'), mode=cv2.RETR_EXTERNAL,
                                                 method=cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(ctr) for ctr in contours]
    max_contour = contours[areas.index(max(areas))]
    x, y, w, h = cv2.boundingRect(max_contour)
    # Draw bounding box and label
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label[:-1], (x + 3, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite('/imatge/ajimenez/workspace/ITR/cam_heatmaps/uu' + str(k) + '.jpg', img)
    cv2.imwrite('/imatge/ajimenez/workspace/ITR/cam_heatmaps/uu' + 'contour' + str(k) + '.jpg', ima2)


def visualize_cam(model, batch_size, images, top_nclass, output_path_heatmaps, image_names, specify_class='a', filter=False, gc=''):
    '''
    Extract and save CAM images for the top N classes, for each image in the dataset
    :param model: The network
    :param batch_size: batch_size
    :param images: images in format [num_total,3,width,height]
    :param image_names: name of the images [num_total]
    :return:
    '''
    tt = time.time()

    num_samples = images.shape[0]
    height = images.shape[2]
    width = images.shape[3]

    fm_w = width / 16
    fm_h = height / 16
    x = np.zeros((batch_size, 3, width, height))

    class_list = np.zeros((num_samples, top_nclass),dtype=np.int32)

    cams = np.zeros((images.shape[0], top_nclass, fm_h, fm_w), dtype=np.float32)
    print 'Num of total samples: ',num_samples
    print 'Batch size: ', batch_size
    sys.stdout.flush()
    num_it = int(math.floor(num_samples/batch_size))
    last_batch = num_samples % batch_size

    weights_fc = model.layers[-1].get_weights()[0]
    # Set convolutional layer to extract the CAM
    final_conv_layer = get_output_layer(model, "CAM_relu")
    get_output = K.function([model.layers[0].input, K.learning_phase()], [final_conv_layer.output, model.layers[-1].output])

    for i in range(0, num_it+1):
        t0 = time.time()
        print('Batch number: ', i)

        if i == num_it:
            if last_batch != 0:
                x = images[i*batch_size:batch_size*i+last_batch, :, :, :]
            else:
                break
        else:
            x = images[i*batch_size:batch_size*(i+1), :, :, :]

        sys.stdout.flush()
        [conv_outputs, scores] = get_output([x,0])

        t = time.time() - t0
        print 'Time elapsed to forward the batch: ', t

        if specify_class == 'a':
            for ii in range(0,batch_size):
                print 'Image number: ', ii
                if filter:
                    indexed_scores = filter_CAMs(scores[ii], gc)
                else:
                    indexed_scores = scores[ii].argsort()[::-1]
                for k in range(0,top_nclass):
                    w_class = weights_fc[:, indexed_scores[k]]
                    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[2:4])
                    for ind, w in enumerate(w_class):
                        cam += w * conv_outputs[ii,ind, :, :]

                    class_list[ii, k] = indexed_scores[k]

                    cam /= np.max(cam)
                    cams[i * batch_size + ii, k, :, :] = cam
                    # draw_bounding_box(images[i * batch_size + ii], cam, 'Church', k)
                    print cam.shape
                    cam_bw = cam
                    cam_bw[np.where(cam < 0)] = 0
                    cv2.imwrite(output_path_heatmaps + image_names[i * batch_size + ii] + '_bw_' + str(k) + '.jpg',
                                cam_bw * 255)
                    cam = cv2.resize(cam, (width, height))

                    print cam.shape
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                    heatmap[np.where(cam < 0.1)] = 0
                    img = heatmap * 0.5 + 0.7 * np.transpose(images[i * batch_size + ii], (1, 2, 0))
                    cv2.imwrite(output_path_heatmaps + image_names[i * batch_size + ii] + '_' + str(k) + '.jpg', img)

        else:
            for ii in range(0, batch_size):
                for k in range(0, top_nclass):
                    w_class = weights_fc[:, specify_class[k]]
                    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[2:4])
                    for ind, w in enumerate(w_class):
                        cam += w * conv_outputs[ii, ind, :, :]
                    cam /= np.max(cam)
                    cam[np.where(cam < 0)] = 0

                    cams[i * batch_size + ii, k, :, :] = cam

                    #draw_bounding_box(images[i * batch_size + ii], cam, 'Church', k)
                    print cam.shape
                    cam_bw = cam
                    cam_bw[np.where(cam < 0)] = 0
                    cv2.imwrite(output_path_heatmaps + image_names[i * batch_size + ii] + '_bw_' + str(k) + '.jpg', cam_bw*255)
                    cam = cv2.resize(cam, (width, height))

                    print cam.shape
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                    heatmap[np.where(cam < 0.1)] = 0
                    img = heatmap * 0.5 + 0.7 * np.transpose(images[i*batch_size+ii], (1,2,0))
                    cv2.imwrite(output_path_heatmaps+image_names[i*batch_size+ii]+'_'+str(k)+'.jpg', img)

                    print 'Time elapsed to compute the batch: ', time.time()-t0

    print 'Total time elapsed: ', time.time()-tt
    if specify_class =='a':
        return class_list, cams
    else:
        return cams


def extract_feat_cam_all(model, layer, batch_size, images, top_nclass=1000):
    '''
    Extract and save CAM masks for all classes, for each image in the dataset. Also extract and save features
    from last conv layer
    :param model: The network
    :param batch_size: batch_size
    :param images: images in format [num_total,3,width,height]
    :return:
    '''

    num_samples = images.shape[0]
    width = images.shape[3]
    height = images.shape[2]

    fm_w = width/16
    fm_h = height/16

    x = np.zeros((batch_size, 3, height, width), dtype=np.float32)

    size_feats = 512

    class_list = np.zeros((num_samples, top_nclass))
    print 'Num of total samples: ', num_samples
    print 'Batch size: ', batch_size
    sys.stdout.flush()

    num_it = int(math.floor(num_samples / batch_size))
    last_batch = num_samples % batch_size
    batch_size_loop = batch_size

    # Set convolutional layer to extract the CAM
    final_conv_layer = get_output_layer(model, "CAM_relu")
    conv_layer_features = get_output_layer(model, layer)
    features_conv = np.zeros((num_samples, size_feats, fm_h, fm_w))

    # Function to get scores, conv_maps
    get_output = K.function([model.layers[0].input, K.learning_phase()], \
                            [final_conv_layer.output, model.layers[-1].output, conv_layer_features.output])
    # Extract weights from FC
    weights_fc = model.layers[-1].get_weights()[0]

    cams = np.zeros((images.shape[0], top_nclass, fm_h, fm_w), dtype=np.float32)

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

        [conv_outputs, scores, features] = get_output([x, 0])
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
    Extract and save CAM masks for the top N classes, for each image in the dataset. Also extract and save features
    from last conv layer
    :param model: The network
    :param batch_size: batch_size
    :param images: images in format [num_total,3,width,height]
    :return:
    '''
    num_samples = images.shape[0]
    width = images.shape[3]
    height = images.shape[2]

    fm_w = width/16
    fm_h = height/16

    tt = time.time()
    x = np.zeros((batch_size, 3, height, width), dtype=np.float32)

    size_feats = 512

    class_list = np.zeros((num_samples, top_nclass), dtype=np.int32)
    print 'Num of total samples: ', num_samples
    print 'Batch size: ', batch_size
    sys.stdout.flush()

    num_it = int(math.floor(num_samples / batch_size))
    last_batch = num_samples % batch_size
    batch_size_loop = batch_size

    # width, height of conv layer
    # 14x14 for 224x224 input image
    # 30x30 for 480x480 input image
    # 30x14 for 480x224

    # Set convolutional layer to extract the CAM
    final_conv_layer = get_output_layer(model, "CAM_relu")
    f_shape =  final_conv_layer.output_shape
    conv_layer_features = get_output_layer(model, layer)
    features_conv = np.zeros((num_samples, size_feats, fm_h, fm_w))
    #features_conv = np.zeros((num_samples, size_feats, f_shape[2], f_shape[3]))
    all_scores = np.zeros((num_samples, classes_places))
    # Function to get scores, conv_maps
    get_output = K.function([model.layers[0].input, K.learning_phase()], \
                            [final_conv_layer.output, model.layers[-1].output, conv_layer_features.output])
    # Extract weights from FC
    weights_fc = model.layers[-1].get_weights()[0]

    #cams = np.zeros((images.shape[0], top_nclass, fm_h, fm_w), dtype=np.float32)
    cams = np.zeros((images.shape[0], top_nclass, f_shape[2], f_shape[3]), dtype=np.float32)

    if roi:
        bbox_coord = np.zeros((num_samples, 5, 4),dtype=np.int16)

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

        [conv_outputs, scores, features] = get_output([x, 0])
        features_conv[i*batch_size:i*batch_size+features.shape[0], :, :, :] = features

        print ('Time elapsed to forward the batch: ', time.time()-t0)

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

                    class_list[ii, k] = indexed_scores[k]

        else:
            for ii in range(0, batch_size_loop):
                # print 'Image number: ', ii
                indexed_scores = scores[ii].argsort()[::-1]
                for k in range(0, top_nclass):
                    w_class = weights_fc[:, specify_class[k]]
                    all_scores[i * batch_size + ii, k] = scores[ii, specify_class[k]]
                    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[2:4])
                    for ind, w in enumerate(w_class):
                        cam += w * conv_outputs[ii, ind, :, :]
                    cam /= np.max(cam)
                    cam[np.where(cam < 0)] = 0

                    cams[i * batch_size + ii, k, :, :] = cam

                if roi:
                    average = True

                    if average:
                        average_cam = np.zeros((cams.shape[2], cams.shape[3]))
                        for cam in cams[ii, 0:2]:
                            average_cam += cam
                        heatmap = average_cam / 2
                    else:
                        heatmap = cams[ii, 0]

                    bbox_coord[ii, 0, :] = extract_ROI(heatmap=heatmap, threshold=0.01)
                    bbox_coord[ii, 1, :] = extract_ROI(heatmap=heatmap, threshold=0.1)
                    bbox_coord[ii, 2, :] = extract_ROI(heatmap=heatmap, threshold=0.2)
                    bbox_coord[ii, 3, :] = extract_ROI(heatmap=heatmap, threshold=0.3)
                    bbox_coord[ii, 4, :] = extract_ROI(heatmap=heatmap, threshold=0.4)

        print 'Time elapsed to compute the batch: ', time.time()-t0

    if specify_class is None:
        return features_conv, cams, class_list
    elif specify_class == 'No':
        return features_conv, cams
    else:
        print bbox_coord.shape
        return features_conv, cams, bbox_coord


