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


def save_cams(cams, features, scores, cams_name):
    with h5py.File(cams_name, 'w') as hf:
        hf.create_dataset('data', data=cams)
        hf.create_dataset('features', data=features)
        hf.create_dataset('scores', data=scores)
    print 'Data saved!'


def load_query(name, query_name):
    with h5py.File(filepath, 'r') as hf:
        data = np.array(hf.get('data'))
        print 'Shape of the array features: ', data.shape
        return data

def load_cams(path, nclass, load_all=''):
    with h5py.File(path, 'r') as hf:
        cams = np.array(hf.get('data'))[:, 0:nclass, :, :]
        if load_all == 'cams':
            print 'Shape of the array cams: ', cams.shape
            return cams
        features = np.array(hf.get('cams'))
        scores = np.array(hf.get('scores'))[:, 0:nclass]

        print 'Shape of the array cams: ', cams.shape
        print 'Shape of the array features: ', features.shape
        print 'Shape of the array scores: ', scores.shape

    return cams, features, scores


def extract_feat_and_cam_masks(model, batch_size, images, top_nclass, cam_path):
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
    features_conv = np.zeros((num_samples, 1024, fm_h, fm_w))
    all_scores = np.zeros((num_samples, classes_places))
    # Function to get scores, conv_maps
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    # Extract weights from FC
    weights_fc = model.layers[-2].get_weights()[0]

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

        [conv_outputs, scores] = get_output([x])
        features_conv[i*batch_size:i*batch_size+conv_outputs.shape[0], :, :, :] = conv_outputs

        print ('Time elapsed to forward the batch: ', time.time()-t0)

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

        print 'Time elapsed to compute the batch: ', time.time()-t0

    save_cams(cams, features_conv, all_scores, cam_path)

    print 'Total time elapsed: ', time.time() - tt


def visualize_cam(model, batch_size, images, top_nclass, output_path_heatmaps, image_names):
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

    x = np.zeros((batch_size, 3, width, height))

    print 'Num of total samples: ',num_samples
    print 'Batch size: ', batch_size
    sys.stdout.flush()
    num_it = int(math.floor(num_samples/batch_size))
    last_batch = num_samples % batch_size

    weights_fc = model.layers[-2].get_weights()[0]
    # Set convolutional layer to extract the CAM
    final_conv_layer = get_output_layer(model, "CAM_relu")
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])


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
        [conv_outputs, scores] = get_output([x])

        t = time.time() - t0
        print 'Time elapsed to forward the batch: ', t
        for ii in range(0,batch_size):
            print 'Image number: ', ii
            indexed_scores = scores[ii].argsort()[::-1]
            for k in range(0,top_nclass):
                w_class = weights_fc[:, indexed_scores[k]]
                cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[2:4])
                for ind, w in enumerate(w_class):
                    cam += w * conv_outputs[ii,ind, :, :]

                cam /= np.max(cam)

                print cam.shape
                cam_bw = cam
                cam_bw[np.where(cam < 0)] = 0
                cv2.imwrite(output_path_heatmaps + image_names[i * batch_size + ii] + '_bw_' + str(k) + '.jpg',cam_bw*255)
                cam = cv2.resize(cam, (width, height))
                print cam.shape
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap[np.where(cam < 0.1)] = 0
                img = heatmap * 0.5 + 0.7 * np.transpose(images[i*batch_size+ii], (1,2,0))
                cv2.imwrite(output_path_heatmaps+image_names[i*batch_size+ii]+'_'+str(k)+'.jpg', img)
            print 'Time elapsed to compute the batch: ', time.time()-t0
    print 'Total time elapsed: ', time.time()-tt