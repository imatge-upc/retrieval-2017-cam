from keras.models import *
from keras.callbacks import *
import keras.backend as K
import h5py
import sys
import math
import time
import numpy as np

def save_chunk(name, images):
    with h5py.File(name+'.h5', 'w') as hf:
        hf.create_dataset('data', data=images)


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def choose_model(model_name, position):
    # Load Model
    t = time.time()
    if model_name == 'googlenet':
        if position == 'h':
            print 'Loading model Googlenet Horizontal'

            model = model_from_json(open('/imatge/ajimenez/work/ITR/models/inception_cam_model_structure_h.json').read())

            model.load_weights('/imatge/ajimenez/work/ITR/models/inception_cam_weights.h5')

        elif position == 'v':
            print 'Loading model Googlenet Vertical'

            model = model_from_json(
                open('/imatge/ajimenez/work/ITR/models/inception_cam_model_structure_v.json').read())

            model.load_weights('/imatge/ajimenez/work/ITR/models/inception_cam_weights.h5')

    elif model_name == 'vgg_16_imagenet':
        print 'Loading model VGG-16'
        model = model_from_json(open('/imatge/ajimenez/work/ITR/models/vgg_16_imagenet.json').read())

        model.load_weights('/imatge/ajimenez/work/ITR/models/vgg16_weights.h5')

    model.summary()
    print 'Time elapsed to load the model: ', time.time()-t
    return model


def extract_features(model, layer, batch_size, images, save_name):
    num_samples = images.shape[0]
    width = images.shape[3]
    height = images.shape[2]
    #VGG 224
    fm_w = width / 16
    fm_h = height / 16

    tt = time.time()

    x = np.zeros((batch_size, 3, width, height), dtype=np.float32)

    print 'Num of total samples: ', num_samples
    print 'Batch size: ', batch_size
    sys.stdout.flush()

    num_it = int(math.floor(num_samples / batch_size))
    last_batch = num_samples % batch_size

    conv_layer = get_output_layer(model, layer)
    get_output = K.function([model.layers[0].input], [conv_layer.output])

    features_conv = np.zeros((num_samples, 512, fm_h, fm_w))

    for i in range(0,num_it+1):
        if i == num_it:
            if last_batch != 0:
                x = images[i*batch_size:batch_size*i+last_batch, :, :, :]
            else:
                break
        else:
            x = images[i*batch_size:batch_size*(i+1), :, :, :]

        print 'Batch number: ', i
        print x.shape
        sys.stdout.flush()
        conv_outputs = get_output([x])
        print conv_outputs[0].shape
        #print conv_outputs
        sys.stdout.flush()
        features_conv[i * batch_size:i * batch_size + conv_outputs[0].shape[0], :, :, :] = conv_outputs[0]
    print features_conv.shape
    sys.stdout.flush()
    save_chunk(save_name, features_conv)
