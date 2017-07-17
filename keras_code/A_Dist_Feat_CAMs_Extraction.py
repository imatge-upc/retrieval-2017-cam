from scipy.misc import imread
import time
import sys
import h5py
import numpy as np
from cam_utils import extract_feat_cam, extract_feat_cam_all
from vgg_cam import vggcam
from utils import create_folders, save_data, preprocess_images
from pooling_functions import weighted_cam_pooling


# Dataset Selection
dataset = 'distractors100k'

# Extract Offline
aggregation_type = 'Offline'

# Image Pre-processing (Size W x H)

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
    VGGCAM_weight_path = '../models/vgg_cam_weights.h5'
    layer = 'relu5_1'
    dim_descriptor = 512
    model = vggcam(nb_classes)
    model.load_weights(VGGCAM_weight_path)
    model.summary()

# CAM Extraction

if aggregation_type == 'Offline':
    num_classes = 64
elif aggregation_type == 'Online':
    num_classes = 1000

# Images to load into the net (+ images, + memory, + fast)
batch_size = 6
# Images to pre-load (+ images, + memory, + fast) (also saves feats & CAMs for this number when saving-CAMs)
image_batch_size = 1000
# Dimension of h5 files (+ images, + memory)
descriptors_batch_size = 10000
chunk_index = 0

# For saving also features & CAMs
saving_CAMs = False
ind = 0

if dataset == 'distractors100k':
    n_img_dataset = 100070
    train_list_path_h = "../lists/list_oxford105k_horizontal.txt"
    train_list_path_v = "../lists/list_oxford105k_vertical.txt"
    path_descriptors = '/data/jim011/distractors100k/descriptors/' + model_name + '/' + layer + '/' + dim + '/'
    descriptors_cams_path_wp = path_descriptors + 'distractor_all_' + str(num_classes) + '_wp'
    descriptors_cams_path_mp = path_descriptors + 'distractor_all_' + str(num_classes) + '_mp'
    create_folders(path_descriptors)

    # If you want to save features & CAMs
    # feature_path = '/imatge/ajimenez/work/ITR/distractors100k/features/' + model_name + '/' + layer + '/' + dim + '/'
    # cam_path = '/imatge/ajimenez/work/ITR/distractors100k/cams/' + model_name + '/' + layer + '/' + dim + '/'
    # create_folders(feature_path)
    # create_folders(cam_path)


def extract_cam_descriptors(model, batch_size, num_classes, size, mean_value, image_train_list_path, desc_wp, chunk_index, ind=0):
    images = [0] * image_batch_size
    image_names = [0] * image_batch_size
    counter = 0
    desc_count = 0
    num_images = 0
    t0 = time.time()

    print 'Horizontal size: ', size[0]
    print 'Vertical size: ', size[1]

    for line in open(image_train_list_path):
        if counter >= image_batch_size:
            print 'Processing image batch: ', ind
            t1 = time.time()
            data = preprocess_images(images, size[0], size[1], mean_value)

            if aggregation_type == 'Offline':
                features, cams, cl = \
                    extract_feat_cam(model, layer, batch_size, data, num_classes)
                if saving_CAMs:
                    save_data(cams, cam_path, 'cams_' + str(ind) + '.h5')
                    save_data(features, feature_path, 'features_' + str(ind) + '.h5')
                d_wp = weighted_cam_pooling(features, cams)
                desc_wp = np.concatenate((desc_wp, d_wp))

            print 'Image batch processed, CAMs descriptors obtained!'
            print 'Time elapsed: ', time.time()-t1
            sys.stdout.flush()
            counter = 0
            desc_count += image_batch_size
            if descriptors_batch_size == desc_count and aggregation_type == 'Offline':
                print 'Saving ...' + descriptors_cams_path_wp + '_' + str(chunk_index)+'.h5'
                save_data(desc_wp, descriptors_cams_path_wp + '_' + str(chunk_index)+'.h5', '')
                desc_count = 0
                chunk_index += 1
                desc_wp = np.zeros((0, dim_descriptor), dtype=np.float32)
            ind += 1

        line = line.rstrip('\n')
        images[counter] = imread(line)
        image_names[counter] = line
        counter += 1
        num_images += 1

    #Last batch
    print 'Last Batch:'
    data = np.zeros((counter, 3, size[1], size[0]), dtype=np.float32)
    data[0:] = preprocess_images(images[0:counter], size[0], size[1], mean_value)

    if aggregation_type == 'Offline':
        features, cams, cl = extract_feat_cam(model, layer, batch_size, data, num_classes)
        if saving_CAMs:
            save_data(cams, cam_path, 'cams_' + str(ind) + '.h5')
            save_data(features, feature_path, 'features_' + str(ind) + '.h5')
        d_wp = weighted_cam_pooling(features, cams)
        desc_wp = np.concatenate((desc_wp, d_wp))
        save_data(desc_wp, descriptors_cams_path_wp + '_' + str(chunk_index) + '.h5', '')
        chunk_index += 1
        desc_wp = np.zeros((0, dim_descriptor), dtype=np.float32)

    ind += 1
    print desc_wp.shape
    print 'Batch processed, CAMs descriptors obtained!'
    print 'Total time elapsed: ', time.time() - t0
    sys.stdout.flush()

    return desc_wp, chunk_index


########################################################################################################################
# Main Script

print 'Num classes: ', num_classes
print 'Mean: ', mean_value

t_0 = time.time()
desc_wp = np.zeros((0, dim_descriptor), dtype=np.float32)

# Horizontal Images
desc_wp, c_ind = \
    extract_cam_descriptors(model, batch_size, num_classes, size_h, mean_value, train_list_path_h, desc_wp, chunk_index)

# Vertical Images
desc_wp, c_ind = \
    extract_cam_descriptors(model, batch_size, num_classes, size_v, mean_value, train_list_path_v, desc_wp, c_ind, ind)


print 'Data Saved'
print 'Total time elapsed: ', time.time() - t_0
