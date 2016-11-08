import numpy as np
import h5py
import sys
import utils_oxford
from scipy.misc import imread, imresize, imsave


image_train_list_path_oxford = "/imatge/ajimenez/workspace/ITR/lists/list_oxford.txt"
image_train_list_path_paris = ""

bound_queries = True

img_width = 480
img_height = 480

file_path = '/imatge/ajimenez/work/datasets_hdf5/oxford/'
#file_path = '/imatge/ajimenez/work/datasets_hdf5/paris/'


name_file = 'oxford_' + str(img_width)
if bound_queries:
    name_file += '_bb'
name_file += '.h5'


#f = h5py.File('../models/mean_places.h5','r')
#mean_value = f.get('/data')
#mean_value = np.array(mean_value)
#f.close()

print mean_value.shape
sys.stdout.flush()
# mean_value = [123.68, 116.779, 103.939] #Imagenet
mean_value = [104, 166.66, 122.67] #Places
#mean_value = [0,0,0]
# Load Data
images, num_images, image_names = utils_oxford.load_data_oxford(image_train_list_path)

image_names = np.array(image_names)

# Pre-process images

x = utils_oxford.preprocess_images(images, num_images, img_width, img_height, mean_value)


if bound_queries:
    q, q_names = utils_oxford.preprocess_queries(img_width, img_height, mean_value)

    print q.shape

    for ind, name in enumerate(q_names):
        for i in range(0, image_names.shape[0]):
            if name == image_names[i]:
                print 'coincidence'
                x[i] = q[ind]
                break


print num_images
print x.shape
print image_names
sys.stdout.flush()

#labels = utils_oxford.label_oxford(image_names)

#print labels.shape

print x[0, 0, 0:10, 0:10]

utils_oxford.save_oxford(file_path, name_file, x, image_names)


