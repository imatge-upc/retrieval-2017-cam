from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import h5py
import sys
import utils_datasets as utils
import evaluate_oxford_paris as eval
import cam_utils as cu
import time

workpath = "/imatge/ajimenez/workspace/ITR/results/"
filename = 'oxford.h5'
save_path = '/imatge/ajimenez/workspace/ITR/results/ranked_oxford_RMAC_PCA/'


def save_rankings(indices, image_names, path):
    for i in range(0, image_names.shape[0]):
        file = open(path + image_names[i] + '.txt', 'w')
        for ind in indices[i]:
            file.write(image_names[ind] + '\n')
        file.close()

#data = utils.load_features(workpath+filename)

name_file = '/imatge/ajimenez/work/ITR/results_ITR/descriptors/vgg_imagenet/oxford_1024_720_sum_pooling.h5'

data = utils.load_data(name_file)

print data[3249]

print 'Data shape: ', data.shape
sys.stdout.flush()

t = time.time()
distances, indices = eval.compute_distances(data, 5063)
print 'Time elapsed computing distances: ', time.time()-t


#print(indices)
#print(distances)
#print(indices.shape)
#print(distances.shape)

image_name = utils.read_dataset('/imatge/ajimenez/workspace/ITR/datasets_hdf5/oxford_128/data4torch.h5','labels')

#save_rankings(indices, image_name, save_path)

#eval.evaluate_oxford(save_path, 'aa.txt')

print 'Evaluated' + name_file