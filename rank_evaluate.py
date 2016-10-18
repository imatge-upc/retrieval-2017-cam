from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import h5py
import sys
import utils_oxford as utils
import cam_utils as cu
import time

workpath = "/imatge/ajimenez/workspace/ITR/results/"
filename = 'oxford.h5'
save_path = '/imatge/ajimenez/workspace/ITR/results/ranked_oxford_RMAC_PCA/'

#data = utils.load_features(workpath+filename)

name_file = '/imatge/ajimenez/work/results_ITR/descriptors/oxford_4_th_0_weigth_mp.h5'


data = cu.load_data(name_file)

print data[0]

print 'Data shape: ', data.shape
sys.stdout.flush()

t = time.time()
distances, indices = utils.compute_distances(data, 5063)
print 'Time elapsed computing distances: ', time.time()-t


#print(indices)
#print(distances)
#print(indices.shape)
#print(distances.shape)

image, image_name = utils.read_data_oxford('/imatge/ajimenez/workspace/ITR/datasets_hfd5/oxford_128/data4torch.h5')


utils.save_rankings(indices, image_name, save_path)

utils.evaluate_oxford(ranking_path=save_path)

print 'Evaluated' + name_file

