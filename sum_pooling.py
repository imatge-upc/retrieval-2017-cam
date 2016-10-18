import cam_utils as cu
import numpy as np
import time
from sklearn.decomposition import PCA
import sys

cam_path = '/imatge/ajimenez/work/results_ITR/cam_masks/googlenet_places/oxford_bb_c_f_s.h5'
results_path = '/imatge/ajimenez/work/results_ITR/results/'
name_result = 'oxford_bb_sum_pooling.h5'

num_classes = 5
# Load features and masks
masks, features, scores = cu.load_cams(cam_path, num_classes)

num_samples = features.shape[0]
num_features = features.shape[1]

image_representations = np.zeros((num_samples, num_features), dtype=np.float32)
descriptor = np.zeros((num_samples, num_features), dtype=np.float32)

max_pooling = False

print 'Num samples = ', num_samples
print 'Num feature maps = ', num_features
print 'Num regions = ', num_classes
sys.stdout.flush()

t = time.time()
for i in range(0, num_samples):
    print 'Image: ', i
    sys.stdout.flush()
    for f in range(0, num_features):
        if max_pooling:
            descriptor[i, f] = np.amax(features[i, f])
        else:
            descriptor[i, f] = features[i, f].sum()


descriptor /= np.linalg.norm(descriptor, axis=1)[:, None]

print 'Time elapsed computing sum_pooling: ', time.time() - t
print descriptor.shape
sys.stdout.flush()

cu.save_data(descriptor, results_path, name_result)

print 'Total time elapsed: ', time.time()-t




