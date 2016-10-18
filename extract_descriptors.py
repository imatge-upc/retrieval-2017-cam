import cam_utils as cu
import numpy as np
import time
from sklearn.decomposition import PCA
import sys

cam_path = '/imatge/ajimenez/work/results_ITR/cam_masks/googlenet_places/oxford_cams_5.h5'
results_path = '/imatge/ajimenez/work/results_ITR/descriptors/'


#PARAMETERS

# THRESHOLD CAMS
thresh_cam = 0.1
# PCA
pca_on = False
pca_dim = 512
# WEIGHT
weight = True
# Perform also max pooling
max_pooling = True
# Load features and masks
num_classes = 4

name_result = 'oxford_' + str(num_classes) + '_th_' + str(thresh_cam)

if weight:
    name_result += '_weigth'

if max_pooling:
    mp_name_result = name_result + '_mp.h5'

wp_name_result = name_result + '_wp.h5'


print 'Beginning Descriptor Extraction: '
print 'Weighted sum pooling: ' + wp_name_result
if max_pooling:
    print 'Max pooling: ' + mp_name_result
sys.stdout.flush()

# Beginning

masks, features, scores = cu.load_cams(cam_path, num_classes)

num_samples = features.shape[0]
num_features = features.shape[1]

wp_image_representations = np.zeros((num_samples, num_features), dtype=np.float32)
wp_regions = np.zeros((num_features, num_classes), dtype=np.float32)

if max_pooling:
    mp_regions = np.zeros((num_features, num_classes), dtype=np.float32)
    mp_image_representations = np.zeros((num_samples, num_features), dtype=np.float32)


if weight:
    # All scores sum 1
    if num_classes < 205:
        for i in range(num_samples):
            scores[i] *= 1/sum(scores[i])

if thresh_cam > 0:
    for i in range(0,num_samples):
        for k in range(0,num_classes):
            masks[i][k][np.where(masks[i][k] < thresh_cam)] = 0



if pca_on:
    all_regions_images = np.zeros((num_samples*num_classes, num_features), dtype=np.float32)
    img_pca = np.zeros((num_samples, pca_dim), dtype=np.float32)

print 'Num samples = ', num_samples
print 'Num feature maps = ', num_features
print 'Num regions = ', num_classes
sys.stdout.flush()

t = time.time()
for i in range(0, num_samples):
    print 'Image: ', i
    sys.stdout.flush()
    for f in range(0, num_features):
        for k in range(0, num_classes):
            # For each region compute avg weighted sum of activations and l2 normalize
            if max_pooling:
                mp_regions[f, k] = np.amax(np.multiply(features[i, f], masks[i, k]))

            wp_regions[f, k] = np.multiply(features[i, f], masks[i, k]).sum()

    #regions [1024, num_classes]
    wp_regions /= np.linalg.norm(wp_regions, axis=0)
    if max_pooling:
        mp_regions /= np.linalg.norm(mp_regions, axis=0)

    if weight:
        print scores[i,0:2]
        print wp_regions[0:2,0:2]
        wp_regions *= scores[i]
        print wp_regions[0:2,0:2]
        if max_pooling:
            mp_regions *= scores[i]

    if pca_on:
        # all [5063*numclasses, 1024]
        all_regions_images[i*num_classes:num_classes*(i+1)] = np.transpose(regions)

    else:
        wp_image_representations[i] = wp_regions.sum(axis=1)
        wp_image_representations[i] /= np.linalg.norm(wp_image_representations[i])
        if max_pooling:
            mp_image_representations[i] = mp_regions.sum(axis=1)
            mp_image_representations[i] /= np.linalg.norm(mp_image_representations[i])

print 'Time elapsed computing R-MAC: ', time.time() - t
print wp_image_representations.shape
sys.stdout.flush()


if pca_on:
    t1 = time.time()
    print 'Applying PCA with dimension reduction to: ', pca_dim
    sys.stdout.flush()
    pca = PCA(n_components=pca_dim, whiten=True)
    images_pca = pca.fit_transform(all_regions_images)
    print 'PCA finished!'
    print 'Time elapsed computing PCA: ', time.time() - t1
    print images_pca.shape
    print 'Normalizing...'
    images_pca /= np.linalg.norm(images_pca, axis=1)[:,None]
    sys.stdout.flush()
    for i in range(0, num_samples):
        if num_classes > 1:
            img_pca[i] = images_pca[i*num_classes:num_classes*(i+1)].sum(axis=0)
            img_pca[i] /= np.linalg.norm(img_pca[i])
        else:
            img_pca[i] = images_pca[i]

    cu.save_data(img_pca, results_path, name_result)

else:
    cu.save_data(wp_image_representations, results_path, wp_name_result)
    if max_pooling:
        cu.save_data(mp_image_representations, results_path, mp_name_result)
print 'Total time elapsed: ', time.time()-t




