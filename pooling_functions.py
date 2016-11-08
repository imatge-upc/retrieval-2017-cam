import cam_utils as cu
import numpy as np
import time
from sklearn.decomposition import PCA
import sys
import utils_datasets as ud
import time


def compute_PCA(descriptors, whiten=True):
    print descriptors.shape
    t1 = time.time()
    print 'Applying PCA with dimension reduction to: ', pca_dim
    sys.stdout.flush()
    pca = PCA(n_components=pca_dim, whiten=True)
    pca.fit(descriptors)
    print pca.components_.shape
    print 'PCA finished!'
    print 'Time elapsed computing PCA: ', time.time() - t1
    return pca


def sum_pooling(features):
    num_samples = features.shape[0]
    sys.stdout.flush()
    descriptors = np.zeros((num_samples, num_features), dtype=np.float32)
    for i in range(0, num_samples):
        #print 'Image: ', i
        #sys.stdout.flush()
        for f in range(0, num_features):
            descriptors[i, f] = features[i, f].sum()
    descriptors /= np.linalg.norm(descriptors, axis=1)[:, None]
    return descriptors


def max_pooling(features):
    num_samples = features.shape[0]
    sys.stdout.flush()
    descriptors = np.zeros((num_samples, num_features), dtype=np.float32)
    for i in range(0, num_samples):
        #print 'Image: ', i
        #sys.stdout.flush()
        for f in range(0, num_features):
            descriptors[i, f] = np.amax(features[i, f])
    descriptors /= np.linalg.norm(descriptors, axis=1)[:, None]
    return descriptors


def weighted_pooling(features, cams, max_pool=False, region_descriptors=False, pca=''):
    t = time.time()
    num_samples = features.shape[0]
    num_features = features.shape[1]
    num_classes = cams.shape[1]

    wp_batch_representations = np.zeros((num_samples, num_features), dtype=np.float32)
    wp_regions = np.zeros((num_features, num_classes), dtype=np.float32)
    wsp_descriptors_reg = np.zeros((num_samples * num_classes, num_features), dtype=np.float32)
    wmp_descriptors_reg = np.zeros((num_samples * num_classes, num_features), dtype=np.float32)

    if max_pool:
        mp_regions = np.zeros((num_features, num_classes), dtype=np.float32)
        mp_batch_representations = np.zeros((num_samples, num_features), dtype=np.float32)

    if pca != '':
        print 'Applying PCA...'
        sys.stdout.flush()

    for i in range(0, num_samples):
        for f in range(0, num_features):
            for k in range(0, num_classes):
                # For each region compute avg weighted sum of activations and l2 normalize
                if max_pool:
                    mp_regions[f, k] = np.amax(np.multiply(features[i, f], cams[i, k]))

                wp_regions[f, k] = np.multiply(features[i, f], cams[i, k]).sum()

        wp_regions /= np.linalg.norm(wp_regions, axis=0)

        if max_pool:
            mp_regions /= np.linalg.norm(mp_regions, axis=0)

        if region_descriptors:
            wsp_descriptors_reg[num_classes*i:num_classes*(i+1)] = np.transpose(wp_regions)
            if max_pool:
                wmp_descriptors_reg[num_classes*i:num_classes*(i+1)] = np.transpose(mp_regions)

        if pca != '':
            wp_regions = np.transpose(pca.transform(np.transpose(wp_regions)))
            wp_regions /= np.linalg.norm(wp_regions, axis=0)
            mp_regions = np.transpose(pca.transform(np.transpose(mp_regions)))
            mp_regions /= np.linalg.norm(mp_regions, axis=0)


        wp_batch_representations[i] = wp_regions.sum(axis=1)
        wp_batch_representations[i] /= np.linalg.norm(wp_batch_representations[i])

        #wp_batch_representations[i][np.where(wp_batch_representations[i] < 0.001)] = 0

        if max_pool:
            mp_batch_representations[i] = mp_regions.sum(axis=1)
            mp_batch_representations[i] /= np.linalg.norm(mp_batch_representations[i])

    print 'Time elapsed computing image representations for the batch: ', time.time() - t

    if region_descriptors and max_pool:
        print wp_batch_representations.shape
        print wsp_descriptors_reg.shape
        return wp_batch_representations, mp_batch_representations, wsp_descriptors_reg, wmp_descriptors_reg
    elif region_descriptors:
        return wp_batch_representations, wsp_descriptors_reg
    elif max_pool:
        return wp_batch_representations, mp_batch_representations
    else:
        return wp_batch_representations





