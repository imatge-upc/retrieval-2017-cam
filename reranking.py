import numpy as np
import os
import h5py
import sys
import evaluate_oxford_paris as eval
import utils_datasets as utils
import time
from sklearn.decomposition import PCA
from vgg_cam import VGGCAM
from utils_datasets import create_folders, save_data, preprocess_images, preprocess_query, load_data
from pooling_functions import weighted_cam_pooling, descriptor_aggregation
from cam_utils import extract_feat_cam
from scipy.misc import imread
import math
from sklearn.metrics.pairwise import cosine_similarity

size_v = [720, 1024]
size_h = [1024, 720]

#Imagenet
mean_value = [123.68, 116.779, 103.939]

dim_descriptors = 512

n_images_oxford = 5063
n_images_paris = 6392

layer = 'relu5_1'


# Sliding windows
def compute_score_sliding(features_query, features_img, pooling='sum'):
    score = np.zeros(features_img.shape[0])
    f_2_max = np.zeros(features_img.shape[1])
    num_samples = features_img.shape[0]
    f_q = features_query[0]
    q_h, q_w = features_query[1], features_query[2]
    f_h, f_w = features_img.shape[2], features_img.shape[3]

    if q_h > f_h or q_w > f_w:
        print 'CHECK'
        pass
    else:
        print 'else'

        n_times_hor = f_w - q_w
        n_times_ver = f_h - q_h
        #print n_times_hor
        #print n_times_ver
        for img_ind in range(0,num_samples):
            for s in range(0, n_times_hor):
                for v in range(0, n_times_ver):
                    for f_ind2 in range(0, features_img.shape[1]):
                        if pooling == 'sum':
                            f_2_max[f_ind2] = features_img[img_ind, f_ind2, v:q_h + v, s:q_w + s].sum()
                        elif pooling == 'max':
                            f_2_max[f_ind2] = np.amax(features_img[img_ind, f_ind2, v:q_h + v, s:q_w + s])

                    f_2_max /= np.linalg.norm(f_2_max)
                    score_aux = np.dot(f_q, f_2_max)
                    # print 'Score aux: ', score_aux
                    sys.stdout.flush()
                    if score_aux > score[img_ind]:
                        #print 'new Score: ', score_aux
                        sys.stdout.flush()
                        score[img_ind] = np.copy(score_aux)
                        #print [s, v]
    return score


# Compute score using CAMs, PCA , Region of interest
def compute_scores_cams(desc_query, features_img, cams, roi, pca_matrix):
    print 'Feat shape:', features_img.shape
    print 'Cams shape', cams.shape
    nthres = 4
    scores = np.zeros(features_img.shape[0])
    feats = np.zeros((1, features_img.shape[1], features_img.shape[2], features_img.shape[3]), dtype=np.float32)
    cams_ = np.zeros((1, cams.shape[1], cams.shape[2], cams.shape[3]), dtype=np.float32)
    final_descriptors = np.zeros((features_img.shape[0],features_img.shape[1]), dtype=np.float32)
    for img_ind in range(features_img.shape[0]):
        print features_img[img_ind].shape
        feats[0] = features_img[img_ind]
        cams_[0] = cams[img_ind]
        scores[img_ind] = -10
        print 'Img: ', img_ind
        x, y, w, h = roi[img_ind, :, 0], roi[img_ind, :, 1], roi[img_ind, :, 2], roi[img_ind, :, 3]
        for th in range(0, nthres):
            print y[th], y[th] + h[th]
            print x[th], x[th] + w[th]
            sys.stdout.flush()
            if h[th] >= 5 and w[th] >= 5:
                d_wp = weighted_cam_pooling(feats[:, :,y[th]:y[th]+h[th], x[th]:x[th]+w[th]],
                                            cams_[:, :, y[th]:y[th]+h[th], x[th]:x[th]+w[th]], max_pool=False)
                descriptor = descriptor_aggregation(d_wp, 1, cams.shape[1], pca_matrix)

                score_aux = np.dot(desc_query, np.transpose(descriptor))

                print 'Thresh: ', th
                print 'Score:', score_aux
                if score_aux > scores[img_ind]:
                    print 'Max in th:', th
                    scores[img_ind] = np.copy(score_aux)
                    final_descriptors[img_ind] = descriptor
            else:
                pass
    return scores, final_descriptors


def compute_scores_cams_only(fmax_q, features_img, cams, pca_matrix):
    print 'Feat shape:', features_img.shape
    print 'Cams shape', cams.shape
    scores = np.zeros(features_img.shape[0])
    vec = np.zeros((features_img.shape[1], cams.shape[1]), dtype=np.float32)
    for img_ind in range(features_img.shape[0]):
        C = np.array(compute_crow_channel_weight(features_img[img_ind]))
        print 'Img: ', img_ind
        for f_ind in range(0, features_img.shape[1]):
            for c_ind in range(0, cams.shape[1]):
                vec[f_ind, c_ind] = np.multiply(features_img[img_ind,f_ind],
                                                cams[img_ind, c_ind]).sum()

        vec = vec*C[:, None]
        vec /= np.linalg.norm(vec, axis=0)
        vec = np.transpose(pca_matrix.transform(np.transpose(vec)))
        vec /= np.linalg.norm(vec, axis=0)
        f_2_max = vec.sum(axis=1)
        f_2_max /= np.linalg.norm(f_2_max)
        scores[img_ind] = np.dot(fmax_q, f_2_max)
    return scores


def compute_scores_roi(fmax_q, features_img, roi, pca_matrix):
    print 'Feat shape:', features_img.shape
    nthres = 5
    scores = np.zeros(features_img.shape[0])
    for img_ind in range(features_img.shape[0]):
        C = np.array(compute_crow_channel_weight(features_img[img_ind]))
        scores[img_ind] = -10
        print 'Img: ', img_ind
        x, y, w, h = roi[img_ind, :, 0], roi[img_ind, :, 1], roi[img_ind, :, 2], roi[img_ind, :, 3]
        for th in range(0, nthres):
            f_2_max = np.zeros(features_img.shape[1])
            if h[th] >= 5 and w[th] >= 5:
                for f_ind in range(0, features_img.shape[1]):
                    f_2_max[f_ind] = features_img[img_ind, f_ind, y[th]:y[th]+h[th], x[th]:x[th]+w[th]].sum()

                f_2_max = f_2_max * C
                f_2_max /= np.linalg.norm(f_2_max)
                print f_2_max.shape
                f_2_max = f_2_max.reshape((1, features_img.shape[1]))
                print f_2_max.shape
                f_2_max = np.transpose(pca_matrix.transform(f_2_max))
                print f_2_max.shape
                sys.stdout.flush()
                f_2_max /= np.linalg.norm(f_2_max)
                score_aux = np.vdot(fmax_q, f_2_max)
                print 'Thresh: ', th
                print 'Score:', score_aux
                if score_aux > scores[img_ind]:
                    print 'Max in th:', th
                    scores[img_ind] = np.copy(score_aux)
    return scores


def re_order(order, vector_h, vector_v):
    vector = list()
    count_h = 0
    count_v = 0
    for pos in order:
        if pos == 0:
            vector.append(vector_h[count_h])
            count_h += 1
        elif pos == 1:
            vector.append(vector_v[count_v])
            count_v += 1
    return vector


def re_ranking(desc_query, class_list, image_names, indices, dataset, top_n_ranking, pca_matrix, model_h, model_v):
    if dataset == 'Oxford':
        images_path = '/imatge/ajimenez/work/datasets_retrieval/Oxford/1_images/'

    elif dataset == 'Paris':
        images_path = '/imatge/ajimenez/work/datasets_retrieval/Paris/imatges_paris/'

    index_q = indices[0:top_n_ranking]
    tt = time.time()
    indexed_names = list()
    i = 0
    batch_size = 12
    n_batches = int(math.floor(top_n_ranking / batch_size))
    last_batch = top_n_ranking % batch_size
    scores = np.zeros(top_n_ranking, dtype=np.float32)
    scores_h = np.zeros(top_n_ranking, dtype=np.float32)
    scores_v = np.zeros(top_n_ranking, dtype=np.float32)
    final_desc_h = np.zeros(top_n_ranking, dtype=np.float32)
    final_desc_v = np.zeros(top_n_ranking, dtype=np.float32)
    print desc_query.shape
    final_descriptors_all = np.zeros((top_n_ranking, desc_query.shape[1]), dtype=np.float32)
    image_ranked_names = image_names[index_q]

    num_cams = class_list.shape[0]

    for k in range(0, n_batches+1):
        x_2_h = np.zeros((0, 3, size_h[1], size_h[0]), dtype=np.float32)
        x_2_v = np.zeros((0, 3, size_v[1], size_v[0]), dtype=np.float32)
        xx_2_v = np.zeros((1, 3, size_v[1], size_v[0]), dtype=np.float32)
        xx_2_h = np.zeros((1, 3, size_h[1], size_h[0]), dtype=np.float32)
        t1 = time.time()
        if k == n_batches:
            #Last Batch
            if last_batch != 0:
                last_ind = batch_size * k + last_batch
            else:
                break
        else:
            last_ind = batch_size * (k+1)

        print image_names[index_q[k*batch_size:last_ind]]

        # Separate the images in Horizontal/Vertical for faster processing
        image_order = list()
        for ind_im, name in enumerate(image_names[index_q[k*batch_size:last_ind]]):
            im = imread(images_path + name.replace('\n', '') + '.jpg')
            if im.shape[0] >= im.shape[1]:
                size = size_v
                xx_2_v[0] = preprocess_query(im, size[0], size[1], mean_value)
                x_2_v = np.concatenate((x_2_v, xx_2_v))
                image_order.append(1)
            else:
                size = size_h
                xx_2_h[0] = preprocess_query(im, size[0], size[1], mean_value)
                x_2_h = np.concatenate((x_2_h, xx_2_h))
                image_order.append(0)

        # Extract Features/CAMs
        print 'Time loading images: ', time.time() - t1
        if x_2_h.shape[0] > 0:
            t2 = time.time()
            features_h, cams_h, roi_h = extract_feat_cam(model_h, layer, batch_size, x_2_h, num_cams,
                                                         class_list, roi=True)
            #features_h, cams_h, roi_h = extract_feat_cam(model_h, layer, batch_size, x_2_h, num_cams)
            print 'Time extracting features: ', time.time() - t2
            t3 = time.time()
            #scores_h = compute_scores_cams_only(f_max, features_h, cams_h, pca_matrix)
            scores_h, final_desc_h = compute_scores_cams(desc_query, features_h, cams_h, roi_h, pca_matrix)
            #scores_h = compute_scores_roi(f_max, features_h, roi_h, pca_matrix)
            #scores_h = compute_score_sliding(f_query, features_h)
            print 'Time computing scores: ', time.time() - t3
            print scores_h

        if x_2_v.shape[0] > 0:
            t2 = time.time()
            features_v, cams_v, roi_v = extract_feat_cam(model_v, layer, batch_size, x_2_v, num_cams,
                                                         class_list, roi=True)
            print 'Time extracting features: ', time.time() - t2
            #features_v, cams_v, roi_v = extract_feat_cam(model_v, layer, batch_size, x_2_v, num_cams)

            t3 = time.time()
            #scores_v = compute_scores_cams_only(f_max, features_v, cams_v, pca_matrix)
            scores_v, final_desc_v = compute_scores_cams(desc_query, features_v, cams_v, roi_v, pca_matrix)
            #scores_v = compute_scores_roi(f_max, features_v, roi_v, pca_matrix)
            #scores_v = compute_score_sliding(f_query, features_v)
            print 'Time computing scores: ', time.time() - t3
            print scores_v


        # Compute Scores
        print image_order
        # Re-order
        scores[k*batch_size:last_ind] = re_order(image_order, scores_h, scores_v)
        final_descriptors_all[k*batch_size:last_ind] = re_order(image_order, final_desc_h, final_desc_v)
        print final_descriptors_all.shape

        print scores[k*batch_size:batch_size*(k+1)]
        print 'Time loading computing scores: ', time.time() - t2
        print 'Time elapsed x image:', time.time() - t1

    print scores
    ordered_sc = scores.argsort()[::-1]
    print ordered_sc
    print image_names[index_q]
    print image_ranked_names[ordered_sc]
    # Index of the in order of relevance
    ordered_ind = index_q[ordered_sc]
    indexed_names.append(np.copy(image_ranked_names[ordered_sc]))
    indices[0:top_n_ranking] = ordered_ind
    i += 1
    print 'Time elapsed:', time.time()-tt
    # Return indices and data ordered by similarity with the query
    return indices, final_descriptors_all[ordered_sc]
