import numpy as np
import os
import math
import h5py
import sys
from sklearn.neighbors import NearestNeighbors
import sklearn
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt


def save_rankings(indices, image_names, path, dataset):
    if dataset == 'Oxford':
        f = open('/imatge/ajimenez/work/ITR/oxford/lists/queries_list_oxford.txt')
    if dataset == 'Paris':
        f = open('/imatge/ajimenez/work/ITR/paris/lists/queries_list_paris.txt')

    for line in f:
        for i in range(0, image_names.shape[0]):
            if image_names[i] == line:
                print line
                file = open(path + image_names[i].replace('\n', '') + '.txt', 'w')
                for ind in indices[i]:
                    file.write(image_names[ind])
                file.close()
                break
    f.close()


def compute_distances(data, neighbors):
    print('Computing distances...')
    sys.stdout.flush()
    nbrs = NearestNeighbors(n_neighbors=neighbors, metric='cosine', algorithm='brute').fit(data)
    distances, indices = nbrs.kneighbors(data)
    print(indices.shape)
    print(distances.shape)
    return distances, indices


def evaluate_oxford(ranking_path, desc_name):
    print('Ranking and Evaluating Oxford...')
    #  queries
    path_gt = "/imatge/ajimenez/work/datasets_retrieval/Oxford/2_groundtruth/"
    query_names = ["all_souls", "ashmolean", "balliol", "bodleian", "christ_church", "cornmarket","hertford","keble","magdalen","pitt_rivers","radcliffe_camera"]

    ap_list = list()
    for query_name in query_names:
        for i in range(1,6):
            f = open(path_gt + query_name + '_' + str(i)+'_query.txt').readline()
            f = f.replace("oxc1_", "")
            f_list = f.split(" ")
            f = f_list[0]
            print f
            cmd = "./compute_ap {} {}{}.txt > tmp.txt".format(path_gt+query_name + '_' + str(i), ranking_path, f)
            # print cmd
            # execute command
            os.system(cmd)
            # retrieve result
            ap = np.loadtxt("tmp.txt", dtype='float32')
            print ('AP: ', ap)
            ap_list.append(ap)

    #ap_file = open(ranking_path+desc_name+'_map.txt', 'w')
    mean_ap = sum(ap_list) / len(ap_list)
    print ("The mean_ap is: ", mean_ap)
    #ap_file.write(str(mean_ap))
    #ap_file.close()
    return mean_ap


def evaluate_paris(ranking_path='/imatge/ajimenez/workspace/ITR/results/lists_paris/'):

    if not os.path.exists(ranking_path):
        os.makedirs(ranking_path)

    print('Ranking and Evaluating Oxford...')
    #  queries
    path_gt = "/imatge/ajimenez/work/datasets_retrieval/Paris/imatges_paris_gt/"
    query_names = ["defense", "eiffel", "invalides", "louvre", "moulinrouge", "museedorsay", "notredame", "pantheon",
                   "pompidou", "sacrecoeur", "triomphe"]
    ap_list = list()
    for query_name in query_names:
        for i in range(1, 6):
            f = open(path_gt + query_name + '_' + str(i)+'_query.txt').readline()
            #f = f.replace("", "")
            f_list = f.split(" ")
            f = f_list[0]
            print f
            cmd = "./compute_ap {} {}{}.txt > tmp.txt".format(path_gt+query_name + '_' + str(i), ranking_path, f)
            # print cmd
            # execute command
            os.system(cmd)
            # retrieve result
            ap = np.loadtxt("tmp.txt", dtype='float32')
            print ('AP: ', ap)
            ap_list.append(ap)

    mean_ap = sum(ap_list) / len(ap_list)
    print ("The mean_ap is: ", mean_ap)
    return mean_ap


def show_images_top(n_images, dataset):

    if dataset == 'Oxford':
        query_list = open('/imatge/ajimenez/workspace/ITR/lists/queries_list_oxford.txt')
        images_path = '/imatge/ajimenez/work/datasets_retrieval/Oxford/1_images/'
        ranking_path = '/imatge/ajimenez/workspace/ITR/results/ranked_oxford_RMAC_PCA/'

    elif dataset == 'Paris':
        query_list = open('/imatge/ajimenez/workspace/ITR/lists/queries_list_paris.txt')

    for query in query_list:
        count = 1
        fig = plt.figure()
        f = open(ranking_path+query+'.txt')
        for line in f:
            if count <= nimages:
                line = line.rstrip('\n')
                fig.add_subplot(1, n_images, count)
                plt.imshow(imresize(imread(images_path+line+'.jpg'), (128, 128)))
                count += 1
            else:
                plt.savefig('/imatge/ajimenez/work/'+query + '.png')
                print 'saving images'
                break






