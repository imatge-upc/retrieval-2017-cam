import numpy as np
import os
import math
import h5py
import sys
from sklearn.neighbors import NearestNeighbors
import sklearn
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
import time

# Query Expansion
def expand_query(n_expand, data, indices):
    ind_data = indices[0:n_expand]
    print ind_data.shape
    desc = np.zeros(data.shape[1])
    for ind in ind_data:
        desc += data[ind]
    desc = desc / n_expand
    return desc


# Save Ranking after Re-ranking
def save_ranking_indices(indices, image_names, image_name, path):
    for i in range(0, image_names.shape[0]):
        if image_names[i].replace('\n', '') == image_name:
            print 'Saving ranking for ... ', image_name
            print i
            file = open(path + image_names[i].replace('\n', '') + '.txt', 'w')
            for ind in indices:
                file.write(image_names[ind])
            file.close()


# Save Ranking for one query
def save_ranking_one_query(data, query_desc, image_names, path, image_name):
    for i in range(0, image_names.shape[0]):
        if image_names[i].replace('\n', '') == image_name:
            print 'Saving ranking for ... ', image_name
            print i
            data_aux = data[i].copy()
            data[i] = query_desc
            data_local = data
            t_t = time.time()
            #distances, indices = compute_distances(data, image_names.shape[0])
            #print distances[i]
            #print 'neighbors: ', time.time()-t_t
            t_o = time.time()
            distances, indices = compute_distances_optim(query_desc, data)
            print 'New: ', time.time() - t_o
            file = open(path + image_names[i].replace('\n', '') + '.txt', 'w')
            for ind in indices:
                file.write(image_names[ind])
            file.close()
            data[i] = data_aux
            return indices, data_local


# Save Rankings for all the queries (GS)
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


# Compute distances and get list of indices
def compute_distances(data, neighbors):
    print('Computing distances...')
    nbrs = NearestNeighbors(n_neighbors=neighbors, metric='cosine', algorithm='brute').fit(data)
    distances, indices = nbrs.kneighbors(data)
    print distances.shape
    print indices.shape
    sys.stdout.flush()
    return distances, indices


# Compute distances and get list of indices
def compute_distances_optim(desc, data):
    print('Computing distances...')
    print desc.shape
    print data.shape
    print np.transpose(data).shape
    dist = np.dot(desc, np.transpose(data))
    ind = dist[0].argsort()[::-1]
    sys.stdout.flush()
    return dist[0], ind


# Compute mAP
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

    ap_file = open(ranking_path+'all_scores_map.txt', 'w')
    for res in ap_list:
        ap_file.write(str(res)+'\n')
    mean_ap = sum(ap_list) / len(ap_list)
    print ("The mean_ap is: ", mean_ap)
    #ap_file.write(str(mean_ap))
    #ap_file.close()
    return mean_ap


def evaluate_paris(ranking_path='/imatge/ajimenez/workspace/ITR/results/lists_paris/'):

    if not os.path.exists(ranking_path):
        os.makedirs(ranking_path)

    print('Ranking and Evaluating Paris...')
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


# Best Images Plot
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


# Statistics about ranking
def show_stats(dataset, results_path):

    if dataset == 'Oxford':
        path_gt = "/imatge/ajimenez/work/datasets_retrieval/Oxford/2_groundtruth/"
        query_names = ["all_souls", "ashmolean", "balliol", "bodleian", "christ_church", "cornmarket", "hertford",
                       "keble",
                       "magdalen", "pitt_rivers", "radcliffe_camera"]
    elif dataset == 'Paris':
        path_gt = "/imatge/ajimenez/work/datasets_retrieval/Paris/imatges_paris_gt/"
        query_names = ["defense", "eiffel", "invalides", "louvre", "moulinrouge", "museedorsay", "notredame",
                       "pantheon",
                       "pompidou", "sacrecoeur", "triomphe"]

    for query in query_names:
        for i in range(1, 6):
            f = open(path_gt + query + '_' + str(i)+'_query.txt').readline()
            if dataset == 'Oxford':
                q = f.replace("oxc1_", "")
                f_list = q.split(" ")
            else:
                f_list = f.split(" ")
            q = f_list[0]
            print q + ':'
            print
            f_good = open(path_gt+query+'_'+str(i)+'_good.txt')
            f_ok = open(path_gt+query+'_'+str(i)+'_ok.txt')
            f_junk = open(path_gt+query+'_'+str(i)+'_junk.txt')
            print 'Good Images'
            counter_total_img = 0
            count_100 = 0
            count_1000 = 0
            count_more = 0
            count_25 = 0
            for line_good in f_good:
                counter_total_img += 1
                line_good = line_good.replace('\n','')
                f_res = open(results_path + q + '.txt')
                for i, line_res in enumerate(f_res):
                    if line_good == line_res.replace('\n',''):
                        if i >= 1000:
                            #print line_good + 'in position ' + str(i) + '<<<<<<<<<'
                            count_more += 1
                        elif i >= 100:
                            count_1000 +=1
                            #print line_good + 'in position ' + str(i) + '<<<<<<'
                        elif i >= 25:
                            count_100 += 1
                            #print line_good + 'in position ' + str(i) + '<<<'
                        else:
                            count_25 += 1
                            #print line_good + 'in position ' + str(i)
            print 'Stats: '
            print 'Total images good: ', counter_total_img
            print 'Images in top 25: ', count_25
            print 'Images between 25-100: ', count_100
            print 'Images between 100-1000: ', count_1000
            print 'Images above rank 1000: ', count_more
            print 'Ok images'
            counter_total_img = 0
            count_100 = 0
            count_1000 = 0
            count_more = 0
            count_25 = 0
            for line_ok in f_ok:
                counter_total_img += 1
                line_ok = line_ok.replace('\n', '')
                f_res = open(results_path + q + '.txt')
                for i, line_res in enumerate(f_res):
                    if line_ok == line_res.replace('\n',''):
                        if i >= 1000:
                            #print line_ok + 'in position ' + str(i) + '<<<<<<<<<'
                            count_more += 1
                        elif i >= 100:
                            count_1000 += 1
                            #print line_ok + 'in position ' + str(i) + '<<<<<<'
                        elif i >= 25:
                            count_100 += 1
                            #print line_ok + 'in position ' + str(i) + '<<<'
                        else:
                            count_25 += 1
                            #print line_ok + 'in position ' + str(i)
            print 'Stats: '
            print 'Total images ok: ', counter_total_img
            print 'Images in top 25: ', count_25
            print 'Images between 25-100: ', count_100
            print 'Images between 100-1000: ', count_1000
            print 'Images above rank 1000: ', count_more

            print '##########################################################################################'


def compare_scores(list_1_path, list_2_path):
    f_1 = open(list_1_path, 'r')
    f_2 = open(list_2_path, 'r')
    for i in range(0,55):
        dif = float(f_1.readline()) - float(f_2.readline())
        print dif


#show_stats('Oxford', '/imatge/ajimenez/work/ITR/oxford/results/vgg_16_CAM_imagenet/relu5_1/1024x720/')





