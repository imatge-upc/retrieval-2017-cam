from scipy.misc import imread, imresize, imsave
from sklearn.neighbors import NearestNeighbors
import sklearn
import numpy as np
import os
import math
import h5py
import matplotlib.pyplot as plt
import sys
from PIL import Image


image_path = '/imatge/ajimenez/work/datasets_retrieval/Oxford/1_images/'

def load_features(path_features):
    print ('Loading data...')
    sys.stdout.flush()
    with h5py.File(path_features, 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = hf.get('feature')
        np_data = np.array(data)
        print('Shape of the array dataset_1: \n', np_data.shape)
    return np_data


def read_data_oxford(path_data):
    with h5py.File(path_data, 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        image_name = hf.get('image_name')
        image_names = np.array(image_name)
        image = hf.get('data')
        images = np.array(image)
    return images, image_names


def compute_distances(data, neighbors):
    print('Computing distances...')
    sys.stdout.flush()
    nbrs = NearestNeighbors(n_neighbors=neighbors, metric='cosine', algorithm='brute').fit(data)
    distances, indices = nbrs.kneighbors(data)
    print(indices.shape)
    print(distances.shape)
    return distances, indices


def save_rankings(indices, image_names, path):
    for i in range(0, image_names.shape[0]):
        file = open(path + image_names[i] + '.txt', 'w')
        for ind in indices[i]:
            file.write(image_names[ind] + '\n')
        file.close()


def evaluate_oxford(ranking_path = '/imatge/ajimenez/workspace/ITR/results/lists_ox_51/'):

    if not os.path.exists(ranking_path):
        os.makedirs(ranking_path)

    print('Ranking and Evaluating Oxford...')
    #  queries
    path_gt = "/imatge/ajimenez/work/datasets_retrieval/Oxford/2_groundtruth/"
    query_names = ["all_souls", "ashmolean", "balliol","bodleian", "christ_church", "cornmarket","hertford","keble","magdalen","pitt_rivers","radcliffe_camera"]

    ap_list = list()
    for query_name in query_names:
        for i in range (1,6):
            f = open(path_gt + query_name + '_' + str(i)+'_query.txt').readline()
            f = f.replace("oxc1_", "")
            f_list = f.split(" ")
            f = f_list[0]
            print f
            cmd = "./compute_ap {} {}{}.txt > tmp.txt".format(path_gt+query_name + '_' + str(i), ranking_path, f )
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


def preprocess_queries(width, height, mean):

    print('Preprocessing Queries...')
    #  queries
    path_gt = "/imatge/ajimenez/work/datasets_retrieval/Oxford/2_groundtruth/"
    query_names = ["all_souls", "ashmolean", "balliol","bodleian", "christ_church", "cornmarket","hertford","keble","magdalen","pitt_rivers","radcliffe_camera"]
    queries_name = list()
    images = list()
    bounding_box = list()
    for query_name in query_names:
        for i in range(1, 6):
            f = open(path_gt + query_name + '_' + str(i) + '_query.txt').readline()
            f = f.replace("oxc1_", "")
            f_list = f.split(" ")
            queries_name.append(f_list[0])
            print f_list[0]
            for k in range(1, 5):
                f_list[k] = (int(math.floor(float(f_list[k]))))
                #print f_list[k]

            img = Image.open(image_path + f_list[0] + '.jpg')
            img_cropped = img.crop((f_list[1], f_list[2], f_list[3], f_list[4]))
            images.append(img_cropped)
    return preprocess_images(images, 55, width, height, mean), queries_name


def label_oxford(image_names):

    print ("Labeling the dataset... ")
    # queries
    path_gt = "/imatge/ajimenez/work/datasets_retrieval/Oxford/2_groundtruth/"
    query_names = ["all_souls", "ashmolean", "balliol", "bodleian", "christ_church", "cornmarket", "hertford", "keble",
                   "magdalen", "pitt_rivers", "radcliffe_camera"]

    labels = np.zeros(image_names.shape[0], dtype=np.int32)
    label_num = 1
    num_images_labeled = 0
    for query_name in query_names:
        for i in range(1, 6):
            with open(path_gt + query_name + '_' + str(i) + '_good.txt') as f:
                images_good = f.readlines()
            print images_good

            with open(path_gt + query_name + '_' + str(i) + '_ok.txt') as f:
                images_ok = f.readlines()
            print images_ok

            for img in images_good:
                for j in range(0,image_names.shape[0]):
                    if (img.replace('\n', '')) == image_names[j]:
                        labels[j] = label_num
                        print(image_names[j] + 'labeled as: ' + str(label_num))
                        num_images_labeled +=1
                        break

            for img in images_ok:
                for j in range(0,image_names.shape[0]):
                    if (img.replace('\n', '')) == image_names[j]:
                        labels[j] = label_num
                        print(image_names[j] + 'labeled as: ' + str(label_num))
                        num_images_labeled += 1
                        break
        label_num += 1

    print ("Num images labeled by weak labels: ", num_images_labeled)
    cc = 0
    for k in range(0, image_names.shape[0]):
        if labels[k] == 0:
            if cc == 2:
                label_num += 1
                cc = 0
            cc += 1
            labels[k] = label_num

    return labels


def load_data_oxford(image_train_list_path="/imatge/ajimenez/workspace/ITR/lists/list_oxford.txt"):
    print ("Loading Data ... ")
    images = list()
    image_names = list()
    num_images = 0
    for line in open(image_train_list_path):
        line = line.rstrip('\n')
        # print ("Loading " + line)
        images.append(imread(line))
        image_names.append(line)
        num_images += 1

    for i in range(0, len(image_names)):
        image_names[i] = image_names[i].replace('/imatge/ajimenez/work/datasets_retrieval/Oxford/1_images/', '')
        image_names[i] = image_names[i].replace('.jpg', '')

    return images, num_images, image_names


def save_oxford(path, name, images, image_names):
    with h5py.File(path+name, 'w') as hf:
        hf.create_dataset('data', data=images)
        #hf.create_dataset('labels', data=labels)
        hf.create_dataset('image_name', data=image_names)


def preprocess_images(images, num_images, img_width, img_height, mean_value):
    print ("Preprocessing Images... ")
    x = np.zeros((num_images, 3, img_width, img_height), dtype=np.float32)
    for i in range(0, num_images):
        print str(i + 1) + "/" + str(num_images)
        images[i] = imresize(images[i], [img_width, img_height]).astype(dtype=np.float32)
        # RGB -> BGR
        R = np.copy(images[i][:, :, 0])
        B = np.copy(images[i][:, :, 2])
        images[i][:, :, 0] = B
        images[i][:, :, 2] = R

        # Subtract mean
        images[i][:, :, 0] -= mean_value[0]
        images[i][:, :, 1] -= mean_value[1]
        images[i][:, :, 2] -= mean_value[2]

        x[i, :, :, :] = np.transpose(images[i], (2, 0, 1))

    return x

