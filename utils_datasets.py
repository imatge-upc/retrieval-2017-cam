from scipy.misc import imread, imresize, imsave
import numpy as np
import os
import math
import h5py
import matplotlib.pyplot as plt
import sys
from PIL import Image


image_path_oxford = '/imatge/ajimenez/work/datasets_retrieval/Oxford/1_images/'
image_path_paris = '/imatge/ajimenez/work/datasets_retrieval/Paris/imatges_paris/'

oxford_list_path = "/imatge/ajimenez/work/ITR/oxford/lists/list_oxford.txt"


def create_folders(path):
    if not os.path.exists(path):
        print 'Creating path: ', path
        os.makedirs(path)
    else:
        print 'Path already exists'


def save_chunk(name, images, image_names, chunk=''):
    print 'Saving chunk number ', chunk
    if chunk != '':
        with h5py.File(name+'_'+str(chunk)+'.h5', 'w') as hf:
            hf.create_dataset('data', data=images)
            hf.create_dataset('image_name', data=image_names)
    else:
        with h5py.File(name+'.h5', 'w') as hf:
            hf.create_dataset('data', data=images)
            hf.create_dataset('image_name', data=image_names)


def save_dataset(path, name, images, image_names):
    with h5py.File(path+name, 'w') as hf:
        hf.create_dataset('data', data=images)
        hf.create_dataset('image_name', data=image_names)


def create_oxford_hdf5(batch_size, size, mean_value, name, image_train_list_path):
    print 'Creating Oxford dataset in hdf5 in' + name
    images = [0] * batch_size
    image_names = [0] * batch_size
    counter = 0
    chunk = 0
    num_images = 0

    for line in open(image_train_list_path):
        if counter >= batch_size:
            counter = 0
            x = preprocess_images(images, size[0], size[1], mean_value)
            # if local_search:
            #     for j, q_name in enumerate(queries_name):
            #         for i, img_name in enumerate(image_names):
            #             if img_name == q_name:
            #                 print query_images.shape
            #                 print j
            #                 x[i] = query_images[j]
            #                 break
            save_chunk(name, x, image_names, chunk)
            sys.stdout.flush()
            chunk += 1

        line = line.rstrip('\n')
        images[counter] = imread(line)
        line = line.replace('/imatge/ajimenez/work/datasets_retrieval/Oxford/1_images/', '')
        image_names[counter] = (line.replace('.jpg', ''))
        counter += 1
        num_images += 1
        #print ("Loading " + line)

    #Last batch
    x = preprocess_images(images[0:num_images % batch_size], size[0], size[1], mean_value)
    # if local_search:
    #     for j, q_name in enumerate(queries_name):
    #         for i, img_name in enumerate(image_names[0:num_images % batch_size]):
    #             if img_name == q_name:
    #                 x[i] = query_images[j]
    #                 break

    save_chunk(name, x, image_names[0:num_images % batch_size], chunk)
    file_ds = open(name+'_info.txt', 'w')
    file_ds.write(str(name) + '\n')
    file_ds.write(str(chunk)+'\n')
    file_ds.write(str(batch_size)+'\n')
    file_ds.write(str(num_images))
    file_ds.close()
    print 'Total images saved: ', num_images
    print 'Total chunks: ', chunk


def create_paris_hdf5(batch_size, size, mean_value, name, image_train_list_path):
    print 'Creating Paris dataset in hdf5 in' + name
    images = [0] * batch_size
    image_names = [0] * batch_size
    counter = 0
    chunk = 0
    num_images = 0

    for line in open(image_train_list_path):
        if counter >= batch_size:
            counter = 0
            x = preprocess_images(images, size[0], size[1], mean_value)
            save_chunk(name, x, image_names, chunk)
            sys.stdout.flush()
            chunk += 1

        line = line.rstrip('\n')
        images[counter] = imread(line)
        line = line.replace(image_path_paris, '')
        image_names[counter] = (line.replace('.jpg', ''))
        counter += 1
        num_images += 1
        # print ("Loading " + line)

    # Last batch
    x = preprocess_images(images[0:num_images % batch_size], size[0], size[1], mean_value)

    # Save check file
    save_chunk(name, x, image_names[0:num_images % batch_size], chunk)
    file_ds = open(name + '_info.txt', 'w')
    file_ds.write(str(name) + '\n')
    file_ds.write(str(chunk) + '\n')
    file_ds.write(str(batch_size) + '\n')
    file_ds.write(str(num_images))
    file_ds.close()
    print 'Total images saved: ', num_images
    print 'Total chunks: ', chunk


def read_dataset_properties(dataset_name):
    f = open(dataset_name)
    name = (f.readline()).replace('\n', '')
    chunk_n = int(f.readline()) + 1
    batch_size = int(f.readline())
    total_images = int(f.readline())
    f.close()
    return name, chunk_n, batch_size, total_images


def preprocess_images(images, img_width, img_height, mean_value):
    print ("Preprocessing Images... ")
    num_images = len(images)
    x = np.zeros((num_images, 3, img_height, img_width), dtype=np.float32)
    for i in range(0, num_images):
        # print str(i + 1) + "/" + str(num_images)
        images[i] = imresize(images[i], [img_height, img_width]).astype(dtype=np.float32)
        # print images[i].shape
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
        #print x[i].shape
        #print 'Preprocessed ' + str(num_images)
    print x.shape
    return x


def preprocess_query(image, img_width, img_height, mean_value):
    print ("Preprocessing query... ")
    image = imresize(image, [img_height, img_width]).astype(dtype=np.float32)
    # RGB -> BGR
    R = np.copy(image[:, :, 0])
    B = np.copy(image[:, :, 2])
    image[:, :, 0] = B
    image[:, :, 2] = R

    # Subtract mean
    image[:, :, 0] -= mean_value[0]
    image[:, :, 1] -= mean_value[1]
    image[:, :, 2] -= mean_value[2]

    image = np.transpose(image, (2, 0, 1))
    #print x[i].shape
    #print 'Preprocessed ' + str(num_images)
    print image.shape
    return image


def read_dataset(path_data, only_labels='all'):
    with h5py.File(path_data, 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        image_name = hf.get('image_name')
        image_names = np.array(image_name)
        if only_labels != 'labels':
            image = hf.get('data')
            images = np.array(image)
            return images, image_names
        else:
            return image_names


def preprocess_queries_oxford(size_h, size_v, mean_value, local_search=False):
    print('Preprocessing Queries Oxford')
    #  queries
    path_gt = "/imatge/ajimenez/work/datasets_retrieval/Oxford/2_groundtruth/"
    query_names = ["all_souls", "ashmolean", "balliol","bodleian", "christ_church", "cornmarket","hertford","keble","magdalen","pitt_rivers","radcliffe_camera"]
    queries_name_h = list()
    queries_name_v = list()
    images_h = list()
    images_v = list()
    #file_queries = open('queries_list_oxford.txt', 'w')
    f_h = open('/imatge/ajimenez/workspace/ITR/lists/list_oxford_queries_horizontal.txt', 'w')
    f_v = open('/imatge/ajimenez/workspace/ITR/lists/list_oxford_queries_vertical.txt', 'w')
    for query_name in query_names:
        for i in range(1, 6):
            f = open(path_gt + query_name + '_' + str(i) + '_query.txt').readline()
            f = f.replace("oxc1_", "")
            f_list = f.split(" ")
            #file_queries.write(f_list[0] + '\n')
            # print f_list[0]
            for k in range(1, 5):
                f_list[k] = (int(math.floor(float(f_list[k]))))
                #print f_list[k]

            img = imread(image_path_oxford + f_list[0] + '.jpg')
            if local_search:
                img_cropped = img[f_list[2]:f_list[4], f_list[1]:f_list[3]]
                print 'Height: ', img_cropped.shape[0]
                print 'Width: ', img_cropped.shape[1]
                print 'Resized into...'
            else:
                img_cropped = img

            # Vertical queries
            if img_cropped.shape[0] >= img_cropped.shape[1]:
                img_cropped = preprocess_query(img_cropped, size_v[0], size_v[1], mean_value)
                images_v.append(img_cropped)
                queries_name_v.append(f_list[0])
                f_v.write(f_list[0]+'\n')

            # Horizontal queries
            elif img_cropped.shape[0] < img_cropped.shape[1]:
                img_cropped = preprocess_query(img_cropped, size_h[0], size_h[1], mean_value)
                images_h.append(img_cropped)
                queries_name_h.append(f_list[0])
                f_h.write(f_list[0]+'\n')

    #file_queries.close()
    f_h.close()
    f_v.close()

    return images_h, images_v, queries_name_h, queries_name_v


def preprocess_queries_paris(size_h, size_v, mean_value, local_search=False):
    print 'Preprocessing Queries Paris ...'
    #  queries
    path_gt = "/imatge/ajimenez/work/datasets_retrieval/Paris/imatges_paris_gt/"
    query_names = ["defense", "eiffel", "invalides", "louvre", "moulinrouge", "museedorsay", "notredame", "pantheon",
                   "pompidou", "sacrecoeur", "triomphe"]
    queries_name_h = list()
    queries_name_v = list()
    images_h = list()
    images_v = list()
    #file_queries = open('queries_list_paris.txt', 'w')
    f_h = open('/imatge/ajimenez/workspace/ITR/lists/list_paris_queries_horizontal.txt', 'w')
    f_v = open('/imatge/ajimenez/workspace/ITR/lists/list_paris_queries_vertical.txt', 'w')
    for query_name in query_names:
        for i in range(1, 6):
            f = open(path_gt + query_name + '_' + str(i) + '_query.txt').readline()
            #f = f.replace("paris_", "")
            f_list = f.split(" ")
            #file_queries.write(f_list[0] + '\n')
            # print f_list[0]
            for k in range(1, 5):
                f_list[k] = (int(math.floor(float(f_list[k]))))
                #print f_list[k]

                img = imread(image_path_paris + f_list[0] + '.jpg')
            if local_search:
                img_cropped = img[f_list[2]:f_list[4], f_list[1]:f_list[3]]
                print 'Height: ', img_cropped.shape[0]
                print 'Width: ', img_cropped.shape[1]
                print 'Resized into...'
            else:
                img_cropped = img

            # Vertical queries
            if img_cropped.shape[0] >= img_cropped.shape[1]:
                img_cropped = preprocess_query(img_cropped, size_v[0], size_v[1], mean_value)
                images_v.append(img_cropped)
                queries_name_v.append(f_list[0])
                f_v.write(f_list[0] + '\n')

            # Horizontal queries
            elif img_cropped.shape[0] < img_cropped.shape[1]:
                img_cropped = preprocess_query(img_cropped, size_h[0], size_h[1], mean_value)
                images_h.append(img_cropped)
                queries_name_h.append(f_list[0])
                f_h.write(f_list[0] + '\n')

    #file_queries.close()
    f_h.close()
    f_v.close()

    return images_h, images_v, queries_name_h, queries_name_v


def load_data_oxford(image_train_list_path=oxford_list_path):
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


def load_data(filepath):
    with h5py.File(filepath, 'r') as hf:
        data = np.array(hf.get('data'))
        print 'Shape of the array features: ', data.shape
        return data


def save_data(data, path, name):
    with h5py.File(path + name, 'w') as hf:
        hf.create_dataset('data', data=data)

# Used for experimental unsupervised learning
# def label_oxford(image_names):
#
#     print ("Labeling the dataset... ")
#     # queries
#     path_gt = "/imatge/ajimenez/work/datasets_retrieval/Oxford/2_groundtruth/"
#     query_names = ["all_souls", "ashmolean", "balliol", "bodleian", "christ_church", "cornmarket", "hertford", "keble",
#                    "magdalen", "pitt_rivers", "radcliffe_camera"]
#
#     labels = np.zeros(image_names.shape[0], dtype=np.int32)
#     label_num = 1
#     num_images_labeled = 0
#     for query_name in query_names:
#         for i in range(1, 6):
#             with open(path_gt + query_name + '_' + str(i) + '_good.txt') as f:
#                 images_good = f.readlines()
#             print images_good
#
#             with open(path_gt + query_name + '_' + str(i) + '_ok.txt') as f:
#                 images_ok = f.readlines()
#             print images_ok
#
#             for img in images_good:
#                 for j in range(0,image_names.shape[0]):
#                     if (img.replace('\n', '')) == image_names[j]:
#                         labels[j] = label_num
#                         print(image_names[j] + 'labeled as: ' + str(label_num))
#                         num_images_labeled +=1
#                         break
#
#             for img in images_ok:
#                 for j in range(0,image_names.shape[0]):
#                     if (img.replace('\n', '')) == image_names[j]:
#                         labels[j] = label_num
#                         print(image_names[j] + 'labeled as: ' + str(label_num))
#                         num_images_labeled += 1
#                         break
#         label_num += 1
#
#     print ("Num images labeled by weak labels: ", num_images_labeled)
#     cc = 0
#     for k in range(0, image_names.shape[0]):
#         if labels[k] == 0:
#             if cc == 2:
#                 label_num += 1
#                 cc = 0
#             cc += 1
#             labels[k] = label_num
#
#     return labels
#

