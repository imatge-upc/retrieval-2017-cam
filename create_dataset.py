import numpy as np
import sys
import utils_datasets as utils
import time

# Parameters
img_width = 1024
img_height = 1024

# 1024x720 // 720x1024
size_h = [img_width, img_height-304]
size_v = [img_width-304, img_height]

dim = '1024x720'

batch_size = 100

local_search = True

mean_from = 'Places'

if mean_from == 'Places':
    mean_value = [104, 166.66, 122.67]
    folder = 'places/'
elif mean_from == 'Imagenet':
    mean_value = [123.68, 116.779, 103.939]
    folder = 'imagenet/'
else:
    mean_value = [0, 0, 0]

#dataset = 'Oxford'
dataset = 'Paris'


if dataset == 'Oxford':
    t = time.time()
    print 'Creating Oxford dataset'
    print 'batch_size ', batch_size
    train_list_path_h = "/imatge/ajimenez/work/ITR/oxford/lists/list_oxford_horizontal_no_queries.txt"
    train_list_path_v = "/imatge/ajimenez/work/ITR/oxford/lists/list_oxford_vertical_no_queries.txt"
    file_path = '/imatge/ajimenez/work/ITR/oxford/datasets_hdf5/' + folder + dim + '/'
    name_file = file_path + 'oxford'

    name_file_h = name_file + '_h'
    name_file_v = name_file + '_v'

    utils.create_folders(file_path)

    q_imgs_h, q_imgs_v, q_name_h, q_name_v = utils.preprocess_queries_oxford(size_h, size_v, mean_value, local_search)

    if local_search:
        utils.save_chunk(name_file + '_queries_h_ls', q_imgs_h, q_name_h)
        utils.save_chunk(name_file + '_queries_v_ls', q_imgs_v, q_name_v)
    else:
        utils.save_chunk(name_file + '_queries_h', q_imgs_h, q_name_h)
        utils.save_chunk(name_file + '_queries_v', q_imgs_v, q_name_v)

    #utils.create_oxford_hdf5(batch_size, size_h, mean_value, name_file_h, train_list_path_h)
    #utils.create_oxford_hdf5(batch_size, size_v, mean_value, name_file_v, train_list_path_v)

    print 'Total time elapsed ', time.time() - t

elif dataset == 'Paris':
    t = time.time()
    print 'Creating Paris dataset'
    print 'batch_size ', batch_size
    train_list_path_h = "/imatge/ajimenez/work/ITR/paris/lists/list_paris_horizontal_no_queries.txt"
    train_list_path_v = "/imatge/ajimenez/work/ITR/paris/lists/list_paris_vertical_no_queries.txt"
    file_path = '/imatge/ajimenez/work/ITR/paris/datasets_hdf5/' + folder + dim + '/'
    name_file = file_path + 'paris'

    name_file_h = name_file + '_h'
    name_file_v = name_file + '_v'

    utils.create_folders(file_path)

    q_imgs_h, q_imgs_v, q_name_h, q_name_v = utils.preprocess_queries_paris(size_h, size_v, mean_value, local_search)

    if local_search:
        utils.save_chunk(name_file + '_queries_h_ls', q_imgs_h, q_name_h)
        utils.save_chunk(name_file + '_queries_v_ls', q_imgs_v, q_name_v)
    else:
        utils.save_chunk(name_file + '_queries_h', q_imgs_h, q_name_h)
        utils.save_chunk(name_file + '_queries_v', q_imgs_v, q_name_v)

    #utils.create_paris_hdf5(batch_size, size_h, mean_value, name_file_h, train_list_path_h)
    #utils.create_paris_hdf5(batch_size, size_v, mean_value, name_file_v, train_list_path_v)

    print 'Total time elapsed ', time.time() - t
