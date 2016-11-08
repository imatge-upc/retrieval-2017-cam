import numpy as np
import h5py
import sys
import utils_oxford


image_train_list_path = "/imatge/ajimenez/workspace/ITR/code/list_all.txt"

img_width = 224
img_height = 224

# mean_value = [123.68, 116.779, 103.939]
mean_value = [0,0,0]

# Load Data
images, num_images, image_names = utils_oxford.load_data_oxford(image_train_list_path)

image_names = np.array(image_names)

# Pre-process images
x = utils_oxford.preprocess_images(images, num_images, img_width, img_height, mean_value)

print num_images
print x.shape
print image_names
sys.stdout.flush()

labels = utils_oxford.label_oxford(image_names)

print labels.shape

for lab in labels:
	print lab

utils_oxford.save_oxford('/imatge/ajimenez/workspace/ITR/datasets_hfd5/oxford_224/', x, labels, image_names)




