import cam_utils as cu
import numpy as np
import time
from sklearn.decomposition import PCA
import sys
import utils_datasets as ud

regions_for_pca = '/imatge/ajimenez/work/ITR/results_ITR/descriptors/vgg_imagenet/paris/regions/'
save_pca_path='/imatge/ajimenez/work/ITR/results_ITR/descriptors/vgg_imagenet/paris/regions/'
pca_name = 'Paris_PCA_1024x720_fusion_mp'
regions_name = '1024x720_fusion_mp'

pca_dim = 512
descriptors = ud.load_data(regions_for_pca+regions_name)

print descriptors.shape
t1 = time.time()
print 'Applying PCA with dimension reduction to: ', pca_dim
sys.stdout.flush()
pca = PCA(n_components=pca_dim, whiten=True)
pca.fit(descriptors)
print pca.components_.shape
ud.save_data(pca, save_pca_path, pca_name+'.h5')
print 'PCA finished!'
print 'Time elapsed computing PCA: ', time.time() - t1