#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

# import os 
# import numpy as np
import torch
from pt_extract_features.utils_ml import FeatureExtractor, dim_reduce
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

# set paths   
image_path = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
featu_path = "./extracted_features"
# feature extraction
fe = FeatureExtractor(model_tag = "ResNet50")
fe.eval_nodes
fe.create("layer1.2.conv3")
fe.extract(image_path, freq_pool = 4, batch_size = 16, n_batches = 10)
# dim-reduction
X_red = dim_reduce(fe.X, n_neigh = 10, n_dims_red = 8)
print(fe.N.shape, fe.X.shape, X_red.shape)





