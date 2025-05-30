#-------------------------
# Author : Serge Zaugg
# Description: reduce dim of long feature vectors via UMAP
#-------------------------

import os 
import numpy as np
import plotly.express as px
from pt_extract_features.utils_ml import dim_reduce

featu_path = "./extracted_features"

li = [
    "features_DenseNet121_denseblock3.npz",
    "features_MaxVit_T_blocks.3.npz",
    "features_ResNet50_layer1.npz",
    "features_ResNet50_layer2.npz",
    "features_ResNet50_layer3.npz",
    "features_vgg16_features.28.npz",
    ]

n_neigh = 10
n_dims_red = 8

for file_name_in in li:
    npzfile_full_path = os.path.join(featu_path, file_name_in)
    npzfile = np.load(npzfile_full_path)
    X = npzfile['X']
    N = npzfile['N']
    X_red = dim_reduce(X, n_neigh, n_dims_red)
    X_2D  = dim_reduce(X, n_neigh, 2)
    # X.shape
    # X_red.shape
    # X_2D.shape
    # N.shape
    # save as npz
    tag_dim_red = "dimred_" + str(n_dims_red) + "_"
    out_name = os.path.join(featu_path, tag_dim_red + file_name_in)
    np.savez(file = out_name, X_red = X_red, X_2D = X_2D, N = N)

