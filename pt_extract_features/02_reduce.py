#-------------------------
# Author : Serge Zaugg
# Description: Batch compute reduced dim from long feature vectors via UMAP
# Save as npz with several slots needed by the frontend 
#-------------------------

import os 
import numpy as np
from pt_extract_features.utils_ml import dim_reduce

# featu_path = "./extracted_features"
featu_path = "./extracted_features_parus"

# get list of all available full feature arrays
li = [a for a in os.listdir(featu_path) if 'full_features_' in a]
# n neighbors of UMAP currently fixed to 10 !!!!
n_neigh = 10
# loop over full feature array and several values of dimred 
for file_name_in in li:
    npzfile_full_path = os.path.join(featu_path, file_name_in)
    npzfile = np.load(npzfile_full_path)
    X = npzfile['X']
    N = npzfile['N']
    # make 2d feats needed for plot 
    X_2D  = dim_reduce(X, n_neigh, 2)
    for n_dims_red in [2,4,8,16]:
        X_red = dim_reduce(X, n_neigh, n_dims_red)
        print(X.shape, X_red.shape, X_2D.shape, N.shape)
        # save as npz
        tag_dim_red = "dimred_" + str(n_dims_red) + "_neigh_" + str(n_neigh) + "_"
        file_name_out = tag_dim_red + '_'.join(file_name_in.split('_')[4:])
        out_name = os.path.join(featu_path, file_name_out)
        np.savez(file = out_name, X_red = X_red, X_2D = X_2D, N = N)
