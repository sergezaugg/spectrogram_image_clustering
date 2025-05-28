#-------------------------
# check actual range of extracted features 

import os 
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from pt_extract_features.utils_ml import dim_reduce

featu_path = "./extracted_features"

file_name_in = "features_DenseNet121_denseblock3.npz"
# file_name_in = "features_MaxVit_T_blocks.3.npz"
# file_name_in = "features_ResNet50_layer1.npz"
# file_name_in = "features_ResNet50_layer2.npz"
# file_name_in = "features_ResNet50_layer3.npz"

npzfile_full_path = os.path.join(featu_path, file_name_in)

npzfile = np.load(npzfile_full_path)
X = npzfile['X']
N = npzfile['N']

n_neigh = 10
n_dims_red = 16
X_red = dim_reduce(X, n_neigh, n_dims_red)
X_2D  = dim_reduce(X, n_neigh, 2)

X.shape
X_red.shape
X_2D.shape
N.shape

tag_dim_red = "dimred_" + str(n_dims_red) + "_"

# save as npz
out_name = os.path.join(featu_path, tag_dim_red + file_name_in)
np.savez(file = out_name, X_red = X_red, X_2D = X_2D, N = N)







# XS, _ = train_test_split(X, train_size=300, random_state=6666, shuffle=True)
# XS.shape

# fig = px.scatter(data_frame = XS)
# fig.show()





# 
# outli_score = np.abs(X).mean(1)
# thld = np.quantile(outli_score, 0.99)
# X.shape
# sel = outli_score < thld
# X[sel].shape

# fig = px.scatter(x = outli_score)
# fig.show()

