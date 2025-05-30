#-------------------------
# Author : Serge Zaugg
# Description: Interactive script to visually explore distribution of feature vectors 
# 
#-------------------------

import os 
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split

featu_path = "./extracted_features"

# explore full-sized feature vectors 
li = [
    "features_DenseNet121_denseblock3.npz", "features_MaxVit_T_blocks.3.npz", "features_vgg16_features.28.npz",
    "features_ResNet50_layer1.npz", "features_ResNet50_layer2.npz", "features_ResNet50_layer3.npz",
    ]

for file_name_in in li:
    npzfile = np.load(os.path.join(featu_path, file_name_in))
    X = npzfile['X']
    XS, _ = train_test_split(X, train_size=200, random_state=6666, shuffle=True)
    px.scatter(data_frame = XS.T, title = file_name_in).show()

        
# explore dim-reduced features 
for file_name_in in [a for a in os.listdir(featu_path) if 'dimred_' in a]:
    npzfile = np.load(os.path.join(featu_path, file_name_in))
    X = npzfile['X_red']
    XS, _ = train_test_split(X, train_size=200, random_state=6666, shuffle=True)
    px.scatter(data_frame = XS.T, title = file_name_in).show()
    

