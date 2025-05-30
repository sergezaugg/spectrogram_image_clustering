#-------------------------
# Author : Serge Zaugg
# check actual range of extracted features 
#-------------------------

import os 
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split

featu_path = "./extracted_features"

li = ["features_DenseNet121_denseblock3.npz", "features_MaxVit_T_blocks.3.npz", "features_vgg16_features.28.npz",
      "features_ResNet50_layer1.npz", "features_ResNet50_layer2.npz", "features_ResNet50_layer3.npz",
    ]

for file_name_in in li:
    print(file_name_in)
    npzfile_full_path = os.path.join(featu_path, file_name_in)
    npzfile = np.load(npzfile_full_path)
    X = npzfile['X']
    # take only a few random samples to spped up plotly plots 
    XS, _ = train_test_split(X, train_size=300, random_state=6666, shuffle=True)
    XS.shape
    fig = px.scatter(data_frame = XS, title = file_name_in)
    fig.show()

        


