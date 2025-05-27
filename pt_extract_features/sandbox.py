#-------------------------
# check actual range of extracted features 

import os 
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split

npzfile_full_path = "extracted_features/20250527_110409_unwrapped_features_MobileNet_V3_Large.npz"
npzfile_full_path = "extracted_features/20250527_123259_unwrapped_features_vgg16.npz"
npzfile_full_path = "extracted_features/20250527_125947_unwrapped_features_ResNet50.npz"

npzfile = np.load(npzfile_full_path)
X = npzfile['X']
X.shape

XS, _ = train_test_split(X, train_size=300, random_state=6666, shuffle=True)
XS.shape

fig = px.line(data_frame = XS)
fig.show()
