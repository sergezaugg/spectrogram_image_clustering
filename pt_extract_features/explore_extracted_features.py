#-------------------------
# check actual range of extracted features 

import os 
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split

npzfile_full_path = "extracted_features/20250527_190946_unwrapped_features_MaxVit_T.npz"
npzfile_full_path = "extracted_features/20250527_194433_unwrapped_features_DenseNet121.npz"
npzfile_full_path = "extracted_features/20250527_200518_unwrapped_features_ResNet50.npz"
npzfile_full_path = "extracted_features/20250527_202434_unwrapped_features_ResNet50layer2.3.conv3.npz"
# npzfile_full_path = "extracted_features/sssssssssss.npz"
# npzfile_full_path = "extracted_features/sssssssssss.npz"

npzfile = np.load(npzfile_full_path)
X = npzfile['X']
X.shape

XS, _ = train_test_split(X, train_size=300, random_state=6666, shuffle=True)
XS.shape

fig = px.scatter(data_frame = XS)
fig.show()






outli_score = np.abs(X).mean(1)
thld = np.quantile(outli_score, 0.99)
X.shape
sel = outli_score < thld
X[sel].shape


fig = px.scatter(x = outli_score)
fig.show()

