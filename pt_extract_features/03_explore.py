#-------------------------
# Author : Serge Zaugg
# Description: Interactive script to visually explore distribution of feature vectors 
#-------------------------

import os 
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split

featu_path = "./extracted_features"

# explore full-sized feature vectors 
for file_name_in in [a for a in os.listdir(featu_path) if 'full_features_' in a]:
    npzfile = np.load(os.path.join(featu_path, file_name_in))
    X = npzfile['X']
    print(X.shape)
    XS, _ = train_test_split(X, train_size=150, random_state=6666, shuffle=True)
    px.scatter(data_frame = XS.T, title = file_name_in).show()
    
# explore dim-reduced features 
for file_name_in in [a for a in os.listdir(featu_path) if 'dimred_' in a]:
    npzfile = np.load(os.path.join(featu_path, file_name_in))
    X = npzfile['X_red']
    print(X.shape)
    XS, _ = train_test_split(X, train_size=150, random_state=6666, shuffle=True)
    px.scatter(data_frame = XS.T, title = file_name_in).show()
    

