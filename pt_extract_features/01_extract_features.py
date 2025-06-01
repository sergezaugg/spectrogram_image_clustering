#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

import os 
import numpy as np
import datetime
import torch
import skimage.measure
from utils_ml import FeatureExtractor
from pt_extract_features.utils_ml import ImageDataset

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-------------------------
# set params  
image_path = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
featu_path = "./extracted_features"
batch_size = 16





#-------------------------
# Step 1: Initialize model with pre-trained weights


fe = FeatureExtractor(model_tag = "ResNet50")
fe.eval_nodes
fe.create("layer3.5.conv3")
freq_pool = 4

fe = FeatureExtractor(model_tag = "ResNet50")
fe.eval_nodes
fe.create("layer2.3.conv3")
freq_pool = 4

fe = FeatureExtractor(model_tag = "ResNet50")
fe.eval_nodes
fe.create("layer1.2.conv3")
freq_pool = 4


fe = FeatureExtractor(model_tag = "vgg16")
fe.eval_nodes
fe.create("features.28")
freq_pool = 4



fe = FeatureExtractor(model_tag = "DenseNet121")
fe.eval_nodes
fe.create("features.denseblock3")
freq_pool = 4


fe = FeatureExtractor(model_tag = "MaxVit_T")
fe.eval_nodes
fe.create("blocks.3.layers.1.layers.MBconv.layers.conv_c")
freq_pool = 1






#-------------------------
#-------------------------
#-------------------------
#-------------------------
#-------------------------





freq_pool = 4


dataset = ImageDataset(image_path, fe.preprocessor)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  shuffle=False, drop_last=False)

X_li = [] # features
N_li = [] # file Nanes
for ii, (batch, finam) in enumerate(loader, 0):
    print('Model:', fe.model_tag )
    print('Feature layer:', fe.fex_tag )
    print('Input resized image:', batch.shape)
    # batch = batch.to(torch.float)
    pred = fe.extractor(batch)['feature_1'].detach().numpy() 
    print('Feature out of net:', pred.shape)
    # blockwise pooling along frequency axe 
    pred = skimage.measure.block_reduce(pred, (1,1,freq_pool,1), np.mean)
    print('After average pool along freq:', pred.shape)
    # full average pool over time (do asap to avoid memory issues later)
    pred = pred.mean(axis=3)
    print('After average pool along time:', pred.shape)
    # unwrap freq int feature dim
    pred = np.reshape(pred, shape=(pred.shape[0], pred.shape[1]*pred.shape[2]))
    print('After reshape:', pred.shape)
    print("")
    # do it dirty
    X_li.append(pred)
    N_li.append(np.array(finam))

    # dev
    if ii > 800:
        break

X = np.concatenate(X_li)
N = np.concatenate(N_li)

# check dims
print(X.shape, N.shape)


# save as npz
tstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")

# save as npz
out_name = os.path.join(featu_path, tstmp + 'features_' + fe.model_tag + fe.fex_tag + '.npz')
np.savez(file = out_name, X = X, N = N)



