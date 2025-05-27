#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

import os 
import numpy as np
import datetime
import torch
import skimage.measure
from sklearn.decomposition import PCA
from torchvision.models.feature_extraction import create_feature_extractor
from pt_extract_features.utils_ml import ImageDataset, load_pretraind_model
from torchvision.models.feature_extraction import get_graph_node_names

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-------------------------
# set params  
image_path = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
featu_path = "./extracted_features"
batch_size = 16


# model_tag = "Vit_b_16"
# model_tag = "MaxVit_T"
# model_tag = "Swin_S"


#-------------------------
# Step 1: Initialize model with pre-trained weights

# model_tag = "MobileNet_V3_Large"
# model, weights = load_pretraind_model(model_tag)
# model = torch.nn.Sequential(*(list(model.children())[:-2]))
# freq_pool = 2

# model_tag = "vgg16"
# model, weights = load_pretraind_model(model_tag)
# model = torch.nn.Sequential(*(list(model.children())[:-2]))
# freq_pool = 1



# # model = torch.nn.Sequential(*(list(model.children())[:-3]))
# model_tag = "ResNet50"
# model, weights = load_pretraind_model(model_tag)
# freq_pool = 4
# return_nodes = {"layer3.5.conv3": "feature_1"}
# model = create_feature_extractor(model, return_nodes=return_nodes)

# # model = torch.nn.Sequential(*(list(model.children())[:-3]))
# model_tag = "DenseNet121"
# model, weights = load_pretraind_model(model_tag)
# freq_pool = 4
# return_nodes = {"features.denseblock3": "feature_1"}
# model = create_feature_extractor(model, return_nodes=return_nodes)


model_tag = "MaxVit_T"
model, weights = load_pretraind_model(model_tag)
train_nodes, eval_nodes = get_graph_node_names(model)
return_nodes = {"blocks.3.layers.1.layers.MBconv.layers.conv_c": "feature_1"}
model = create_feature_extractor(model, return_nodes=return_nodes)




#-------------------------
# Step 2: Extract features 
_ = model.eval()
preprocess = weights.transforms()
dataset = ImageDataset(image_path, preprocess)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  shuffle=False, drop_last=False)

X_li = [] # features
N_li = [] # file Nanes
for ii, (batch, finam) in enumerate(loader, 0):
    print('inp', batch.shape)
    # batch = batch.to(torch.float)
    pred = model(batch)['feature_1'].detach().numpy() 
    print('out of net', pred.shape)
    # blockwise pooling along frequency axe 
    pred = skimage.measure.block_reduce(pred, (1,1,freq_pool,1), np.mean)
    # full average pool over time (do asap to avoid memory issues later)
    pred = pred.mean(axis=3)
    # unwrap freq int feature dim
    pred = np.reshape(pred, shape=(pred.shape[0], pred.shape[1]*pred.shape[2]))
    print('pooled-reshaped', pred.shape)
    print("")
    # do it dirty
    X_li.append(pred)
    N_li.append(np.array(finam))

    # dev
    if ii > 400:
        break

X = np.concatenate(X_li)
N = np.concatenate(N_li)

# check dims
print(X.shape, N.shape)


# save as npz
tstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")

# save as npz
out_name = os.path.join(featu_path, tstmp + 'unwrapped_features_' + model_tag + '.npz')
np.savez(file = out_name, X = X, N = N)

# # further reduce dim with pca 
# pca = PCA(n_components=1024)
# pca.fit(X)
# print(pca.explained_variance_ratio_.sum())
# X_red = pca.transform(X)
# X_red.shape

# # save as npz
# out_name = os.path.join(featu_path, tstmp + 'unwrapped_feat_pca_' + model_tag + '.npz')
# np.savez(file = out_name, X = X_red, N = N)


