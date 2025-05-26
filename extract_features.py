#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

import os 
import numpy as np
import datetime
import torch
from utils_ml import ImageDataset, load_pretraind_model
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-------------------------
# set params  D:\image_clust\xc_spectrograms
# image_path = "D:/image_clust/xc_spectrograms_long"
image_path = "D:/image_clust/xc_spectrograms"
featu_path = "./extracted_features"
batch_size = 64

# model_tag = "ResNet50"
# model_tag = "DenseNet121"
model_tag = "MobileNet_V3_Large"
# model_tag = "Vit_b_16"
# model_tag = "vgg16"
# model_tag = 'MobileNet_randinit'

#-------------------------
# Step 1: Initialize model with pre-trained weights
model, weights = load_pretraind_model(model_tag)
# (32, 1000)

# model = torch.nn.Sequential(*(list(model.children())[:-2]))
# print(model)

#  "MobileNet_V3_Large"
model = torch.nn.Sequential(*(list(model.children())[:-2]))
print(model)





#-------------------------
# Step 2: Extract features 
model.eval()
preprocess = weights.transforms()
# preprocess = None
dataset = ImageDataset(image_path, preprocess)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  shuffle=False, drop_last=False)



# batch.dtype
# inp torch.Size([32, 3, 224, 224])
# inp torch.Size([32, 3, 128, 256])

X_li = [] # features
N_li = [] # file Nanes
for ii, (batch, finam) in enumerate(loader, 0):
    print('inp', batch.shape)
    # batch = batch.to(torch.float)
    prediction = model(batch).detach().numpy()  #.squeeze(0)
    file_names = np.array(finam)
    print('out', prediction.shape)
    print("")
    X_li.append(prediction)
    N_li.append(file_names)

X = np.concatenate(X_li)
N = np.concatenate(N_li)

# check dims
print(X.shape, N.shape)

# averag pool over time 
X = X.mean(axis=3)
print(X.shape)
# unwrap freq int feature dim
X = np.reshape(X, shape=(X.shape[0], X.shape[1]*X.shape[2]))
print(X.shape)



# save as npz
tstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
out_name = os.path.join(featu_path, tstmp + 'Feat_from_' + model_tag + '.npz')
np.savez(file = out_name, X = X, N = N)



