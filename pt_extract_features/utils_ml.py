#--------------------             
# Author : Serge Zaugg
# Description : Functions an classes specific to ML/PyTorch backend
#--------------------

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import decode_image
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

class ImageDataset(Dataset):
    """
    Description: A simple PyTorch dataset (loader) to batch process images from file
    """
    def __init__(self, imgpath, preprocess):
        """
        imgpath (str) : path to a dir that contains JPG images. 
        label_path (str) : path to a csv file which matches PNG filenames with labels
        preprocess ('torchvision.transforms._presets.ImageClassification'>) : preprocessing transforms provided with the pretrained models
        """
        self.all_img_files = np.array([a for a in os.listdir(imgpath) if '.png' in a])
        self.imgpath = imgpath   
        self.preprocess = preprocess  
   

    def __getitem__(self, index):     
        img = decode_image(os.path.join(self.imgpath,  self.all_img_files[index]))  
        # Apply inference preprocessing transforms
        if self.preprocess is not None:
            img = self.preprocess(img) # .unsqueeze(0)

        filename = self.all_img_files[index]
        return (img, filename)
    
    def __len__(self):
        return (len(self.all_img_files))
    

def load_pretraind_model(model_tag):
    """
    """
    if model_tag == "ResNet50":
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
    elif model_tag == "DenseNet121":
        from torchvision.models import densenet121, DenseNet121_Weights 
        weights = DenseNet121_Weights.IMAGENET1K_V1
        model = densenet121(weights=weights)
    elif model_tag == "MobileNet_V3_Large":
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = mobilenet_v3_large(weights=weights)
    elif model_tag == "MobileNet_randinit":
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = mobilenet_v3_large(weights=None)
    elif model_tag == "vgg16":
        from torchvision.models import vgg16, VGG16_Weights
        weights = VGG16_Weights.IMAGENET1K_V1
        model = vgg16(weights=weights)
    # Transformers 
    elif model_tag == "Vit_b_16":
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        model = vit_b_16(weights=weights)
    elif model_tag == "MaxVit_T":
        from torchvision.models import maxvit_t, MaxVit_T_Weights  
        weights = MaxVit_T_Weights.IMAGENET1K_V1     
        model = maxvit_t(weights=weights)  
    elif model_tag == "Swin_S":
        from torchvision.models import swin_s, Swin_S_Weights
        weights = Swin_S_Weights.IMAGENET1K_V1
        model = swin_s(weights=weights)
    else:
        print("not a valid model_tag")
    return(model, weights)    


def dim_reduce(X, n_neigh, n_dims_red):
    """
    UMAP dim reduction for clustering
    """
    scaler = StandardScaler()
    reducer = umap.UMAP(
        n_neighbors = n_neigh, 
        n_components = n_dims_red, 
        metric = 'euclidean',
        n_jobs = -1
        )
    X_scaled = scaler.fit_transform(X)
    X_trans = reducer.fit_transform(X_scaled)
    X_out = scaler.fit_transform(X_trans)
    return(X_out)