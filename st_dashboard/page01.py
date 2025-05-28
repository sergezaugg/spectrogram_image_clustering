#--------------------             
# Author : Serge Zaugg
# Description : Info text
#--------------------

import streamlit as st
from streamlit import session_state as ss
import gc
import pandas as pd
gc.collect()

c00, c01  = st.columns([0.15, 0.20])

with c00:
    with st.container(border=True) :  
           st.markdown(''' 

             ### Data
            * Spectrograms obtained from recordings from [xeno-canto](https://xeno-canto.org/).
            * Over 27000 mini spectrograms stored as RGB images. 
            * A smaller random subsample of N=10000 images used in dashboard to avoid memory issues.   
            * Extracted features and images are [here](https://www.kaggle.com/datasets/sezaugg/spectrogram-clustering-01)     
         
            ### Feature extraction 
            * Feature extraction was performed offline on a GPU machine.
            * Spectrogram converted to color images            
            * The output of convolutional layers of pre-trained DNNs was used for feature extraction.
            * The DNN output was an array of dim (batch, channel, frequency, time).
            * Post process:
              * Frequency dim was reduced by blocks with average pooling.
              * Time dim was reduced by global average pooling.  
              * The remaining frequency dims were wrapped into the channel dim.                  
              * Details are given in the right panel. 
            * Extracted feature are stored on a dedicated [Kaggle dataset](https://www.kaggle.com/datasets/sezaugg/spectrogram-clustering-01)          

            ### Dimensionality reduction
            * UMAP is slow and was pre-computed.           
            * We used Uniform Manifold Approximation and Projection ([UMAP](https://umap-learn.readthedocs.io)), a technique for general non-linear dimension reduction.  
            * UMAP is a [stochastic algorithm](https://umap-learn.readthedocs.io/en/latest/reproducibility.html) -> expect small differences between runs!     

            ### Clustering
            * Clustering done with [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).
            * Samples not assigned to a cluster by DBSCAN are given the cluster ID '-01'
            * The numerical value of Cluster IDs is arbitrary

            ''')  

with c01:
    with st.container(border=True) : 

        st.markdown(''' 
                 
            #### features_ResNet50_layer3.npz
            * Feature layer: :orange[layer3.5.conv3] 
            * Input resized image: :orange[(, 3, 224, 224)] 
            * Feature out of net: :orange[(, 1024, 14, 14)] 
            * After average pool along freq: :orange[(, 1024, 4, 14)] 
            * After average pool along time: :orange[(, 1024, 4)] 
            * After reshape: :orange[(, 4096)] 
                    
            #### features_ResNet50_layer2.npz
            * Feature layer: :orange[layer2.3.conv3]
            * Input resized image: :orange[(, 3, 224, 224)]
            * Feature out of net: :orange[(, 512, 28, 28)]
            * After average pool along freq: :orange[(, 512, 7, 28)]
            * After average pool along time: :orange[(, 512, 7)]
            * After reshape: :orange[(, 3584)]
                    
            #### features_ResNet50_layer1.npz
            * Feature layer: :orange[layer1.2.conv3]
            * Input resized image: :orange[(16, 3, 224, 224)]
            * Feature out of net: :orange[(16, 256, 56, 56)]
            * After average pool along freq: :orange[(16, 256, 14, 56)]
            * After average pool along time: :orange[(16, 256, 14)]
            * After reshape: :orange[(16, 3584)]
      
            #### features_DenseNet121_denseblock3.npz
            * Feature layer: :orange[features.denseblock3]
            * Input resized image: :orange[(, 3, 224, 224)]
            * Feature out of net: :orange[(, 1024, 14, 14)]
            * After average pool along freq: :orange[(, 1024, 4, 14)]
            * After average pool along time: :orange[(, 1024, 4)]
            * After reshape: :orange[(, 4096)]
                        
            #### features_MaxVit_T_blocks.3.npz
            * Feature layer: :orange[blocks.3.layers.1.layers.MBconv.layers.conv_c]
            * Input resized image:  :orange[( , 3, 224, 224)]
            * Feature out of net:  :orange[( , 512, 7, 7)]
            * After average pool along freq:  :orange[( , 512, 7, 7)]
            * After average pool along time:  :orange[( , 512, 7)]
            * After reshape:  :orange[( , 3584)]


           
            ''')
        
   
 

      