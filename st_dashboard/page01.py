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
          
            ### Clustering

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
        
   
 

      