#--------------------             
# Author : Serge Zaugg
# Description : Info text
#--------------------

import streamlit as st
from streamlit import session_state as ss
import gc
import pandas as pd
gc.collect()

c00, c01  = st.columns([0.20, 0.20])

with c00:
    with st.container(border=True) :  
        st.markdown(''' 

        ### Data and audio pre-processing
        * Spectrograms obtained from recordings from [xeno-canto](https://xeno-canto.org/).
        * Standardized preparation of spectrogram images done with [this tool](https://github.com/sergezaugg/xeno_canto_organizer).   
        * MP3 recording were converted to WAVE and resampled to 24'000 sample per second.
        * Short segments of 0.4 seconds where transformed to mini-spectrograms of 128 frequency by 256 time bins.             
        * These 25157 mini-spectrograms stored as RGB images. 
        * A smaller random subsample of N=10'000 images used in dashboard to avoid memory issues.   
        * Extracted features and images are [here](https://www.kaggle.com/datasets/sezaugg/spectrogram-clustering-01)     

        ### Feature extraction 
        * Spectrogram converted to color images and fed into pre-trained image DNNs.            
        * The output of internal convolutional layers was used for feature extraction.
        * 3D arrays converted to linear feature vectors (Details in the right panel).  
        * Feature extraction was performed offline on a GPU machine.    
                 
        ### Dimensionality reduction   
        * Uniform Manifold Approximation and Projection ([UMAP](https://umap-learn.readthedocs.io)) was used for non-linear dimension reduction. 
        * UMAP is intense and kills the app on my free-low-resource deployment platform, thus it was also pre-computed.  
        * Based on recommendations from [Best et al. 2023](https://doi.org/10.1371/journal.pone.0283396), we provide dim reduced data of 4, 8 and 16 dims.               
        * UMAP is a [stochastic algorithm](https://umap-learn.readthedocs.io/en/latest/reproducibility.html) -> expect small differences between runs! 

        ### Data storage   
        * Extracted feature are stored on a dedicated [Kaggle dataset](https://www.kaggle.com/datasets/sezaugg/spectrogram-clustering-01)           

        ### Clustering
        * Clustering can be done interactively with [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).
        * Samples not assigned to a cluster by DBSCAN are given the cluster ID '-01'
        * The numerical value of Cluster IDs is arbitrary

        ''')

        st.subheader("Related reading")

        st.markdown('''Applications of clustering to acoustic data''')

        st.page_link("https://doi.org/10.1371/journal.pone.0283396", label=":red[P. Best et al.]")

        st.page_link("https://doi.org/10.1016/j.aiig.2021.12.002", label=":red[Q. Kong et al.]")

        st.page_link("https://doi.org/10.3389/frsen.2024.1429227", label=":red[A. Noble et al.]")

        st.markdown('''Dive into the methods''')

        st.page_link("https://arxiv.org/abs/1802.03426", label=":red[McInnes et al. : UMAP]")

        st.page_link("https://cdn.aaai.org/KDD/1996/KDD96-037.pdf", label=":red[Ester et al : DBSCAN]")

        st.page_link("https://doi.org/10.1145/3068335", label=":red[Schubert et al : DBSCAN ]")



with c01:

        
        
    with st.container(border=True) : 
        
        st.markdown(''' 

            # Details on extraction from image DNNs

            To date, I assessed 3 CNNs (DenseNet121, vgg16, ResNet50) and one vision transformer (MaxVit_T). 
            Features were extracted a 3 different depth from ResNet50.
            In all cases the output array from convolutional layer was used (i.e. before the non-linear transfer function).
            Block-wise pooling was used on frequency to keep information of frequency location of patterns.
            Current values chosen such that unwrapped (reshaped) vector is not longer than 5000.
            Full pooling over time was used to make final features invariant to time position in time.
                            
            ''')        

        st.image(image = "pics/DNN_data_flow_chart_01.png", 
                 caption="Schematics of how linear feature vectors were obtained from internal conv layers, in this example *layer2.3.conv3* of *ResNet50*)", 
                 width=650, 
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)

        st.divider()

        st.markdown('''

            Array dims below: (, channel, frequency, time)
                                           
            #### DenseNet121_denseblock3.npz
            * Feature layer: :red[features.denseblock3]
            * Input resized image: :red[(, 3, 224, 224)]
            * Feature out of net: :red[(, 1024, 14, 14)]
            * After average pool along freq: :red[(, 1024, 4, 14)]
            * After average pool along time: :red[(, 1024, 4)]
            * After reshape: :red[(, 4096)]
                        
            #### MaxVit_T_blocks.3.npz
            * Feature layer: :red[blocks.3.layers.1.layers.MBconv.layers.conv_c]
            * Input resized image:  :red[( , 3, 224, 224)]
            * Feature out of net:  :red[( , 512, 7, 7)]
            * After average pool along freq:  :red[( , 512, 7, 7)]
            * After average pool along time:  :red[( , 512, 7)]
            * After reshape:  :red[( , 3584)]
                    
            #### Model: vgg16_features.28.npz
            * Feature layer: :red[features.28]
            * Input resized image: :red[(, 3, 224, 224)]
            * Feature out of net: :red[(, 512, 14, 14)]
            * After average pool along freq: :red[(, 512, 4, 14)]
            * After average pool along time: :red[(, 512, 4)]
            * After reshape: :red[(, 2048)]

            #### ResNet50_layer1.npz
            * Feature layer: :red[layer1.2.conv3]
            * Input resized image: :red[(, 3, 224, 224)]
            * Feature out of net: :red[(, 256, 56, 56)]
            * After average pool along freq: :red[(, 256, 14, 56)]
            * After average pool along time: :red[(, 256, 14)]
            * After reshape: :red[(, 3584)]
                    
            #### ResNet50_layer2.npz
            * Feature layer: :red[layer2.3.conv3]
            * Input resized image: :red[(, 3, 224, 224)]
            * Feature out of net: :red[(, 512, 28, 28)]
            * After average pool along freq: :red[(, 512, 7, 28)]
            * After average pool along time: :red[(, 512, 7)]
            * After reshape: :red[(, 3584)]
                            
            #### ResNet50_layer3.npz
            * Feature layer: :red[layer3.5.conv3] 
            * Input resized image: :red[(, 3, 224, 224)] 
            * Feature out of net: :red[(, 1024, 14, 14)] 
            * After average pool along freq: :red[(, 1024, 4, 14)] 
            * After average pool along time: :red[(, 1024, 4)] 
            * After reshape: :red[(, 4096)]       
            ''')            
