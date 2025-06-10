#--------------------             
# Author : Serge Zaugg
# Description : Info text
#--------------------

import streamlit as st
from streamlit import session_state as ss
import gc
import pandas as pd
gc.collect()

c00, c01 = st.columns([0.20, 0.20])

with c00:
    with st.container(border=True, height=370) :  
        st.markdown(''' 
        ### Audio pre-processing, feature extraction, and dim reduction  
        * Spectrograms obtained with recordings from [xeno-canto](https://xeno-canto.org/).
        * Standardized acoustic data preparation performed with [this tool](https://github.com/sergezaugg/xeno_canto_organizer).   
        * MP3 converted to WAVE, resampled to 24'000 sps and sequentially chunked to pieces of 0.4 seconds . 
        * Pieces where transformed to spectrograms of 128 frequency by 256 time bins (Nyq. freq = 12 KHz). 
        * RGB spectrograms were fed into pre-trained image DNNs or encoders (Details below).            
        * Arrays returned by internal layers converted to linear feature vectors (Details below).  
        * [UMAP](https://umap-learn.readthedocs.io) was used for non-linear dimension reduction (Details below). 
        * Feature extraction and dim-reduction were pre-computed on a GPU machine.
        * RGB spectrograms and features stored on a dedicated Kaggle dataset [1](https://www.kaggle.com/datasets/sezaugg/spectrogram-clustering-01) and
                    [2](https://www.kaggle.com/datasets/sezaugg/spectrogram-clustering-parus-major)             
        ''')
with c01:
    with st.container(border=True, height=370) :  
        st.markdown(''' 
        ### Unsupervised clustering
        * Clustering can be done interactively with [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).
        * Samples not assigned to a cluster by DBSCAN are given the cluster ID '-01'
        * The numerical value of Cluster IDs is arbitrary
        ''')
        st.subheader("Related reading")
        st.markdown(''' 
            * Applications of clustering to acoustic data: 
            [Best et al.](https://doi.org/10.1371/journal.pone.0283396); 
            [Kong et al.](https://doi.org/10.1016/j.aiig.2021.12.002); 
            [Noble et al.](https://doi.org/10.3389/frsen.2024.1429227)     
            * Dive into the methods: 
            [McInnes et al. : UMAP](https://arxiv.org/abs/1802.03426); 
            [Ester et al : DBSCAN](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf); 
            [Schubert et al : DBSCAN](https://doi.org/10.1145/3068335)      
            ''')
 

c00, c01 = st.columns([0.20, 0.20])
with c00:
    with st.container(border=True) : 
        st.markdown('''### Feature extraction from image DNNs ''')    
        st.markdown('''For detail of feature extraction from images DNNs see [this github project](https://github.com/sergezaugg/spectro_image_feature_extract)''')        
        st.image(image = "pics/spectro_imDNN_data_flow.png", 
                 caption="Schematics of how linear feature vectors were obtained from internal conv layers, in this example *layer2.3.conv3* of *ResNet50*)", 
                 width=580, 
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)
with c01:   
    with st.container(border=True) : 
        st.markdown('''### Feature extraction from Spectrogram Auto-Encoders''')  
        st.markdown('''For detail of feature extraction with Spectrogram Auto-Encoders see [this github project](https://github.com/sergezaugg/spectro_aec_clust)''')    
        st.image(image = "pics/spectro_SAEC_data_flow.png", 
                 caption="Schematics of how linear feature vectors were obtained from Spectrogram auto-Encoders", 
                 width=580, 
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)





