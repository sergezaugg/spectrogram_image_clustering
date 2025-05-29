#--------------------             
# Author : Serge Zaugg
# Description : Info text
#--------------------

import streamlit as st
from streamlit import session_state as ss
import gc
import pandas as pd
gc.collect()

c00, c01  = st.columns([0.20, 0.10])

with c00:
    with st.container(border=True) :  
        st.markdown(''' 

        ### Data
        * Spectrograms obtained from recordings from [xeno-canto](https://xeno-canto.org/).
        * Standardized preparation of spectrogram images done with [this tool](https://github.com/sergezaugg/xeno_canto_organizer).            
        * These 25157 mini spectrograms stored as RGB images. 
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

        st.subheader("Related reading")

        st.markdown('''A few related applications''')

        st.page_link("https://doi.org/10.1016/j.aiig.2021.12.002", label=":blue[Q. Kong et al. : Deep convolutional autoencoders as generic feature extractors in seismological applications]")

        st.page_link("https://doi.org/10.1371/journal.pone.0283396", label=":blue[P. Best et al. : Deep audio embeddings for vocalisation clustering]")

        st.page_link("https://doi.org/10.3389/frsen.2024.1429227", label=":blue[A. Noble et al. : Unsupervised clustering reveals acoustic diversity and niche differentiation in pulsed calls from a coral reef ecosystem]")

        st.markdown('''Dive into the methods''')

        st.page_link("https://arxiv.org/abs/1802.03426", label=":blue[McInnes, Healy,Melville : UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction]")

        st.page_link("https://cdn.aaai.org/KDD/1996/KDD96-037.pdf", label=":blue[Ester et al : A density-based algorithm for discovering clusters in large spatial databases with noise]")

        st.page_link("https://doi.org/10.1145/3068335", label=":blue[Schubert et al : DBSCAN revisited, revisited: why and how you should (still) use DBSCAN]")



   


      