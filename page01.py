#--------------------             
# Author : Serge Zaugg
# Description : Info text
#--------------------

import streamlit as st
from streamlit import session_state as ss

c00, c01  = st.columns([0.1, 0.18])

with c00:
    with st.container(border=True) : 
        st.image(image = "pics/data_flow_chart_2.png", caption="Data flow chart", width=None, 
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)
         
with c01:
    with st.container(border=True) : 
        
        st.markdown(''' 

            ### Data
            * Image data come from [Food Classification dataset](https://www.kaggle.com/datasets/bjoernjostein/food-classification) published on Kaggle 
            * Original dataset is [here](https://www.aicrowd.com/challenges/kiit-ai-mini-blitz/problems/foodch)        
            * Over 9300 hand-annotated images with 61 classes
            * Features were pre-extracted on a GPU machine and stored [here](https://www.kaggle.com/datasets/sezaugg/food-classification-features-v01)      
            * Smaller random subsample of N=3000 images used in dashboard to avoid memory issues.        

            ### Feature extraction (image to vector)
            * Features extracted with image classification models pre-trained with the Imagenet datataset
            * Details see official [PyTorch documentation](https://docs.pytorch.org/vision/main/models.html)
            * As output we used the last linear layer which outputs 1000 continuous features (ommited Softamx) 
            * These models were trained specifically for the Imagenet classes, so let's hope the feature are informative for our task
            * As a baseline, one dataset with random.noise features is provided        

            ### Dimensionality reduction
            * We used Uniform Manifold Approximation and Projection ([UMAP](https://umap-learn.readthedocs.io)), a technique for general non-linear dimension reduction.  
            * UMAP is a [stochastic algorithm](https://umap-learn.readthedocs.io/en/latest/reproducibility.html) -> expect small differences between runs!     

            ### Clustering
            * Clustering done with [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).
            * Here the focus is on clustering and ground truth classes are not used during training
            * Ground truth classes are only used to assess the quality of clustering
            * Samples not assigned to a cluster by DBSCAN are given the cluster ID '-01'
            * The numerical value of Cluster IDs is arbitrary
                    
            ### Clustering metrics
            * Adjusted Rand score is a consensus measures between true classes and predicted clusters, values in [-0.5, 1]
            * Adjusted mutual info score (AMI) is a consensus measures between true classes and predicted clusters, values in [~0, 1]
            * More info on clustering metrics [Vinh, Epps, Bailey (2010)](https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf)
                    
            ''')
        
   
 

      