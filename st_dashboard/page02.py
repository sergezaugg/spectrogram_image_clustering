#--------------------             
# Author : Serge Zaugg
# Description : Main interactive streamlit page
#--------------------

import os
import streamlit as st
from streamlit import session_state as ss
import numpy as np
import pandas as pd
import gc
# streamlit need it like that:
from utils import dim_reduction_for_2D_plot, dim_reduction_for_clustering, perform_dbscan_clusterin, update_ss
from utils import make_sorted_df, make_scatter_plot, display_mini_images_by_file
# streamlit does not find the module !!!
# from st_dashboard.utils import dim_reduction_for_2D_plot, dim_reduction_for_clustering, perform_dbscan_clusterin, update_ss
# from st_dashboard.utils import make_sorted_df, make_scatter_plot, display_mini_images_by_file
gc.collect()

cols = st.columns([0.1, 0.35, 0.1, 0.35, 0.15])

# Handle start-up of app
if ss['dapar']['dataset_name'] == 'empty' :
    st.page_link("page03.py", label="Click to select a dataset")
else :
    if len(ss['dapar']['X']) <= 0:
        st.text("haha")
      
# main dashboard
if len(ss['dapar']['X']) > 0 :

    with cols[0]: 
        with st.container(border=True, height = 250):  
            st.text("Dimension") 
            st.info(str(ss['dapar']['X'].shape[0]) + '  imgs')
            st.info(str(ss['dapar']['X'].shape[1]) + '  feats') 
   
    with cols[1]:
        with st.container(border=True, height = 250):   
            _ = st.select_slider(label = "UMAP reduce dim", options=[2,4,8,16,32,64], disabled = ss['upar']['skip_umap'],
                                key = "k_UMAP_dim", value = ss['upar']["umap_n_dims_red"], on_change=update_ss, args=["k_UMAP_dim", "umap_n_dims_red"])
            _ = st.select_slider(label = "UMAP nb neighbors", options=[2,5,10,15,20,30,40,50,75,100], disabled = ss['upar']['skip_umap'], 
                            key = "k_UMAP_n_neigh", value=ss['upar']["umap_n_neighbors"], on_change=update_ss, args=["k_UMAP_n_neigh", "umap_n_neighbors"])   
    
    #-------------------------------------------
    # computational block 1 (st-cached)
    X2D_scaled = dim_reduction_for_2D_plot(X = ss['dapar']['X'], n_neigh = ss['upar']['umap_n_neighbors'], n_components = 2)
    X_scaled = dim_reduction_for_clustering(X = ss['dapar']['X'], n_neigh = ss['upar']['umap_n_neighbors'], n_dims_red = ss['upar']['umap_n_dims_red'], 
                                            skip_umap = ss['upar']['skip_umap'])
    gc.collect()
    #-------------------------------------------

    with cols[2]:
        with st.container(border=True, height = 250): 
            st.text("Dimension")  
            st.info(str(X_scaled.shape[0]) + '  imgs')
            st.info(str(X_scaled.shape[1]) + '  feats')

    with cols[3]:
        with st.container(border=True, height = 250): 
            eps_options = (10.0**(np.arange(-3.0, 1.01, 0.05))).round(3)
            _ = st.select_slider(label = "DBSCAN eps", options = eps_options, 
                                key = "k_dbscan_eps", value=ss['upar']["dbscan_eps"], on_change=update_ss, args=["k_dbscan_eps", "dbscan_eps"])
            _ = st.select_slider(label = "DBSCAN min samples", options=np.arange(5, 31, 5), 
                                key = "k_dbscan_min", value=ss['upar']["dbscan_min_samples"], on_change=update_ss, args=["k_dbscan_min", "dbscan_min_samples"])

    #-------------------------------------------
    # computational block 2 (st-cached)
    clusters_pred = perform_dbscan_clusterin(X = X_scaled, eps = ss['upar']['dbscan_eps'], min_samples = ss['upar']['dbscan_min_samples']) 
    num_unasigned = (clusters_pred == -1).sum()
    num_asigned = len(clusters_pred) - num_unasigned
    num_clusters = len(np.unique(clusters_pred))
    ss['dapar']['clusters_pred_str'] = np.array([format(a, '03d') for a in clusters_pred])
    df_pred = make_sorted_df(cat = ss['dapar']['clusters_pred_str'], cat_name = 'Predicted cluster', X = X2D_scaled)
    fig02 = make_scatter_plot(df = df_pred, cat_name = 'Predicted cluster', title = "Predicted clusters", height = 900, width = 1000, b_margin = 300)
    gc.collect()
    #-------------------------------------------


    

    # st.write('clusters_pred.shape', clusters_pred.shape)
    # st.write("X", ss['dapar']['X'].sum(1).shape)
    # aa = pd.DataFrame({'clusters_pred': clusters_pred, 'featusum': (np.abs((ss['dapar']['X']))).mean(1)})
    # bb = aa.groupby('clusters_pred').std().reset_index()
    # bb = bb.sort_values(by = 'featusum', ascending = False)
    # st.write(bb)


    with cols[4]:
        with st.container(border=True, height = 250): 
            st.text("Clustering metrics")
            coco = st.columns(2)
            coco[0].metric("N images assigned ", num_asigned)
            coco[0].metric("N clusters", num_clusters)
          
    # show plots 
    c01, c02 = st.columns([0.4, 0.5])
    with c01:
        st.plotly_chart(fig02, use_container_width=False, theme=None)  
    with c02:
        clu_id_list = np.unique(ss['dapar']['clusters_pred_str'])
        clu_selected = st.segmented_control(label = "Select a cluster ID", options = clu_id_list, selection_mode="single", key = "k_img_clu",
                                        default = clu_id_list[-1], label_visibility="hidden")        
        
        st.text("Cluster content preview (up to 60 random images from cluster)")
    
        # select all images in a given cluster 
        sel = ss['dapar']['clusters_pred_str'] == clu_selected
        images_in_cluster = ss['dapar']['im_filenames'][sel]
        # take a smaller subsample 
        rand_index = np.random.choice(np.arange(len(images_in_cluster)), size=min(120, len(images_in_cluster)), replace=False)    
        images_in_cluster_sample = images_in_cluster[rand_index]
        display_mini_images_by_file(sel_imgs = images_in_cluster_sample)


gc.collect()

        
     
        
  

   


