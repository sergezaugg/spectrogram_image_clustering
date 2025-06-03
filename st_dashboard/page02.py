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
from utils import perform_dbscan_clusterin, update_ss, display_bar_plot, select_random_image_subset
from utils import make_sorted_df, make_scatter_plot, display_mini_images_by_file
# streamlit does not find the module !!!
# from st_dashboard.utils import dim_reduction_for_2D_plot, dim_reduction_for_clustering, perform_dbscan_clusterin, update_ss
# from st_dashboard.utils import make_sorted_df, make_scatter_plot, display_mini_images_by_file
gc.collect()


# Handle start-up of app
if ss['dapar']['dataset_name'] == 'empty' :
    st.page_link("page03.py", label="Click to select a dataset")
else :
    if len(ss['dapar']['X_dimred']) <= 0:
        st.text("haha")
      
# main dashboard
if len(ss['dapar']['X_dimred']) > 0 :

    cols = st.columns([0.30, 0.35, 0.15])

    with cols[0]:
        c01, c02 = st.columns([0.2, 0.6])    
        with c01:
            with st.container(border=True, height = 275): 
                st.text("Input to DBSCAN")  
                txt01 = ':red-background[' + str(ss['dapar']['X_dimred'].shape[0]) + ' images' + ']'
                txt02 = ':red-background[' + str(ss['dapar']['X_dimred'].shape[1]) + ' features' + ']'
                st.markdown(txt01)
                st.markdown(txt02)

        with c02:
            with st.container(border=True, height = 275): 
                # eps_options = (10.0**(np.arange(-3.0, 0.50, 0.05))).round(3)
                eps_options = np.concatenate([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 2.1, 0.02)]).round(2)
                min_samples_options = np.arange(10, 51, 5)
                # # callback version (too fast for streamlit.app ?)
                # _ = st.select_slider(label = "DBSCAN eps", options = eps_options, 
                #     key = "k_dbscan_eps", value=ss['upar']["dbscan_eps"], on_change=update_ss, args=["k_dbscan_eps", "dbscan_eps"])
                # _ = st.select_slider(label = "DBSCAN min samples", options = min_samples_options, 
                #     key = "k_dbscan_min", value=ss['upar']["dbscan_min_samples"], on_change=update_ss, args=["k_dbscan_min", "dbscan_min_samples"])
           
                # form-submit version
                with st.form("form_01", border=False):
                    eps_value = st.select_slider(label = "DBSCAN eps", options = eps_options, value=ss['upar']["dbscan_eps"])
                    min_samples_value = st.select_slider(label = "DBSCAN min samples", options = min_samples_options, value=ss['upar']["dbscan_min_samples"])
                    # Every form must have a submit button.
                    submitted = st.form_submit_button("Recompute DBSCAN", type = "primary")
                    if submitted:
                        ss['upar']["dbscan_eps"] = eps_value
                        ss['upar']["dbscan_min_samples"] = min_samples_value
                        st.rerun()
                        
        _ = st.checkbox("Hide dots not assigned to a cluster",
                        key = "k_suppress", value=ss['upar']["exclude_non_assigned"], on_change=update_ss, args=["k_suppress", "exclude_non_assigned"])

        #-------------------------------------------
        # computational block 2 (st-cached)
        clusters_pred = perform_dbscan_clusterin(X = ss['dapar']['X_dimred'] , eps = ss['upar']['dbscan_eps'], min_samples = ss['upar']['dbscan_min_samples']) 
        num_unasigned = (clusters_pred == -1).sum()
        num_asigned = len(clusters_pred) - num_unasigned
        num_clusters = len(np.unique(clusters_pred))
        ss['dapar']['clusters_pred_str'] = np.array([format(a, '03d') for a in clusters_pred])
        df_pred = make_sorted_df(cat = ss['dapar']['clusters_pred_str'], cat_name = 'Predicted cluster', X = ss['dapar']['X2D'])
        fig02 = make_scatter_plot(df = df_pred, cat_name = 'Predicted cluster', 
                                title = "Predicted clusters      (Scatterplot from 2D UMAP mapping)", height = 800, width = 1000, b_margin = 300,
                                exclude_non_assigned = ss['upar']["exclude_non_assigned"])
        #-------------------------------------------

        # show plots 
        st.plotly_chart(fig02, use_container_width=False, theme=None)  
    
    with cols[1]:
        with st.container(border=True): 
            c01, c02, _ = st.columns([0.1, 0.1, 0.3])    
            c01.metric("N images assigned ", num_asigned)
            c02.metric("N clusters", num_clusters)

        clu_id_list = np.unique(ss['dapar']['clusters_pred_str'])
        clu_id_list = clu_id_list[clu_id_list != '-01'] # remove -01 from options
        clu_selected = st.segmented_control(label = "Select a cluster ID", options = clu_id_list, selection_mode="single", key = "k_img_clu",
                                        default = clu_id_list[-1], label_visibility="visible")        
        st.text("Cluster content preview (max 120 random images from cluster)")
        # select all images in a given cluster 
        sel = ss['dapar']['clusters_pred_str'] == clu_selected
        images_in_cluster = ss['dapar']['im_filenames'][sel]
        # take a smaller subsample 
        images_in_cluster_sample = select_random_image_subset(images_in_cluster, max_n_images = 120)
        # st.write(images_in_cluster_sample)

        c01, c02, c03 = st.columns([0.3, 0.3, 0.6]) 
        do_save = c01.button("Add to image pool", type="primary")
        do_reset = c02.button("Reset image pool", type="primary")
        if do_save: 
            ss['dapar']['image_pool'].extend(images_in_cluster)
            ss['dapar']['image_pool'] = list(set(ss['dapar']['image_pool'])) # remove dups 
            ss['dapar']['image_pool'].sort()
        if do_reset: 
            ss['dapar']['image_pool'] = list()
        with c03:  
            st.info('Images in pool: ' + str(len(ss['dapar']['image_pool'])))      
        display_mini_images_by_file(sel_imgs = images_in_cluster_sample)

         

      





    # display_bar_plot
    with cols[2]:
        with st.container(border=True): 
            st.text("Origin of spectrograms in cluster")
            display_bar_plot(images_in_cluster_sample)







gc.collect()

        
     
        
  

   


