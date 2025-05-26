#--------------------             
# Author : Serge Zaugg
# Description : Computation steps defined as functions here
#--------------------

import os
import streamlit as st
from streamlit import session_state as ss
import pandas as pd 
import plotly.express as px
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
# from sklearn.cluster import OPTICS


def update_ss(kname, ssname):
    """
    description : helper callback fun to implement statefull apps
    kname : key name of widget
    ssname : key name of variable in session state (ss)
    """
    ss["upar"][ssname] = ss[kname]      

def get_short_class_name(a):
    """ a : a string"""
    return("-".join(a.split("-")[0:2]))








@st.cache_data
def dim_reduction_for_2D_plot(X, n_neigh, n_components = 2):
    """
    UMAP dim reduction for 2D plot 
    """


    # make a smaller random subsample for training
    rand_index = np.random.choice(np.arange(len(X)), size=3000, replace=False)    
    X_small = X[rand_index]
    print('X.shape', X.shape)
    print('X_small.shape', X_small.shape)

    reducer = umap.UMAP(
        n_neighbors = n_neigh, 
        n_components = n_components, 
        metric = 'euclidean',
        n_jobs = -1
        )
    
    reducer.fit(X_small, ensure_all_finite=True)

    X2D_trans = reducer.transform(X)

    scaler = StandardScaler()
    X2D_scaled = scaler.fit_transform(X2D_trans)
    return(X2D_scaled)















@st.cache_data
def dim_reduction_for_clustering(X, n_neigh, n_dims_red, skip_umap = False):
    """
    UMAP dim reduction for clustering
    """
    scaler = StandardScaler()
    if skip_umap == True:
        X_scaled = scaler.fit_transform(X)
        return(X_scaled)
    else:    

        # make a smaller random subsample for training
        rand_index = np.random.choice(np.arange(len(X)), size=3000, replace=False)    
        X_small = X[rand_index]
        print('X.shape', X.shape)
        print('X_small.shape', X_small.shape)


        reducer = umap.UMAP(
            n_neighbors = n_neigh, 
            n_components = n_dims_red, 
            metric = 'euclidean',
            n_jobs = -1
            )
        
        # X_trans = reducer.fit_transform(X, ensure_all_finite=True)
        reducer.fit(X_small, ensure_all_finite=True)
        X_trans = reducer.transform(X)



        X_scaled = scaler.fit_transform(X_trans)
        return(X_scaled)




















@st.cache_data
def perform_dbscan_clusterin(X, eps, min_samples):
    """ 
    """
    clu = DBSCAN(eps = eps, min_samples = min_samples, metric='euclidean', n_jobs = 4) 
    # clu = OPTICS(min_samples=min_samples, max_eps=eps, metric='euclidean', n_jobs=-1)
    clusters_pred = clu.fit_predict(X)
    return(clusters_pred)

@st.cache_data
def make_sorted_df(cat, cat_name, X):
    if X.shape[1] == 2:
        df = pd.DataFrame({ cat_name : cat, 'Dim-1' : X[:,0] , 'Dim-2' : X[:,1]})
    if X.shape[1] == 3:
        df = pd.DataFrame({ cat_name : cat, 'Dim-1' : X[:,0] , 'Dim-2' : X[:,1], 'Dim-3' : X[:,2]})
    df = df.sort_values(by=cat_name)
    return(df)

@st.cache_data
def make_scatter_plot(df, cat_name, title = "not set", height = 900, width = 1000, b_margin=300):
    fig = px.scatter(
        data_frame = df,
        x = 'Dim-1',
        y = 'Dim-2',
        color = cat_name,
        template='plotly_dark',
        height= height,
        width = width,
        color_discrete_sequence = px.colors.qualitative.Light24,
        title = title,
        # labels = {'aaa', ""}
        )
    _ = fig.update_layout(margin=dict(t=30, b=b_margin, l=15, r=15))
    _ = fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0.0))
    _ = fig.update_layout(showlegend=True,legend_title=None)
    _ = fig.update_layout(yaxis_title=None)
    _ = fig.update_layout(xaxis_title=None)
    _ = fig.update_xaxes(showline=True, linewidth=2, linecolor='white', mirror=True)
    _ = fig.update_yaxes(showline=True, linewidth=2, linecolor='white', mirror=True)
    return(fig)

@st.cache_data
def make_scatter_3d_plot(df, cat_name, title = "not set"):
    fig = px.scatter_3d(
        data_frame = df,
        x = 'Dim-1',
        y = 'Dim-2',
        z = 'Dim-3',
        color = cat_name,
        template='plotly_dark',
        height=1200,
        width =1000,
        color_discrete_sequence = px.colors.qualitative.Light24,
        title = title,
        )
    _ = fig.update_layout(margin=dict(t=30, b=300, l=15, r=15))
    _ = fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0.0))
    _ = fig.update_layout(showlegend=True,legend_title=None)
    _ = fig.update_xaxes(showline=True, linewidth=2, linecolor='white', mirror=True)
    _ = fig.update_yaxes(showline=True, linewidth=2, linecolor='white', mirror=True)
    _ = fig.update_scenes(aspectmode='cube')
    _ = fig.update_traces(marker_size = 5)
    return(fig)

@st.fragment
def show_cluster_details(conf_table):
    c03, c04 = st.columns([0.5, 0.5])
    with c03:
        st.text("Select a cluster ID")
        clu_id_list = conf_table.index  
        clu_selected = st.segmented_control(label = "Select a cluster ID", options = clu_id_list, selection_mode="single", default = clu_id_list[0], label_visibility="hidden")                
        clu_row = pd.DataFrame(conf_table.loc[clu_selected]  )
        clu_summary = (clu_row.loc[(clu_row!=0).any(axis=1)]).reset_index() 
        clu_summary.columns = ['Ground truth', 'Count']
        clu_summary = clu_summary.sort_values(by='Count', ascending = False)     
    with c04:
        st.text("Cluster content")
        st.dataframe(clu_summary, hide_index = True, height =800, use_container_width = True)


@st.fragment
def display_mini_images_by_file(sel_imgs):
    num_cols = 15
    grid = st.columns(num_cols)
    col = 0
    for ii, im_filname in enumerate(sel_imgs):
        try:
            with grid[col]:
                st.image(os.path.join(ss['dapar']['imgs_path'], im_filname), use_container_width=True)
            col += 1
            if ii % num_cols == (num_cols-1):
                col = 0
            print('OK')    
        except:
            print('shit') 


@st.fragment
def display_imags_from_cluster():
    clu_id_list = np.unique(ss['dapar']['clusters_pred_str'])
    clu_selected = st.segmented_control(label = "Select a cluster ID", options = clu_id_list, selection_mode="single", key = "k_img_clu",
                                        default = clu_id_list[-1], label_visibility="hidden")                
    # select all images in a given cluster 
    sel = ss['dapar']['clusters_pred_str'] == clu_selected
    images_in_cluster = ss['dapar']['im_filenames'][sel]
    # images_in_clu_tru = ss['dapar']['clusters_true'][sel]
    # take a smaller subsample 
    rand_index = np.random.choice(np.arange(len(images_in_cluster)), size=min(60, len(images_in_cluster)), replace=False)    
    images_in_cluster_sample = images_in_cluster[rand_index]
    # images_clu_tru_sample = images_in_clu_tru[rand_index]
    display_mini_images_by_file(sel_imgs = images_in_cluster_sample)





