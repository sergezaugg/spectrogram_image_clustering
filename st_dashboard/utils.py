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
# from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, OPTICS, k_means
import numpy as np


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
def perform_dbscan_clusterin(X, eps, min_samples):
    """ 
    """
    clu = DBSCAN(eps = eps, min_samples = min_samples, metric='euclidean', n_jobs = -1) 
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
def make_scatter_plot(df, cat_name, title = "not set", height = 900, width = 1000, b_margin=300, exclude_non_assigned = False):

    if exclude_non_assigned:
        print(df.shape)
        df = df[df[cat_name] != '-01']
        colsec = px.colors.qualitative.Light24
        print(df.shape)
    else:
        colsec = ["#777777"] + px.colors.qualitative.Light24


    fig = px.scatter(
        data_frame = df,
        x = 'Dim-1',
        y = 'Dim-2',
        color = cat_name,
        template='plotly_dark',
        height= height,
        width = width,
        color_discrete_sequence = colsec,
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
    _ = fig.update_layout(paper_bgcolor="#000000", plot_bgcolor="#000000") 
    _ = fig.update_layout(xaxis_title_font_size=15)
    _ = fig.update_layout(yaxis_title_font_size=15)
    _ = fig.update_layout(xaxis_tickfont_size=15)
    _ = fig.update_layout(legend_font_size=15)
    return(fig)

@st.fragment
def display_mini_images_by_file(sel_imgs):
    num_cols = 5
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






