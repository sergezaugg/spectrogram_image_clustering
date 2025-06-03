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

def data_source_format(s):
    """ helper finction for st.segmented_control below"""
    if s == "spectrogram-clustering-01":
        return("Crows & tits SW-Eur")
    elif s == "spectrogram-clustering-parus-major":
        return("Parus major Eur")
    else:
        return("error")

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

# @st.cache_data
@st.fragment
def make_scatter_plot(df, cat_name, title = "not set", height = 900, width = 1000, b_margin=300, exclude_non_assigned = False):

    # heuristics to exclude outliers from plot
    std_1 = df['Dim-1'].std()
    qua_1_lo = np.quantile(df['Dim-1'], q=0.005) - 0.5*std_1
    qua_1_up = np.quantile(df['Dim-1'], q=0.995) + 0.5*std_1
    std_2 = df['Dim-1'].std()
    qua_2_lo = np.quantile(df['Dim-2'], q=0.005) - 0.5*std_2
    qua_2_up = np.quantile(df['Dim-2'], q=0.995) + 0.5*std_2
  
   
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
    _ = fig.update_layout(xaxis=dict(showgrid=False),yaxis=dict(showgrid=False))
    _ = fig.update_xaxes(range=[qua_1_lo, qua_1_up])
    _ = fig.update_yaxes(range=[qua_2_lo, qua_2_up])

    return(fig)

@st.fragment
def display_mini_images_by_file(sel_imgs, num_cols = 5):
    grid = st.columns(num_cols)
    col = 0
    for ii, im_filname in enumerate(sel_imgs):
        try:
            with grid[col]:
                st.image(os.path.join(ss['dapar']['imgs_path'], im_filname), use_container_width=True)
                # st.image(os.path.join(ss['dapar']['imgs_path'], im_filname), use_container_width=True, caption = im_filname)
            col += 1
            if ii % num_cols == (num_cols-1):
                col = 0
            print('OK')    
        except:
            print('shit') 


@st.fragment
def display_bar_plot(x):
    """
    Arguments : 
    files_counts : A 1D numpy array
    """
    xx = pd.Series(x).str.slice(start=0, stop=8)
    xx = xx.value_counts().reset_index()
    xx.columns = ["File name", "Mini-image counts per XC-file"]
    st.bar_chart(xx, x = "File name", y = "Mini-image counts per XC-file", 
                 horizontal = True, use_container_width = True, color = "#cccccc",
                 y_label = '')


@st.cache_data
def select_random_image_subset(images_in_cluster, max_n_images = 60):
    rand_index = np.random.choice(np.arange(len(images_in_cluster)), size=min(max_n_images, len(images_in_cluster)), replace=False)    
    return(images_in_cluster[rand_index])


@st.cache_data
def update_label_data_frame(cluster_list):
    """
    Description : get list of files in the pool of clusters and make a dummy-coded class-indicator dataframe 
    cluster_list (list) : list of filenames that were added to the pool 
    """
    # if necessary, initialize df and set the list of files as first column 
    if ss['dapar']['df_prelim_labels'].shape[0] == 0:
        ss['dapar']['df_prelim_labels'] = pd.DataFrame({'filename' : ss['dapar']['im_filenames']})
        ss['dapar']['df_prelim_labels'] = ss['dapar']['df_prelim_labels'].sort_values(by = 'filename')
    # get sequentially numberd names for the label indicator variable 
    ii = ss['dapar']['df_prelim_labels'].shape[1]
    var_name = 'class'+ str(ii)
    # create a temp df with label indicators 
    dft = pd.DataFrame({
        'filename' : ss['dapar']['im_filenames'], 
        var_name : np.zeros(ss['dapar']['im_filenames'].shape[0])
        })
    dft[var_name] = dft[var_name].mask(cond = dft['filename'].isin(cluster_list), other=1)
    # merge the new label indicators into the main df (in ss) 
    ss['dapar']['df_prelim_labels'] = ss['dapar']['df_prelim_labels'].merge(right= dft, how='left', on='filename')
    


