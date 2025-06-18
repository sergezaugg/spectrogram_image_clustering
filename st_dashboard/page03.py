#--------------------             
# Author : Serge Zaugg
# Description : select a dataset to be loaded into session state
#--------------------

import os
import streamlit as st
from streamlit import session_state as ss
import numpy as np
import pandas as pd
import kagglehub
import gc
from sklearn.model_selection import train_test_split
from utils import data_source_format, set_default_eps
gc.collect()

c00, c01, _  = st.columns([0.20, 0.10, 0.10])
# first, user selects a data source 
with c00:    
    with st.container(border=True): 
        st.subheader("Select data source")  
        data_source_options = ["spectrogram-clustering-01", "spectrogram-clustering-parus-major"]
        kgl_datasource = st.segmented_control("(Changing data source will erase the image pool)", 
                                              options = data_source_options, format_func=data_source_format, default=ss['upar']["datsou"], label_visibility="visible")
    # (download) and put data source data into ss
    if ss['dapar']['feat_path'] == 'empty' or kgl_datasource != ss['dapar']['kgl_datasource']:
        st.text("Preparing data ...")
        ss['dapar']['kgl_datasource'] = kgl_datasource
        kgl_ds = "sezaugg/" + kgl_datasource 
        kgl_path = kagglehub.dataset_download(kgl_ds, force_download = False) # get local path where downloaded
        ss['dapar']['feat_path'] = kgl_path
        ss['dapar']['imgs_path'] = os.path.join(ss['dapar']['feat_path'], 'xc_spectrograms', 'xc_spectrograms')
        ss['dapar']['li_npz'] = [a for a in os.listdir(ss['dapar']['feat_path']) if ('.npz' in a) and (('dimred_' in a))]
        ss['dapar']['li_npz'].sort()
        # prelim labels dfs not supported over multiple data sources, thus must be re-initialized
        ss['dapar']['image_pool'] = list()
        ss['dapar']['df_prelim_labels'] = np.array([])
        st.rerun()
    # Then, choose a dataset
    else :
        with st.container(border=True): 
            st.subheader("Select features used for clustering") 
            # select a model type
            mod_sel_short = list(set(["_".join(x.split("_")[4:])  for x in ss['dapar']['li_npz']]))
            selected_model = st.radio("Model used to extracted features", options = mod_sel_short, index=0)
            npz_sub_finame = [a for a in ss['dapar']['li_npz'] if selected_model in a]
            # get dimred options that are available for this model    
            dimred_options = ["_".join(x.split("_")[0:2])  for x in npz_sub_finame]
            # dirty trick to get dimred_options sorted logically for user
            dimred_full_sorted = ['dimred_2', 'dimred_4', 'dimred_8', 'dimred_16', 'dimred_32', 'dimred_64']
            dimred_options = [a for a in dimred_full_sorted if a in dimred_options]
            #  select dim reduction 
            with st.form("form02", border=False):
                ndim_sel = st.radio("Level of UMAP dim reduction", options = dimred_options, index=2, format_func=lambda x: x.split("_")[1])
                set_default_eps(ndim_sel)
                submitted_2 = st.form_submit_button("Activate features dataset", type = "primary")  
                if submitted_2:      
                    npz_finame = [a for a in ss['dapar']['li_npz'] if ndim_sel in a and selected_model in a]
                    npz_finame = npz_finame[0]
                    npzfile_full_path = os.path.join(ss['dapar']['feat_path'], npz_finame)
                    npzfile = np.load(npzfile_full_path)
                    # take a subset of data (else public streamlit.app will crash) 
                    # X_red, _, X_2D, _, N, _, = train_test_split(npzfile['X_red'], npzfile['X_2D'], npzfile['N'], train_size=0.999999, random_state=6666, shuffle=False)
                    X_red = npzfile['X_red'] 
                    X_2D  = npzfile['X_2D']
                    N     = npzfile['N']
                    
                    # put selected data into ss
                    ss['dapar']['dataset_name']  = npz_finame 
                    ss['upar']["datsou"] = kgl_datasource
                    ss['dapar']['X2D']           = X_2D.astype(np.float16)
                    ss['dapar']['X_dimred']      = X_red.astype(np.float16)
                    ss['dapar']['im_filenames']  = N
                    del(X_red, X_2D, N, npzfile)

                    # Get the sorting indices for the first array
                    sorted_indices = np.argsort(ss['dapar']['im_filenames'])
                    # Sort  arrays using these indices
                    ss['dapar']['X2D']          = ss['dapar']['X2D'][sorted_indices]
                    ss['dapar']['X_dimred']     = ss['dapar']['X_dimred'][sorted_indices]
                    ss['dapar']['im_filenames'] = ss['dapar']['im_filenames'][sorted_indices]

                    # load meta-data 
                    path_meat = os.path.join(ss['dapar']['feat_path'], 'downloaded_data_meta.pkl')
                    ss['dapar']['df_meta'] = pd.read_pickle(path_meat)
                    st.rerun() # to update sidebar!
        

     

        # Experimental
        with st.container(border=True): 
            selected_model_b = st.radio("Secondary Model used to extracted features", options = mod_sel_short, index=0, key = "spec01")
            with st.form("form02_b", border=False):
                submitted_3 = st.form_submit_button("Activate secondary features dataset", type = "primary")  
                if submitted_3:      
                    npz_finame = [a for a in ss['dapar']['li_npz'] if ndim_sel in a and selected_model_b in a]
                    npz_finame = npz_finame[0]
                    npzfile_full_path = os.path.join(ss['dapar']['feat_path'], npz_finame)
                    npzfile = np.load(npzfile_full_path)
                    X_red = npzfile['X_red']
                    N     = npzfile['N']
                    ss['dapar']['im_filenames_b']  = N
                    ss['dapar']['X_dimred_b'] = X_red.astype(np.float16)
                    # Sort arrays using these indices
                    sorted_indices = np.argsort(ss['dapar']['im_filenames_b'])
                    ss['dapar']['im_filenames_b'] = ss['dapar']['im_filenames_b'][sorted_indices]
                    ss['dapar']['X_dimred_b']     = ss['dapar']['X_dimred_b'][sorted_indices]
                    # concatenate featue spaces 
                    ss['dapar']['X_dimred'] = np.concatenate([ss['dapar']['X_dimred'], ss['dapar']['X_dimred_b']], axis = 1)
                    st.rerun() # to update sidebar!

        with st.container(border=True):              
                    st.page_link("page02.py", label="Go to analysis")   

                    
with c01:    
    with st.container(border=True): 
        st.text("Recomended models:")  
        st.text("ResNet50_layer3.5.conv3.npz")  
        st.text("saec_20250617_150956.npz")  


        

st.write(ss['dapar']['X_dimred'].shape)
# st.write(pd.DataFrame({"aa": ss['dapar']['im_filenames'],   "bb": ss['dapar']['im_filenames_b']})  )            







gc.collect() 

