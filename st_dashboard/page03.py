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
from utils import data_source_format, set_default_eps
gc.collect()

# first, user selects a data source 
c0x, c1x, _  = st.columns([0.20, 0.20, 0.10])
with c0x:  
    with st.container(border=True, height = 170): 
        st.subheader("Select data source")  
        data_source_options = ["spectrogram-clustering-01", "spectrogram-clustering-parus-major", "xc-data-02-corvidae"]
        kgl_datasource = st.segmented_control("(Changing data source will erase the image pool)", 
                                            options = data_source_options, format_func=data_source_format, default=ss['upar']["datsou"], label_visibility="visible")                 
with c1x:    
    with st.container(border=True, height = 170): 
        st.text("Recomended: Take one SAEC and one IDNN model")  
        st.text("SAEC: features from bottle-neck layer of spectrogram auto-encoders")  
        st.text("IDNN: features from inner layer of image dnns (currently, ResNet and MaxVit)")  

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
    c00, c01, _  = st.columns([0.20, 0.20, 0.10])
    with c00:   
        with st.container(border=True, height = 500): 
            st.subheader("Select 1st features") 
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
                
                submitted_2 = st.form_submit_button("Activate features dataset", type = "primary")  
                if submitted_2:      
                    npz_finame = [a for a in ss['dapar']['li_npz'] if ndim_sel in a and selected_model in a]
                    npz_finame = npz_finame[0]
                    npzfile_full_path = os.path.join(ss['dapar']['feat_path'], npz_finame)
                    npzfile = np.load(npzfile_full_path, allow_pickle=True)
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

                    # temp 
                    ss['dapar']['X_dimred_conc'] = ss['dapar']['X_dimred']
                    set_default_eps()
                    # ugly temp construct to re-initialise second feat
                    ss['dapar']['dataset_name_b']  = 'empty'
                    ss['dapar']['im_filenames_b']  = np.array([]),
                    ss['dapar']['X_dimred_b'] = np.array([]),

                    # load meta-data 
                    path_meat = os.path.join(ss['dapar']['feat_path'], 'downloaded_data_meta.pkl')
                    ss['dapar']['df_meta'] = pd.read_pickle(path_meat)
                    st.rerun() # to update sidebar!
        # concatenate a second feature array (Experimental)
        with c01: 
            with st.container(border=True, height = 500): 
                st.subheader("Select 2nd features") 

                # exclude already selected model
                mod_sel_short_b = [a for a in mod_sel_short if a != selected_model]
                selected_model_b = st.radio("Model used to extracted features", options = mod_sel_short_b, index=0, key = "spec01")
                with st.form("form02_b", border=False):
                    submitted_3 = st.form_submit_button("Add features dataset", type = "primary")  
                    if submitted_3:      
                        npz_finame = [a for a in ss['dapar']['li_npz'] if ndim_sel in a and selected_model_b in a]
                        npz_finame = npz_finame[0]
                        npzfile_full_path = os.path.join(ss['dapar']['feat_path'], npz_finame)
                        npzfile = np.load(npzfile_full_path, allow_pickle=True)
                        X_red = npzfile['X_red']
                        N     = npzfile['N']
                        # put selected data into ss
                        ss['dapar']['dataset_name_b'] = npz_finame 
                        ss['dapar']['im_filenames_b']  = N
                        ss['dapar']['X_dimred_b'] = X_red.astype(np.float16)
                        # Sort arrays using these indices
                        sorted_indices = np.argsort(ss['dapar']['im_filenames_b'])
                        ss['dapar']['im_filenames_b'] = ss['dapar']['im_filenames_b'][sorted_indices]
                        ss['dapar']['X_dimred_b']     = ss['dapar']['X_dimred_b'][sorted_indices]
                        # concatenate feature spaces 
                        ss['dapar']['X_dimred_conc'] = np.concatenate([ss['dapar']['X_dimred'], ss['dapar']['X_dimred_b']], axis = 1)
                        set_default_eps()
                        
                        st.rerun() # to update sidebar!
                st.text("Dim reduction value taken from left")

    c0x, _, _  = st.columns([0.40, 0.05, 0.05])
    with c0x:  
        with st.container(border=True):   
            st.write('Dim of feature array: ', ss['dapar']['X_dimred_conc'].shape)           
            st.page_link("page02.py", label="Go to analysis")   

# st.write(pd.DataFrame({"aa": ss['dapar']['im_filenames'],   "bb": ss['dapar']['im_filenames_b']})  ) 
           
gc.collect() 

