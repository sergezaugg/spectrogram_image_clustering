#--------------------             
# Author : Serge Zaugg
# Description : page to select a dataset to be loaded into session state
#--------------------

import os
import streamlit as st
from streamlit import session_state as ss
import numpy as np
import pandas as pd
import kagglehub
import gc
from sklearn.model_selection import train_test_split
gc.collect()

c00, c01, c02  = st.columns([0.1, 0.10, 0.10])

# First, get data into ss
if ss['dapar']['feat_path'] == 'empty' :
    st.text("Preparing data ...")
    kgl_ds = "sezaugg/" + 'spectrogram-clustering-01' 
    kgl_path = kagglehub.dataset_download(kgl_ds, force_download = False) # get local path where downloaded
    ss['dapar']['feat_path'] = kgl_path
    ss['dapar']['imgs_path'] = os.path.join(ss['dapar']['feat_path'], 'xc_spectrograms', 'xc_spectrograms')
    ss['dapar']['li_npz'] = [a for a in os.listdir(ss['dapar']['feat_path']) if ('.npz' in a) and (('dimred_4' in a) or ('dimred_8' in a) or ('dimred_16' in a))]
    ss['dapar']['li_npz'].sort()
    # load meta data 
    path_meat = os.path.join(ss['dapar']['feat_path'], 'downloaded_data_meta.pkl')
    ss['dapar']['df_meta'] = pd.read_pickle(path_meat)
    st.rerun()
# Then, choose a dataset
else :
    with c00:
        with st.container(border=True):  
            # first pre-select datasets based on the dim reduction 
            ndim_sel = st.radio("Select level of UMAP dim reduction", options = ['dimred_4', 'dimred_8', 'dimred_16'], index=2, format_func=lambda x: x.split("_")[1])
            npz_sel = [a for a in ss['dapar']['li_npz'] if ndim_sel in a]
            # pre select good default for the selected dim
            if ndim_sel == 'dimred_4':
                ss['upar']['dbscan_eps'] =  0.20
            if ndim_sel == 'dimred_8':
                ss['upar']['dbscan_eps'] =  0.36
            if ndim_sel == 'dimred_16':
                ss['upar']['dbscan_eps'] =  0.46

            npz_sel.sort()
            with st.form("form01", border=False):
                # seconf selec DNN model used for fex
                npz_finame = st.radio("Select model used to extracted features", options = npz_sel, index=3, format_func=lambda x: "_".join(x.split("_")[3:]) )
                submitted_1 = st.form_submit_button("Activate dataset", type = "primary")  
                if submitted_1:
                    npzfile_full_path = os.path.join(ss['dapar']['feat_path'], npz_finame)
                    npzfile = np.load(npzfile_full_path)
                    # take a subset of data (else public streamlit.app will crash) 
                    X_red, _, X_2D, _, N, _, = train_test_split(npzfile['X_red'], npzfile['X_2D'], npzfile['N'], train_size=10000, random_state=6666, shuffle=True)
                    # put selected data into ss
                    ss['dapar']['dataset_name']  = npz_finame 
                    ss['dapar']['X2D']           = X_2D.astype(np.float16)
                    ss['dapar']['X_dimred']      = X_red.astype(np.float16)
                    ss['dapar']['im_filenames']  = N
                    del(X_red, X_2D, N, npzfile)
                    st.rerun() # to update sidebar!

gc.collect() 



