#--------------------             
# Author : Serge Zaugg
# Description : Utility functions used by other scripts
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

c00, c01  = st.columns([0.1, 0.18])

# First, get data into ss
if ss['dapar']['feat_path'] == 'empty' :
    st.text("Preparing data ...")
    kgl_ds = "sezaugg/" + 'spectrogram-clustering-01' 
    kgl_path = kagglehub.dataset_download(kgl_ds, force_download = False) # get local path where downloaded
    ss['dapar']['feat_path'] = kgl_path
    ss['dapar']['imgs_path'] = os.path.join(ss['dapar']['feat_path'], 'xc_spectrograms', 'xc_spectrograms')
    ss['dapar']['li_npz'] = [a for a in os.listdir(ss['dapar']['feat_path']) if ('.npz' in a) and (('dimred_16_features' in a))]
    st.rerun()
# Then, choose a dataset
else :
    with c00:
        with st.container(border=True, height = 200):   
            with st.form("form01", border=False):
                npz_finame = st.selectbox("Select data with extracted features", options = ss['dapar']['li_npz'])
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
                    del(X_red, X_2D, N)
                    st.rerun() # to update sidebar - 

gc.collect()
        





