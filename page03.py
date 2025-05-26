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
# from utils import get_short_class_name
gc.collect()




c00, c01  = st.columns([0.1, 0.18])

# First, get data into ss
# download the data from kaggle (https://www.kaggle.com/datasets/sezaugg/food-classification-features-v01)
if ss['dapar']['feat_path'] == 'empty' :
    st.text("Preparing data ...")
    
    kgl_ds = "sezaugg/" + 'spectrogra-images-experiment' 

    kgl_path = kagglehub.dataset_download(kgl_ds, force_download = False) # get local path where downloaded
    ss['dapar']['feat_path'] = kgl_path
    ss['dapar']['imgs_path'] = os.path.join(ss['dapar']['feat_path'], 'xc_spectrograms', 'xc_spectrograms')
    di = dict()
    li_npz = [a for a in os.listdir(ss['dapar']['feat_path']) if ('.npz' in a) and (('Spectro_from_' in a) or ('unwrapped' in a))]
    for npz_finame in li_npz:
        npzfile_full_path = os.path.join(ss['dapar']['feat_path'], npz_finame)
        npzfile = np.load(npzfile_full_path)
        # take a subset of data (else public streamlit.app will crash) 
        X_train, X_test, N_train, N_test, = train_test_split(npzfile['X'], npzfile['N'], train_size=3000, random_state=6666, shuffle=True)
        di[npz_finame] = {'X' : X_train , 'im_filenames' : N_train}
    ss['dapar']['npdata'] = di
    gc.collect()
    st.rerun()
# Then, choose a dataset
else :
    with c00:
        with st.container(border=True, height = 200):   
            with st.form("form01", border=False):
                npz_finame = st.selectbox("Select data with extracted features", options = ss['dapar']['npdata'].keys())
                submitted_1 = st.form_submit_button("Activate dataset", type = "primary")   
                if submitted_1:
                    # copy selected data into dedicated dict 
                    ss['dapar']['dataset_name']   = npz_finame 
                    ss['dapar']['X']              = ss['dapar']['npdata'][npz_finame]['X']  
                    ss['dapar']['im_filenames']  = ss['dapar']['npdata'][npz_finame]['im_filenames'] 
                    st.rerun()  # mainly to update sidebar   
        # st.page_link("page02.py", label="Go to analysis")                
        

gc.collect()
        