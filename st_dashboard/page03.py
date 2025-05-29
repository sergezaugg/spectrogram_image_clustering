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

c00, c01, c02  = st.columns([0.1, 0.10, 0.10])

# First, get data into ss
if ss['dapar']['feat_path'] == 'empty' :
    st.text("Preparing data ...")
    kgl_ds = "sezaugg/" + 'spectrogram-clustering-01' 
    kgl_path = kagglehub.dataset_download(kgl_ds, force_download = False) # get local path where downloaded
    ss['dapar']['feat_path'] = kgl_path
    ss['dapar']['imgs_path'] = os.path.join(ss['dapar']['feat_path'], 'xc_spectrograms', 'xc_spectrograms')
    ss['dapar']['li_npz'] = [a for a in os.listdir(ss['dapar']['feat_path']) if ('.npz' in a) and (('dimred_4' in a) or ('dimred_16' in a))]
    ss['dapar']['li_npz'].sort()
    # load meta data 
    path_meat = os.path.join(ss['dapar']['feat_path'], 'downloaded_data_meta.pkl')
    # st.write(path_meat)  
    ss['dapar']['df_meta'] = pd.read_pickle(path_meat)
    st.rerun()
# Then, choose a dataset
else :
    with c00:
        with st.container(border=True):   
            with st.form("form01", border=False):
                # npz_finame = st.selectbox("Select data with extracted features", options = ss['dapar']['li_npz'])
                npz_finame = st.radio("Select data with extracted features", options = ss['dapar']['li_npz'])

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


    with c01:
        with st.container(border=True) : 
            st.markdown('''                
                #### features_DenseNet121_denseblock3.npz
                * Feature layer: :orange[features.denseblock3]
                * Input resized image: :orange[(, 3, 224, 224)]
                * Feature out of net: :orange[(, 1024, 14, 14)]
                * After average pool along freq: :orange[(, 1024, 4, 14)]
                * After average pool along time: :orange[(, 1024, 4)]
                * After reshape: :orange[(, 4096)]
                            
                #### features_MaxVit_T_blocks.3.npz
                * Feature layer: :orange[blocks.3.layers.1.layers.MBconv.layers.conv_c]
                * Input resized image:  :orange[( , 3, 224, 224)]
                * Feature out of net:  :orange[( , 512, 7, 7)]
                * After average pool along freq:  :orange[( , 512, 7, 7)]
                * After average pool along time:  :orange[( , 512, 7)]
                * After reshape:  :orange[( , 3584)]
                ''')
    with c02:
        with st.container(border=True) : 
            st.markdown('''      
                #### features_ResNet50_layer1.npz
                * Feature layer: :orange[layer1.2.conv3]
                * Input resized image: :orange[(16, 3, 224, 224)]
                * Feature out of net: :orange[(16, 256, 56, 56)]
                * After average pool along freq: :orange[(16, 256, 14, 56)]
                * After average pool along time: :orange[(16, 256, 14)]
                * After reshape: :orange[(16, 3584)]
                        
                #### features_ResNet50_layer2.npz
                * Feature layer: :orange[layer2.3.conv3]
                * Input resized image: :orange[(, 3, 224, 224)]
                * Feature out of net: :orange[(, 512, 28, 28)]
                * After average pool along freq: :orange[(, 512, 7, 28)]
                * After average pool along time: :orange[(, 512, 7)]
                * After reshape: :orange[(, 3584)]
                               
                #### features_ResNet50_layer3.npz
                * Feature layer: :orange[layer3.5.conv3] 
                * Input resized image: :orange[(, 3, 224, 224)] 
                * Feature out of net: :orange[(, 1024, 14, 14)] 
                * After average pool along freq: :orange[(, 1024, 4, 14)] 
                * After average pool along time: :orange[(, 1024, 4)] 
                * After reshape: :orange[(, 4096)]       
                ''')            
    
gc.collect() 



