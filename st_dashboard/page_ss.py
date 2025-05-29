#--------------------             
# Author : Serge Zaugg
# Description : display info on session state directly on frontend
#--------------------

import streamlit as st
import os
from streamlit import session_state as ss
import gc
gc.collect()

st.text("Paths:")
st.write(ss['dapar']['feat_path'])
st.write(ss['dapar']['imgs_path'])

# st.write(ss['dapar']['df_meta'].head())

st.divider()
st.text("Active data:")
st.write('Nb images:', len(os.listdir(ss['dapar']['imgs_path'])))
st.write('Dataset name: ', ss['dapar']['dataset_name'])
st.write('Filename array:', ss['dapar']['im_filenames'].shape, ss['dapar']['im_filenames'].dtype)
st.write('Dim reduced for plots:', ss['dapar']['X2D'].shape, ss['dapar']['X2D'].dtype)
st.write('Dim reduced for clustering:', ss['dapar']['X_dimred'].shape, ss['dapar']['X_dimred'].dtype)


