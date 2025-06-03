
#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

import streamlit as st
from streamlit import session_state as ss
import numpy as np
import gc
from utils import display_mini_images_by_file, update_label_data_frame
gc.collect()

if len(ss['dapar']['image_pool']) == 0:
    st.text("The image pool is empty, got to Analyse to assign clusters to the pool")
else:
    st.text("Spectrograms of all clusters that were pooled")
    st.text("Check the consistency of spectrograms and if OK, save as a preliminary label")
    do_add_to_df = st.button("Add to preliminary labels", type="primary")
    if do_add_to_df: 
        varname = update_label_data_frame(cluster_list = ss['dapar']['image_pool'])
        # reset the list to empty
        ss['dapar']['image_pool'] = list()
        st.text('Saved in data frame "Preliminary labels" as ' + varname)

    display_mini_images_by_file(sel_imgs = np.array(ss['dapar']['image_pool']),  num_cols = 10)



