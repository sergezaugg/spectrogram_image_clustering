
#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

import streamlit as st
from streamlit import session_state as ss
import numpy as np
import gc
from utils import display_mini_images_by_file, update_label_data_frame, select_random_image_subset
gc.collect()


@st.cache_data
def convert_for_download(df):
    return df.to_csv().encode("utf-8")

c1,c2=st.columns([0.3, 0.5])
c1.subheader("Spectrograms of all clusters that were pooled")
c2.text("")
# c2.text("Check the consistency of spectrograms and then add them to labels data frame")

if len(ss['dapar']['image_pool']) == 0:
    c1.info("Image pool is empty, got to 'Cluster spectrograms' to assign clusters to the pool")
else:   
    do_add_to_df = st.button("Add to labels data frame", type="primary")
    if do_add_to_df: 
        varname = update_label_data_frame(cluster_list = ss['dapar']['image_pool'])
        # reset the list to empty
        ss['dapar']['image_pool'] = list()
        st.text('Saved in data frame "Preliminary labels" as ' + varname)
    # safeguard to prevent plotting of thousans of images 
    images_in_pool_sample = select_random_image_subset(np.array(ss['dapar']['image_pool']), max_n_images = 300)  
    display_mini_images_by_file(sel_imgs = images_in_pool_sample ,  num_cols = 10)

st.divider()

c1,c2=st.columns([0.3, 0.5])
c1.subheader("Data frame with clustering-based labels")
c2.text("")
if (len(ss['dapar']['df_prelim_labels'])==0):
    c1.info("Data frame is empty")
else:    
    # c2.text("N rows: " +  str(ss['dapar']['df_prelim_labels'].shape[0]) + "   N cols: " +  str(ss['dapar']['df_prelim_labels'].shape[1]))
    c1.download_button(label="Download CSV", type="primary", file_name="clusterin_based_labels.csv",mime="text/csv",icon=":material/download:", 
                            data=convert_for_download(ss['dapar']['df_prelim_labels'])) 
    c2.write("Please download spectrogram images directly here:")
    c2.write("https://www.kaggle.com/datasets/sezaugg/" + ss['dapar']['kgl_datasource'])

    st.dataframe(ss['dapar']['df_prelim_labels'], width=1000)





