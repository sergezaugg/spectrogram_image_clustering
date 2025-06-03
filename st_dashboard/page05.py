
#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

import os
import streamlit as st
from streamlit import session_state as ss
import numpy as np
import pandas as pd
import gc
from utils import display_mini_images_by_file, update_label_data_frame

gc.collect()

do_add_to_df = st.button("Add to dataframe", type="primary")
if do_add_to_df: 
    update_label_data_frame(cluster_list = ss['dapar']['saved_clusterd'])
    # reset the list to empty
    ss['dapar']['saved_clusterd'] = list()

display_mini_images_by_file(sel_imgs = np.array(ss['dapar']['saved_clusterd']),  num_cols = 10)



